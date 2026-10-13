import asyncio
import logging
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

from app.config import settings
from app.llm import LLMService
from app.vector_store import VectorStore

logger = logging.getLogger("omnibound.graph")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AgentState(TypedDict):
    tenant_id: str
    original_question: str
    current_question: str
    tenant_has_data: bool
    retrieved_chunks: List[Dict[str, Any]]
    relevant_chunks: List[Dict[str, Any]]
    retry_count: int
    answer: str
    found_in_corpus: bool
    citations: List[Dict[str, Any]]
    trace: List[Dict[str, str]]


def build_graph(llm: LLMService, vector_store: VectorStore):
    """
    Constructs the LangGraph StateGraph with retrieval, grading, retry, and answer decisions:
    
    START -> retrieve -> (tenant has data?)
                          ├─ no  -> not_found -> END
                          └─ yes -> grade -> (route)
                                             ├─ relevant found          -> answer -> END
                                             ├─ none & retry_count == 0 -> rewrite -> retrieve
                                             └─ none & already retried  -> not_found -> END
    """

    async def retrieve_node(state: AgentState) -> Dict[str, Any]:
        tenant_id = state["tenant_id"]
        query = state["current_question"]
        logger.info(f"🔵 [Node: retrieve] Tenant: '{tenant_id}' | Query: '{query}'")

        has_tenant_data = await vector_store.has_data(tenant_id)
        if not has_tenant_data:
            logger.info("   ↳ No data exists for this tenant.")
            trace_entry = {
                "node": "retrieve",
                "detail": "no data for tenant"
            }
            return {
                "tenant_has_data": False,
                "retrieved_chunks": [],
                "relevant_chunks": [],
                "trace": state["trace"] + [trace_entry]
            }

        embedding = await llm.get_single_embedding(query)
        chunks = await vector_store.query(
            tenant_id=tenant_id,
            query_embedding=embedding,
            top_k=settings.TOP_K
        )
        logger.info(f"   ↳ Retrieved {len(chunks)} chunks from vector store.")

        trace_entry = {
            "node": "retrieve",
            "detail": f"{len(chunks)} chunks retrieved"
        }
        return {
            "tenant_has_data": True,
            "retrieved_chunks": chunks,
            "trace": state["trace"] + [trace_entry]
        }

    async def grade_node(state: AgentState) -> Dict[str, Any]:
        query = state["original_question"]
        chunks = state["retrieved_chunks"]
        logger.info(f"🟡 [Node: grade] Grading {len(chunks)} chunks against: '{query}'")

        if not chunks:
            logger.info("   ↳ 0 chunks available to grade.")
            trace_entry = {
                "node": "grade",
                "detail": "0 relevant, 0 rejected"
            }
            return {
                "relevant_chunks": [],
                "trace": state["trace"] + [trace_entry]
            }

        tasks = [llm.grade_chunk(query, chunk["text"]) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        relevant = [chunk for chunk, is_rel in zip(chunks, results) if is_rel]
        rejected_count = len(chunks) - len(relevant)
        logger.info(f"   ↳ Results: {len(relevant)} relevant, {rejected_count} rejected.")

        trace_entry = {
            "node": "grade",
            "detail": f"{len(relevant)} relevant, {rejected_count} rejected"
        }
        return {
            "relevant_chunks": relevant,
            "trace": state["trace"] + [trace_entry]
        }

    async def rewrite_node(state: AgentState) -> Dict[str, Any]:
        original_q = state["original_question"]
        logger.info(f"🟠 [Node: rewrite] No relevant chunks found. Rewriting original query: '{original_q}'")
        rewritten = await llm.rewrite_question(original_q)
        new_retry_count = state["retry_count"] + 1
        logger.info(f"   ↳ Rewritten query: '{rewritten}' (retry_count: {new_retry_count})")

        trace_entry = {
            "node": "rewrite",
            "detail": f"query rewritten to '{rewritten}'"
        }
        return {
            "current_question": rewritten,
            "retry_count": new_retry_count,
            "trace": state["trace"] + [trace_entry]
        }

    async def answer_node(state: AgentState) -> Dict[str, Any]:
        relevant = state["relevant_chunks"]
        query = state["original_question"]
        logger.info(f"🟢 [Node: answer] Generating final answer using {len(relevant)} verified chunks.")

        chunk_texts = [c["text"] for c in relevant]
        answer_text = await llm.generate_answer(query, chunk_texts)

        citations = []
        for c in relevant:
            citations.append({
                "source": c.get("source", ""),
                "brand": c.get("brand", ""),
                "chunk_text": c.get("text", "")
            })

        trace_entry = {
            "node": "answer",
            "detail": f"answer generated from {len(relevant)} chunks"
        }
        return {
            "answer": answer_text,
            "found_in_corpus": True,
            "citations": citations,
            "trace": state["trace"] + [trace_entry]
        }

    async def not_found_node(state: AgentState) -> Dict[str, Any]:
        logger.info("🔴 [Node: not_found] Search retried and still no relevant chunks. Setting found_in_corpus=False.")
        trace_entry = {
            "node": "not_found",
            "detail": "no relevant chunks found after search"
        }
        return {
            "answer": "The corpus does not cover this question.",
            "found_in_corpus": False,
            "citations": [],
            "trace": state["trace"] + [trace_entry]
        }

    def route_decision(state: AgentState) -> str:
        """
        Pure deterministic routing function (conditional edge).
        """
        if len(state["relevant_chunks"]) > 0:
            logger.info("🔀 [Edge: route] Decision -> 'answer' (relevant chunks exist)")
            return "answer"
        elif state["retry_count"] == 0:
            logger.info("🔀 [Edge: route] Decision -> 'rewrite' (0 relevant, retry_count == 0)")
            return "rewrite"
        else:
            logger.info("🔀 [Edge: route] Decision -> 'not_found' (0 relevant, already retried)")
            return "not_found"

    def route_after_retrieve(state: AgentState) -> str:
        if not state["tenant_has_data"]:
            logger.info("🔀 [Edge: retrieve] Decision -> 'not_found' (tenant has no data)")
            return "not_found"
        logger.info("🔀 [Edge: retrieve] Decision -> 'grade' (tenant has data)")
        return "grade"

    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("not_found", not_found_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "grade": "grade",
            "not_found": "not_found"
        }
    )

    workflow.add_conditional_edges(
        "grade",
        route_decision,
        {
            "answer": "answer",
            "rewrite": "rewrite",
            "not_found": "not_found"
        }
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("answer", END)
    workflow.add_edge("not_found", END)

    return workflow.compile()
