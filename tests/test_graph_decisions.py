import pytest
from app.schemas import Chunk
from app.graph import build_graph
from tests.fakes import FakeLLM, FakeVectorStore


@pytest.mark.asyncio
async def test_decision_logic_pathways():
    """
    Test 2 (B.6): Decision logic in LangGraph StateGraph.
    - Path 1: Relevant chunks found -> answers directly with citations.
    - Path 2: Irrelevant chunks -> rewrites question, retries.
    - Path 3: Irrelevant chunks after retry -> not_found, found_in_corpus == False.
    """
    tenant_id = "test-tenant"
    chunks = [
        Chunk(
            chunk_id="crm-0",
            text="Zoho CRM Standard plan starts at $14 per user monthly.",
            tenant_id=tenant_id,
            doc_id="crm",
            chunk_index=0,
            source="zoho.com",
            brand="Zoho"
        )
    ]

    # --- Scenario 1: Relevant chunks found -> Answer ---
    fake_llm_yes = FakeLLM(should_grade_yes=True)
    fake_store = FakeVectorStore()
    embeddings = await fake_llm_yes.get_embeddings([c.text for c in chunks])
    await fake_store.upsert_chunks(tenant_id, chunks, embeddings)

    graph_yes = build_graph(fake_llm_yes, fake_store)
    state_1 = {
        "tenant_id": tenant_id,
        "original_question": "How much is Zoho Standard?",
        "current_question": "How much is Zoho Standard?",
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "retry_count": 0,
        "answer": "",
        "found_in_corpus": False,
        "citations": [],
        "trace": []
    }
    res_1 = await graph_yes.ainvoke(state_1)
    assert res_1["found_in_corpus"] is True
    assert fake_llm_yes.answer_call_count == 1
    assert fake_llm_yes.rewrite_call_count == 0
    nodes_executed_1 = [t["node"] for t in res_1["trace"]]
    assert "retrieve" in nodes_executed_1
    assert "grade" in nodes_executed_1
    assert "answer" in nodes_executed_1

    # --- Scenario 2: Chunks rejected -> Rewrites -> Retries -> Fails -> Not Found ---
    fake_llm_no = FakeLLM(should_grade_yes=False)
    graph_no = build_graph(fake_llm_no, fake_store)
    state_2 = {
        "tenant_id": tenant_id,
        "original_question": "What is the capital of France?",
        "current_question": "What is the capital of France?",
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "retry_count": 0,
        "answer": "",
        "found_in_corpus": False,
        "citations": [],
        "trace": []
    }
    res_2 = await graph_no.ainvoke(state_2)
    assert res_2["found_in_corpus"] is False
    assert res_2["citations"] == []
    assert fake_llm_no.rewrite_call_count == 1
    assert fake_llm_no.answer_call_count == 0

    nodes_executed_2 = [t["node"] for t in res_2["trace"]]
    # Trace must show: retrieve -> grade -> rewrite -> retrieve -> grade -> not_found
    assert nodes_executed_2.count("retrieve") == 2
    assert nodes_executed_2.count("grade") == 2
    assert "rewrite" in nodes_executed_2
    assert "not_found" in nodes_executed_2
