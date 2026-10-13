import asyncio
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Path, status
from fastapi.responses import JSONResponse

from app.schemas import (
    IngestRequest,
    IngestResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    Citation,
    TraceEntry
)
from app.errors import UnreachableURLError, LLMTimeoutError
from app.llm import LLMService
from app.vector_store import VectorStore
from app.ingestion import process_document_to_chunks
from app.graph import build_graph


llm_service: LLMService = None
vector_store: VectorStore = None
qa_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_service, vector_store, qa_graph
    llm_service = LLMService()
    vector_store = VectorStore()
    qa_graph = build_graph(llm_service, vector_store)
    yield


app = FastAPI(
    title="OmniBound Brand Q&A Agent",
    description="Multi-tenant RAG service with LangGraph for Brand Intelligence",
    version="1.0.0",
    lifespan=lifespan
)


TENANT_ID_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"


@app.exception_handler(UnreachableURLError)
async def handle_unreachable_url(request, exc: UnreachableURLError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "unreachable_url", "detail": str(exc), "url": exc.url}
    )


@app.exception_handler(LLMTimeoutError)
async def handle_llm_timeout(request, exc: LLMTimeoutError):
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"error": "llm_timeout", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "detail": "Unexpected server error"}
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Checks reachability of vector database and LLM.
    Returns 200 with vector_db and llm statuses.
    """
    vector_ok = await vector_store.ping() if vector_store else False
    llm_ok = await llm_service.ping() if llm_service else False

    status_str = "ok" if (vector_ok and llm_ok) else "degraded"
    return HealthResponse(
        status=status_str,
        vector_db="up" if vector_ok else "down",
        llm="up" if llm_ok else "down"
    )


@app.post("/tenants/{tenant_id}/ingest", response_model=IngestResponse)
async def ingest_documents(
    payload: IngestRequest,
    tenant_id: str = Path(..., pattern=TENANT_ID_PATTERN, description="Tenant identifier")
):
    """
    Ingests raw text or fetched URLs concurrently, chunks them, embeds them, and upserts into Pinecone.
    Idempotent: Re-ingesting existing docs overwrites without growing vector counts.
    """
    all_chunks = []
    unique_doc_ids = set()

    async with httpx.AsyncClient() as client:
        # Concurrent processing across all documents using asyncio.gather
        tasks = [
            process_document_to_chunks(tenant_id, doc, client)
            for doc in payload.documents
        ]
        results = await asyncio.gather(*tasks)

        for doc_id, chunks in results:
            unique_doc_ids.add(doc_id)
            all_chunks.extend(chunks)

    # Step 2: Generate embeddings in batch
    chunk_texts = [c.text for c in all_chunks]
    embeddings = await llm_service.get_embeddings(chunk_texts)

    # Step 3: Upsert into Pinecone under tenant namespace
    stored_count = await vector_store.upsert_chunks(tenant_id, all_chunks, embeddings)

    return IngestResponse(
        documents_stored=len(unique_doc_ids),
        chunks_stored=stored_count
    )


@app.post("/tenants/{tenant_id}/ask", response_model=AskResponse)
async def ask_question(
    payload: AskRequest,
    tenant_id: str = Path(..., pattern=TENANT_ID_PATTERN, description="Tenant identifier")
):
    """
    Executes LangGraph RAG workflow:
    retrieve -> grade -> route (answer | rewrite -> retrieve | not_found)
    """
    initial_state = {
        "tenant_id": tenant_id,
        "original_question": payload.question,
        "current_question": payload.question,
        "tenant_has_data": False,
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "retry_count": 0,
        "answer": "",
        "found_in_corpus": False,
        "citations": [],
        "trace": []
    }

    final_state = await qa_graph.ainvoke(initial_state)

    citations = [
        Citation(
            source=c.get("source", ""),
            brand=c.get("brand", ""),
            chunk_text=c.get("chunk_text", "")
        )
        for c in final_state.get("citations", [])
    ]

    trace = [
        TraceEntry(node=t.get("node", ""), detail=t.get("detail", ""))
        for t in final_state.get("trace", [])
    ]

    return AskResponse(
        answer=final_state.get("answer", ""),
        found_in_corpus=final_state.get("found_in_corpus", False),
        citations=citations,
        trace=trace
    )
