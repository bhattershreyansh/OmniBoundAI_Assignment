import pytest
from app.schemas import Chunk
from app.graph import build_graph
from tests.fakes import FakeLLM, FakeVectorStore


@pytest.mark.asyncio
async def test_tenant_isolation():
    """
    Test 1 (B.6): Tenant isolation.
    Data ingested for tenant A is NEVER returned when tenant B asks.
    Tenant A retrieves data; tenant B receives empty / found_in_corpus == False.
    """
    fake_llm = FakeLLM(should_grade_yes=True)
    fake_store = FakeVectorStore()
    graph = build_graph(fake_llm, fake_store)

    # Ingest document for Tenant A
    tenant_a = "tenant-alpha"
    chunks_a = [
        Chunk(
            chunk_id="doc1-0",
            text="HubSpot CRM offers free contact management.",
            tenant_id=tenant_a,
            doc_id="doc1",
            chunk_index=0,
            source="hubspot.com",
            brand="HubSpot"
        )
    ]
    embeddings = await fake_llm.get_embeddings(["HubSpot CRM offers free contact management."])
    await fake_store.upsert_chunks(tenant_a, chunks_a, embeddings)

    # 1. Ask as Tenant A -> Should find data and return answer
    state_a = {
        "tenant_id": tenant_a,
        "original_question": "What is HubSpot's pricing?",
        "current_question": "What is HubSpot's pricing?",
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "retry_count": 0,
        "answer": "",
        "found_in_corpus": False,
        "citations": [],
        "trace": []
    }
    result_a = await graph.ainvoke(state_a)
    assert result_a["found_in_corpus"] is True
    assert len(result_a["citations"]) > 0
    assert result_a["citations"][0]["brand"] == "HubSpot"

    # 2. Ask as Tenant B (which ingested nothing) -> Should NOT find data
    tenant_b = "tenant-beta"
    state_b = {
        "tenant_id": tenant_b,
        "original_question": "What is HubSpot's pricing?",
        "current_question": "What is HubSpot's pricing?",
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "retry_count": 0,
        "answer": "",
        "found_in_corpus": False,
        "citations": [],
        "trace": []
    }
    result_b = await graph.ainvoke(state_b)
    assert result_b["found_in_corpus"] is False
    assert result_b["citations"] == []
