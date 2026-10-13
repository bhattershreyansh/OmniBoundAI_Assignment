from typing import List, Dict, Any, Optional
from app.schemas import Chunk


class FakeLLM:
    def __init__(self, should_grade_yes: bool = True):
        self.should_grade_yes = should_grade_yes
        self.grade_call_count = 0
        self.rewrite_call_count = 0
        self.answer_call_count = 0

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Deterministic 1536-dimensional mock embedding
        return [[0.1] * 1536 for _ in texts]

    async def get_single_embedding(self, text: str) -> List[float]:
        return [0.1] * 1536

    async def grade_chunk(self, question: str, chunk_text: str) -> bool:
        self.grade_call_count += 1
        return self.should_grade_yes

    async def rewrite_question(self, question: str) -> str:
        self.rewrite_call_count += 1
        return f"{question} (clarified)"

    async def generate_answer(self, question: str, relevant_chunks_text: List[str]) -> str:
        self.answer_call_count += 1
        return "This is a synthesized test answer based on relevant chunks."

    async def ping(self) -> bool:
        return True


class FakeVectorStore:
    def __init__(self):
        # In-memory tenant isolation storage: {tenant_id: {chunk_id: chunk_dict}}
        self.storage: Dict[str, Dict[str, Dict[str, Any]]] = {}

    async def upsert_chunks(self, tenant_id: str, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        if tenant_id not in self.storage:
            self.storage[tenant_id] = {}

        # Remove existing chunks for docs being re-ingested to emulate fresh overwrite
        new_doc_ids = {chunk.doc_id for chunk in chunks}
        self.storage[tenant_id] = {
            cid: c for cid, c in self.storage[tenant_id].items()
            if c["doc_id"] not in new_doc_ids
        }

        for chunk in chunks:
            self.storage[tenant_id][chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "brand": chunk.brand,
                "tenant_id": tenant_id,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index
            }
        return len(chunks)


    async def query(self, tenant_id: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        tenant_data = self.storage.get(tenant_id, {})
        # Return at most top_k items for this tenant only
        results = list(tenant_data.values())[:top_k]
        return results

    async def has_data(self, tenant_id: str) -> bool:
        return tenant_id in self.storage and len(self.storage[tenant_id]) > 0

    async def ping(self) -> bool:
        return True
