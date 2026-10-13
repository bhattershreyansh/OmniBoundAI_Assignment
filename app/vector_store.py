import asyncio
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from app.config import settings
from app.schemas import Chunk


class VectorStore:
    def __init__(self, api_key: Optional[str] = None, index_name: Optional[str] = None):
        self.api_key = api_key or settings.PINECONE_API_KEY
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(self.index_name)

    async def upsert_chunks(self, tenant_id: str, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        """
        Upserts chunk vectors and metadata into Pinecone under the tenant namespace.
        Vector ID format: {doc_id}-{chunk_index} guarantees idempotency.
        """
        if not chunks:
            return 0

        doc_ids = {chunk.doc_id for chunk in chunks}
        for doc_id in doc_ids:
            await asyncio.to_thread(
                self.index.delete,
                namespace=tenant_id,
                filter={
                    "tenant_id": {"$eq": tenant_id},
                    "doc_id": {"$eq": doc_id}
                }
            )

        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": {
                    "tenant_id": tenant_id,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "source": chunk.source,
                    "brand": chunk.brand,
                    "text": chunk.text,
                }
            })

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            await asyncio.to_thread(
                self.index.upsert,
                vectors=batch,
                namespace=tenant_id
            )

        return len(chunks)


    async def query(self, tenant_id: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Queries top_k closest chunks for a tenant.
        Strict isolation: namespace=tenant_id AND metadata filter on tenant_id.
        Handles both dict-style and object-style responses across Pinecone SDK versions.
        """
        response = await asyncio.to_thread(
            self.index.query,
            vector=query_embedding,
            top_k=top_k,
            namespace=tenant_id,
            include_metadata=True,
            filter={"tenant_id": {"$eq": tenant_id}}
        )

        # Support both response["matches"] and response.matches
        matches = response.matches if hasattr(response, "matches") else response.get("matches", [])
        results = []
        for match in matches:
            # Support both match["metadata"] and match.metadata
            metadata = match.metadata if hasattr(match, "metadata") else match.get("metadata", {})
            mid = match.id if hasattr(match, "id") else match.get("id")
            score = match.score if hasattr(match, "score") else match.get("score")

            results.append({
                "chunk_id": mid,
                "score": score,
                "text": metadata.get("text", "") if isinstance(metadata, dict) else getattr(metadata, "text", ""),
                "source": metadata.get("source", "") if isinstance(metadata, dict) else getattr(metadata, "source", ""),
                "brand": metadata.get("brand", "") if isinstance(metadata, dict) else getattr(metadata, "brand", ""),
                "doc_id": metadata.get("doc_id", "") if isinstance(metadata, dict) else getattr(metadata, "doc_id", ""),
                "chunk_index": metadata.get("chunk_index", 0) if isinstance(metadata, dict) else getattr(metadata, "chunk_index", 0),
            })
        return results

    async def has_data(self, tenant_id: str) -> bool:
        """
        Checks whether the specified tenant has any ingested vectors.
        Does NOT swallow exceptions: real Pinecone outages surface as errors.
        Handles both dict and object structures across SDK versions.
        """
        stats = await asyncio.to_thread(self.index.describe_index_stats)
        namespaces = stats.namespaces if hasattr(stats, "namespaces") else stats.get("namespaces", {})
        
        if namespaces and tenant_id in namespaces:
            ns_info = namespaces[tenant_id]
            vector_count = ns_info.vector_count if hasattr(ns_info, "vector_count") else ns_info.get("vector_count", 0)
            return vector_count > 0
        return False

    async def ping(self) -> bool:
        """Verifies Pinecone index reachability."""
        try:
            await asyncio.to_thread(self.index.describe_index_stats)
            return True
        except Exception:
            return False
