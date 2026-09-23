"""
search/retrieve.py

Retrieves candidate passages from Qdrant vector database for RAG synthesis.
"""

import asyncio
from qdrant_client import AsyncQdrantClient, QdrantClient, models as qm

from app.config import settings
from ingest.embed import embed

# Initialize clients using settings
async_qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
sync_qdrant_client = QdrantClient(url=settings.qdrant_url)


async def retrieve_passages(question: str, top_k: int = 5) -> list[dict]:
    """
    Async retrieval: Embeds question, queries Qdrant, and returns 
    formatted passage objects for synthesis.
    """
    # 1. Generate query embedding (offload CPU work to threadpool)
    query_vectors = await asyncio.to_thread(
        embed, [question], model_name=settings.embed_model
    )
    query_vector = query_vectors[0]

    # 2. Search Qdrant
    response = await async_qdrant_client.query_points(
        collection_name=settings.collection_name,
        query=query_vector.tolist(),
        limit=top_k,
        query_filter=qm.Filter(
            must=[
                qm.FieldCondition(
                    key="chunker",
                    match=qm.MatchValue(value=settings.chunker),
                )
            ]
        ),
    )

    # 3. Format hits into passage dictionary list
    passages = []
    for point in response.points:
        payload = point.payload or {}
        passages.append({
            "chunk_id": str(point.id),
            "doc_id": payload.get("doc_id", "unknown"),
            "section": payload.get("section", "N/A"),
            "text": payload.get("text", ""),
            "score": round(float(point.score), 4),
        })

    return passages


def retrieve_passages_sync(question: str, top_k: int = 5) -> list[dict]:
    """
    Synchronous fallback for simple scripts/testing.
    """
    query_vector = embed([question], model_name=settings.embed_model)[0]

    response = sync_qdrant_client.query_points(
        collection_name=settings.collection_name,
        query=query_vector.tolist(),
        limit=top_k,
        query_filter=qm.Filter(
            must=[
                qm.FieldCondition(
                    key="chunker",
                    match=qm.MatchValue(value=settings.chunker),
                )
            ]
        ),
    )

    passages = []
    for point in response.points:
        payload = point.payload or {}
        passages.append({
            "chunk_id": str(point.id),
            "doc_id": payload.get("doc_id", "unknown"),
            "section": payload.get("section", "N/A"),
            "text": payload.get("text", ""),
            "score": round(float(point.score), 4),
        })

    return passages