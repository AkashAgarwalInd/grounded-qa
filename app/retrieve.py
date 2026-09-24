from qdrant_client import QdrantClient, models as qm
from app.config import settings
from ingest.embed import embed

# Initialize Qdrant client from central settings
client = QdrantClient(url=settings.qdrant_url)


async def dense_search(query: str, k: int = 5) -> list[dict]:
    """
    Config A: Dense-only vector retrieval using BAAI/bge-small-en-v1.5.
    No BM25 sparse matching, no reranker, no LLM query re-writer.

    Args:
        query: User query string
        k: Number of results to retrieve

    Returns:
        List of result dictionaries with chunk_id, score, doc_id, section, text, char_start
    """
    # 1. Embed query with the SAME model and L2 normalization
    query_vector = embed([query], model_name=settings.embed_model)[0]

    # 2. Query Qdrant
    response = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector.tolist(),
        limit=k,
        query_filter=qm.Filter(
            must=[
                qm.FieldCondition(
                    key="chunker",
                    match=qm.MatchValue(value=settings.chunker),
                )
            ]
        ),
    )

    # 3. Format hits with payload metadata for citations
    results = []
    for point in response.points:
        payload = point.payload or {}
        results.append({
            "chunk_id": point.id,
            "score": round(float(point.score), 4),
            "doc_id": payload.get("doc_id", "unknown"),
            "section": payload.get("section", "N/A"),
            "text": payload.get("text", ""),
            "char_start": payload.get("char_start", 0),
        })

    return results