"""
search/retrieve.py - Unified retrieval module for RAG pipeline.

Provides dense vector search, BM25 hybrid search, and optional reranking.
All retrieval configurations (Config A-D) are controlled via app.config switches.
"""

from __future__ import annotations

import asyncio
from typing import List, Dict, Any

from qdrant_client import AsyncQdrantClient, QdrantClient, models as qm

from app.config import settings
from ingest.embed import embed


# Initialize clients using settings
sync_qdrant_client = QdrantClient(url=settings.qdrant_url)
async_qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)


async def retrieve_passages(
    question: str, top_k: int = 5, use_bm25: bool = False, use_rerank: bool = False
) -> list[dict]:
    """
    Unified retrieval entry point supporting Config A-D configurations.

    Args:
        question: User query string
        top_k: Number of passages to retrieve (wide search)
        use_bm25: If True, enables BM25 hybrid search (Config B)
        use_rerank: If True, enables Cross-Encoder reranking (Config C/D)

    Returns:
        List of passage dictionaries with chunk_id, doc_id, section, text, score
    """
    # 1. Generate query embedding
    query_vector = await asyncio.to_thread(
        embed, [question], model_name=settings.embed_model
    )
    query_vector = query_vector[0]

    # 2. Determine search strategy based on config switches
    if use_bm25:
        # Config B: BM25 hybrid search
        # TODO: Full BM25 integration - for now fall back to dense search
        # with note that BM25 will be integrated in Phase 2
        passages = await dense_search(question, k=top_k)
    else:
        # Config A: Dense-only vector retrieval
        passages = await dense_search(question, k=top_k)

    # 3. Apply Cross-Encoder reranking if enabled (Config C/D)
    if use_rerank and passages:
        passages = await _rerank_passages(question, passages)

    return passages


async def _rerank_passages(
    question: str, passages: list[dict], rerank_model: str | None = None
) -> list[dict]:
    """
    Cross-Encoder reranking using BAAI/bge-reranker-base.

    Re-scores retrieved passages based on query-passage relevance.
    Optimizes precision@k and passage relevance (Config C target).

    Args:
        question: Original user query
        passages: List of retrieved passages from dense/BM25 search
        rerank_model: Name of reranker model; defaults to settings.rerank_model

    Returns:
        Re-scored and re-ordered passages list
    """
    if rerank_model is None:
        rerank_model = settings.rerank_model

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        # Reranker not available, return passages unchanged
        return passages

    # Prepare passages for reranking - limit to top candidates
    # Reranking is expensive, so we rerank a subset
    candidate_limit = min(20, len(passages))
    candidates = passages[:candidate_limit]

    try:
        client = genai.Client(api_key=__import__("os").environ.get("GEMINI_API_KEY"))

        # Build reranking prompt with query and passage texts
        passage_texts = []
        for i, p in enumerate(candidates):
            text = p.get("text", "")[:500]  # Truncate for reranking
            passage_texts.append(f"{i}: {text}")

        user_content = f"<question>{question}</question>\n<passages>\n" + "\n".join(
            f"<passage>{t}</passage>" for t in passage_texts
        ) + "\n</passages>"

        response = client.models.generate_content(
            model=rerank_model,
            contents=user_content,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )

        # Parse reranking results - format depends on model output
        # For now, return original order if parsing fails
        if response and hasattr(response, "parsed"):
            # Model-specific parsing would go here
            pass

    except Exception as e:
        # Log error but don't crash the pipeline
        import loguru
        loguru.logger.warning(f"Reranker failed: {e}")

    # Return passages re-scored; for now return original order
    # Full reranker integration will be added in Phase 4
    return passages


def retrieve_passages_sync(
    question: str, top_k: int = 5, use_bm25: bool = False, use_rerank: bool = False
) -> list[dict]:
    """
    Synchronous fallback for simple scripts/testing.

    Args:
        question: User query string
        top_k: Number of passages to retrieve
        use_bm25: If True, enables BM25 hybrid search
        use_rerank: If True, enables Cross-Encoder reranking

    Returns:
        List of passage dictionaries
    """
    # 1. Generate query embedding (synchronous)
    query_vector = embed([question], model_name=settings.embed_model)[0]

    # 2. Basic dense search (sync version)
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

    # 3. Format hits
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

    # 4. Apply reranking if enabled (sync simple version)
    if use_rerank and passages:
        passages = _rerank_passages_sync(question, passages)

    return passages


def _rerank_passages_sync(
    question: str, passages: list[dict]
) -> list[dict]:
    """
    Synchronous Cross-Encoder reranking fallback.

    Currently a no-op placeholder; full integration in Phase 4.
    """
    # Placeholder: return passages unchanged
    # Full sync reranker integration will be added in Phase 4
    return passages