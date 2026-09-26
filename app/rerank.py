import time
from contextvars import ContextVar

# Re-use the stage ContextVar from app.retrieve
from app.retrieve import stage  # noqa: F401 imported for side-effect of ContextVar setup
from sentence_transformers import CrossEncoder

_ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)


def rerank(query: str, candidates: list[dict], final_k: int) -> list[dict]:
    """
    Cross-Encoder reranking over a list of candidate passages.

    Uses a single batched predict call over all (query, passage) pairs,
    then sorts candidates by the reranker's relevance scores.

    Args:
        query: Original user query string
        candidates: List of passage dicts with at least a 'text' key
        final_k: Number of top results to return

    Returns:
        Reranked list of passage dicts, truncated to final_k
    """
    if not candidates:
        return []

    t0 = time.perf_counter()

    # ONE batched call: all query-passage pairs at once
    pairs = [(query, c['text']) for c in candidates]
    scores = _ce.predict(pairs, batch_size=16)

    # Sort candidates by rerank score descending
    ranked = [c for _, c in sorted(zip(scores, candidates),
                                   key=lambda t: t[0], reverse=True)]

    stage('rerank', t0)
    return ranked[:final_k]


def find_rescue_case() -> str | None:
    """
    Find a query where the reranker promotes a passage from RRF rank 8+ to top 5.

    Returns a description string of the rescue case, or None if not found.
    """
    import asyncio
    from app.retrieve import retrieve_passages, hybrid_search
    from app.rerank import rerank

    # Run retrieval with RRF
    # Use a set of queries that are likely to have rank-8 rescues
    queries = [
        "What is the penalty for failing to notify a personal data breach?",
        "When is consent NOT required for processing?",
        "What does Section 43A require?",
    ]

    for query in queries:
        # Retrieve with RRF (Config B)
        passages = asyncio.run(retrieve_passages(
            query, top_k=20, use_bm25=True, use_rerank=False
        ))

        # Check RRF order
        before = [c['chunk_id'] for c in passages]

        # Rerank to top 5
        after = asyncio.run(rerank(query, passages, 5))
        after_ids = [c['chunk_id'] for c in after]

        # Check if any passage moved from rank > 5 to top 5
        for cid in after_ids:
            if cid in before:
                moved = before.index(cid) + 1  # 1-indexed rank
                if moved > 5:
                    return (f"Query: '{query}'\\n"
                            f"RESCUED: chunk {cid} went from rank {moved} -> top 5\\n"
                            f"RRF order: {before}\\n"
                            f"After rerank: {after_ids}")

    return None