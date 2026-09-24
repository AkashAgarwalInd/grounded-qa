"""
app/run_rag.py
"""

import asyncio
from app.config import settings
from app.retrieve import dense_search
from app.synthesise import synthesise, GroundedAnswer


async def main():
    question = "What are the duties of a data fiduciary regarding security safeguards?"

    # Config toggles from settings - can be overridden at runtime
    use_bm25 = getattr(settings, "use_bm25", False)
    use_rerank = getattr(settings, "use_rerank", False)

    passages = await dense_search(question, k=settings.top_k)

    # Apply config switches
    if use_bm25:
        # BM25 hybrid search - placeholder, full integration in Phase 2
        print("[INFO] BM25 hybrid search enabled (Config B)")
        # TODO: Integrate BM25 search here

    if use_rerank:
        # Cross-Encoder reranking - placeholder, full integration in Phase 4
        print("[INFO] Cross-Encoder reranking enabled (Config C)")
        # TODO: Integrate reranker here

    result, grounding_errors = synthesise(question=question, passages=passages)

    print(f"Sufficient Context: {result.sufficient_context}")
    print(f"Answer:\n{result.answer}\n")
    print("Citations:")
    for c in result.citations:
        print(f" - [{c.doc_id} | {c.section}]: \"{c.quote}\"")

    if grounding_errors:
        print("🚨 GROUNDING / HALLUCINATION ERRORS DETECTED:")
        for err in grounding_errors:
            print(f"  - {err}")
    else:
        print("✅ VERIFIED: All citations resolve verbatim to source chunks.")


if __name__ == "__main__":
    asyncio.run(main())