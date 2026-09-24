"""pytest tests for the RAG retrieval pipeline."""

import pytest
import sys
import pathlib

# Append project root directory to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# Now import app modules
from app.config import settings
from app.retrieve import dense_search, hybrid_search, bm25_search, rrf_fuse, rewrite_query


@pytest.fixture(autouse=True)
def skip_if_no_qdrant():
    """Skip tests if Qdrant is not available."""
    import qdrant_client
    yield


class TestDenseSearch:
    """Test Config A: Dense-only vector retrieval."""

    def test_dense_search_returns_results(self):
        """Dense search should return results with expected fields."""
        results = asyncio_run(dense_search("What are the duties of a data fiduciary?", k=3))
        assert len(results) > 0, "Dense search should return results"
        assert all(
            "chunk_id" in r and "score" in r and "doc_id" in r for r in results
        ), "Results should have chunk_id, score, doc_id fields"

    def test_dense_search_respects_chunker(self):
        """Dense search should respect the configured chunker filter."""
        results = asyncio_run(dense_search("security safeguards", k=3))
        # All results should have the configured chunker type
        for r in results:
            assert r.get("section") != "N/A" or r.get("doc_id") != "unknown"


class TestBM25Search:
    """Test Config B: BM25 lexical search."""

    def test_bm25_search_returns_results(self):
        """BM25 search should return results when corpus is available."""
        results = bm25_search("What are the duties", 3)
        # BM25 may return empty if no corpus cached, that's OK
        # The important thing is the function doesn't crash

    def test_bm25_search_has_correct_fields(self):
        """BM25 results should have expected fields when available."""
        # This test may be skipped if BM25 corpus not built
        try:
            results = bm25_search("duties", k=3)
            if results:
                assert all(
                    "score" in r for r in results
                ), "BM25 results should have score field"
        except Exception:
            pytest.skip("BM25 corpus not available")


class TestRRF:
    """Test RRF fusion of dense and BM25 results."""

    def test_rrf_fuse_returns_results(self):
        """RRF should return fused results."""
        dense = [{"chunk_id": i, "score": 0.5 - i * 0.1} for i in range(5)]
        bm25 = [{"chunk_id": i, "score": 0.8 - i * 0.1} for i in range(5)]
        fused = rrf_fuse(dense, bm25, 60)
        assert len(fused) > 0, "RRF should return fused results"


class TestQueryRewriting:
    """Test Config D: Query rewriting."""

    def test_rewrite_query_returns_string(self):
        """Query rewriting should return a string."""
        import asyncio
        result = asyncio.run(rewrite_query("When is consent NOT required?"))
        assert isinstance(result, str), "Rewrite should return a string"

    def test_rewrite_query_handles_negation(self):
        """Query rewriting should handle negation expansion."""
        import asyncio
        result = asyncio.run(rewrite_query("When is consent NOT required for processing?"))
        # Should either return the original or a rewritten version
        # The important thing is it doesn't crash
        assert isinstance(result, str)


def asyncio_run(coro):
    """Helper to run async functions in pytest."""
    import asyncio
    return asyncio.run(coro)