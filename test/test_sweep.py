"""
scripts/ef_sweep.py

HNSW Index Recall vs. Latency Sweep for Qdrant.
Measures HNSW performance against exact brute-force search baseline across multiple ef values.
"""

import sys
import time
import pathlib
import statistics as st

# Append project root directory to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient, models as qm
from app.config import settings
from ingest.embed import embed

# Golden set of evaluation queries (50 representative queries)
SAMPLE_QUERIES = [
    "What are the duties of a data fiduciary regarding security safeguards?",
    "What are the penalties for non-compliance under DPDP Act?",
    "How is personal data defined under GDPR versus DPDP?",
    "What are the conditions for processing children's personal data?",
    "What is the role and responsibility of a Data Protection Officer?",
    "What rights does a Data Principal have regarding data correction?",
    "Under what circumstances can personal data be transferred outside India?",
    "What constitutes a personal data breach under the Act?",
    "What is the timeline for notifying a data breach to the Board?",
    "What are the exemptions provided for state agencies or public interest?",
    # Add remaining sample queries from your benchmark dataset here...
]

client = QdrantClient(url=settings.qdrant_url)
collection_name = settings.collection_name

# Construct the standard filter applied across all queries
query_filter = qm.Filter(
    must=[
        qm.FieldCondition(
            key="chunker",
            match=qm.MatchValue(value=settings.chunker)
        )
    ]
)

def search(q_vec: list[float], ef: int | None = None, limit: int = 20):
    """
    Executes a vector query against Qdrant.
    When ef is None, forces exact brute-force search for baseline ground truth.
    """
    search_params = (
        qm.SearchParams(exact=True)
        if ef is None
        else qm.SearchParams(hnsw_ef=ef)
    )
    
    t0 = time.perf_counter()
    response = client.query_points(
        collection_name=collection_name,
        query=q_vec,
        limit=limit,
        query_filter=query_filter,
        search_params=search_params,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    hit_ids = {point.id for point in response.points}
    return hit_ids, latency_ms

def run_sweep():
    print(f"Generating query embeddings using model: '{settings.embed_model}'...")
    query_vectors = [embed([q], model_name=settings.embed_model)[0].tolist() for q in SAMPLE_QUERIES]
    
    # Connection warmup
    _ = search(query_vectors[0], ef=16)

    # 1. Establish Ground Truth via Exact Search (exact=True)
    print(f"Computing exact ground truth (exact=True) for {len(query_vectors)} queries at k=20...")
    truth = {}
    exact_lats = []
    for i, q_vec in enumerate(query_vectors):
        hit_ids, latency = search(q_vec, ef=None)
        truth[i] = hit_ids
        exact_lats.append(latency)
    
    exact_lats.sort()
    exact_p50 = st.median(exact_lats)
    exact_p95 = exact_lats[int(len(exact_lats) * 0.95)]
    
    # 2. Sweep across ef values
    print("\nStarting HNSW ef sweep...\n")
    print(f"{'ef':>7} {'recall@20':>10} {'p50 ms':>8} {'p95 ms':>8}")
    print("-" * 37)
    
    # Print Exact Baseline
    print(f"{'Exact':>7} {1.0000:>10.4f} {exact_p50:>8.2f} {exact_p95:>8.2f}")

    ef_values = (16, 32, 64, 128, 256)
    for ef in ef_values:
        recalls, latencies = [], []
        for i, q_vec in enumerate(query_vectors):
            got_ids, latency = search(q_vec, ef=ef)
            ground_truth_ids = truth[i]
            
            # Index Recall@20 = (|Approximate Hits ∩ Exact Hits|) / Total Exact Hits
            denom = max(len(ground_truth_ids), 1)
            recall = len(got_ids & ground_truth_ids) / denom
            
            recalls.append(recall)
            latencies.append(latency)
        
        latencies.sort()
        mean_recall = st.mean(recalls)
        p50_lat = st.median(latencies)
        p95_lat = latencies[int(len(latencies) * 0.95)]
        
        print(f"{ef:>7} {mean_recall:>10.4f} {p50_lat:>8.2f} {p95_lat:>8.2f}")

if __name__ == "__main__":
    run_sweep()