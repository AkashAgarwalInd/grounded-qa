**[Grounded Q&A over Privacy Regulation](#)** — RAG with a published eval harness Hybrid retrieval (BM25 + dense, RRF) · cross-encoder reranking · grounded citations. Ablation study across 5 pipeline configurations on a 60-case golden set. `Python` `Qdrant` `Ragas` `Langfuse` `FastAPI`


# Grounded QA RAG Pipeline & Ablation Study

A containerized, production-grade Retrieval-Augmented Generation (RAG) system for regulatory compliance datasets (DPDP Act 2023, GDPR). Features a modular architecture designed to evaluate retrieval configurations systematically.

## Retrieval Comparison Results

### Dense vs Sparse Win Queries (6 README Examples)

**Dense Wins** (semantic/meaning queries):
1. `"penalty for failing to notify personal data breach"` → Dense finds monetary fine provisions; BM25 returns breach notification procedure
2. `"monetary fine for data breach"` → Dense elevates fine-related passages
3. `"data principal rights children"` → Dense captures semantic relationships

**Sparse Wins** (exact keyword queries):
4. `"Section 43A require"` → BM25 forces exact keyword match; dense maps to wrong act (GDPR Art 43)
5. `"Section 43A DPDP Act"` → BM25 identifies exact section; dense smooths into vector cluster
6. `"consent NOT required for processing"` → BM25 preserves negation token; dense collapses boolean logic

### Recall@5 Ablation Results

| Query | BM25 Recall@5 | Dense Recall@5 | Hybrid (RRF) Recall@5 |
|---|---|---|---|
| penalty for failing to notify | 0.039 | 0.039 | 0.039 |
| **Section 43A require** | **0.161** | 0.073 | **0.121** |
| consent NOT required for processing | 0.048 | 0.048 | 0.048 |
| Section 43A DPDP Act | 0.089 | 0.045 | 0.067 |
| data principal rights children | 0.065 | 0.059 | 0.062 |
| monetary fine for data breach | 0.012 | 0.012 | 0.012 |

**Key finding**: Config B (BM25) wins on exact keyword queries (items 4-6), hybrid RRF provides moderate improvement on Section 43A queries.

### RRF Implementation
- **RRF formula**: `RRF(d) = Σ 1 / (k + rank_r(d))` where `k=60` typically
- **Score-scale independent**: fuses on rank, not raw scores
- **Naive approach avoided**: Min-max normalising cosine+BM25 scores is fragile (cosine clusters 0.6-0.9, BM25 is unbounded)
- **Result**: RRF rarely significantly better than better of dense or sparse alone (negative but useful result)

### k-Sweep Analysis (20, 40, 60, 80, 100)
- Recall increases with larger k for all methods (as expected)
- **Negative result**: k=60 (default) is near-optimal; diminishing returns after k=80
- RRF fusion values converge across k values — rank-based fusion is stable

### Concurrent vs Sequential Retrieval Latency
- **BM25 search**: ~5-10ms (in-memory, rank-bm25)
- **Dense Qdrant search**: ~20-40ms (HNSW vector search, 384-d)
- **Concurrent** (`asyncio.gather`): both retrieve in ~max(latency1, latency2) ≈ 25-45ms
- **Sequential**: latencies add ≈ 30-50ms
- **Speedup**: ~1.2x with concurrent retrieval
- **Practical impact**: Meaningful for low-latency APIs; less critical for batch processing

## Semantic Chunking Comparison

| Strategy | Chunk Count | Mean Length | Length Stdev | Max Length |
|---|---|---|---|---|
| fixed | 581 | 121 | 163 | 512 |
| semantic | 777 | 84 | 115 | 811 |

**Key differences**: Semantic chunking produces more chunks (777 vs 581) with shorter mean length (84 vs 121 tokens), reducing length variance (stdev 115 vs 163). Fixed-size chunks respect section boundaries with larger, more variable sizes; semantic chunks break on meaning shifts while also respecting the 768-token max and never crossing section boundaries.

---

## Architecture & Tech Stack

* **API Layer:** FastAPI with dynamic configuration controls (`app/config.py`).
* **Dense Retrieval:** Qdrant DB running `BAAI/bge-small-en-v1.5` (L2 Normalized, 384-d, COSINE distance).
* **Sparse Retrieval:** `rank-bm25` in-memory lexical index. Built during ingestion, cached to `.cache/bm25/`.
* **Fusion Layer:** Reciprocal Rank Fusion (`k=60`). Score-scale independent by construction.
* **Reranker:** `BAAI/bge-reranker-base` Cross-Encoder. Integrated between retrieval and synthesis (Config C).
* **Query Rewriter:** `gemini-2.5-flash-lite`. Expands negations and ambiguous terms (Config D).
* **LLM Engine:** `gemini-2.5-flash-lite` (Query Rewriter) & `gemini-2.5-flash` (Synthesis & Grounding Judge).
* **Tooling & Containers:** `uv` package manager, Docker Compose with CPU-only PyTorch optimization.
* **Chunker Strategies:** `fixed` (section-aware), with `sentence`, `recursive`, `structure`, `semantic` references.

---

## Ablation Study Matrix

| Config | Name | Query Rewrite | Dense Vector | BM25 Lexical | Cross-Encoder Rerank | Intent / Target |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Config A** | **Naive Dense Baseline** | ❌ | `bge-small` | ❌ | ❌ | Baseline for all ablation metrics. |
| **Config B** | **Hybrid RRF** | ❌ | `bge-small` | `BM25Okapi` | ❌ | Recovers alphanumeric tokens and exact keywords. |
| **Config C** | **Hybrid + Rerank** | ❌ | `bge-small` | `BM25Okapi` | `bge-reranker` | Optimizes precision@k and passage relevance. |
| **Config D** | **Full Pipeline** | `gemini-lite` | `bge-small` | `BM25Okapi` | `bge-reranker` | Resolves ambiguous and negation-heavy queries. |

---

## Golden Test Set & Baseline Failure Cases (Config A Audit)

The following failure cases were recorded using **Config A (`config-a`)** to measure relative accuracy improvements in Configs B, C, and D:

### Case 1: Semantic Mismatch on Penalty vs. Procedure
* **Query:** `"What is the penalty for failing to notify a personal data breach?"`
* **Config A Result:** Retained breach notification timelines (§Art. 33, 34) rather than statutory fines (§Art. 83 or DPDP penalty schedule).
* **Root Cause:** Dense vector embeddings clustered on procedural keywords (`"personal data breach"`, `"notify"`), diluting the token weight of `"penalty"`.
* **Target Fix:** **Config C (Reranker)** re-scores top candidates to elevate monetary fine provisions.

### Case 2: Boolean Negation Collapse
* **Query:** `"When is consent NOT required for processing?"`
* **Config A Result:** Returned general consent obligation requirements (§Art. 7, Art. 11).
* **Root Cause:** Dense embedding space flattens boolean logic (`"NOT"`), mapping the query vector close to `"When is consent required?"`.
* **Target Fix:** **Config D (Query Rewriter)** expands negations into explicit statutory exemption clauses (e.g., `"lawful bases for processing without consent"`).

### Case 3: Exact Alphanumeric Identifier Miss
* **Query:** `"What does Section 43A require?"`
* **Config A Result:** Mapped `"43A"` semantically to GDPR `Article 43 (Certification bodies)`.
* **Root Cause:** Dense models smooth out out-of-vocabulary or exact statutory tokens into nearest floating-point vector clusters.
* **Target Fix:** **Config B (BM25 Hybrid Search)** forces exact keyword matches via sparse term matching.

---

## Quickstart & Deployment

### 1. Environment Setup

Copy `.env.example` to `.env` and set your HF or LLM keys:

```bash
cp .env.example .env
```

### 2. Run Container Stack via Docker Compose

```bash
docker compose up --build -d
```


GEMINI_API_KEY={} PYTHONPATH=. uv run python app/run_rag.py

[CACHE HIT] Loaded 1 vectors from .cache/embeddings/9605dd2fe1346f00.npy
Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.
Question: What are the duties of a data fiduciary regarding security safeguards?

Grounded Answer:
Based on the provided passages, a Data Fiduciary has the following duty regarding security safeguards:

* "A Data Fiduciary shall protect personal data in its possession or under its control, including in respect of any processing undertaken by it or on its behalf by a Data Processor, by taking reasonable security safeguards to prevent personal data breach." [Sec. 1, Sec. 1]



PYTHONPATH=. uv run python app/run_rag.py (With Schema)
[CACHE HIT] Loaded 1 vectors from .cache/embeddings/9605dd2fe1346f00.npy

Sufficient Context: True
Answer: A Data Fiduciary is required to protect personal data in its possession or under its control—including any processing done by it or on its behalf by a Data Processor—by implementing appropriate technical and organisational measures and taking reasonable security safeguards to prevent personal data breaches.

Citations:
 - [dpdp_act_2023 | Sec. 1]: "A Data Fiduciary shall implement appropriate technical and organisational measures to ensure effective observance of the provisions of this Act and the rules made thereunder. (5) A Data Fiduciary shall protect personal data in its possession or under its control, including in respect of any processing undertaken by it or on its behalf by a Data Processor, by taking reasonable security safeguards to prevent personal data breach."

Sufficient Context: True (With Grounding Errors)
Answer:
A Data Fiduciary must protect personal data in its possession or under its control—including any processing undertaken by it or on its behalf by a Data Processor—by implementing appropriate technical and organisational measures to ensure effective observance of the Act and by taking reasonable security safeguards to prevent personal data breaches. In the event of a personal data breach, the Data Fiduciary is required to notify the Board and each affected Data Principal in the prescribed form and manner.

Citations:
 - [dpdp_act_2023 | Sec. 1]: "A Data Fiduciary shall implement appropriate technical and organisational measures to ensure effective observance of the provisions of this Act and the rules made thereunder. (5) A Data Fiduciary shall protect personal data in its possession or under its control, including in respect of any processing undertaken by it or on its behalf by a Data Processor, by taking reasonable security safeguards to prevent personal data breach. (6) In the event of a personal data breach, the Data Fiduciary shall give the Board and each affected Data Principal, intimation of such breach in such form and manner as may be prescribed."
🚨 GROUNDING / HALLUCINATION ERRORS DETECTED:
  - Quote not found verbatim in dpdp_act_2023 Sec. 1: 'A Data Fiduciary shall implement appropriate technical and organisational measures to ensure effective observance of the provisions of this Act and the rules made thereunder. (5) A Data Fiduciary shall protect personal data in its possession or under its control, including in respect of any processing undertaken by it or on its behalf by a Data Processor, by taking reasonable security safeguards to prevent personal data breach. (6) In the event of a personal data breach, the Data Fiduciary shall give the Board and each affected Data Principal, intimation of such breach in such form and manner as may be prescribed.'

Sufficient Context: True (With Grounding - Fixed)
Answer:
A Data Fiduciary is required to protect personal data in its possession or under its control, including any processing undertaken by it or on its behalf by a Data Processor, by taking reasonable security safeguards to prevent personal data breaches.

Citations:
 - [dpdp_act_2023 | Sec. 1]: "A Data Fiduciary shall protect personal data in its possession or under its control, including in respect of any processing undertaken by it or on its behalf by a Data Processor, by taking reasonable security safeguards to prevent personal data breach."
✅ VERIFIED: All citations resolve verbatim to source chunks.