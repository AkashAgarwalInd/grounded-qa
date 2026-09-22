**[Grounded Q&A over Privacy Regulation](#)** — RAG with a published eval harness Hybrid retrieval (BM25 + dense, RRF) · cross-encoder reranking · grounded citations. Ablation study across 5 pipeline configurations on a 60-case golden set. `Python` `Qdrant` `Ragas` `Langfuse` `FastAPI`


# Grounded QA RAG Pipeline & Ablation Study

A containerized, production-grade Retrieval-Augmented Generation (RAG) system for regulatory compliance datasets (DPDP Act 2023, GDPR). Features a modular architecture designed to evaluate retrieval configurations systematically.

---

## 🛠 Architecture & Tech Stack

* **API Layer:** FastAPI with dynamic configuration controls (`app/config.py`).
* **Dense Retrieval:** Qdrant DB running `BAAI/bge-small-en-v1.5` (L2 Normalized, 384-d).
* **Sparse Retrieval:** `rank-bm25` in-memory lexical index.
* **Fusion Layer:** Reciprocal Rank Fusion ($k=60$).
* **Reranker:** `BAAI/bge-reranker-base` Cross-Encoder.
* **LLM Engine:** `gemini-2.5-flash-lite` (Query Rewriter) & `gemini-2.5-flash` (Synthesis & Grounding Judge).
* **Tooling & Containers:** `uv` package manager, Docker Compose with CPU-only PyTorch optimization.

---

## 🧪 Ablation Study Matrix

| Config | Name | Query Rewrite | Dense Vector | BM25 Lexical | Cross-Encoder Rerank | Intent / Target |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Config A** | **Naive Dense Baseline** | ❌ | `bge-small` | ❌ | ❌ | Baseline for all ablation metrics. |
| **Config B** | **Hybrid RRF** | ❌ | `bge-small` | `BM25Okapi` | ❌ | Recovers alphanumeric tokens and exact keywords. |
| **Config C** | **Hybrid + Rerank** | ❌ | `bge-small` | `BM25Okapi` | `bge-reranker` | Optimizes precision@k and passage relevance. |
| **Config D** | **Full Pipeline** | `gemini-lite` | `bge-small` | `BM25Okapi` | `bge-reranker` | Resolves ambiguous and negation-heavy queries. |

---

## 📊 Golden Test Set & Baseline Failure Cases (Config A Audit)

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

## 🚀 Quickstart & Deployment

### 1. Environment Setup

Copy `.env.example` to `.env` and set your HF or LLM keys:

```bash
cp .env.example .env
```

### 2. Run Container Stack via Docker Compose

```bash
docker compose up --build -d
```

