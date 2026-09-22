from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import settings
from app.retrieve import dense_search

# Fields that must never be echoed back to a client.
SECRET_FIELDS = {"llm_api_key"}


def public_config() -> dict:
    """Active settings, minus secrets, plus the derived collection name."""
    cfg = settings.model_dump(exclude=SECRET_FIELDS)
    cfg["collection_name"] = settings.collection_name
    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared HTTP client for the app's lifetime (reused by health checks now,
    # and available for other outbound calls later).
    app.state.http = httpx.AsyncClient(timeout=2.0)
    yield
    await app.state.http.aclose()


app = FastAPI(title="grounded-qa", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class CitationSnippet(BaseModel):
    doc_id: str
    section: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[str] = []
    passages: list[CitationSnippet] = []
    config: dict


@app.get("/health")
async def health() -> dict:
    """Liveness plus a best-effort Qdrant check.

    Always returns 200 while the API is up; the Qdrant field tells you whether
    the vector store is reachable.
    """
    try:
        r = await app.state.http.get(f"{settings.qdrant_url}/healthz")
        qdrant = "ok" if r.status_code == 200 else f"http {r.status_code}"
    except httpx.HTTPError:
        qdrant = "unreachable"
    return {"status": "ok", "qdrant": qdrant}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Config A Baseline: Naive dense retrieval directly from Qdrant.
    
    Returns raw passages and citations without LLM synthesis.
    """
    # 1. Retrieve top-k passages using dense search baseline
    raw_passages = await dense_search(req.question, k=settings.final_k)

    # 2. Extract formatted citation identifiers (e.g., "dpdp_act_2023 - Sec. 1")
    citation_labels = [
        f"{p['doc_id']} - §{p['section']}" for p in raw_passages
    ]

    # 3. Map retrieved points to response model
    passage_models = [
        CitationSnippet(
            doc_id=p["doc_id"],
            section=p["section"],
            text=p["text"],
            score=p["score"],
        )
        for p in raw_passages
    ]

    return AskResponse(
        answer=f"[Config A Baseline] Retrieved {len(raw_passages)} passages using dense vector search.",
        citations=citation_labels,
        passages=passage_models,
        config=public_config(),
    )