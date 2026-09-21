from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import settings

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


class AskResponse(BaseModel):
    answer: str
    citations: list[str] = []
    config: dict


@app.get("/health")
async def health() -> dict:
    """Liveness plus a best-effort Qdrant check.

    Always returns 200 while the API is up; the Qdrant field tells you whether
    the vector store is reachable. (Compose "done when": 200 here and Qdrant on 6333.)
    """
    try:
        r = await app.state.http.get(f"{settings.qdrant_url}/healthz")
        qdrant = "ok" if r.status_code == 200 else f"http {r.status_code}"
    except httpx.HTTPError:
        qdrant = "unreachable"
    return {"status": "ok", "qdrant": qdrant}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Stub. Real pipeline (rewrite -> retrieve -> rerank -> synthesise -> validate)
    lands on Days 2-7, each stage gated by a switch in app/config.py."""
    return AskResponse(
        answer=f"[stub] pipeline not implemented yet. You asked: {req.question}",
        citations=[],
        config=public_config(),
    )