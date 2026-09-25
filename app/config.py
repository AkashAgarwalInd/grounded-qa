import hashlib
import json
import subprocess
import pathlib
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM: any OpenAI-compatible endpoint. Default = Gemini via AI Studio ---
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_api_key: str = "dummy"
    rewrite_model: str = "gemini-2.5-flash-lite"   # cheap route
    llm_model: str = "gemini-2.5-flash"            # synthesis
    judge_model: str = "gemini-2.5-flash"          # Ragas judge
    llm_max_concurrency: int = 2                   # stay under free-tier RPM
    llm_cache_dir: str = ".llm_cache"              # never pay twice for the same call

    # --- Vector store ---
    qdrant_url: str = "http://localhost:6333"
    collection: str = "regs"

    # --- Ablation switches: every row of the Day-12 table is one of these ---
    use_rewrite:          bool = False   # query rewriting
    use_bm25:             bool = False   # config B
    use_rerank:           bool = False   # config C
    chunker: Literal["fixed", "sentence", "recursive", "structure", "semantic"] = "fixed"  # config D
    use_contextual:       bool = False   # config E
    use_grounding_check:  bool = False   # validate_grounding()

    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    top_k: int = 20        # retrieve wide
    final_k: int = 5       # rerank narrow

    @property
    def collection_name(self) -> str:
        # Only index-affecting switches go in the name; query-time switches share an index.
        tag = f"{self.chunker}-{'ctx' if self.use_contextual else 'plain'}"
        return f"{self.collection}_{self.embed_model.split('/')[-1]}_{tag}"

    def validate(self) -> list[str]:
        """
        Validate configuration settings and return list of errors.
        
        Checks:
        - LLM API key validity
        - Qdrant URL accessibility
        - Model names validity
        - Reasonable parameter ranges
        
        Returns:
            List of error strings (empty if valid)
        """
        errors: list[str] = []
        
        # Check LLM API key
        if self.llm_api_key == "dummy" or not self.llm_api_key:
            errors.append("LLM API key is using default dummy value - set GEMINI_API_KEY environment variable")
        
        # Check Qdrant URL
        if not self.qdrant_url.startswith("http"):
            errors.append(f"Invalid Qdrant URL: {self.qdrant_url}")
        
        # Check model names
        if not self.embed_model:
            errors.append("Embed model name is empty")
        
        if not self.rerank_model:
            errors.append("Rerank model name is empty")
        
        # Check parameter ranges
        if self.top_k <= 0:
            errors.append(f"top_k must be positive, got {self.top_k}")
        
        if self.final_k <= 0 or self.final_k > self.top_k:
            errors.append(f"final_k must be positive and <= top_k, got {self.final_k}")
        
        # Check chunker is valid
        valid_chunks = ["fixed", "sentence", "recursive", "structure", "semantic"]
        if self.chunker not in valid_chunks:
            errors.append(f"Invalid chunker: {self.chunker}. Must be one of {valid_chunks}")
        
        return errors


@dataclass(frozen=True)
class RunConfig:
    """Frozen dataclass mirroring runtime config flags plus git SHA and timestamp."""
    use_bm25: bool
    use_rerank: bool
    chunker: str
    use_contextual: bool
    embed_model: str
    top_k: int
    final_k: int
    golden_hash: str = ""

    @property
    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:8]

    def save(self, root: str = "runs") -> str:
        d = pathlib.Path(root) / self.hash
        d.mkdir(parents=True, exist_ok=True)
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        (d / "config.json").write_text(json.dumps(
            {**asdict(self), "git": sha,
             "at": dt.datetime.now(dt.UTC).isoformat()}, indent=2))
        return str(d)


settings = Settings()