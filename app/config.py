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


settings = Settings()