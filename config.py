"""Central configuration for MedRAG.

Loads settings from environment variables (via a `.env` file when present)
and exposes them as a single `settings` object the rest of the pipeline imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional at runtime
    pass


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Redis
    redis_url: str | None
    redis_host: str | None
    redis_port: int
    redis_password: str | None
    redis_index_name: str

    # Embeddings
    embed_provider: str  # "bge" | "voyage" | "stub"

    # Voyage (cloud, paid)
    voyage_api_key: str | None
    voyage_model: str
    voyage_dim: int

    # BGE (local, free)
    bge_model: str
    bge_dim: int
    bge_query_instruction: str

    # Anthropic
    anthropic_api_key: str | None
    anthropic_model: str

    # Behavior
    use_stub_embedder: bool
    retrieval_top_k: int
    retrieval_per_pair_k: int
    retrieval_max_chunks: int

    @property
    def embed_dim(self) -> int:
        """Vector dimensionality for the active embedding provider."""
        if self.use_stub_embedder:
            return self.bge_dim if self.embed_provider == "bge" else self.voyage_dim
        if self.embed_provider == "voyage":
            return self.voyage_dim
        return self.bge_dim

    def resolved_redis_url(self) -> str | None:
        """Return a connection URL, building one from parts if needed."""
        if self.redis_url:
            return self.redis_url
        if self.redis_host:
            auth = f"default:{self.redis_password}@" if self.redis_password else ""
            return f"redis://{auth}{self.redis_host}:{self.redis_port}"
        return None


def load_settings() -> Settings:
    return Settings(
        redis_url=os.getenv("REDIS_URL") or None,
        redis_host=os.getenv("REDIS_HOST") or None,
        redis_port=_get_int("REDIS_PORT", 6379),
        redis_password=os.getenv("REDIS_PASSWORD") or None,
        redis_index_name=os.getenv("REDIS_INDEX_NAME", "medrag_kb"),
        embed_provider=(os.getenv("EMBED_PROVIDER", "bge") or "bge").lower(),
        voyage_api_key=os.getenv("VOYAGE_API_KEY") or None,
        voyage_model=os.getenv("VOYAGE_MODEL", "voyage-3-large"),
        voyage_dim=_get_int("VOYAGE_DIM", 1024),
        bge_model=os.getenv("BGE_MODEL", "BAAI/bge-large-en-v1.5"),
        bge_dim=_get_int("BGE_DIM", 1024),
        bge_query_instruction=os.getenv(
            "BGE_QUERY_INSTRUCTION",
            "Represent this sentence for searching relevant passages:",
        ),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8"),
        use_stub_embedder=_get_bool("USE_STUB_EMBEDDER", False),
        retrieval_top_k=_get_int("RETRIEVAL_TOP_K", 8),
        retrieval_per_pair_k=_get_int("RETRIEVAL_PER_PAIR_K", 3),
        retrieval_max_chunks=_get_int("RETRIEVAL_MAX_CHUNKS", 14),
    )


settings = load_settings()
