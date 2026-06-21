"""Embedding layer for MedRAG.

Provides a single interface over three backends:

- ``BGEEmbedder``     -> local BAAI BGE embeddings via sentence-transformers.
  Runs on-device with no API cost (default provider).
- ``VoyageEmbedder``  -> cloud ``voyage-3-large`` embeddings (requires an API key).
- ``StubEmbedder``    -> deterministic local vectors for building/testing the
  pipeline without downloading a model. NOT semantically meaningful; never use
  for a real knowledge base you intend to query for clinical relevance.

BGE (v1.5 English) and Voyage both benefit from distinguishing ``document`` vs
``query`` inputs, so the interface exposes both ``embed_documents`` and
``embed_query``.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from config import Settings, settings as default_settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class BGEEmbedder:
    """Local BAAI BGE embeddings via sentence-transformers (no API cost).

    For BGE v1.5 English models, retrieval quality improves when a short
    instruction is prepended to *queries* (not documents). Vectors are
    L2-normalized so cosine similarity in Redis is well-behaved.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        dim: int = 1024,
        query_instruction: str = "",
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers`."
            ) from exc

        self.model_name = model_name
        self.query_instruction = query_instruction
        self._model = SentenceTransformer(model_name)
        # Method was renamed across sentence-transformers versions.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        actual = get_dim() if get_dim else None
        if actual and actual != dim:
            # Trust the model's real dimensionality over the configured value.
            print(
                f"[embeddings] note: {model_name} outputs {actual} dims; "
                f"using {actual} (configured BGE_DIM was {dim})."
            )
            dim = actual
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return [v.astype("float32").tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        payload = f"{self.query_instruction} {text}".strip() if self.query_instruction else text
        vector = self._model.encode(
            [payload], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        return vector.astype("float32").tolist()


class VoyageEmbedder:
    """Wraps the Voyage AI client for ``voyage-3-large`` embeddings."""

    def __init__(self, api_key: str, model: str = "voyage-3-large", dim: int = 1024):
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "voyageai is not installed. Run `pip install voyageai`."
            ) from exc

        if not api_key:
            raise ValueError("VOYAGE_API_KEY is required for VoyageEmbedder.")

        self._client = voyageai.Client(api_key=api_key)
        self.model = model
        self.dim = dim

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        result = self._client.embed(
            texts,
            model=self.model,
            input_type=input_type,
            output_dimension=self.dim,
        )
        return result.embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts, input_type="document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="query")[0]


class StubEmbedder:
    """Deterministic, dependency-free embedder for offline pipeline testing.

    Vectors are derived from a hash of the text, so identical text always maps
    to an identical (unit-normalized) vector. This is enough to exercise Redis
    indexing, upsert, and top-k search mechanics, but carries no real semantics.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        # Seed a PRNG from a stable hash of the text for reproducibility.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def get_embedder(settings: Settings = default_settings) -> Embedder:
    """Factory: return the embedder for the configured provider.

    ``USE_STUB_EMBEDDER=true`` always wins (offline pipeline testing). Otherwise
    the provider is chosen by ``EMBED_PROVIDER`` (default "bge").
    """
    if settings.use_stub_embedder:
        return StubEmbedder(dim=settings.embed_dim)

    provider = settings.embed_provider

    if provider == "voyage":
        if not settings.voyage_api_key:
            print(
                "[embeddings] WARNING: EMBED_PROVIDER=voyage but no VOYAGE_API_KEY "
                "set; falling back to StubEmbedder (not semantically meaningful)."
            )
            return StubEmbedder(dim=settings.voyage_dim)
        return VoyageEmbedder(
            api_key=settings.voyage_api_key,
            model=settings.voyage_model,
            dim=settings.voyage_dim,
        )

    if provider == "stub":
        return StubEmbedder(dim=settings.embed_dim)

    # Default: local BGE.
    return BGEEmbedder(
        model_name=settings.bge_model,
        dim=settings.bge_dim,
        query_instruction=settings.bge_query_instruction,
    )
