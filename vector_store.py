"""Redis vector store for the MedRAG knowledge base.

Uses Redis 8's native **vector sets** (the ``vectorset`` module: VADD / VSIM /
VGETATTR), which ship in core Redis 8 and require no separate Redis Stack
install. Each knowledge-base chunk is stored as one element of a single vector
set, with its embedding plus a JSON attribute blob holding the text + metadata:

    element name  -> chunk id
    vector        -> FLOAT32 embedding (dim = settings.embed_dim)
    attributes    -> {"text", "source", "drug_name", "section_type", "date"}

KNN search uses ``VSIM ... WITHSCORES`` (cosine similarity in [0, 1], higher is
more similar); attributes for each hit are fetched with ``VGETATTR``.

Note: earlier revisions targeted the RediSearch ``FT.*`` query engine. Native
vector sets are used instead so the system runs on a stock local Redis 8.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import redis

from config import Settings, settings as default_settings


@dataclass
class Chunk:
    """A single knowledge-base chunk to be indexed."""

    id: str
    text: str
    source: str = ""
    drug_name: str = ""
    section_type: str = ""
    date: str = ""
    url: str = ""
    embedding: list[float] = field(default_factory=list)


@dataclass
class SearchHit:
    id: str
    score: float
    text: str
    source: str
    drug_name: str
    section_type: str
    date: str
    url: str = ""


class VectorStore:
    def __init__(self, settings: Settings = default_settings):
        url = settings.resolved_redis_url()
        if not url:
            raise ValueError(
                "No Redis connection configured. Set REDIS_URL (or REDIS_HOST/"
                "PORT/PASSWORD) in your environment / .env file."
            )
        # Fail fast rather than hang forever if the endpoint is unresponsive.
        self.client = redis.Redis.from_url(
            url,
            decode_responses=False,
            socket_timeout=10,
            socket_connect_timeout=10,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        # The vector set key holding all chunks.
        self.key = settings.redis_index_name
        self.dim = settings.embed_dim

    # -- connectivity ----------------------------------------------------
    def ping(self) -> bool:
        return bool(self.client.ping())

    def index_exists(self) -> bool:
        return bool(self.client.exists(self.key))

    def supports_vectorset(self) -> bool:
        """True if this Redis build exposes the vector-set commands."""
        try:
            modules = self.client.execute_command("MODULE", "LIST")
        except redis.ResponseError:
            return False
        flat = b" ".join(
            part if isinstance(part, bytes) else str(part).encode()
            for row in modules for part in (row if isinstance(row, (list, tuple)) else [row])
        )
        return b"vectorset" in flat or b"search" in flat

    # -- index management ------------------------------------------------
    def create_index(self, recreate: bool = False) -> None:
        """Vector sets are created lazily on first VADD; only handle recreate."""
        if recreate and self.index_exists():
            self.client.delete(self.key)

    # -- writes ----------------------------------------------------------
    @staticmethod
    def _vector_bytes(embedding: list[float]) -> bytes:
        return np.asarray(embedding, dtype=np.float32).tobytes()

    def upsert(self, chunks: list[Chunk]) -> int:
        pipe = self.client.pipeline(transaction=False)
        for chunk in chunks:
            attrs = json.dumps(
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "drug_name": chunk.drug_name,
                    "section_type": chunk.section_type,
                    "date": chunk.date,
                    "url": chunk.url,
                }
            )
            pipe.execute_command(
                "VADD",
                self.key,
                "FP32",
                self._vector_bytes(chunk.embedding),
                chunk.id,
                "SETATTR",
                attrs,
            )
        pipe.execute()
        return len(chunks)

    def count(self) -> int:
        if not self.index_exists():
            return 0
        try:
            return int(self.client.execute_command("VCARD", self.key))
        except redis.ResponseError:
            return 0

    # -- reads -----------------------------------------------------------
    def search(
        self,
        query_vector: list[float],
        k: int = 8,
        drug_name: Optional[str] = None,
        source: Optional[str] = None,
        section_types: Optional[list[str]] = None,
    ) -> list[SearchHit]:
        if not self.index_exists():
            return []

        args: list[Any] = [
            "VSIM", self.key, "FP32", self._vector_bytes(query_vector),
            "WITHSCORES", "COUNT", k,
        ]
        clauses: list[str] = []
        if drug_name:
            clauses.append(f'.drug_name == "{drug_name.replace(chr(34), "")}"')
        if source:
            clauses.append(f'.source == "{source.replace(chr(34), "")}"')
        if section_types:
            quoted = ", ".join(f'"{s.replace(chr(34), "")}"' for s in section_types)
            clauses.append(f".section_type in [{quoted}]")
        if clauses:
            args += ["FILTER", " && ".join(clauses)]

        raw = self.client.execute_command(*args)
        pairs = _parse_withscores(raw)
        if not pairs:
            return []

        # Fetch attributes for each returned element in one round-trip.
        pipe = self.client.pipeline(transaction=False)
        for element, _ in pairs:
            pipe.execute_command("VGETATTR", self.key, element)
        attr_blobs = pipe.execute()

        hits: list[SearchHit] = []
        for (element, score), blob in zip(pairs, attr_blobs):
            meta = _load_attrs(blob)
            hits.append(
                SearchHit(
                    id=_decode(element),
                    score=float(score),
                    text=meta.get("text", ""),
                    source=meta.get("source", ""),
                    drug_name=meta.get("drug_name", ""),
                    section_type=meta.get("section_type", ""),
                    date=meta.get("date", ""),
                    url=meta.get("url", ""),
                )
            )
        return hits


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if value is not None else ""


def _load_attrs(blob: Any) -> dict:
    text = _decode(blob)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _parse_withscores(raw: Any) -> list[tuple[Any, float]]:
    """Normalize VSIM WITHSCORES output across RESP2 (flat) and RESP3 (mapping)."""
    if raw is None:
        return []
    # RESP3 may return a dict {element: score}.
    if isinstance(raw, dict):
        return [(el, float(sc)) for el, sc in raw.items()]
    # RESP2 returns a flat list [element, score, element, score, ...].
    if isinstance(raw, (list, tuple)):
        # Some clients may nest as [[element, score], ...].
        if raw and isinstance(raw[0], (list, tuple)):
            return [(item[0], float(item[1])) for item in raw]
        out: list[tuple[Any, float]] = []
        for i in range(0, len(raw) - 1, 2):
            out.append((raw[i], float(raw[i + 1])))
        return out
    return []
