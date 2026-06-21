"""Retrieval layer for MedRAG.

Retrieval quality directly determines whether a dangerous interaction is
surfaced, so this module does more than a single similarity search.

Strategy (multi-query with guaranteed per-pair coverage):

1. A **general** query (proposed drug + patient context) retrieves chunks about
   the proposed drug itself — contraindications, dosing, monitoring, etc.
2. A **focused interaction** query is run for *each* current medication paired
   with the proposed drug. This prevents the failure mode where a single blended
   query returns only proposed-drug chunks and silently misses the interaction
   chunk for one of the patient's current medications.
3. Results are merged and de-duplicated (keeping the best score per chunk). The
   top interaction chunk for each drug pair is **guaranteed** a slot in the
   final result so it cannot be crowded out by higher-scoring general chunks,
   up to ``retrieval_max_chunks``.

Each query is embedded with the same model used at index time.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from config import settings
from embeddings import Embedder, get_embedder
from vector_store import SearchHit, VectorStore


def build_query(
    drug_name: str,
    patient_record: dict,
    indication: Optional[str] = None,
) -> str:
    """Compose the general retrieval query: drug + diagnoses + indication."""
    parts: list[str] = [f"Medication: {drug_name}."]

    if indication:
        parts.append(f"Indication: {indication}.")

    diagnoses = [d.get("name") for d in patient_record.get("diagnoses", []) if d.get("name")]
    if diagnoses:
        parts.append("Active diagnoses: " + ", ".join(diagnoses) + ".")

    current_meds = [m.get("name") for m in patient_record.get("medications", []) if m.get("name")]
    if current_meds:
        parts.append("Current medications: " + ", ".join(current_meds) + ".")

    allergies = [a.get("substance") for a in patient_record.get("allergies", []) if a.get("substance")]
    if allergies:
        parts.append("Allergies: " + ", ".join(allergies) + ".")

    return " ".join(parts)


def build_pair_query(proposed_drug: str, current_med: str) -> str:
    """Compose a focused drug-pair interaction query."""
    return (
        f"Drug-drug interaction between {proposed_drug} and {current_med}: "
        f"mechanism, severity, contraindication, and dose adjustment."
    )


def _merge(pool: dict[str, SearchHit], hits: list[SearchHit]) -> None:
    """Insert hits into the pool keyed by id, keeping the highest score."""
    for h in hits:
        existing = pool.get(h.id)
        if existing is None or h.score > existing.score:
            pool[h.id] = h


def retrieve_chunks(
    drug_name: str,
    patient_record: dict,
    indication: Optional[str] = None,
    top_k: Optional[int] = None,
    store: Optional[VectorStore] = None,
    embedder: Optional[Embedder] = None,
    per_pair_k: Optional[int] = None,
    max_chunks: Optional[int] = None,
) -> list[SearchHit]:
    """Retrieve relevant chunks using general + per-pair interaction queries.

    Returns a de-duplicated list ordered by relevance, with the top interaction
    chunk for each current medication guaranteed inclusion (subject to
    ``max_chunks``).
    """
    top_k = top_k or settings.retrieval_top_k
    per_pair_k = per_pair_k or settings.retrieval_per_pair_k
    max_chunks = max_chunks or settings.retrieval_max_chunks
    store = store or VectorStore(settings)
    embedder = embedder or get_embedder(settings)

    pool: dict[str, SearchHit] = {}
    guaranteed_ids: list[str] = []

    # 1. General query about the proposed drug + patient context.
    general_q = build_query(drug_name, patient_record, indication)
    general_hits = store.search(embedder.embed_query(general_q), k=top_k)
    _merge(pool, general_hits)

    # 2. Focused interaction query per current medication.
    proposed_lower = drug_name.strip().lower()
    seen_meds: set[str] = set()
    for med in patient_record.get("medications", []):
        name = (med.get("name") or "").strip()
        key = name.lower()
        if not name or key == proposed_lower or key in seen_meds:
            continue
        seen_meds.add(key)

        pair_hits = store.search(embedder.embed_query(build_pair_query(drug_name, name)), k=per_pair_k)
        if not pair_hits:
            continue
        _merge(pool, pair_hits)
        # Guarantee the single best interaction chunk for this pair a slot.
        if pair_hits[0].id not in guaranteed_ids:
            guaranteed_ids.append(pair_hits[0].id)

    if not pool:
        return []

    # 3. Assemble final list: guaranteed per-pair hits first, then fill the
    #    remaining budget with the highest-scoring remaining chunks.
    ordered: list[SearchHit] = []
    used: set[str] = set()

    for cid in guaranteed_ids:
        if cid in pool and cid not in used:
            ordered.append(pool[cid])
            used.add(cid)

    remaining = sorted(
        (h for cid, h in pool.items() if cid not in used),
        key=lambda h: h.score,
        reverse=True,
    )
    for h in remaining:
        if len(ordered) >= max_chunks:
            break
        ordered.append(h)
        used.add(h.id)

    # Present in descending relevance for the physician / prompt.
    ordered.sort(key=lambda h: h.score, reverse=True)
    return ordered[:max_chunks]
