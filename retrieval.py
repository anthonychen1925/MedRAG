"""Retrieval layer for MedRAG.

Retrieval quality directly determines whether a dangerous interaction is
surfaced, so this module does more than a single similarity search. The index
holds three source types (openFDA labels, DDInter interaction pairs, openFDA
FAERS adverse-event reports). Because there are far more interaction-pair chunks
than label chunks, a naive top-k search returns *only* interaction chunks and
crowds out the proposed drug's own contraindications/dosing. To prevent that,
retrieval allocates a **per-source quota** via several targeted queries:

1. **Label safety** — a query for the proposed drug's contraindications,
   warnings, renal/hepatic dosing, etc., filtered server-side to openFDA label
   safety sections. Chunks for the proposed drug itself are prioritized.
2. **Per-pair interactions** — one focused query for each current medication
   paired with the proposed drug. The single best interaction chunk per pair is
   **guaranteed** a slot so a dangerous combination can never be crowded out.
3. **Adverse-event signal** — the proposed drug's FAERS report summary.
4. **General fill** — remaining budget filled by a blended query.

Each query is embedded with the same model used at index time.
"""

from __future__ import annotations

from typing import Optional

from config import settings
from embeddings import Embedder, get_embedder
from vector_store import SearchHit, VectorStore

# openFDA label sections that carry the proposed drug's own safety profile.
LABEL_SAFETY_SECTIONS = [
    "boxed_warning",
    "contraindications",
    "warnings_and_cautions",
    "drug_interactions",
    "dosage_and_administration",
    "use_in_specific_populations",
    "pregnancy",
    "geriatric_use",
    "renal_impairment",
    "hepatic_impairment",
    "adverse_reactions",
    "clinical_pharmacology",
]
SOURCE_LABEL = "openFDA"
SOURCE_FAERS = "openFDA FAERS"


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


def build_safety_query(drug_name: str, patient_record: dict, indication: Optional[str]) -> str:
    """Compose a query targeting the proposed drug's own label safety profile."""
    parts = [
        f"{drug_name} contraindications, boxed warning, warnings and precautions, "
        f"adverse reactions, dose adjustment in renal and hepatic impairment, "
        f"use in elderly and specific populations."
    ]
    diagnoses = [d.get("name") for d in patient_record.get("diagnoses", []) if d.get("name")]
    if diagnoses:
        parts.append("Patient conditions: " + ", ".join(diagnoses) + ".")
    if indication:
        parts.append(f"Indication: {indication}.")
    return " ".join(parts)


def build_faers_query(drug_name: str) -> str:
    return f"Most frequently reported adverse events and safety signals for {drug_name}."


def build_renal_dose_query(drug_name: str) -> str:
    return (
        f"{drug_name} dosage and administration with dose adjustment for renal "
        f"impairment, creatinine clearance, and reduced kidney function."
    )


# Label sections that carry dose-adjustment guidance for organ impairment.
DOSING_SECTIONS = ["dosage_and_administration", "renal_impairment", "use_in_specific_populations"]


def _has_renal_impairment(patient_record: dict) -> bool:
    """Detect reduced kidney function from labs (eGFR/creatinine) or diagnoses."""
    for lab in patient_record.get("labs", []):
        test = (lab.get("test") or "").lower()
        try:
            value = float(str(lab.get("value")).split()[0])
        except (TypeError, ValueError, IndexError):
            continue
        if "egfr" in test and value < 60:
            return True
        if "creatinine" in test and value > 1.3:
            return True
    for dx in patient_record.get("diagnoses", []):
        name = (dx.get("name") or "").lower()
        if "kidney" in name or "renal" in name or "ckd" in name:
            return True
    return False


def _merge(pool: dict[str, SearchHit], hits: list[SearchHit]) -> None:
    """Insert hits into the pool keyed by id, keeping the highest score."""
    for h in hits:
        existing = pool.get(h.id)
        if existing is None or h.score > existing.score:
            pool[h.id] = h


def _current_med_names(patient_record: dict, proposed_lower: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for med in patient_record.get("medications", []):
        name = (med.get("name") or "").strip()
        key = name.lower()
        if not name or key == proposed_lower or key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def retrieve_chunks(
    drug_name: str,
    patient_record: dict,
    indication: Optional[str] = None,
    top_k: Optional[int] = None,
    store: Optional[VectorStore] = None,
    embedder: Optional[Embedder] = None,
    per_pair_k: Optional[int] = None,
    max_chunks: Optional[int] = None,
    n_label_safety: int = 6,
) -> list[SearchHit]:
    """Retrieve a source-balanced chunk set for the proposed drug + patient.

    Allocates guaranteed slots across label-safety, per-pair interaction, and
    adverse-event sources so no single (much larger) source crowds out the
    others. Returns a de-duplicated list ordered by relevance.
    """
    top_k = top_k or settings.retrieval_top_k
    per_pair_k = per_pair_k or settings.retrieval_per_pair_k
    max_chunks = max_chunks or settings.retrieval_max_chunks
    store = store or VectorStore(settings)
    embedder = embedder or get_embedder(settings)

    proposed_lower = drug_name.strip().lower()
    base_token = proposed_lower.split()[0] if proposed_lower else ""
    pool: dict[str, SearchHit] = {}
    guaranteed_ids: list[str] = []  # ordered by priority

    def add_guaranteed(hit: SearchHit) -> None:
        _merge(pool, [hit])
        if hit.id not in guaranteed_ids:
            guaranteed_ids.append(hit.id)

    # 1. Per-pair interactions (highest priority — a missed DDI is the worst
    #    failure mode). Guarantee the single best chunk for each current med.
    for name in _current_med_names(patient_record, proposed_lower):
        pair_hits = store.search(
            embedder.embed_query(build_pair_query(drug_name, name)), k=per_pair_k
        )
        if not pair_hits:
            continue
        _merge(pool, pair_hits)
        add_guaranteed(pair_hits[0])

    # 2. Proposed drug's own label safety profile. Filter to label safety
    #    sections, then prioritize chunks that are actually about the proposed
    #    drug (its drug_name contains the requested base token).
    safety_hits = store.search(
        embedder.embed_query(build_safety_query(drug_name, patient_record, indication)),
        k=max(n_label_safety * 3, top_k),
        source=SOURCE_LABEL,
        section_types=LABEL_SAFETY_SECTIONS,
    )
    _merge(pool, safety_hits)
    own = [h for h in safety_hits if base_token and base_token in h.drug_name.lower()]
    other = [h for h in safety_hits if h not in own]
    for hit in (own + other)[:n_label_safety]:
        add_guaranteed(hit)

    # 2b. If the patient has renal impairment, guarantee a renal dose-adjustment
    #     chunk for the proposed drug — a high-stakes, commonly-needed fact.
    if _has_renal_impairment(patient_record):
        dose_hits = store.search(
            embedder.embed_query(build_renal_dose_query(drug_name)),
            k=max(top_k, 6),
            source=SOURCE_LABEL,
            section_types=DOSING_SECTIONS,
        )
        _merge(pool, dose_hits)
        dose_own = [h for h in dose_hits if base_token and base_token in h.drug_name.lower()]
        if dose_own:
            add_guaranteed(dose_own[0])
        elif dose_hits:
            add_guaranteed(dose_hits[0])

    # 3. Proposed drug's FAERS adverse-event signal (one slot).
    faers_hits = store.search(
        embedder.embed_query(build_faers_query(drug_name)),
        k=3,
        source=SOURCE_FAERS,
    )
    _merge(pool, faers_hits)
    faers_own = [h for h in faers_hits if base_token and base_token in h.drug_name.lower()]
    if faers_own:
        add_guaranteed(faers_own[0])
    elif faers_hits:
        add_guaranteed(faers_hits[0])

    # 4. General blended query to fill any remaining budget with broad context.
    general_hits = store.search(
        embedder.embed_query(build_query(drug_name, patient_record, indication)), k=top_k
    )
    _merge(pool, general_hits)

    if not pool:
        return []

    # Assemble: guaranteed slots first (priority order), then fill remaining
    # budget with the highest-scoring leftover chunks.
    ordered: list[SearchHit] = []
    used: set[str] = set()
    for cid in guaranteed_ids:
        if len(ordered) >= max_chunks:
            break
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

    ordered.sort(key=lambda h: h.score, reverse=True)
    return ordered[:max_chunks]
