"""Knowledge-base ingestion for MedRAG.

Fetches FDA structured product labels from the openFDA Drug Label API
(https://open.fda.gov/apis/drug/label/ — free, public, no auth required),
chunks each label by clinically relevant SPL section, embeds the chunks, and
upserts them into the Redis vector index.

Usage
-----
    python ingest.py                      # ingest the default seed drug list
    python ingest.py --drugs warfarin metformin lisinopril
    python ingest.py --recreate           # drop & rebuild the index first
    python ingest.py --dry-run            # fetch + chunk, but do not write

Notes
-----
- openFDA labels reflect FDA-approved labeling. ``effective_time`` is captured
  as the chunk ``date`` so the reasoning engine can flag stale labeling.
- Only quantitative, safety-relevant sections are indexed (see SECTION_FIELDS).
- Licensed/paywalled sources (DrugBank commercial, TERIS, ACOG) are deliberately
  NOT fetched here; add them only if you hold the appropriate rights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, Optional

import requests

from config import settings
from embeddings import get_embedder
from vector_store import Chunk, VectorStore

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
# Canonical, physician-facing label page keyed by SPL set id.
DAILYMED_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
CACHE_DIR = Path("data/cache")

# openFDA SPL field -> human-readable section_type stored as metadata.
# Ordered roughly by clinical importance for prescribing safety.
SECTION_FIELDS: dict[str, str] = {
    "boxed_warning": "boxed_warning",
    "contraindications": "contraindications",
    "warnings_and_cautions": "warnings_and_cautions",
    "warnings": "warnings",
    "drug_interactions": "drug_interactions",
    "dosage_and_administration": "dosage_and_administration",
    "use_in_specific_populations": "use_in_specific_populations",
    "pregnancy": "pregnancy",
    "geriatric_use": "geriatric_use",
    "renal_impairment": "renal_impairment",
    "hepatic_impairment": "hepatic_impairment",
    "adverse_reactions": "adverse_reactions",
    "clinical_pharmacology": "clinical_pharmacology",
}

# Default seed list: the README's demo drugs + common Synthea polypharmacy meds.
DEFAULT_DRUGS: list[str] = [
    "warfarin sodium",
    "metformin hydrochloride",
    "lisinopril",
    "amiodarone hydrochloride",
    "digoxin",
    "atorvastatin calcium",
    "amlodipine besylate",
    "metoprolol tartrate",
    "furosemide",
    "spironolactone",
    "clopidogrel bisulfate",
    "sertraline hydrochloride",
    "ibuprofen",
    "potassium chloride",
    "insulin glargine",
]

MAX_CHUNK_CHARS = 1600
MIN_CHUNK_CHARS = 40


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _cache_path(generic_name: str) -> Path:
    safe = re.sub(r"[^a-z0-9]+", "_", generic_name.lower()).strip("_")
    return CACHE_DIR / f"{safe}.json"


def fetch_label(generic_name: str, use_cache: bool = True) -> Optional[dict]:
    """Fetch one drug label from openFDA, with on-disk caching."""
    cache_file = _cache_path(generic_name)
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())

    params = {
        "search": f'openfda.generic_name:"{generic_name}"',
        "limit": 1,
    }
    try:
        resp = requests.get(OPENFDA_LABEL_URL, params=params, timeout=30)
    except requests.RequestException as exc:
        print(f"  [fetch] network error for {generic_name!r}: {exc}")
        return None

    if resp.status_code == 404:
        print(f"  [fetch] no label found for {generic_name!r}")
        return None
    if resp.status_code != 200:
        print(f"  [fetch] HTTP {resp.status_code} for {generic_name!r}")
        return None

    results = resp.json().get("results") or []
    if not results:
        return None

    label = results[0]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(label))
    return label


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split long section text into <=max_chars chunks on paragraph/sentence."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    paragraphs = re.split(r"\n\s*\n", text)
    buf = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 1 <= max_chars:
            buf = f"{buf}\n{para}".strip()
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        if len(para) <= max_chars:
            buf = para
        else:
            # Hard-split an oversized paragraph on sentence boundaries.
            sentences = re.split(r"(?<=[.;])\s+", para)
            for sent in sentences:
                if len(buf) + len(sent) + 1 <= max_chars:
                    buf = f"{buf} {sent}".strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = sent[:max_chars]
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def chunk_label(label: dict, requested_name: str) -> list[Chunk]:
    """Turn one openFDA label into section-level chunks with metadata."""
    openfda = label.get("openfda", {}) or {}
    names = openfda.get("generic_name") or [requested_name]
    drug_name = (names[0] if names else requested_name).lower()
    date = label.get("effective_time", "") or ""
    if re.fullmatch(r"\d{8}", date):  # YYYYMMDD -> YYYY-MM-DD
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

    # Build a clickable DailyMed link to the full label so physicians can verify.
    set_id = label.get("set_id") or (openfda.get("spl_set_id") or [""])[0]
    url = DAILYMED_URL.format(set_id=set_id) if set_id else ""

    chunks: list[Chunk] = []
    for field, section_type in SECTION_FIELDS.items():
        raw = label.get(field)
        if not raw:
            continue
        section_text = "\n\n".join(raw) if isinstance(raw, list) else str(raw)
        for i, piece in enumerate(_split_text(section_text)):
            uid = hashlib.sha1(
                f"{drug_name}|{section_type}|{i}|{piece[:64]}".encode()
            ).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=uid,
                    text=f"[{drug_name} — {section_type}] {piece}",
                    source="openFDA",
                    drug_name=drug_name,
                    section_type=section_type,
                    date=date,
                    url=url,
                    embedding=[],
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def ingest(
    drugs: Iterable[str],
    recreate: bool = False,
    dry_run: bool = False,
    embed_batch: int = 64,
) -> int:
    drugs = list(drugs)
    print(f"Ingesting {len(drugs)} drug label(s) from openFDA...")

    all_chunks: list[Chunk] = []
    for name in drugs:
        label = fetch_label(name)
        if not label:
            continue
        drug_chunks = chunk_label(label, name)
        print(f"  {name}: {len(drug_chunks)} chunks")
        all_chunks.extend(drug_chunks)
        time.sleep(0.3)  # be polite to the public API

    print(f"Total chunks: {len(all_chunks)}")
    if not all_chunks:
        print("Nothing to ingest.")
        return 0

    embedder = get_embedder(settings)
    print(f"Embedding with: {type(embedder).__name__} (dim={embedder.dim})")
    texts = [c.text for c in all_chunks]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), embed_batch):
        batch = texts[start : start + embed_batch]
        vectors.extend(embedder.embed_documents(batch))
        print(f"  embedded {min(start + embed_batch, len(texts))}/{len(texts)}")
    for chunk, vec in zip(all_chunks, vectors):
        chunk.embedding = vec

    if dry_run:
        print("Dry run: skipping Redis write.")
        return len(all_chunks)

    store = VectorStore(settings)
    if not store.ping():
        raise ConnectionError("Could not reach Redis. Check REDIS_URL / network.")
    store.create_index(recreate=recreate)
    written = store.upsert(all_chunks)
    print(f"Upserted {written} chunks. Index now holds {store.count()} docs.")
    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest drug labels into MedRAG's vector store.")
    p.add_argument("--drugs", nargs="*", help="Generic drug names (default: seed list).")
    p.add_argument("--recreate", action="store_true", help="Drop and rebuild the index.")
    p.add_argument("--dry-run", action="store_true", help="Fetch + chunk + embed, no write.")
    p.add_argument("--no-cache", action="store_true", help="Bypass on-disk label cache.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.no_cache:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
    drug_list = args.drugs if args.drugs else DEFAULT_DRUGS
    ingest(drug_list, recreate=args.recreate, dry_run=args.dry_run)
