"""Knowledge-base ingestion for MedRAG.

Builds the vector index from multiple trusted, citable sources:

1. **openFDA drug labels** (FDA structured product labels) — per-drug
   contraindications, warnings, interactions, dosing, etc. Cited to DailyMed.
2. **openFDA FAERS** (adverse-event reports) — most-reported adverse events per
   drug from real-world spontaneous reports. Noisy; treated as signal only.
3. **DDInter 2.0** — structured, severity-rated drug-drug interaction pairs
   (CC BY-NC-SA 4.0; non-commercial use with attribution).

All three are free and require no auth. Chunks carry a ``source`` and
``section_type`` so the reasoning engine and UI can attribute every claim.

Usage
-----
    python ingest.py                      # all sources, drug list from file
    python ingest.py --drugs warfarin metformin lisinopril
    python ingest.py --recreate           # drop & rebuild the index first
    python ingest.py --dry-run            # fetch + chunk, but do not write
    python ingest.py --no-faers --no-ddinter   # labels only

Notes
-----
- openFDA labels reflect FDA-approved labeling. ``effective_time`` is captured
  as the chunk ``date`` so the reasoning engine can flag stale labeling.
- Licensed/paywalled sources (DrugBank commercial, TERIS, ACOG) are deliberately
  NOT fetched here; add them only if you hold the appropriate rights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterable, Optional

import requests

from config import settings
from embeddings import get_embedder
from vector_store import Chunk, VectorStore

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_EVENT_URL = "https://api.fda.gov/drug/event.json"
# Canonical, physician-facing label page keyed by SPL set id.
DAILYMED_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
DDINTER_BASE = "https://ddinter2.scbdd.com"
DDINTER_CSV_URL = DDINTER_BASE + "/static/media/download/ddinter_downloads_code_{code}.csv"
DDINTER_CODES = ["A", "B", "D", "H", "L", "P", "R", "V"]
DDINTER_DATE = "2024-05-14"  # DDInter 2.0 last update

CACHE_DIR = Path("data/cache")
DDINTER_DIR = Path("data/ddinter")
DRUG_LIST_PATH = Path("data/drug_list.txt")
FAERS_TOP_N = 15

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

# Small fallback list if the drug list file is missing.
DEFAULT_DRUGS: list[str] = [
    "warfarin", "metformin", "lisinopril", "amiodarone", "digoxin",
    "atorvastatin", "amlodipine", "metoprolol", "furosemide", "spironolactone",
    "clopidogrel", "sertraline", "ibuprofen", "potassium chloride", "insulin glargine",
]


def load_drug_list(path: Path = DRUG_LIST_PATH) -> list[str]:
    """Read the curated drug list (one generic name per line; '#' comments)."""
    if not path.exists():
        return list(DEFAULT_DRUGS)
    drugs: list[str] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        key = name.lower()
        if key not in seen:
            seen.add(key)
            drugs.append(name)
    return drugs


MAX_CHUNK_CHARS = 1600
MIN_CHUNK_CHARS = 40

# How many candidate labels to fetch before selecting the cleanest match.
CANDIDATE_LIMIT = 25


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _cache_path(generic_name: str) -> Path:
    safe = re.sub(r"[^a-z0-9]+", "_", generic_name.lower()).strip("_")
    return CACHE_DIR / f"{safe}.json"


def _section_richness(label: dict) -> int:
    """How many of the sections we index are present in this label."""
    return sum(1 for field in SECTION_FIELDS if label.get(field))


# Routes that represent systemic exposure (what matters for DDI / dosing).
SYSTEMIC_ROUTES = {"ORAL", "INTRAVENOUS", "INTRAMUSCULAR", "SUBCUTANEOUS"}
# Local/topical routes whose labels (e.g. eye drops, creams) are usually the
# wrong monograph for systemic medication-safety analysis.
LOCAL_ROUTES = {
    "OPHTHALMIC", "TOPICAL", "OTIC", "NASAL", "DENTAL", "RECTAL",
    "VAGINAL", "CUTANEOUS", "TRANSDERMAL", "INHALATION", "RESPIRATORY (INHALATION)",
}


def _route_score(label: dict) -> float:
    """Prefer systemic-route labels over local/topical formulations."""
    routes = {r.upper() for r in ((label.get("openfda", {}) or {}).get("route") or [])}
    if not routes:
        return 0.0
    if routes & SYSTEMIC_ROUTES:
        return 6.0
    if routes <= LOCAL_ROUTES:  # only local routes -> wrong monograph for DDI
        return -6.0
    return 0.0


def _score_candidate(label: dict, requested: str) -> float:
    """Rank a candidate label; higher = cleaner single-ingredient match.

    Combination products (e.g. "SITAGLIPTIN AND METFORMIN HYDROCHLORIDE") are
    penalized so a standalone monograph for the requested drug wins.
    """
    openfda = label.get("openfda", {}) or {}
    generic_names = [g.lower() for g in (openfda.get("generic_name") or [])]
    substances = openfda.get("substance_name") or []
    joined = " ".join(generic_names)
    req = requested.lower().strip()

    score = 0.0
    # Single active ingredient is the strongest signal of a clean monograph.
    n_sub = len(substances)
    if n_sub == 1:
        score += 10
    elif n_sub > 1:
        score -= 5 * (n_sub - 1)  # penalize combination products

    # Name match against the requested drug.
    if req in generic_names:
        score += 5
    elif any(req in g for g in generic_names):
        score += 2

    # Combination-product name markers.
    if " and " not in joined and "/" not in joined and ";" not in joined:
        score += 3

    if "HUMAN PRESCRIPTION DRUG" in (openfda.get("product_type") or []):
        score += 1

    # Prefer systemic routes (oral/IV) over topical/ophthalmic formulations.
    score += _route_score(label)

    # Prefer richer labels (more of the sections we care about).
    score += min(_section_richness(label), 6) * 0.2
    return score


def fetch_label(generic_name: str, use_cache: bool = True) -> Optional[dict]:
    """Fetch the cleanest single-ingredient label from openFDA, with caching.

    Retrieves several candidates and selects the best single-ingredient match
    rather than blindly taking the first result (which may be a combination
    product).
    """
    cache_file = _cache_path(generic_name)
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())

    params = {
        "search": f'openfda.generic_name:"{generic_name}"',
        "limit": CANDIDATE_LIMIT,
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

    label = max(results, key=lambda lbl: _score_candidate(lbl, generic_name))
    chosen = (label.get("openfda", {}) or {}).get("generic_name") or ["?"]
    n_sub = len((label.get("openfda", {}) or {}).get("substance_name") or [])
    req = generic_name.lower()
    matched = any(req in g.lower() for g in chosen)
    # Only warn on genuine mismatches (combo product or name not found).
    if n_sub > 1 or not matched:
        print(f"    [select] {generic_name!r} -> {chosen} ({n_sub} active ingredient(s))")

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
# Source 2: openFDA FAERS adverse-event reports
# ---------------------------------------------------------------------------

def fetch_faers_reactions(generic_name: str) -> list[tuple[str, int]]:
    """Return the top reported adverse-event terms + counts for a drug.

    Uses the openFDA /drug/event count endpoint over FAERS spontaneous reports.
    """
def _faers_cache_path(generic_name: str) -> Path:
    safe = re.sub(r"[^a-z0-9]+", "_", generic_name.lower()).strip("_")
    return CACHE_DIR / f"faers_{safe}.json"


def fetch_faers_reactions(generic_name: str, use_cache: bool = True) -> list[tuple[str, int]]:
    """Top reported adverse-event terms + counts (cached to limit API calls)."""
    cache_file = _faers_cache_path(generic_name)
    if use_cache and cache_file.exists():
        return [tuple(x) for x in json.loads(cache_file.read_text())]

    params = {
        "search": f'patient.drug.openfda.generic_name:"{generic_name}"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": FAERS_TOP_N,
    }
    try:
        resp = requests.get(OPENFDA_EVENT_URL, params=params, timeout=30)
    except requests.RequestException:
        return []
    if resp.status_code != 200:
        return []
    results = resp.json().get("results") or []
    reactions = [(r.get("term", ""), int(r.get("count", 0))) for r in results if r.get("term")]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(reactions))
    return reactions


def build_faers_chunk(generic_name: str, reactions: list[tuple[str, int]]) -> Optional[Chunk]:
    if not reactions:
        return None
    drug_name = generic_name.lower()
    listed = "; ".join(f"{term.lower()} ({count:,} reports)" for term, count in reactions)
    text = (
        f"[{drug_name} — adverse_event_reports] Most frequently reported adverse "
        f"events for {drug_name} in the FDA Adverse Event Reporting System (FAERS): "
        f"{listed}. NOTE: FAERS reports are voluntary, spontaneous reports; counts "
        f"reflect reporting frequency and do NOT establish causation, incidence, or "
        f"that the drug caused the event."
    )
    url = (
        "https://fis.fda.gov/sense/app/95239e26-e0be-42d9-a960-9a5f7f1c25ee/"
        "sheet/7a47a261-d58b-4203-a8aa-6d3021737452/state/analysis"
    )
    uid = hashlib.sha1(f"faers|{drug_name}".encode()).hexdigest()[:16]
    return Chunk(
        id=uid, text=text, source="openFDA FAERS", drug_name=drug_name,
        section_type="adverse_event_reports", date="", url=url, embedding=[],
    )


def build_faers_chunks(drugs: Iterable[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    drugs = list(drugs)
    print(f"Fetching FAERS adverse-event signals for {len(drugs)} drugs...")
    for name in drugs:
        reactions = fetch_faers_reactions(name)
        chunk = build_faers_chunk(name, reactions)
        if chunk:
            chunks.append(chunk)
        time.sleep(0.2)
    print(f"  FAERS chunks: {len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# Source 3: DDInter 2.0 structured drug-drug interactions
# ---------------------------------------------------------------------------

def _download_ddinter(use_cache: bool = True) -> list[Path]:
    """Download DDInter CSVs (by ATC code) into the cache, return their paths."""
    DDINTER_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for code in DDINTER_CODES:
        dest = DDINTER_DIR / f"code_{code}.csv"
        if not (use_cache and dest.exists()):
            try:
                resp = requests.get(DDINTER_CSV_URL.format(code=code), timeout=60)
                if resp.status_code == 200 and resp.content:
                    dest.write_bytes(resp.content)
                else:
                    print(f"  [ddinter] HTTP {resp.status_code} for code {code}")
            except requests.RequestException as exc:
                print(f"  [ddinter] download error for code {code}: {exc}")
        if dest.exists():
            paths.append(dest)
        time.sleep(0.3)
    return paths


def build_ddinter_chunks(drugs: Iterable[str], use_cache: bool = True) -> list[Chunk]:
    """Build one chunk per severity-rated interaction pair where BOTH drugs are
    in our target set (keeps the set relevant and bounded)."""
    target = {d.lower() for d in drugs}
    paths = _download_ddinter(use_cache=use_cache)
    if not paths:
        print("  [ddinter] no data files available; skipping.")
        return []

    chunks: list[Chunk] = []
    seen_pairs: set[tuple[str, str]] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                a = (row.get("Drug_A") or "").strip()
                b = (row.get("Drug_B") or "").strip()
                level = (row.get("Level") or "Unknown").strip()
                if not a or not b:
                    continue
                la, lb = a.lower(), b.lower()
                if la not in target or lb not in target:
                    continue
                pair_key = tuple(sorted((la, lb)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                text = (
                    f"[drug interaction — {level}] {a} + {b}: this combination is "
                    f"classified as a {level} severity drug-drug interaction in "
                    f"DDInter 2.0 (a pharmacist-curated interaction database)."
                )
                uid = hashlib.sha1(f"ddinter|{pair_key[0]}|{pair_key[1]}".encode()).hexdigest()[:16]
                chunks.append(
                    Chunk(
                        id=uid, text=text, source="DDInter 2.0",
                        drug_name=la, section_type="drug_interaction",
                        date=DDINTER_DATE, url=DDINTER_BASE, embedding=[],
                    )
                )
    print(f"  DDInter interaction pairs (both drugs in set): {len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_label_chunks(drugs: Iterable[str]) -> list[Chunk]:
    drugs = list(drugs)
    print(f"Fetching openFDA labels for {len(drugs)} drugs...")
    chunks: list[Chunk] = []
    found = 0
    for name in drugs:
        label = fetch_label(name)
        if not label:
            continue
        found += 1
        chunks.extend(chunk_label(label, name))
        time.sleep(0.2)  # be polite to the public API
    print(f"  labels found: {found}/{len(drugs)} | label chunks: {len(chunks)}")
    return chunks


def ingest(
    drugs: Iterable[str],
    recreate: bool = False,
    dry_run: bool = False,
    embed_batch: int = 64,
    use_labels: bool = True,
    use_faers: bool = True,
    use_ddinter: bool = True,
    use_cache: bool = True,
) -> int:
    drugs = list(drugs)
    all_chunks: list[Chunk] = []

    if use_labels:
        all_chunks.extend(build_label_chunks(drugs))
    if use_ddinter:
        all_chunks.extend(build_ddinter_chunks(drugs, use_cache=use_cache))
    if use_faers:
        all_chunks.extend(build_faers_chunks(drugs))

    print(f"\nTotal chunks across all sources: {len(all_chunks)}")
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

    # Upsert in batches to keep memory/pipeline sizes reasonable at scale.
    written = 0
    for start in range(0, len(all_chunks), 500):
        written += store.upsert(all_chunks[start : start + 500])
    print(f"Upserted {written} chunks. Index now holds {store.count()} docs.")
    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest medical sources into MedRAG's vector store.")
    p.add_argument("--drugs", nargs="*", help="Generic drug names (default: data/drug_list.txt).")
    p.add_argument("--recreate", action="store_true", help="Drop and rebuild the index.")
    p.add_argument("--dry-run", action="store_true", help="Fetch + chunk + embed, no write.")
    p.add_argument("--no-cache", action="store_true", help="Bypass on-disk caches (labels + DDInter).")
    p.add_argument("--no-labels", action="store_true", help="Skip openFDA drug labels.")
    p.add_argument("--no-faers", action="store_true", help="Skip openFDA FAERS adverse events.")
    p.add_argument("--no-ddinter", action="store_true", help="Skip DDInter interactions.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.no_cache:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
    drug_list = args.drugs if args.drugs else load_drug_list()
    ingest(
        drug_list,
        recreate=args.recreate,
        dry_run=args.dry_run,
        use_labels=not args.no_labels,
        use_faers=not args.no_faers,
        use_ddinter=not args.no_ddinter,
        use_cache=not args.no_cache,
    )
