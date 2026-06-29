# MedRAG — Medication Decision Support via RAG

A Retrieval-Augmented Generation system that helps physicians evaluate whether a new medication is appropriate for a specific patient, given their medical history, active diagnoses, and current medication regimen.

---

## What This System Does

When a physician is considering prescribing a new drug, MedRAG:

1. Accepts a **FHIR R4 patient bundle** (`.json`) uploaded by the physician and parses it into a structured clinical summary.
2. Accepts a **proposed medication** and a **free-text clinical question** from the physician.
3. Retrieves the most relevant medical knowledge from a curated index (openFDA drug labels, DDInter 2.0 interaction pairs, openFDA FAERS adverse-event signals).
4. Feeds the parsed patient record + retrieved documents to **Claude Opus 4.8**, which reasons over the combined context.
5. Returns a structured safety report covering interactions, contraindications, relevant lab flags, monitoring recommendations, and a top-line recommendation.

The physician reviews this report and makes the final prescribing decision.

---

## System Architecture

```
── INDEX BUILD (ingest.py, runs once) ───────────────────────────────

Three sources (free, citable):
  • openFDA drug labels      (per-drug safety/dosing)
  • DDInter 2.0              (severity-rated interaction pairs)
  • openFDA FAERS            (real-world adverse-event signals)
  ▼
Chunker
  │  Section-level chunking with metadata
  │  (source, drug name, section type, date, verification URL)
  ▼
BGE  (BAAI/bge-large-en-v1.5, local, 1024-dim)
  │  Embed each chunk → vector   (no API cost)
  ▼
Redis 8 Vector Set  (redis-py client)
  │  Store vectors + JSON attributes (VADD); search with VSIM,
  │  with server-side attribute FILTER for source/section

── QUERY (runs per physician request) ───────────────────────────────

Physician Input
  │  (1) FHIR R4 .json file upload
  │  (2) Proposed medication (name, dose, indication)
  │  (3) Free-text clinical question
  ▼
FHIR Parser
  │  Extract: Patient, MedicationRequest, Condition,
  │           AllergyIntolerance, Observation resources
  │  Output: structured patient record + data quality flags
  ▼
Source-balanced Retrieval (retrieval.py)
  │  Multiple targeted BGE-embedded queries with guaranteed quotas:
  │   • per-pair interaction chunk for EACH current medication
  │   • proposed drug's label safety profile
  │   • renal dose-adjustment chunk (if reduced kidney function)
  │   • proposed drug's FAERS adverse-event signal
  │   • general fill to the remaining budget
  ▼
Prompt Assembler
  │  Combine: parsed patient record + retrieved chunks +
  │           proposed medication + physician question
  ▼
Claude Opus 4.8  ◄──── CONTEXT.md defines behavior here
  │  Reason, synthesize, flag gaps, apply severity scale,
  │  cite each grounded claim as [chunk N]
  ▼
Structured Report  (with clickable citations → source URLs)
  │  Recommendation · Interactions · Contraindications ·
  │  Lab Flags · Monitoring · Alternatives · Uncertainty
  │  References panel: DailyMed · DDInter drug page · FAERS viewer
  ▼
Physician Review Interface (Flask, app.py — single process serves UI + pipeline)
```

---

## How It Runs (Runtime)

MedRAG is a **single-process Flask application**. There is no separate frontend server — `app.py` renders the styled UI and orchestrates the full pipeline in one Python process. The static files in `frontend/` are design mockups only; the live UI is embedded in `app.py`.

### Processes required at runtime

| Process | Purpose | Start command |
|---|---|---|
| **Redis 8** | Stores ~20k embedded knowledge chunks | `brew services start redis` or `redis-server` |
| **Flask app** | UI + pipeline orchestration | `source .venv/bin/activate && python app.py` |

The **BGE embedding model** is loaded into RAM when `app.py` starts (not on every request). You will see:

```
[MedRAG] Loading embedder…
[MedRAG] Embedder ready.
* Running on http://127.0.0.1:5001
```

The app listens on **port 5001** (5000 is often occupied by macOS AirPlay). Open **http://127.0.0.1:5001** in a browser.

> **Use the `.venv` Python.** Running `python app.py` from conda base or system Python will fail at analysis time because `sentence-transformers` is installed in the project venv only.

### Startup sequence (`python app.py`)

1. Load settings from `.env` via `config.py`
2. Preload **BGE** (`BAAI/bge-large-en-v1.5`) — takes ~10–30 seconds on first start
3. Start Flask on `127.0.0.1:5001` (debug mode on, auto-reloader **off** — the reloader conflicts with ML model loading)

The embedder is cached as a **process singleton** in `embeddings.py` so it is loaded once per server lifetime.

### Per-request flow (when you submit an analysis)

1. **FHIR parse** (`fhir_parser.py`) — uploaded `.json` (or demo patient) → structured patient record + data quality flags
2. **Retrieval** (`retrieval.py`) — multiple targeted vector searches against Redis (~14 chunks max by default):
   - Best interaction chunk per current medication (DDInter)
   - Proposed drug's label safety sections (openFDA → DailyMed)
   - Renal dose chunk if patient has reduced kidney function
   - FAERS adverse-event signal for the proposed drug
   - General semantic fill to the remaining budget
3. **Prompt assembly** (`prompt_assembly.py`) — patient + chunks + proposed drug + question
4. **Claude Opus 4.8** (`api_client.py`) — `CONTEXT.md` as system prompt; returns markdown report with `[chunk N]` citations (~30–60 s)
5. **Report render** (`app.py`) — markdown parsed into styled section cards; citations link to References & Sources panel

### Index build vs. query

| Step | When | Command |
|---|---|---|
| **Index build** | One-time (or when drugs/sources change) | `python ingest.py --recreate` (~25 min, ~20k chunks) |
| **Query / UI** | Every session | `python app.py` (requires Redis already populated) |

`ingest.py` is **not** run on every startup.

## Tech Stack

### Core pipeline

| Layer | Technology | Role |
|---|---|---|
| **Language** | Python 3.11+ | Entire backend pipeline, ingestion, retrieval, and web server |
| **Reasoning engine** | [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) → **Claude Opus 4.8** (`claude-opus-4-8`) | Reads retrieved chunks + patient record; generates structured safety report. System prompt = `CONTEXT.md`. |
| **Embedding model (default)** | [sentence-transformers](https://github.com/UKPLab/sentence-transformers) → **BGE** `BAAI/bge-large-en-v1.5` (1024-dim, via [Hugging Face Hub](https://huggingface.co/BAAI/bge-large-en-v1.5)) | Local document/query embeddings for semantic search — no per-embedding API cost |
| **Embedding model (optional)** | [Voyage AI SDK](https://github.com/voyage-ai/voyageai-python) → `voyage-3-large` | Cloud embeddings when `EMBED_PROVIDER=voyage` |
| **Vector database** | **Redis 8** native **vector sets** via [redis-py](https://github.com/redis/redis-py) (`redis>=5.0.0`) | Stores ~20k chunk embeddings + JSON metadata. Commands: `VADD`, `VSIM`, `VGETATTR`, `VSETATTR`, `VCARD`. Cosine KNN with server-side `FILTER` on attributes. |
| **Numerics** | [NumPy](https://numpy.org/) | Vector serialization (FLOAT32), stub embedder, similarity ops |
| **Config** | [python-dotenv](https://github.com/theskumar/python-dotenv) | Loads `.env` secrets and tuning parameters |
| **HTTP client** | [Requests](https://requests.readthedocs.io/) | Fetches openFDA APIs and DDInter CSVs during ingestion |

### Web UI & orchestration

| Layer | Technology | Role |
|---|---|---|
| **Web framework** | [Flask](https://flask.palletsprojects.com/) 3.x (`app.py`) | Single process: serves UI, orchestrates pipeline, hosts FAERS viewer at `/source/faers/<drug>` |
| **Templating** | Jinja2 (via Flask `render_template_string`) | Server-side HTML for setup page, report cards, and FAERS viewer |
| **CSS / styling** | [Tailwind CSS](https://tailwindcss.com/) (CDN) | Dark glass-panel "bento" layout |
| **Typography / icons** | Google Fonts (Hanken Grotesk, Inter, Geist) + Material Symbols | UI typography and iconography |

### Patient data standards

| Standard | Role in MedRAG |
|---|---|
| **FHIR R4** | Patient input format — uploaded `.json` Bundle parsed by `fhir_parser.py` |
| **LOINC** | Lab/demographic Observation codes (e.g. eGFR, body weight, smoking status) |
| **ICD-10** | Condition codes extracted from `Condition` resources when present |
| **[Synthea](https://github.com/synthetichealth/synthea)** | Recommended tool for generating demo FHIR R4 bundles |

### External knowledge APIs & databases (ingestion)

| Source | API / access | Used for |
|---|---|---|
| **openFDA Drug Label API** | `https://api.fda.gov/drug/label.json` | FDA structured product labels → chunked by section |
| **openFDA FAERS API** | `https://api.fda.gov/drug/event.json` | Top reported adverse events per drug (signal only) |
| **[DailyMed](https://dailymed.nlm.nih.gov/)** | SPL set-id URLs | Physician-facing citation target for label chunks |
| **[DDInter 2.0](https://ddinter2.scbdd.com/)** | Downloadable CSVs + drug-detail web pages | Severity-rated drug–drug interaction pairs |
| **FDA FAERS Public Dashboard** | Linked from FAERS viewer | Broader FAERS exploration (no per-drug deep link) |

### Python dependencies (`requirements.txt`)

| Package | Purpose |
|---|---|
| `redis>=5.0.0` | Redis client (vector sets, ping, attribute storage) |
| `sentence-transformers>=3.0.0` | Local BGE embeddings (default) |
| `anthropic>=0.40.0` | Claude API client |
| `voyageai>=0.3.0` | Optional cloud embeddings |
| `flask>=3.0.0` | Web UI and pipeline orchestration |
| `python-dotenv>=1.0.0` | Environment variable loading |
| `requests>=2.31.0` | openFDA / DDInter HTTP fetching |
| `numpy>=1.26.0` | Vector operations |

### CLI & utilities

| Tool | Role |
|---|---|
| `ingest.py` | Index builder — fetch, chunk, embed, upsert to Redis |
| `run_case.py` | CLI end-to-end runner (`--retrieval-only` for testing without Claude) |
| `argparse` (stdlib) | CLI argument parsing for `ingest.py` and `run_case.py` |

**Why Claude Opus 4.8:** This task requires deep multi-step reasoning over long interleaved documents (drug monographs can be lengthy), reliable handling of complex patient scenarios with multiple comorbidities, and high accuracy on safety-critical output. The model is instructed by `CONTEXT.md` at system-prompt time.

**Why local BGE embeddings:** `BAAI/bge-large-en-v1.5` is a strong open retrieval model that runs locally via `sentence-transformers`, so there is **no per-embedding API cost** — important when embedding ~20k chunks and re-running ingestion freely. Weights are downloaded from Hugging Face Hub on first use. The embedding provider is pluggable (`EMBED_PROVIDER` in `.env`); Voyage AI is also supported if an API key is preferred. Query and document vectors must use the same model, so changing providers requires re-ingesting.

**Why Redis 8 vector sets:** Redis 8 ships native **vector sets** in core (no separate Redis Stack / RediSearch module needed), so the system runs against a stock local Redis install. `VSIM` gives low-latency cosine KNN, and its `FILTER` expression supports server-side filtering on chunk attributes (e.g. restrict a query to a drug's label-safety sections), which the source-balanced retrieval relies on. Running locally also keeps PHI off third-party infrastructure. Redis is a sponsor of this hackathon.

---

## Physician-Facing Input

The UI presents the physician with three inputs:

**1. FHIR File Upload**
A `.json` file containing the patient's **FHIR R4 Bundle** (`"resourceType": "Bundle"` with an `"entry"` array). Synthea exports work. Single-resource JSON (e.g. a lone `Patient` object) will parse but produce an empty clinical record. Alternatively, check **Use demo patient** to skip upload.

**2. Proposed Medication Field**
A structured input capturing: generic drug name, proposed dose and route, and the indication being considered. Generic name is preferred since that is how drug monographs and interaction databases are indexed.

**3. Free-Text Clinical Question**
An open text field where the physician asks their specific question (e.g., "Is this dose safe given the patient's renal function?" or "Are there interactions with her current warfarin regimen?"). If left blank, the system defaults to a general safety assessment. This field is the physician's direct line to Claude — it focuses the report on what matters most to the clinician in this case.

---

## FHIR Parsing

FHIR R4 bundles are JSON files. The parser targets five resource types and maps them to Claude's expected input structure. Everything else in the bundle is ignored for the MVP.

### Resources Parsed

| FHIR Resource | Fields Extracted | Maps To |
|---|---|---|
| `Patient` | `birthDate`, `gender`, `extension` (weight, smoking, pregnancy) | Demographics |
| `MedicationRequest` | `medicationCodeableConcept`, `dosageInstruction`, `status` | Current Medications |
| `Condition` | `code` (ICD-10), `clinicalStatus`, `verificationStatus` | Active Diagnoses |
| `AllergyIntolerance` | `code`, `reaction.manifestation`, `criticality` | Known Allergies |
| `Observation` | `code`, `valueQuantity`, `referenceRange`, `effectiveDateTime` | Lab Values |

### Key Limitations to Surface in the UI

**Medications reflect prescriptions, not adherence.** `MedicationRequest` records what was prescribed. The patient may have stopped a medication, be non-adherent, or be taking OTC drugs, supplements, or medications from outside providers that will not appear in this record. The physician should always confirm the current medication list before relying on it for interaction checking.

**FHIR records are point-in-time snapshots.** The uploaded file reflects the record at the time of export. It may not include recent labs, new diagnoses, or medication changes made after the export date.

**Condition coding quality varies.** ICD-10 codes may be present or absent depending on the originating system. The parser falls back to the condition display name when codes are missing.

### Parsed Output Format

```json
{
  "demographics": {
    "age": 67,
    "sex": "female",
    "weight_kg": 72,
    "pregnancy_status": null,
    "smoking_status": "former smoker"
  },
  "diagnoses": [
    { "name": "Type 2 diabetes mellitus", "icd10": "E11.9", "status": "active" },
    { "name": "Chronic kidney disease, stage 3", "icd10": "N18.3", "status": "active" }
  ],
  "medications": [
    {
      "name": "metformin",
      "dose": "500mg",
      "route": "oral",
      "frequency": "twice daily",
      "status": "active",
      "source": "MedicationRequest"
    }
  ],
  "allergies": [
    { "substance": "penicillin", "reaction": "anaphylaxis", "criticality": "high" }
  ],
  "labs": [
    {
      "test": "eGFR",
      "value": 38,
      "unit": "mL/min/1.73m²",
      "reference_range": ">60",
      "date": "2026-05-14"
    }
  ],
  "data_quality_flags": [
    "Hepatic function labs (ALT, AST, bilirubin) not present in FHIR record.",
    "Body weight sourced from Observation — verify currency."
  ]
}
```

### FHIR Parser Implementation (Skeleton)

```python
def parse_fhir_bundle(bundle: dict) -> dict:
    patient_record = {
        "demographics": {},
        "diagnoses": [],
        "medications": [],
        "allergies": [],
        "labs": [],
        "data_quality_flags": []
    }

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")

        if resource_type == "Patient":
            patient_record["demographics"] = extract_demographics(resource)

        elif resource_type == "MedicationRequest":
            if resource.get("status") == "active":
                patient_record["medications"].append(extract_medication(resource))

        elif resource_type == "Condition":
            if resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code") == "active":
                patient_record["diagnoses"].append(extract_condition(resource))

        elif resource_type == "AllergyIntolerance":
            patient_record["allergies"].append(extract_allergy(resource))

        elif resource_type == "Observation":
            lab = extract_observation(resource)
            if lab:
                patient_record["labs"].append(lab)

    patient_record["data_quality_flags"] = generate_data_quality_flags(patient_record)
    return patient_record
```

---

## Prompt Assembly

The prompt assembler takes the parsed patient record, retrieved knowledge chunks, proposed medication, and physician question and formats them into the structured input Claude expects. See `CONTEXT.md` for the exact input contract.

Key assembly decisions:
- Retrieved chunks are ordered by relevance score descending and labeled `[chunk N]` so Claude can cite each grounded claim; the UI turns those citations into clickable links to the source.
- Each chunk includes its source name, section type, update date, and verification URL so Claude can surface freshness concerns and the physician can verify. Citation targets in the UI:
  - **openFDA labels** → DailyMed label page (by SPL set id)
  - **DDInter 2.0** → drug-detail page on ddinter2.scbdd.com (specific to the indexed drug)
  - **openFDA FAERS** → human-readable viewer at `/source/faers/<drug>` (reaction-term table, with links to the raw openFDA API query and the FDA FAERS Public Dashboard)
- Data quality flags from the FHIR parser are included in the patient record section so Claude is aware of what is missing.
- The physician's free-text question appears last, immediately before Claude's response, so it is the most proximal instruction.

---

## Output Structure

Claude returns a structured report with seven sections:

1. **Recommendation Summary** — Safe to Prescribe / Proceed with Caution / Contraindicated, with one-sentence rationale and direct answer to the physician's question.
2. **Drug–Drug Interactions** — Severity-rated list (Minor / Moderate / Major / Contraindicated) with mechanism and recommended action.
3. **Contraindications and Precautions** — Absolute and relative, tied to this patient's specific record.
4. **Relevant Lab Values** — Flags for values affecting prescribing safety, with threshold references from retrieved guidelines.
5. **Monitoring Recommendations** — Specific labs, vitals, symptoms, and timing.
6. **Alternative Considerations** — Only from retrieved context; never fabricated.
7. **Knowledge Gaps and Uncertainty** — Explicit flags for what the system does not know, including FHIR data limitations and retrieval gaps.

---

## MVP Scope (Hackathon)

The current implementation focuses on demonstrating the core pipeline end-to-end. The following are explicitly out of scope for the MVP:

- Patient authentication and SMART on FHIR OAuth flows
- Real-time EHR integration (FHIR is uploaded manually as a file)
- Production vector database (Redis runs locally for demo; no cloud persistence or HA)
- Multi-patient session management
- Audit logging and HIPAA-compliant data handling

**For demo purposes:** Use synthetic FHIR bundles generated by [Synthea](https://github.com/synthetichealth/synthea). Synthea produces realistic FHIR R4 patient records with populated diagnoses, medications, labs, and allergies — ideal for showcasing complex polypharmacy scenarios (e.g., an elderly patient on warfarin, digoxin, and amiodarone being considered for a new drug).

---

## Files in This Repository

| File | Purpose |
|---|---|
| `README.md` | This file. Project overview, architecture, and implementation guide. |
| `CONTEXT.md` | System prompt loaded into Claude Opus 4.8. Defines input/output structure, behavioral rules, and clinical role. |
| `config.py` | Centralized, typed settings loaded from environment / `.env` via `python-dotenv`. |
| `fhir_parser.py` | Parses FHIR R4 bundles into structured patient records. Handles `Patient`, `MedicationRequest`, `Condition`, `AllergyIntolerance`, `Observation`. Uses LOINC and ICD-10 where present. |
| `embeddings.py` | Embedder interface + implementations: `BGEEmbedder` (default, local), `VoyageEmbedder`, `StubEmbedder`. Cached as a process singleton; preloaded at app startup. |
| `vector_store.py` | Redis 8 vector-set wrapper (`redis-py`): upsert (`VADD`), KNN search with attribute `FILTER` (`VSIM`), attribute read/update (`VGETATTR`/`VSETATTR`). |
| `ingest.py` | Builds the index from all three sources via `requests` (openFDA APIs, DDInter CSVs); chunks, embeds with BGE, upserts to Redis. Reads drug universe from `data/drug_list.txt`. |
| `retrieval.py` | Source-balanced multi-query retrieval with per-source quotas (per-pair interactions, label safety, renal dosing, FAERS). |
| `prompt_assembly.py` | Combines parsed patient record + retrieved chunks + proposed medication + physician question into Claude's expected input format. |
| `api_client.py` | Anthropic SDK wrapper for calling `claude-opus-4-8`; loads `CONTEXT.md` as the system prompt. |
| `app.py` | Flask web UI: FHIR upload, medication/question input, rendered report with clickable `[chunk N]` citations + a References & Sources panel. Serves a human-readable FAERS viewer at `/source/faers/<drug>`. |
| `run_case.py` | CLI to run a single patient + drug case end-to-end (or `--retrieval-only`) for testing without the web UI. |
| `data/drug_list.txt` | Curated ~270 commonly-prescribed generic drug names (the index's drug universe). Edit to expand coverage. |
| `synthetic_patients/` | Sample FHIR R4 bundles for demo and testing (e.g. elderly polypharmacy, CKD metformin contraindication, low-risk control). |
| `frontend/` | Standalone styled UI mockups (`index.html`, `setup.html`, `report.html`). Not yet wired to the backend — the live UI is served by `app.py`. |

> Generated/large artifacts (`dump.rdb`, `data/cache/`, `data/ddinter/`) are git-ignored and rebuilt by `ingest.py`.

---

## Knowledge Base & Data Sources

The vector index is built by `ingest.py` from three free, citable sources. Every
chunk records its `source`, `section_type`, `date`, and a verification `url`, so
the report can attribute each grounded claim back to a primary source.

| Source | What it provides | `section_type`(s) | Citation target (UI) | License |
|---|---|---|---|---|
| **openFDA Drug Label** | FDA structured product labeling — contraindications, warnings, interactions, dosing, use in specific populations, renal/hepatic adjustments | `boxed_warning`, `contraindications`, `warnings_and_cautions`, `drug_interactions`, `dosage_and_administration`, `use_in_specific_populations`, `renal_impairment`, `hepatic_impairment`, ... | DailyMed label page | Public domain |
| **DDInter 2.0** | Pharmacist-curated, **severity-rated** drug–drug interaction pairs (Major / Moderate / Minor) | `drug_interaction` | DDInter drug-detail page (`/server/drug-detail/{id}/`) | CC BY-NC-SA 4.0 (non-commercial) |
| **openFDA FAERS** | Most-reported real-world adverse events per drug (spontaneous reports; signal only, not causation) | `adverse_event_reports` | MedRAG FAERS viewer (`/source/faers/<drug>`) — table of reaction terms + counts; links to raw openFDA API and FDA dashboard | Public domain |

The drug universe is defined in `data/drug_list.txt` (~270 commonly-prescribed
generics; edit this file to add or remove drugs). DDInter interaction chunks are
created only for pairs where **both** drugs are in this set, keeping the index
relevant and bounded.

### Ingestion commands

```bash
# Build the full index from all three sources (drops & rebuilds)
python ingest.py --recreate

# A subset of drugs
python ingest.py --drugs warfarin amiodarone digoxin --recreate

# Labels only (skip interactions / adverse events)
python ingest.py --no-ddinter --no-faers --recreate

# Fetch + chunk + embed without writing to Redis (sanity check)
python ingest.py --dry-run
```

> **Note on licensing:** DDInter is CC BY-NC-SA 4.0 (non-commercial use with
> attribution) — appropriate for this proof-of-concept, but commercial
> deployment would require a properly licensed interaction source (e.g.
> DrugBank, Lexicomp, Micromedex). openFDA data is U.S. public domain.

---

## Running It

### One-time setup

```bash
cd /path/to/MedRAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY; defaults: BGE + local Redis

brew services start redis       # or: redis-server
python ingest.py --recreate     # ~20k chunks, ~25 min on CPU — skip if index already built
```

### Every session (web UI)

```bash
source .venv/bin/activate
python app.py                   # wait for [MedRAG] Embedder ready.
```

Open **http://127.0.0.1:5001** — upload a FHIR R4 **Bundle** (`.json`), enter the proposed drug, and submit.

**Stop and restart** (if port 5001 is in use):

```bash
kill $(lsof -t -i:5001) 2>/dev/null
source .venv/bin/activate && python app.py
```

**Open in Chrome from terminal:**

```bash
open -a "Google Chrome" http://127.0.0.1:5001
```

### CLI (no web UI)

```bash
source .venv/bin/activate

python run_case.py \
  --patient synthetic_patients/elderly_polypharmacy.json \
  --drug ciprofloxacin --dose 500mg --route oral --indication "complicated UTI" \
  --question "Is ciprofloxacin safe with her warfarin, and what dose given eGFR 34?"
# add --retrieval-only to inspect retrieved chunks without a (paid) Claude call
```

### Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for report generation |
| `REDIS_URL` | `redis://localhost:6379` | Local Redis connection |
| `EMBED_PROVIDER` | `bge` | `bge` (local) \| `voyage` \| `stub` |
| `RETRIEVAL_TOP_K` | `8` | Chunks from general proposed-drug query |
| `RETRIEVAL_PER_PAIR_K` | `3` | Chunks per current-medication interaction query |
| `RETRIEVAL_MAX_CHUNKS` | `14` | Hard cap on chunks passed to Claude |

See `.env.example` for BGE model settings and full list.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `Address already in use` on 5001 | `kill $(lsof -t -i:5001)` then restart |
| `ModuleNotFoundError: sentence_transformers` | Activate venv: `source .venv/bin/activate` |
| `BrokenPipeError` on first analysis | Restart with current `app.py` (embedder preloads at startup) |
| Empty / weak report | Patient JSON may not be a FHIR R4 **Bundle** with `"entry": [...]` |
| Redis connection error | `redis-cli ping` → should print `PONG` |
| Blank page in browser | Use port **5001**, not 5000 |

---

## Quick Start (library usage)

```python
import json
from anthropic import Anthropic
from fhir_parser import parse_fhir_bundle
from retrieval import retrieve_chunks
from prompt_assembly import build_prompt

client = Anthropic()

def evaluate_medication(
    fhir_path: str,
    proposed_drug: dict,
    physician_question: str
) -> str:

    # 1. Parse the uploaded FHIR file
    with open(fhir_path, "r") as f:
        bundle = json.load(f)
    patient_record = parse_fhir_bundle(bundle)

    # 2. Retrieve relevant medical knowledge
    chunks = retrieve_chunks(proposed_drug["name"], patient_record)

    # 3. Assemble the prompt
    prompt = build_prompt(patient_record, proposed_drug, chunks, physician_question)

    # 4. Load system context
    with open("CONTEXT.md", "r") as f:
        system_context = f.read()

    # 5. Call Claude Opus 4.8
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=4096,
        system=system_context,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


# Example usage
report = evaluate_medication(
    fhir_path="synthetic_patients/elderly_polypharmacy.json",
    proposed_drug={
        "name": "lisinopril",
        "dose": "10mg",
        "route": "oral",
        "indication": "hypertension"
    },
    physician_question="Any concerns given the patient's CKD and current potassium levels?"
)

print(report)
```

---

## Safety Design Principles

**Grounding over hallucination.** Claude is instructed to base all specific clinical claims on the retrieved context, not parametric memory. When retrieved context is silent on a question, the system says so rather than inferring.

**Explicit uncertainty.** The system distinguishes between "not a concern per retrieved documents" and "not addressed in retrieved documents." These are not the same thing, and physicians see the difference.

**Medication list reliability flagging.** FHIR `MedicationRequest` records reflect prescriptions, not confirmed adherence. The system is explicitly instructed to flag this limitation when a potential interaction is severe, and to recommend the physician confirm current use before relying on the interaction assessment.

**Missing data flagging.** The FHIR parser generates data quality flags for clinically relevant missing fields. Claude is instructed to surface these flags and not assume normal values for missing data.

**No autonomous decisions.** Every report ends with a statement that the prescribing decision rests with the licensed clinician.

---

## Intended Users

- **Primary:** Licensed physicians and prescribing clinicians using the system as a decision-support tool during medication review.
- **Secondary:** Clinical pharmacists reviewing complex polypharmacy cases.
- **Out of scope:** Patients, non-clinical staff, or any use case where the output would bypass physician review.

---

## Regulatory Note

Systems that support clinical decision-making may fall under FDA oversight as Software as a Medical Device (SaMD). This MVP is a proof-of-concept only and is not intended for use with real patients. Before any clinical deployment, legal review of the FDA clinical decision support framework and applicable HIPAA requirements is required.
