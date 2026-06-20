# MedRAG — Medication Decision Support via RAG

A Retrieval-Augmented Generation system that helps physicians evaluate whether a new medication is appropriate for a specific patient, given their medical history, active diagnoses, and current medication regimen.

---

## What This System Does

When a physician is considering prescribing a new drug, MedRAG:

1. Accepts a **FHIR R4 patient bundle** (`.json`) uploaded by the physician and parses it into a structured clinical summary.
2. Accepts a **proposed medication** and a **free-text clinical question** from the physician.
3. Retrieves the most relevant medical knowledge from a curated index (drug monographs, interaction databases, clinical guidelines, safety literature).
4. Feeds the parsed patient record + retrieved documents to **Claude Opus 4.8**, which reasons over the combined context.
5. Returns a structured safety report covering interactions, contraindications, relevant lab flags, monitoring recommendations, and a top-line recommendation.

The physician reviews this report and makes the final prescribing decision.

---

## System Architecture

```
── INDEX BUILD (runs once) ──────────────────────────────────────────

Medical Documents (drug monographs, guidelines, interaction data)
  ▼
Chunker
  │  Section-level chunking with metadata
  │  (source, drug name, section type, date)
  ▼
Voyage AI voyage-3-large
  │  Embed each chunk → high-dimensional vector
  ▼
Redis Vector Store
  │  Store vectors + metadata via Redis Vector Search (HNSW index)

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
Query Builder
  │  Compose retrieval query: drug name + patient diagnoses + indication
  ▼
Voyage AI voyage-3-large
  │  Embed query → vector
  ▼
Redis Vector Search
  │  Nearest-neighbor search → top-k most relevant chunks
  ▼
Prompt Assembler
  │  Combine: parsed patient record + retrieved chunks +
  │           proposed medication + physician question
  ▼
Claude Opus 4.8  ◄──── CONTEXT.md defines behavior here
  │  Reason, synthesize, flag gaps, apply severity scale
  ▼
Structured Report
  │  Recommendation · Interactions · Contraindications ·
  │  Lab Flags · Monitoring · Alternatives · Uncertainty
  ▼
Physician Review Interface
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Reasoning engine | Claude Opus 4.8 (`claude-opus-4-8`) | Reads retrieved chunks + patient record, generates structured safety report |
| Embedding model | Voyage AI `voyage-3-large` | Converts documents and queries into vectors for semantic search |
| Vector database | Redis (Redis Vector Search) | Stores and searches embeddings via HNSW index; returns top-k relevant chunks |
| Patient data | FHIR R4 (uploaded `.json`) | Source of structured patient record parsed at query time |
| Web framework | Python (Flask or FastAPI) | Serves the physician UI and orchestrates the pipeline |

**Why Claude Opus 4.8:** This task requires deep multi-step reasoning over long interleaved documents (drug monographs can be lengthy), reliable handling of complex patient scenarios with multiple comorbidities, and high accuracy on safety-critical output. The model is instructed by `CONTEXT.md` at system-prompt time.

**Why Voyage AI `voyage-3-large`:** Voyage AI's models are optimized specifically for retrieval tasks and measurably outperform general-purpose embedding models on medical and clinical text. Since retrieval quality directly determines whether a dangerous drug interaction gets surfaced, embedding accuracy is not a place to cut corners.

**Why Redis:** Redis Vector Search provides in-memory nearest-neighbor search with very low latency, making the demo feel responsive. Redis Cloud's free tier requires no infrastructure setup, and its Python client (`redis-py`) integrates cleanly with the rest of the pipeline. Redis is a sponsor of this hackathon.

---

## Physician-Facing Input

The UI presents the physician with three inputs:

**1. FHIR File Upload**
A `.json` file containing the patient's FHIR R4 bundle. This is the primary source of patient data. The FHIR parser extracts the five relevant resource types and surfaces data quality flags for any clinically important missing fields.

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
- Retrieved chunks are ordered by relevance score descending.
- Each chunk includes its source name, section type, and update date so Claude can surface freshness concerns.
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
- Production vector database (an in-memory or local index is sufficient for demo)
- Multi-patient session management
- Audit logging and HIPAA-compliant data handling

**For demo purposes:** Use synthetic FHIR bundles generated by [Synthea](https://github.com/synthetichealth/synthea). Synthea produces realistic FHIR R4 patient records with populated diagnoses, medications, labs, and allergies — ideal for showcasing complex polypharmacy scenarios (e.g., an elderly patient on warfarin, digoxin, and amiodarone being considered for a new drug).

---

## Files in This Repository

| File | Purpose |
|---|---|
| `README.md` | This file. Project overview, architecture, and implementation guide. |
| `CONTEXT.md` | System prompt loaded into Claude Opus 4.8. Defines input/output structure, behavioral rules, and clinical role. |

Additional files to be built:

| File | Purpose |
|---|---|
| `fhir_parser.py` | Parses FHIR R4 bundles into structured patient records. Handles `Patient`, `MedicationRequest`, `Condition`, `AllergyIntolerance`, `Observation`. |
| `retrieval.py` | Query embedding + semantic search over the medical knowledge base (top-k chunks). |
| `prompt_assembly.py` | Combines parsed patient record + retrieved chunks + proposed medication + physician question into Claude's expected input format. |
| `api_client.py` | Anthropic SDK wrapper for calling `claude-opus-4-8`. |
| `app.py` | Web UI: FHIR file upload, medication input, question field, report display. |
| `synthetic_patients/` | Sample Synthea-generated FHIR bundles for demo and testing. |

---

## Quick Start

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
