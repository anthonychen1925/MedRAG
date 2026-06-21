"""Prompt assembly for MedRAG.

Combines the parsed patient record, retrieved knowledge chunks, proposed
medication, and physician question into the exact input contract Claude expects
(see CONTEXT.md "Input Structure").

Assembly decisions (from README.md):
- Retrieved chunks are ordered by relevance score descending.
- Each chunk includes source / drug name / section type / date so Claude can
  surface freshness concerns.
- Data quality flags are included so Claude is aware of what is missing.
- The physician's free-text question appears LAST, immediately before Claude's
  response, so it is the most proximal instruction.
"""

from __future__ import annotations

from typing import Iterable, Union

from vector_store import SearchHit

DEFAULT_QUESTION = "Please provide a general safety assessment for this medication."

ChunkLike = Union[SearchHit, dict]


def _g(chunk: ChunkLike, attr: str, default: str = "") -> str:
    if isinstance(chunk, dict):
        return chunk.get(attr, default) or default
    return getattr(chunk, attr, default) or default


def _format_demographics(demo: dict) -> str:
    lines = []
    age = demo.get("age")
    sex = demo.get("sex")
    lines.append(f"    - Age: {age if age is not None else 'unknown'}, "
                 f"Sex: {sex or 'unknown'}")
    weight = demo.get("weight_kg")
    bmi = demo.get("bmi")
    if weight is not None or bmi is not None:
        wparts = []
        if weight is not None:
            wparts.append(f"{weight} kg")
        if bmi is not None:
            wparts.append(f"BMI {bmi}")
        lines.append(f"    - Weight/BMI: {', '.join(wparts)}")
    if demo.get("pregnancy_status") is not None:
        lines.append(f"    - Pregnancy status: {demo['pregnancy_status']}")
    if demo.get("smoking_status") is not None:
        lines.append(f"    - Smoking status: {demo['smoking_status']}")
    return "\n".join(lines)


def _format_diagnoses(diagnoses: list[dict]) -> str:
    if not diagnoses:
        return "    - None recorded"
    out = []
    for d in diagnoses:
        icd = f" ({d['icd10']})" if d.get("icd10") else ""
        out.append(f"    - {d.get('name', 'unknown')}{icd} [{d.get('status', 'unknown')}]")
    return "\n".join(out)


def _format_medications(meds: list[dict]) -> str:
    if not meds:
        return "    - None recorded"
    out = []
    for m in meds:
        bits = [m.get("name", "unknown")]
        if m.get("dose"):
            bits.append(m["dose"])
        if m.get("route"):
            bits.append(m["route"])
        if m.get("frequency"):
            bits.append(m["frequency"])
        out.append(f"    - {', '.join(bits)} (status: {m.get('status', 'unknown')})")
    return "\n".join(out)


def _format_allergies(allergies: list[dict]) -> str:
    if not allergies:
        return "    - None recorded"
    out = []
    for a in allergies:
        parts = [a.get("substance", "unknown")]
        if a.get("reaction"):
            parts.append(f"reaction: {a['reaction']}")
        if a.get("criticality"):
            parts.append(f"criticality: {a['criticality']}")
        out.append(f"    - {', '.join(parts)}")
    return "\n".join(out)


def _format_labs(labs: list[dict]) -> str:
    if not labs:
        return "    - None recorded"
    out = []
    for lab in labs:
        ref = f", ref {lab['reference_range']}" if lab.get("reference_range") else ""
        date = f", {lab['date']}" if lab.get("date") else ""
        out.append(
            f"    - {lab.get('test', 'unknown')}: {lab.get('value')} "
            f"{lab.get('unit', '')}{ref}{date}"
        )
    return "\n".join(out)


def _format_flags(flags: list[str]) -> str:
    if not flags:
        return "    - None"
    return "\n".join(f"    - {f}" for f in flags)


def _format_chunks(chunks: Iterable[ChunkLike]) -> str:
    chunks = list(chunks)
    if not chunks:
        return ("  No knowledge-base chunks were retrieved for this query. "
                "Treat the retrieved context as empty and flag this explicitly.")
    out = []
    for i, c in enumerate(chunks, start=1):
        url = _g(c, "url")
        header = (
            f"  [chunk {i}] source: {_g(c, 'source', 'unknown')} | "
            f"drug: {_g(c, 'drug_name', 'unknown')} | "
            f"section: {_g(c, 'section_type', 'unknown')} | "
            f"date: {_g(c, 'date', 'unknown')}"
        )
        if url:
            header += f" | url: {url}"
        out.append(f"{header}\n  {_g(c, 'text')}")
    return "\n\n".join(out)


def build_prompt(
    patient_record: dict,
    proposed_drug: dict,
    chunks: Iterable[ChunkLike],
    physician_question: str = "",
) -> str:
    demo = patient_record.get("demographics", {}) or {}
    question = physician_question.strip() if physician_question else DEFAULT_QUESTION

    drug_line = proposed_drug.get("name", "unknown")
    dose = proposed_drug.get("dose")
    route = proposed_drug.get("route")
    dose_route = ", ".join(x for x in [dose, route] if x) or "not specified"
    indication = proposed_drug.get("indication", "not specified")

    return f"""PATIENT RECORD:
  Demographics:
{_format_demographics(demo)}

  Active Diagnoses:
{_format_diagnoses(patient_record.get('diagnoses', []))}

  Current Medications:
{_format_medications(patient_record.get('medications', []))}
    NOTE: This list is sourced from MedicationRequest resources and reflects
    what was prescribed, not necessarily what the patient is currently taking.

  Known Allergies:
{_format_allergies(patient_record.get('allergies', []))}

  Relevant Lab Values:
{_format_labs(patient_record.get('labs', []))}

  Data Quality Flags:
{_format_flags(patient_record.get('data_quality_flags', []))}

PROPOSED MEDICATION:
  - Drug name (generic): {drug_line}
  - Proposed dose and route: {dose_route}
  - Indication being considered: {indication}

RETRIEVED CONTEXT:
{_format_chunks(chunks)}

PHYSICIAN QUESTION:
  {question}
"""
