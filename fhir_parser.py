"""FHIR R4 bundle parser for MedRAG.

Parses a FHIR R4 patient bundle (as produced by Synthea or an EHR export)
into the structured patient record described in README.md. Targets five
resource types: Patient, MedicationRequest, Condition, AllergyIntolerance,
and Observation. Everything else is ignored for the MVP.

Design notes
------------
- Real-world bundles (incl. Synthea) frequently store weight, BMI, and smoking
  status as ``Observation`` resources rather than ``Patient.extension``. We read
  ``Patient`` first and then enrich demographics from those Observations.
- Numeric ``Observation`` resources in the ``laboratory`` category become labs.
  Vital-sign Observations are used to enrich demographics, not listed as labs.
- We never assume normal values for missing data. Clinically relevant gaps are
  surfaced as ``data_quality_flags`` for Claude to act on.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

# LOINC codes used to enrich demographics from Observation resources.
LOINC_BODY_WEIGHT = "29463-7"
LOINC_BMI = "39156-5"
LOINC_SMOKING_STATUS = "72166-2"
LOINC_PREGNANCY_STATUS = "82810-3"

_DEMOGRAPHIC_LOINCS = {
    LOINC_BODY_WEIGHT,
    LOINC_BMI,
    LOINC_SMOKING_STATUS,
    LOINC_PREGNANCY_STATUS,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _first_coding(codeable: Optional[dict]) -> dict:
    """Return the first coding dict from a CodeableConcept, or {}."""
    if not codeable:
        return {}
    coding = codeable.get("coding") or []
    return coding[0] if coding else {}


def _codeable_text(codeable: Optional[dict]) -> Optional[str]:
    """Best-effort human-readable label for a CodeableConcept."""
    if not codeable:
        return None
    if codeable.get("text"):
        return codeable["text"]
    coding = _first_coding(codeable)
    return coding.get("display") or coding.get("code")


def _calculate_age(birth_date: Optional[str], as_of: Optional[date] = None) -> Optional[int]:
    if not birth_date:
        return None
    as_of = as_of or date.today()
    try:
        born = datetime.strptime(birth_date[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    years = as_of.year - born.year - ((as_of.month, as_of.day) < (born.month, born.day))
    return years if years >= 0 else None


def _coding_system_is_icd10(coding: dict) -> bool:
    system = (coding.get("system") or "").lower()
    return "icd-10" in system or "icd10" in system or "sid/icd-10" in system


def _extract_icd10(codeable: Optional[dict]) -> Optional[str]:
    if not codeable:
        return None
    for coding in codeable.get("coding") or []:
        if _coding_system_is_icd10(coding):
            return coding.get("code")
    return None


def _status_code(codeable_or_status: Any) -> Optional[str]:
    """Pull a status code from either a plain string or a CodeableConcept."""
    if isinstance(codeable_or_status, str):
        return codeable_or_status
    if isinstance(codeable_or_status, dict):
        return _first_coding(codeable_or_status).get("code")
    return None


# ---------------------------------------------------------------------------
# Resource extractors
# ---------------------------------------------------------------------------

def extract_demographics(resource: dict) -> dict:
    demographics: dict[str, Any] = {
        "age": _calculate_age(resource.get("birthDate")),
        "sex": resource.get("gender"),
        "weight_kg": None,
        "bmi": None,
        "pregnancy_status": None,
        "smoking_status": None,
    }

    # Some bundles encode weight/smoking/pregnancy as Patient extensions.
    for ext in resource.get("extension", []) or []:
        url = (ext.get("url") or "").lower()
        if "weight" in url and ext.get("valueQuantity"):
            demographics["weight_kg"] = ext["valueQuantity"].get("value")
        elif "smoking" in url or "tobacco" in url:
            demographics["smoking_status"] = (
                ext.get("valueString") or _codeable_text(ext.get("valueCodeableConcept"))
            )
        elif "pregnan" in url:
            demographics["pregnancy_status"] = (
                ext.get("valueString")
                or ext.get("valueBoolean")
                or _codeable_text(ext.get("valueCodeableConcept"))
            )

    return demographics


def extract_medication(resource: dict) -> dict:
    name = _codeable_text(resource.get("medicationCodeableConcept"))

    dose = route = frequency = None
    instructions = resource.get("dosageInstruction") or []
    if instructions:
        instr = instructions[0]
        route = _codeable_text(instr.get("route"))

        dosed = (instr.get("doseAndRate") or [{}])[0]
        dq = dosed.get("doseQuantity") or {}
        if dq.get("value") is not None:
            unit = dq.get("unit") or dq.get("code") or ""
            dose = f"{dq.get('value')}{unit}".strip()

        timing = (instr.get("timing") or {}).get("repeat") or {}
        freq = timing.get("frequency")
        period = timing.get("period")
        period_unit = timing.get("periodUnit")
        if freq and period and period_unit:
            frequency = f"{freq}x per {period}{period_unit}"
        elif instr.get("text"):
            frequency = instr["text"]

    return {
        "name": name.lower() if name else None,
        "dose": dose,
        "route": route,
        "frequency": frequency,
        "status": resource.get("status"),
        "source": "MedicationRequest",
    }


def extract_condition(resource: dict) -> dict:
    code = resource.get("code") or {}
    return {
        "name": _codeable_text(code),
        "icd10": _extract_icd10(code),
        "status": _status_code(resource.get("clinicalStatus")) or "unknown",
    }


def extract_allergy(resource: dict) -> dict:
    reactions = resource.get("reaction") or []
    manifestation = None
    if reactions:
        manifestations = reactions[0].get("manifestation") or []
        if manifestations:
            manifestation = _codeable_text(manifestations[0])

    return {
        "substance": _codeable_text(resource.get("code")),
        "reaction": manifestation,
        "criticality": resource.get("criticality"),
    }


def _observation_category_codes(resource: dict) -> set[str]:
    codes: set[str] = set()
    for cat in resource.get("category") or []:
        for coding in cat.get("coding") or []:
            if coding.get("code"):
                codes.add(coding["code"])
    return codes


def extract_observation(resource: dict) -> Optional[dict]:
    """Return a lab dict for laboratory Observations; otherwise None.

    Vital-sign and social-history Observations (weight, BMI, smoking) are
    handled separately to enrich demographics, so they are skipped here.
    """
    code = resource.get("code") or {}
    coding = _first_coding(code)
    loinc = coding.get("code")

    if loinc in _DEMOGRAPHIC_LOINCS:
        return None

    categories = _observation_category_codes(resource)
    # Skip non-lab observations (e.g. vital-signs) unless clearly numeric labs.
    if categories and "laboratory" not in categories:
        return None

    value = resource.get("valueQuantity")
    if not value or value.get("value") is None:
        return None  # MVP only handles quantitative labs

    ref_range = None
    ranges = resource.get("referenceRange") or []
    if ranges:
        low = (ranges[0].get("low") or {}).get("value")
        high = (ranges[0].get("high") or {}).get("value")
        if low is not None and high is not None:
            ref_range = f"{low}-{high}"
        elif high is not None:
            ref_range = f"<{high}"
        elif low is not None:
            ref_range = f">{low}"
        elif ranges[0].get("text"):
            ref_range = ranges[0]["text"]

    effective = resource.get("effectiveDateTime")
    return {
        "test": _codeable_text(code),
        "loinc": loinc,
        "value": value.get("value"),
        "unit": value.get("unit") or value.get("code"),
        "reference_range": ref_range,
        "date": effective[:10] if effective else None,
    }


def _enrich_demographics_from_observation(demographics: dict, resource: dict) -> None:
    """Fold weight/BMI/smoking/pregnancy Observations into demographics."""
    loinc = _first_coding(resource.get("code")).get("code")
    if loinc not in _DEMOGRAPHIC_LOINCS:
        return

    if loinc == LOINC_BODY_WEIGHT and demographics.get("weight_kg") is None:
        vq = resource.get("valueQuantity") or {}
        if vq.get("value") is not None:
            demographics["weight_kg"] = vq["value"]
    elif loinc == LOINC_BMI and demographics.get("bmi") is None:
        vq = resource.get("valueQuantity") or {}
        if vq.get("value") is not None:
            demographics["bmi"] = vq["value"]
    elif loinc == LOINC_SMOKING_STATUS and not demographics.get("smoking_status"):
        demographics["smoking_status"] = _codeable_text(resource.get("valueCodeableConcept"))
    elif loinc == LOINC_PREGNANCY_STATUS and not demographics.get("pregnancy_status"):
        demographics["pregnancy_status"] = _codeable_text(resource.get("valueCodeableConcept"))


# ---------------------------------------------------------------------------
# Data quality flags
# ---------------------------------------------------------------------------

def generate_data_quality_flags(record: dict) -> list[str]:
    flags: list[str] = []
    demo = record.get("demographics") or {}

    if demo.get("age") is None:
        flags.append("Patient age/birthDate missing from FHIR record.")
    if not demo.get("sex"):
        flags.append("Patient sex/gender missing from FHIR record.")
    if demo.get("weight_kg") is None:
        flags.append("Body weight not present in FHIR record (affects weight-based dosing).")
    if demo.get("sex") == "female" and demo.get("pregnancy_status") is None:
        flags.append("Pregnancy status not documented for a female patient.")

    if not record.get("labs"):
        flags.append("No quantitative lab Observations found in FHIR record.")
    else:
        lab_names = " ".join((lab.get("test") or "").lower() for lab in record["labs"])
        if not any(k in lab_names for k in ("creatinine", "egfr", "gfr")):
            flags.append(
                "No renal function labs (creatinine/eGFR) found — required to "
                "evaluate renally-cleared drugs."
            )
        if not any(k in lab_names for k in ("alt", "ast", "bilirubin", "alkaline")):
            flags.append(
                "Hepatic function labs (ALT, AST, bilirubin) not present in FHIR record."
            )

    if not record.get("medications"):
        flags.append("No active MedicationRequest resources found.")
    if not record.get("allergies"):
        flags.append("No AllergyIntolerance resources found (allergy list may be incomplete).")

    return flags


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_fhir_bundle(bundle: dict) -> dict:
    """Parse a FHIR R4 Bundle into MedRAG's structured patient record."""
    patient_record: dict[str, Any] = {
        "demographics": {},
        "diagnoses": [],
        "medications": [],
        "allergies": [],
        "labs": [],
        "data_quality_flags": [],
    }

    # First pass: pull the Patient resource so demographics exist before we
    # try to enrich them from Observations (order in a Bundle is not guaranteed).
    entries = bundle.get("entry", []) or []
    observations: list[dict] = []

    for entry in entries:
        resource = entry.get("resource", {}) or {}
        rtype = resource.get("resourceType")

        if rtype == "Patient" and not patient_record["demographics"]:
            patient_record["demographics"] = extract_demographics(resource)

        elif rtype == "MedicationRequest":
            if resource.get("status") == "active":
                patient_record["medications"].append(extract_medication(resource))

        elif rtype == "Condition":
            if _status_code(resource.get("clinicalStatus")) == "active":
                patient_record["diagnoses"].append(extract_condition(resource))

        elif rtype == "AllergyIntolerance":
            patient_record["allergies"].append(extract_allergy(resource))

        elif rtype == "Observation":
            observations.append(resource)

    if not patient_record["demographics"]:
        patient_record["demographics"] = {
            "age": None, "sex": None, "weight_kg": None,
            "bmi": None, "pregnancy_status": None, "smoking_status": None,
        }

    # Second pass over Observations: demographics enrichment + labs.
    for resource in observations:
        _enrich_demographics_from_observation(patient_record["demographics"], resource)
        lab = extract_observation(resource)
        if lab:
            patient_record["labs"].append(lab)

    patient_record["data_quality_flags"] = generate_data_quality_flags(patient_record)
    return patient_record
