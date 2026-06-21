"""CLI driver to run one MedRAG case end-to-end from the terminal.

Useful for spot-checking the pipeline on different patients/drugs without the
web UI.

Examples
--------
    python run_case.py --patient synthetic_patients/elderly_polypharmacy.json \
        --drug amiodarone --dose 200mg --route oral \
        --indication "atrial fibrillation" \
        --question "Concerns with her warfarin and digoxin given CKD?"

    # Skip the (paid) Claude call and just inspect retrieval:
    python run_case.py --patient <file> --drug metformin --retrieval-only
"""

from __future__ import annotations

import argparse
import json

from fhir_parser import parse_fhir_bundle
from prompt_assembly import build_prompt
from retrieval import retrieve_chunks


def main() -> None:
    p = argparse.ArgumentParser(description="Run a single MedRAG evaluation case.")
    p.add_argument("--patient", required=True, help="Path to a FHIR R4 bundle .json")
    p.add_argument("--drug", required=True, help="Proposed generic drug name")
    p.add_argument("--dose", default="", help="Proposed dose (e.g. 1000mg)")
    p.add_argument("--route", default="", help="Route (e.g. oral)")
    p.add_argument("--indication", default="", help="Indication being considered")
    p.add_argument("--question", default="", help="Physician question")
    p.add_argument("--retrieval-only", action="store_true",
                   help="Show retrieved chunks and skip the Claude call.")
    p.add_argument("--max-tokens", type=int, default=2500)
    args = p.parse_args()

    with open(args.patient) as f:
        record = parse_fhir_bundle(json.load(f))

    print("=" * 78)
    demo = record["demographics"]
    dx = ", ".join(d["name"] for d in record["diagnoses"])
    meds = ", ".join(m["name"] for m in record["medications"]) or "none"
    labs = ", ".join(f"{lab['test']}={lab['value']}" for lab in record["labs"]) or "none"
    print(f"PATIENT: {demo.get('age')}{(demo.get('sex') or '?')[:1].upper()}  | dx: {dx}")
    print(f"  meds: {meds}")
    print(f"  labs: {labs}")
    print(f"PROPOSED: {args.drug} {args.dose} {args.route} for {args.indication or 'n/a'}")
    print("=" * 78)

    proposed = {"name": args.drug, "dose": args.dose, "route": args.route,
                "indication": args.indication}
    chunks = retrieve_chunks(args.drug, record, indication=args.indication)

    print(f"\nRETRIEVED {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] {c.score:.3f} | {c.drug_name} / {c.section_type}")

    if args.retrieval_only:
        return

    from api_client import generate_report
    prompt = build_prompt(record, proposed, chunks, args.question)
    print("\n" + "=" * 78 + "\nREPORT\n" + "=" * 78)
    print(generate_report(prompt, max_tokens=args.max_tokens))


if __name__ == "__main__":
    main()
