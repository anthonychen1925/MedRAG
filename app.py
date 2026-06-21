"""MedRAG web UI.

A single-file Flask app that orchestrates the pipeline:
  FHIR upload -> parse -> retrieve -> assemble prompt -> Claude -> report.

Run:
    python app.py
then open http://127.0.0.1:5000
"""

from __future__ import annotations

import html
import json
import re
import traceback

from flask import Flask, render_template_string, request

from config import settings
from fhir_parser import parse_fhir_bundle
from prompt_assembly import build_prompt
from retrieval import retrieve_chunks

app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MedRAG — Medication Decision Support</title>
  <style>
    :root { --bg:#0f1720; --panel:#172230; --ink:#e6edf3; --muted:#9fb0c0;
            --accent:#3da9fc; --line:#26344a; --warn:#f4b740; --danger:#ff6b6b;
            --ok:#3ecf8e; }
    * { box-sizing: border-box; }
    body { margin:0; font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
           background:var(--bg); color:var(--ink); }
    header { padding:22px 28px; border-bottom:1px solid var(--line); background:var(--panel); }
    header h1 { margin:0; font-size:20px; letter-spacing:.2px; }
    header p { margin:4px 0 0; color:var(--muted); font-size:13px; }
    .wrap { display:grid; grid-template-columns: 420px 1fr; gap:0; min-height:calc(100vh - 86px); }
    .form { padding:24px 28px; border-right:1px solid var(--line); }
    .report { padding:24px 32px; overflow:auto; }
    label { display:block; font-size:13px; color:var(--muted); margin:16px 0 6px; }
    input[type=text], textarea, input[type=file] {
      width:100%; padding:10px 12px; background:#0c1420; color:var(--ink);
      border:1px solid var(--line); border-radius:8px; font:inherit; }
    textarea { min-height:90px; resize:vertical; }
    .row { display:flex; gap:10px; }
    .row > div { flex:1; }
    button { margin-top:22px; width:100%; padding:12px; border:0; border-radius:8px;
      background:var(--accent); color:#04121f; font-weight:700; font-size:15px; cursor:pointer; }
    button:hover { filter:brightness(1.07); }
    .hint { font-size:12px; color:var(--muted); margin-top:6px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:12px;
            padding:20px 24px; margin-bottom:18px; }
    .report h2 { font-size:15px; color:var(--accent); border-bottom:1px solid var(--line);
                 padding-bottom:6px; margin-top:22px; }
    .report h3 { font-size:14px; margin-top:16px; }
    .status { font-weight:800; padding:10px 14px; border-radius:8px; display:inline-block; }
    .status.ok { background:rgba(62,207,142,.15); color:var(--ok); }
    .status.caution { background:rgba(244,183,64,.15); color:var(--warn); }
    .status.contra { background:rgba(255,107,107,.15); color:var(--danger); }
    .err { background:rgba(255,107,107,.12); border:1px solid var(--danger);
           color:#ffd5d5; padding:14px 16px; border-radius:10px; white-space:pre-wrap; }
    .muted { color:var(--muted); }
    .pill { display:inline-block; font-size:11px; padding:2px 8px; border:1px solid var(--line);
            border-radius:999px; color:var(--muted); margin:2px 4px 2px 0; }
    pre { white-space:pre-wrap; word-wrap:break-word; }
    code { background:#0c1420; padding:1px 5px; border-radius:4px; }
    .empty { color:var(--muted); margin-top:40px; text-align:center; }
    a { color:var(--accent); }
    .refs { padding-left:22px; margin:0; }
    .refs li { margin-bottom:14px; padding-top:6px; }
    .refs li:target { background:rgba(61,169,252,.12); border-radius:6px;
                      box-shadow:0 0 0 6px rgba(61,169,252,.12); }
    .refmeta { font-size:13px; }
    .reflink { display:inline-block; margin-left:8px; font-size:12px; font-weight:600; }
    .refsnippet { color:var(--muted); font-size:12px; margin-top:4px;
                  border-left:2px solid var(--line); padding-left:10px; }
    .cite { color:var(--accent); text-decoration:none; font-weight:600; }
    .cite:hover { text-decoration:underline; }
  </style>
</head>
<body>
  <header>
    <h1>MedRAG · Medication Decision Support</h1>
    <p>Retrieval-augmented safety assessment for prescribing decisions. Decision support only — the final decision rests with the licensed clinician.</p>
  </header>
  <div class="wrap">
    <form class="form" method="post" enctype="multipart/form-data">
      <label>FHIR R4 patient bundle (.json)</label>
      <input type="file" name="fhir_file" accept="application/json,.json">
      <div class="hint">Or load the bundled demo patient (79F, AFib/CKD, polypharmacy).</div>
      <label><input type="checkbox" name="use_demo" value="1" style="width:auto"> Use demo patient instead of upload</label>

      <label>Proposed medication (generic name)</label>
      <input type="text" name="drug_name" placeholder="e.g. amiodarone" value="{{ form.drug_name }}">

      <div class="row">
        <div>
          <label>Dose</label>
          <input type="text" name="dose" placeholder="200mg" value="{{ form.dose }}">
        </div>
        <div>
          <label>Route</label>
          <input type="text" name="route" placeholder="oral" value="{{ form.route }}">
        </div>
      </div>

      <label>Indication</label>
      <input type="text" name="indication" placeholder="e.g. atrial fibrillation rate control" value="{{ form.indication }}">

      <label>Clinical question (optional)</label>
      <textarea name="question" placeholder="e.g. Any concern combining with her warfarin given the CKD?">{{ form.question }}</textarea>

      <button type="submit">Generate safety report</button>
      <div class="hint" style="margin-top:14px">
        Embedder: <span class="pill">{{ embedder }}</span>
        Model: <span class="pill">{{ model }}</span>
        Top-k: <span class="pill">{{ top_k }}</span>
      </div>
    </form>

    <div class="report">
      {% if error %}
        <div class="err">{{ error }}</div>
      {% elif report_html %}
        {% if patient_summary %}
        <div class="card">
          <strong>Patient</strong> · {{ patient_summary }}
          <div style="margin-top:8px">
          {% for f in flags %}<span class="pill">⚠ {{ f }}</span>{% endfor %}
          </div>
        </div>
        {% endif %}
        <div class="card">
          <div class="muted" style="margin-bottom:8px">Retrieved {{ n_chunks }} knowledge chunk(s)
            {% for s in sources %}<span class="pill">{{ s }}</span>{% endfor %}
          </div>
          {{ report_html|safe }}
        </div>
        {% if references %}
        <div class="card">
          <h2 style="margin-top:0">References &amp; Sources</h2>
          <div class="muted" style="font-size:12px;margin-bottom:10px">
            Each citation in the report (e.g. <code>[chunk 1]</code>) links to its entry below.
            Click “View source label” to verify against the original FDA label on DailyMed.
          </div>
          <ol class="refs">
            {% for r in references %}
            <li id="ref-{{ r.n }}">
              <span class="refmeta"><strong>{{ r.drug }}</strong> · {{ r.section }} · {{ r.date or "date n/a" }} · <span class="muted">{{ r.source }}</span></span>
              {% if r.url %}<a class="reflink" href="{{ r.url }}" target="_blank" rel="noopener">View source label ↗</a>{% else %}<span class="muted">(no source link)</span>{% endif %}
              <div class="refsnippet">{{ r.snippet }}</div>
            </li>
            {% endfor %}
          </ol>
        </div>
        {% endif %}
      {% else %}
        <div class="empty">
          <p>Upload a FHIR bundle (or use the demo patient), enter a proposed medication, and generate a report.</p>
        </div>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""

DEMO_PATH = "synthetic_patients/elderly_polypharmacy.json"


def _embedder_label() -> str:
    if settings.use_stub_embedder:
        return "stub"
    if settings.embed_provider == "voyage":
        return settings.voyage_model if settings.voyage_api_key else "stub (no Voyage key)"
    return settings.bge_model


def _status_class(report: str) -> str:
    head = report[:400].lower()
    if "contraindicated" in head or "not recommended" in head:
        return "contra"
    if "proceed with caution" in head:
        return "caution"
    if "safe to prescribe" in head:
        return "ok"
    return ""


def markdown_to_html(text: str) -> str:
    """Minimal, safe markdown rendering for the report (no external deps)."""
    esc = html.escape(text)
    lines = esc.split("\n")
    out: list[str] = []
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    status_cls = _status_class(text)
    for ln in lines:
        s = ln.rstrip()
        if re.match(r"^#{3}\s+", s):
            close_list()
            heading = re.sub(r"^#{3}\s+", "", s)
            out.append(f"<h3>{heading}</h3>")
        elif re.match(r"^#{1,2}\s+", s):
            close_list()
            heading = re.sub(r"^#{1,2}\s+", "", s)
            out.append(f"<h2>{heading}</h2>")
        elif re.match(r"^\s*[-*]\s+", s):
            if not in_list:
                out.append("<ul>"); in_list = True
            item = re.sub(r"^\s*[-*]\s+", "", s)
            out.append(f"<li>{item}</li>")
        elif s.strip() == "":
            close_list(); out.append("<br>")
        else:
            close_list(); out.append(f"<p>{s}</p>")

    close_list()
    rendered = "\n".join(out)
    rendered = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"`(.+?)`", r"<code>\1</code>", rendered)
    # Highlight the leading status line if present.
    if status_cls:
        rendered = re.sub(
            r"<strong>(Safe to Prescribe|Proceed with Caution|Contraindicated[^<]*)</strong>",
            rf'<span class="status {status_cls}">\1</span>',
            rendered, count=1,
        )
    return rendered


_CITATION_RE = re.compile(
    r"(\[)?\b(chunks?)\s+(\d+(?:\s*(?:,|and|&amp;)\s*\d+)*)(\])?",
    re.IGNORECASE,
)


def linkify_citations(report_html: str, n_refs: int) -> str:
    """Turn the model's chunk citations into anchor links.

    Handles both bracketed (``[chunk 1]``, ``[chunks 1, 5]``) and bare
    (``chunk 1``, ``chunks 2 and 7``) forms. Each cited number links to the
    matching entry in the References panel (``#ref-N``) so a physician can jump
    straight to the source.
    """
    def repl(match: "re.Match") -> str:
        open_b = match.group(1) or ""
        word = match.group(2)
        close_b = match.group(4) or ""
        nums = re.findall(r"\d+", match.group(3))
        linked = []
        for num in nums:
            if 1 <= int(num) <= n_refs:
                linked.append(f'<a class="cite" href="#ref-{num}">{num}</a>')
            else:
                linked.append(num)
        return f"{open_b}{word} {', '.join(linked)}{close_b}"

    return _CITATION_RE.sub(repl, report_html)


def build_references(chunks) -> list[dict]:
    refs = []
    for i, c in enumerate(chunks, start=1):
        text = getattr(c, "text", "")
        # Strip the leading "[drug — section] " tag we add at ingest time.
        snippet = re.sub(r"^\[[^\]]+\]\s*", "", text)[:240]
        refs.append({
            "n": i,
            "drug": getattr(c, "drug_name", "") or "unknown",
            "section": (getattr(c, "section_type", "") or "").replace("_", " "),
            "date": getattr(c, "date", ""),
            "source": getattr(c, "source", ""),
            "url": getattr(c, "url", ""),
            "snippet": snippet + ("…" if len(text) > 240 else ""),
        })
    return refs


def _patient_summary(record: dict) -> str:
    d = record.get("demographics", {})
    dx = ", ".join(x["name"] for x in record.get("diagnoses", [])[:4] if x.get("name"))
    return (f"{d.get('age', '?')}{(d.get('sex') or '?')[:1].upper()} · "
            f"{len(record.get('medications', []))} active meds · {dx or 'no active dx'}")


@app.route("/", methods=["GET", "POST"])
def index():
    form = {k: "" for k in ("drug_name", "dose", "route", "indication", "question")}
    ctx = {
        "form": form, "error": None, "report_html": None,
        "embedder": _embedder_label(),
        "model": settings.anthropic_model, "top_k": settings.retrieval_top_k,
        "patient_summary": None, "flags": [], "n_chunks": 0, "sources": [],
        "references": [],
    }

    if request.method == "GET":
        return render_template_string(PAGE, **ctx)

    for k in form:
        form[k] = request.form.get(k, "").strip()

    try:
        if request.form.get("use_demo"):
            with open(DEMO_PATH) as f:
                bundle = json.load(f)
        else:
            upload = request.files.get("fhir_file")
            if not upload or not upload.filename:
                raise ValueError("Please upload a FHIR .json bundle or check 'Use demo patient'.")
            bundle = json.load(upload.stream)

        if not form["drug_name"]:
            raise ValueError("Proposed medication (generic name) is required.")

        record = parse_fhir_bundle(bundle)
        proposed = {
            "name": form["drug_name"], "dose": form["dose"],
            "route": form["route"], "indication": form["indication"],
        }

        chunks = retrieve_chunks(form["drug_name"], record, indication=form["indication"])
        prompt = build_prompt(record, proposed, chunks, form["question"])

        # Importing here so the page still loads if anthropic isn't configured.
        from api_client import generate_report
        report = generate_report(prompt)

        ctx.update(
            report_html=linkify_citations(markdown_to_html(report), len(chunks)),
            patient_summary=_patient_summary(record),
            flags=record.get("data_quality_flags", []),
            n_chunks=len(chunks),
            sources=sorted({c.source for c in chunks if getattr(c, "source", "")}),
            references=build_references(chunks),
        )
    except Exception as exc:  # surface a helpful message instead of a 500
        ctx["error"] = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=2)}"

    return render_template_string(PAGE, **ctx)


if __name__ == "__main__":
    import os
    # Default to 5001 — macOS Control Center / AirPlay Receiver occupies 5000.
    port = int(os.getenv("PORT", "5001"))
    app.run(debug=True, host="127.0.0.1", port=port)
