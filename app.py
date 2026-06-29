"""MedRAG web UI.

A single-file Flask app that orchestrates the pipeline:
  FHIR upload -> parse -> retrieve -> assemble prompt -> Claude -> report.

The UI keeps the dark, glass-panel "bento" aesthetic of the frontend/ mockups
(index.html, setup.html, report.html) while being fully wired to the backend:
Claude's markdown report is parsed into sections and rendered as styled cards,
with the recommendation shown as a status badge and clickable [chunk N] citations
linking to a References & Sources panel.

Run:
    python app.py
then open http://127.0.0.1:5001
"""

from __future__ import annotations

import html
import json
import re
import traceback
import urllib.parse

from flask import Flask, abort, render_template_string, request, url_for

from config import settings
from fhir_parser import parse_fhir_bundle
from prompt_assembly import build_prompt
from ingest import FAERS_TOP_N, OPENFDA_EVENT_URL, fetch_faers_reactions
from retrieval import retrieve_chunks

app = Flask(__name__)

FAERS_DASHBOARD_URL = (
    "https://www.fda.gov/drugs/surveillance/questions-and-answers-fda-adverse-event-"
    "reporting-system-faers/fda-adverse-event-reporting-system-faers-public-dashboard"
)

DEMO_PATH = "synthetic_patients/elderly_polypharmacy.json"

# Shared <head>: Tailwind config + theme tokens + report-body styling. Mirrors
# the design language of the frontend/ mockups.
HEAD = """
<head>
<meta charset="utf-8">
<meta content="width=device-width, initial-scale=1.0" name="viewport">
<title>MedRAG — Medication Decision Support</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;600;700&family=Inter:wght@400;600&family=Geist:wght@500&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
<script>
  tailwind.config = {
    darkMode: "class",
    theme: { extend: {
      colors: {
        "error-container": "#93000a", "surface-container-low": "#0e1d25",
        "clinical-teal": "#00E5BC", "tertiary": "#d1bcff",
        "secondary-container": "#00f1fe", "background": "#06151d",
        "surface-variant": "#28373f", "on-error-container": "#ffdad6",
        "primary-fixed-dim": "#aec6ff", "inverse-primary": "#0059c5",
        "on-primary-container": "#edf0ff", "outline": "#8c90a0",
        "on-background": "#d5e5f0", "surface-bright": "#2c3b44",
        "on-surface": "#d5e5f0", "tertiary-fixed-dim": "#d1bcff",
        "surface": "#06151d", "surface-container-high": "#1d2c34",
        "inverse-surface": "#d5e5f0", "secondary": "#ddfcff",
        "on-surface-variant": "#c2c6d6", "primary-fixed": "#d8e2ff",
        "surface-glass": "rgba(28, 43, 51, 0.6)", "surface-container": "#122129",
        "surface-container-highest": "#28373f", "secondary-fixed-dim": "#00dbe7",
        "outline-variant": "#424754", "tertiary-container": "#803fff",
        "deep-indigo": "#0A0F1E", "surface-dim": "#06151d", "on-primary": "#002e6b",
        "error": "#ffb4ab", "tertiary-fixed": "#e9ddff", "primary": "#aec6ff",
        "primary-container": "#0668e1", "surface-tint": "#aec6ff",
        "secondary-fixed": "#74f5ff", "surface-container-lowest": "#021017"
      },
      borderRadius: { "DEFAULT": "0.25rem", "lg": "0.5rem", "xl": "0.75rem", "full": "9999px" },
      spacing: { "margin-desktop": "64px", "margin-mobile": "20px", "container-max": "1440px", "gutter": "24px", "base": "8px" },
      fontFamily: {
        "display-lg": ["Hanken Grotesk"], "headline-lg-mobile": ["Hanken Grotesk"],
        "body-lg": ["Inter"], "title-md": ["Inter"], "label-sm": ["Geist"],
        "body-md": ["Inter"], "headline-lg": ["Hanken Grotesk"]
      },
      fontSize: {
        "display-lg": ["48px", {"lineHeight": "56px", "letterSpacing": "-0.02em", "fontWeight": "700"}],
        "headline-lg-mobile": ["28px", {"lineHeight": "36px", "fontWeight": "600"}],
        "body-lg": ["18px", {"lineHeight": "28px", "fontWeight": "400"}],
        "title-md": ["20px", {"lineHeight": "28px", "fontWeight": "600"}],
        "label-sm": ["12px", {"lineHeight": "16px", "letterSpacing": "0.05em", "fontWeight": "500"}],
        "body-md": ["16px", {"lineHeight": "24px", "fontWeight": "400"}],
        "headline-lg": ["32px", {"lineHeight": "40px", "letterSpacing": "-0.01em", "fontWeight": "600"}]
      }
    }}
  }
</script>
<style>
  body {
    background-color: #0A0F1E; color: #d5e5f0;
    background-image:
      radial-gradient(circle at 15% 50%, rgba(0, 242, 255, 0.05), transparent 25%),
      radial-gradient(circle at 85% 30%, rgba(128, 63, 255, 0.05), transparent 25%);
    background-attachment: fixed; min-height: 100vh;
  }
  .glass-panel {
    background-color: rgba(28, 43, 51, 0.6);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }
  .glow-hover:hover { box-shadow: 0 0 20px rgba(0, 242, 255, 0.15); }
  .glow-focus:focus-within { box-shadow: 0 0 20px 0 rgba(0, 242, 255, 0.15); border-bottom-color: #00E5BC; }
  .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; }
  /* Rendered Claude markdown inside report cards */
  .report-body p { margin: 0.5rem 0; color: #c2c6d6; }
  .report-body h2 { font-size: 18px; font-weight: 600; color: #00E5BC; margin: 1rem 0 0.4rem; }
  .report-body h3 { font-size: 15px; font-weight: 600; color: #aec6ff; margin: 0.8rem 0 0.3rem; }
  .report-body ul { list-style: disc; padding-left: 1.3rem; margin: 0.4rem 0; }
  .report-body li { margin: 0.3rem 0; color: #c2c6d6; }
  .report-body strong { color: #eef4f8; font-weight: 600; }
  .report-body code { background: rgba(0,0,0,0.35); padding: 1px 6px; border-radius: 5px; font-size: 0.85em; }
  .report-body a.cite { color: #00E5BC; font-weight: 600; text-decoration: none; }
  .report-body a.cite:hover { text-decoration: underline; }
  .report-body br { display: none; }
  .ref-item:target { box-shadow: 0 0 0 2px rgba(0,229,188,0.6); border-radius: 0.5rem; }
</style>
</head>
"""

NAV = """
<nav class="glass-panel backdrop-blur-xl border-b border-white/10 flex justify-between items-center px-margin-mobile md:px-margin-desktop h-20 w-full z-50 sticky top-0">
  <div class="flex items-center gap-4">
    <span class="material-symbols-outlined text-clinical-teal text-3xl" style="font-variation-settings: 'FILL' 1;">medical_services</span>
    <a href="/" class="font-display-lg text-headline-lg-mobile md:text-headline-lg font-bold text-primary tracking-tight">MedRAG</a>
  </div>
  <div class="hidden md:flex gap-8">
    <span class="text-on-surface-variant font-medium">Medication Decision Support</span>
  </div>
  <div>
    <span class="font-label-sm text-label-sm text-clinical-teal uppercase tracking-widest border border-clinical-teal/30 px-3 py-1 rounded-full flex items-center gap-2">
      <span class="w-2 h-2 rounded-full bg-clinical-teal animate-pulse"></span> Engine Ready
    </span>
  </div>
</nav>
"""

SETUP_PAGE = """
<!DOCTYPE html><html class="dark" lang="en">
""" + HEAD + """
<body class="antialiased overflow-x-hidden flex flex-col min-h-screen">
""" + NAV + """
<main class="flex-grow w-full max-w-container-max mx-auto px-margin-mobile md:px-margin-desktop py-12 flex flex-col gap-10">

  <section class="flex flex-col gap-2">
    <h1 class="font-display-lg text-display-lg text-on-surface">Analysis Setup</h1>
    <p class="font-body-lg text-body-lg text-on-surface-variant max-w-2xl">Initialize the reasoning engine with patient context, the proposed intervention, and your clinical query. The final prescribing decision rests with the licensed clinician.</p>
  </section>

  {% if error %}
  <div class="glass-panel rounded-xl p-5 border-l-4 border-error flex items-start gap-3">
    <span class="material-symbols-outlined text-error">error</span>
    <pre class="font-body-md text-body-md text-on-error-container whitespace-pre-wrap">{{ error }}</pre>
  </div>
  {% endif %}

  <form method="post" enctype="multipart/form-data" class="grid grid-cols-1 md:grid-cols-12 gap-gutter">
    <div class="col-span-1 md:col-span-8 flex flex-col gap-gutter">

      <!-- FHIR upload -->
      <div class="glass-panel rounded-xl p-gutter flex flex-col gap-5">
        <h2 class="font-title-md text-title-md text-on-surface flex items-center gap-2">
          <span class="material-symbols-outlined text-clinical-teal">upload_file</span> Patient Context (FHIR R4)
        </h2>
        <label id="drop" class="border-2 border-dashed border-outline-variant rounded-lg p-8 flex flex-col items-center justify-center gap-3 bg-surface-container-low/50 hover:bg-surface-container-low/80 hover:border-clinical-teal/50 transition-colors cursor-pointer text-center">
          <span class="material-symbols-outlined text-4xl text-outline">cloud_upload</span>
          <span id="fname" class="font-title-md text-title-md text-on-surface">Click to upload a FHIR R4 .json bundle</span>
          <span class="font-label-sm text-label-sm text-outline uppercase tracking-widest">Supports .json</span>
          <input id="fhir_file" type="file" name="fhir_file" accept="application/json,.json" class="hidden">
        </label>
        <label class="flex items-center gap-3 text-on-surface-variant font-body-md cursor-pointer">
          <input type="checkbox" name="use_demo" value="1" class="rounded bg-surface-container-low border-outline-variant text-clinical-teal focus:ring-clinical-teal">
          Use bundled demo patient instead (79F, AFib/CKD, polypharmacy)
        </label>
      </div>

      <!-- Proposed intervention -->
      <div class="glass-panel rounded-xl p-gutter flex flex-col gap-5">
        <h2 class="font-title-md text-title-md text-on-surface flex items-center gap-2">
          <span class="material-symbols-outlined text-primary">medication</span> Proposed Intervention
        </h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-5">
          {{ field("Generic Name", "drug_name", "e.g. amiodarone", form.drug_name) }}
          {{ field("Dose & Frequency", "dose", "e.g. 200mg daily", form.dose) }}
          {{ field("Route", "route", "e.g. oral", form.route) }}
          {{ field("Primary Indication", "indication", "e.g. atrial fibrillation", form.indication) }}
        </div>
      </div>

      <!-- Clinical query -->
      <div class="glass-panel rounded-xl p-gutter flex flex-col gap-5">
        <h2 class="font-title-md text-title-md text-on-surface flex items-center gap-2">
          <span class="material-symbols-outlined text-tertiary-container">help_clinic</span> Clinical Query
        </h2>
        <div class="flex flex-col glow-focus transition-all border-b-2 border-outline-variant bg-surface-container-low/50 rounded-t-lg h-36">
          <textarea name="question" class="w-full h-full bg-transparent border-none text-on-surface font-body-md focus:ring-0 p-4 resize-none placeholder:text-outline-variant" placeholder="e.g. Any concern combining with her warfarin given the CKD?">{{ form.question }}</textarea>
        </div>
      </div>
    </div>

    <!-- Action column -->
    <div class="col-span-1 md:col-span-4 flex flex-col gap-gutter">
      <div class="glass-panel rounded-xl p-gutter flex flex-col gap-6 sticky top-28">
        <h3 class="font-title-md text-title-md text-on-surface border-b border-white/5 pb-4">Run Analysis</h3>
        <p class="font-body-md text-body-md text-on-surface-variant">Provide a patient bundle (or demo) and at least a generic drug name, then start the analysis.</p>
        <button type="submit" class="w-full bg-primary-container text-white py-4 px-6 rounded-lg font-title-md text-title-md flex items-center justify-center gap-2 hover:bg-inverse-primary transition-colors glow-hover">
          <span class="material-symbols-outlined" style="font-variation-settings:'FILL' 1;">rocket_launch</span> Start Analysis
        </button>
        <div class="flex flex-col gap-2 pt-2 border-t border-white/5 text-on-surface-variant">
          <div class="flex justify-between font-label-sm text-label-sm uppercase tracking-widest"><span class="text-outline">Embedder</span><span>{{ embedder }}</span></div>
          <div class="flex justify-between font-label-sm text-label-sm uppercase tracking-widest"><span class="text-outline">Model</span><span>{{ model }}</span></div>
          <div class="flex justify-between font-label-sm text-label-sm uppercase tracking-widest"><span class="text-outline">Max chunks</span><span>{{ top_k }}</span></div>
        </div>
      </div>
    </div>
  </form>
</main>
<script>
  const inp = document.getElementById('fhir_file');
  inp.addEventListener('change', () => {
    const f = inp.files[0];
    document.getElementById('fname').textContent = f ? f.name : 'Click to upload a FHIR R4 .json bundle';
  });
</script>
</body></html>
"""

REPORT_PAGE = """
<!DOCTYPE html><html class="dark" lang="en">
""" + HEAD + """
<body class="antialiased overflow-x-hidden flex flex-col min-h-screen">
""" + NAV + """
<main class="flex-grow px-margin-mobile md:px-margin-desktop py-12 flex flex-col gap-10 max-w-container-max mx-auto w-full">

  <!-- Header & status -->
  <header class="flex flex-col md:flex-row justify-between items-start md:items-end gap-4">
    <div>
      <h1 class="font-display-lg text-display-lg text-on-surface mb-2">Clinical Safety Report</h1>
      <p class="font-body-lg text-body-lg text-on-surface-variant">{{ patient_summary }}</p>
      <p class="font-label-sm text-label-sm text-outline uppercase tracking-widest mt-2">
        Retrieved {{ n_chunks }} knowledge chunks · {{ sources|join(" · ") }}
      </p>
    </div>
    <div class="glass-panel rounded-xl px-6 py-4 border-l-4 flex items-center gap-4 glow-hover transition-all" style="border-color: {{ status.hex }};">
      <span class="material-symbols-outlined text-4xl" style="font-variation-settings:'FILL' 1; color: {{ status.hex }};">{{ status.icon }}</span>
      <div>
        <span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-widest block mb-1">Recommendation</span>
        <span class="font-headline-lg text-title-md" style="color: {{ status.hex }};">{{ status.label }}</span>
      </div>
    </div>
  </header>

  {% if flags %}
  <div class="flex flex-wrap gap-2">
    {% for f in flags %}
    <span class="inline-flex items-center gap-1 bg-surface-container/60 border border-white/10 rounded-full px-3 py-1 font-label-sm text-label-sm text-on-surface-variant">
      <span class="material-symbols-outlined text-base text-orange-400">flag</span>{{ f }}
    </span>
    {% endfor %}
  </div>
  {% endif %}

  <!-- Bento grid of report sections -->
  <div class="grid grid-cols-1 md:grid-cols-2 gap-gutter">
    {% for s in sections %}
    <section class="{{ s.span }} glass-panel rounded-xl p-6 flex flex-col gap-3 glow-hover transition-all">
      <div class="flex items-center gap-3 border-b border-white/10 pb-3">
        <span class="material-symbols-outlined" style="color: {{ s.hex }};">{{ s.icon }}</span>
        <h2 class="font-title-md text-title-md text-on-surface">{{ s.title }}</h2>
      </div>
      <div class="report-body font-body-md text-body-md">{{ s.body|safe }}</div>
    </section>
    {% endfor %}

    <!-- References -->
    {% if references %}
    <section class="md:col-span-2 glass-panel rounded-xl p-6 flex flex-col gap-3">
      <div class="flex items-center gap-3 border-b border-white/10 pb-3">
        <span class="material-symbols-outlined text-secondary-fixed-dim">menu_book</span>
        <h2 class="font-title-md text-title-md text-on-surface">References &amp; Sources</h2>
      </div>
      <p class="font-body-md text-on-surface-variant text-sm">Each <code class="bg-black/30 px-1 rounded">[chunk N]</code> citation links to its entry below. Open the source to verify against the original.</p>
      <ol class="flex flex-col gap-3 mt-1">
        {% for r in references %}
        <li id="ref-{{ r.n }}" class="ref-item bg-surface-container/50 rounded-lg p-4 border border-white/5">
          <div class="flex flex-wrap items-center gap-2">
            <span class="font-label-sm text-label-sm text-clinical-teal">[{{ r.n }}]</span>
            <strong class="text-on-surface">{{ r.drug }}</strong>
            <span class="text-on-surface-variant">· {{ r.section }} · {{ r.date or "date n/a" }}</span>
            <span class="font-label-sm text-label-sm uppercase tracking-widest text-outline border border-white/10 rounded-full px-2 py-0.5">{{ r.source }}</span>
            {% if r.url %}<a href="{{ r.url }}" target="_blank" rel="noopener" class="text-clinical-teal text-sm font-semibold hover:underline ml-auto">Open source ↗</a>{% endif %}
          </div>
          <p class="text-on-surface-variant text-sm mt-2 border-l-2 border-white/10 pl-3">{{ r.snippet }}</p>
        </li>
        {% endfor %}
      </ol>
    </section>
    {% endif %}
  </div>

  {% if disclaimer %}
  <p class="font-body-md text-on-surface-variant text-sm italic text-center border-t border-white/5 pt-6">{{ disclaimer }}</p>
  {% endif %}

  <div class="flex justify-center">
    <a href="/" class="bg-surface-container text-on-surface border border-white/10 py-3 px-8 rounded-full font-title-md text-title-md flex items-center gap-2 hover:bg-surface-container-high transition-colors">
      <span class="material-symbols-outlined">refresh</span> New Analysis
    </a>
  </div>
</main>
<footer class="bg-surface-container-lowest border-t border-white/5 flex justify-center py-base px-margin-mobile md:px-margin-desktop w-full mt-8">
  <div class="font-label-sm text-label-sm uppercase tracking-widest text-primary py-4">© 2026 MedRAG · Decision support only</div>
</footer>
</body></html>
"""

# Jinja macro for a styled labeled input, used in the setup form.
FIELD_MACRO = """
{% macro field(label, name, placeholder, value) %}
<div class="flex flex-col glow-focus transition-all border-b-2 border-outline-variant bg-surface-container-low/50 rounded-t-lg">
  <label class="font-label-sm text-label-sm text-outline px-4 pt-3">{{ label }}</label>
  <input class="w-full bg-transparent border-none text-on-surface font-body-md focus:ring-0 px-4 pb-3 placeholder:text-outline-variant" placeholder="{{ placeholder }}" type="text" name="{{ name }}" value="{{ value }}">
</div>
{% endmacro %}
"""


# ---------------------------------------------------------------------------
# Report parsing / rendering helpers
# ---------------------------------------------------------------------------

# (keyword match, Material Symbol, accent hex). First match wins.
_SECTION_META = [
    ("recommendation", "summarize", "#aec6ff"),
    ("interaction", "medication", "#00E5BC"),
    ("contraindication", "block", "#ffb4ab"),
    ("precaution", "block", "#ffb4ab"),
    ("lab", "science", "#00dbe7"),
    ("monitoring", "monitor_heart", "#aec6ff"),
    ("alternative", "alt_route", "#d1bcff"),
    ("knowledge", "psychology_alt", "#d1bcff"),
    ("uncertainty", "psychology_alt", "#d1bcff"),
]
_WIDE_KEYWORDS = ("recommendation", "interaction", "knowledge", "uncertainty")


def _section_style(title: str) -> tuple[str, str, str]:
    t = title.lower()
    icon, hexc = "article", "#c2c6d6"
    for key, ic, hx in _SECTION_META:
        if key in t:
            icon, hexc = ic, hx
            break
    span = "md:col-span-2" if any(k in t for k in _WIDE_KEYWORDS) else "md:col-span-1"
    return icon, hexc, span


def detect_status(report: str) -> dict:
    head = report[:600].lower()
    if "contraindicated" in head or "not recommended" in head:
        return {"label": "Contraindicated / Not Recommended", "hex": "#ffb4ab", "icon": "dangerous"}
    if "proceed with caution" in head:
        return {"label": "Proceed with Caution", "hex": "#f4b740", "icon": "warning"}
    if "safe to prescribe" in head:
        return {"label": "Safe to Prescribe", "hex": "#00E5BC", "icon": "check_circle"}
    return {"label": "Assessment", "hex": "#aec6ff", "icon": "summarize"}


def split_report_sections(report: str) -> tuple[list[tuple[str, str]], str]:
    """Split the markdown report into (title, body) sections by ##/### headings.

    Returns (sections, disclaimer) where disclaimer is the trailing closing line.
    """
    sections: list[tuple[str, list[str]]] = []
    cur_title: str | None = None
    cur_body: list[str] = []
    for ln in report.split("\n"):
        m = re.match(r"^#{1,3}\s+(.*\S)\s*$", ln.strip())
        if m:
            if cur_title is not None:
                sections.append((cur_title, cur_body))
            cur_title = re.sub(r"^\d+\.\s*", "", m.group(1)).strip()
            cur_body = []
        elif cur_title is not None:
            cur_body.append(ln)
    if cur_title is not None:
        sections.append((cur_title, cur_body))

    # Pull a trailing disclaimer line (the mandated closing sentence) out of the
    # last section body so it can render as a footer note.
    disclaimer = ""
    if sections:
        title, body = sections[-1]
        kept: list[str] = []
        for ln in body:
            if "final prescribing decision rests" in ln.lower():
                disclaimer = re.sub(r"^[*_\s>-]+|[*_\s]+$", "", ln).strip()
            else:
                kept.append(ln)
        sections[-1] = (title, kept)

    return [(t, "\n".join(b).strip()) for t, b in sections], disclaimer


def markdown_to_html(text: str) -> str:
    """Minimal, safe markdown rendering for a report section (no external deps)."""
    esc = html.escape(text)
    out: list[str] = []
    in_list = False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for ln in esc.split("\n"):
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
    return rendered


_CITATION_RE = re.compile(
    r"(\[)?\b(chunks?)\s+(\d+(?:\s*(?:,|and|&amp;)\s*\d+)*)(\])?",
    re.IGNORECASE,
)


def linkify_citations(report_html: str, n_refs: int) -> str:
    """Turn the model's chunk citations into anchor links to the References panel."""
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


def build_sections(report: str, n_refs: int) -> tuple[list[dict], str]:
    raw_sections, disclaimer = split_report_sections(report)
    sections: list[dict] = []
    for title, body in raw_sections:
        icon, hexc, span = _section_style(title)
        body_html = linkify_citations(markdown_to_html(body), n_refs)
        sections.append({"title": title, "body": body_html, "icon": icon, "hex": hexc, "span": span})
    return sections, disclaimer


def faers_api_url(drug_name: str) -> str:
    search = urllib.parse.quote(f'patient.drug.openfda.generic_name:"{drug_name.lower()}"')
    return (
        f"{OPENFDA_EVENT_URL}?search={search}"
        f"&count=patient.reaction.reactionmeddrapt.exact&limit={FAERS_TOP_N}"
    )


def faers_viewer_url(drug_name: str) -> str:
    return url_for("faers_source", drug_name=drug_name.lower())


def build_references(chunks) -> list[dict]:
    refs = []
    for i, c in enumerate(chunks, start=1):
        text = getattr(c, "text", "")
        snippet = re.sub(r"^\[[^\]]+\]\s*", "", text)[:240]
        source = getattr(c, "source", "")
        drug = getattr(c, "drug_name", "") or "unknown"
        url = getattr(c, "url", "")
        if source == "openFDA FAERS":
            url = faers_viewer_url(drug)
        refs.append({
            "n": i,
            "drug": drug,
            "section": (getattr(c, "section_type", "") or "").replace("_", " "),
            "date": getattr(c, "date", ""),
            "source": source,
            "url": url,
            "snippet": snippet + ("…" if len(text) > 240 else ""),
        })
    return refs


def _embedder_label() -> str:
    if settings.use_stub_embedder:
        return "stub"
    if settings.embed_provider == "voyage":
        return settings.voyage_model if settings.voyage_api_key else "stub (no Voyage key)"
    return settings.bge_model


def _patient_summary(record: dict) -> str:
    d = record.get("demographics", {})
    dx = ", ".join(x["name"] for x in record.get("diagnoses", [])[:4] if x.get("name"))
    return (f"{d.get('age', '?')}{(d.get('sex') or '?')[:1].upper()} · "
            f"{len(record.get('medications', []))} active meds · {dx or 'no active dx'}")


FAERS_VIEWER_PAGE = """
<!DOCTYPE html><html class="dark" lang="en">
""" + HEAD + """
<body class="antialiased overflow-x-hidden flex flex-col min-h-screen">
""" + NAV + """
<main class="flex-grow w-full max-w-container-max mx-auto px-margin-mobile md:px-margin-desktop py-12 flex flex-col gap-8">
  <section class="flex flex-col gap-2">
    <p class="font-label-sm text-label-sm text-clinical-teal uppercase tracking-widest">openFDA FAERS</p>
    <h1 class="font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface">
      Adverse event reports — {{ drug }}
    </h1>
    <p class="text-on-surface-variant font-body-md max-w-3xl">
      Most frequently reported reaction terms for <strong class="text-on-surface">{{ drug }}</strong>
      in the FDA Adverse Event Reporting System. Counts reflect reporting frequency only —
      they do <em>not</em> establish causation, incidence, or that the drug caused the event.
    </p>
  </section>

  <section class="glass-panel rounded-xl p-6 md:p-8 glow-hover">
    <div class="overflow-x-auto">
      <table class="w-full text-left border-collapse">
        <thead>
          <tr class="border-b border-white/10">
            <th class="py-3 pr-4 font-title-md text-title-md text-primary">#</th>
            <th class="py-3 pr-4 font-title-md text-title-md text-primary">Reaction term</th>
            <th class="py-3 font-title-md text-title-md text-primary text-right">Reports</th>
          </tr>
        </thead>
        <tbody>
          {% for term, count in reactions %}
          <tr class="border-b border-white/5 hover:bg-white/5">
            <td class="py-3 pr-4 text-on-surface-variant">{{ loop.index }}</td>
            <td class="py-3 pr-4 text-on-surface">{{ term }}</td>
            <td class="py-3 text-on-surface text-right font-semibold">{{ "{:,}".format(count) }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </section>

  <section class="flex flex-col md:flex-row gap-4 text-sm">
    <a href="{{ api_url }}" target="_blank" rel="noopener"
       class="glass-panel rounded-lg px-4 py-3 text-clinical-teal hover:underline">
      View raw openFDA API data ↗
    </a>
    <a href="{{ dashboard_url }}" target="_blank" rel="noopener"
       class="glass-panel rounded-lg px-4 py-3 text-on-surface-variant hover:text-clinical-teal hover:underline">
      FDA FAERS Public Dashboard ↗
    </a>
    <a href="/" class="glass-panel rounded-lg px-4 py-3 text-on-surface-variant hover:text-clinical-teal hover:underline ml-auto">
      ← Back to MedRAG
    </a>
  </section>
</main>
</body></html>
"""


@app.route("/source/faers/<drug_name>")
def faers_source(drug_name: str):
    drug = re.sub(r"[^a-z0-9\-]+", "", drug_name.lower().strip())
    if not drug:
        abort(404)
    reactions = fetch_faers_reactions(drug)
    if not reactions:
        abort(404)
    return render_template_string(
        FAERS_VIEWER_PAGE,
        drug=drug,
        reactions=reactions,
        api_url=faers_api_url(drug),
        dashboard_url=FAERS_DASHBOARD_URL,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    form = {k: "" for k in ("drug_name", "dose", "route", "indication", "question")}
    meta = {
        "embedder": _embedder_label(),
        "model": settings.anthropic_model,
        "top_k": settings.retrieval_max_chunks,
    }

    if request.method == "GET":
        return render_template_string(FIELD_MACRO + SETUP_PAGE, form=form, error=None, **meta)

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

        from api_client import generate_report
        report = generate_report(prompt)

        sections, disclaimer = build_sections(report, len(chunks))
        return render_template_string(
            REPORT_PAGE,
            status=detect_status(report),
            sections=sections,
            disclaimer=disclaimer,
            patient_summary=_patient_summary(record),
            flags=record.get("data_quality_flags", []),
            n_chunks=len(chunks),
            sources=sorted({c.source for c in chunks if getattr(c, "source", "")}),
            references=build_references(chunks),
        )
    except Exception as exc:  # surface a helpful message instead of a 500
        error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=2)}"
        return render_template_string(FIELD_MACRO + SETUP_PAGE, form=form, error=error, **meta)


if __name__ == "__main__":
    import os

    # Default to 5001 — macOS Control Center / AirPlay Receiver occupies 5000.
    port = int(os.getenv("PORT", "5001"))

    # Preload the embedder before serving so the first analysis request is fast
    # and so model weights are not loaded inside Flask's debug reloader child
    # (which can raise BrokenPipeError when tqdm writes progress to stdout).
    from embeddings import get_embedder

    print("[MedRAG] Loading embedder…")
    get_embedder(settings)
    print("[MedRAG] Embedder ready.")

    # Debug reloader disabled: reloading + heavy ML model init is unreliable.
    app.run(debug=True, host="127.0.0.1", port=port, use_reloader=False)
