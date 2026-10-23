"""Optional PDF evidence extraction (stretch feature).

1. Extract text from a text-based PDF with PyMuPDF.
2. Locate passages mentioning product quantity, gas flow, duration, blanks,
   impurities and isotope controls.
3. Propose candidate values with a deterministic regex pass, and optionally
   ask Claude for structured candidates (needs the `anthropic` package and
   credentials). Missing values must be null; nothing is filled in.
4. Candidates are returned separately from confirmed inputs. They never enter
   an audit until a person confirms them in the Experiment tab.
"""
from __future__ import annotations

import json
import re

try:
    import pymupdf  # PyMuPDF >= 1.24
except ImportError:  # pragma: no cover
    try:
        import fitz as pymupdf  # older PyMuPDF
    except ImportError:
        pymupdf = None

CLAUDE_MODEL = "claude-opus-5-5"


class ExtractionError(ValueError):
    pass


def available() -> bool:
    return pymupdf is not None


TOPICS = {
    "product": r"\b(NH3|NH4\+?|ammonia|ammonium)\b.{0,80}\b(yield|amount|concentration|produced|nmol|µg|ug|mg L|µM)\b",
    "gas_flow": r"\b(flow(?: rate)?|purg\w*|bubbl\w*|sccm|mL\s?min)\b",
    "duration": r"\b(\d+(?:\.\d+)?)\s?(h|hours?|min|minutes?)\b",
    "blank": r"\b(blank|background|control|open[- ]circuit|OCV|argon|Ar\b)",
    "impurity": r"\b(NOx|NO2?|N2O|nitrate|nitrite|ppm|ppb|impurit\w*|contaminat\w*)\b",
    "isotope": r"(15N|¹⁵N|isotop\w*|labell?ed)",
}

VALUE_PATTERNS = {
    "flow": r"(\d+(?:\.\d+)?)\s?(mL\s?min[−-]1|mL/min|sccm|L\s?min[−-]1)",
    "duration": r"(\d+(?:\.\d+)?)\s?(h|min)\b",
    "gas_fraction": r"(\d+(?:\.\d+)?)\s?(ppm|ppb|ppbmol|ppmv|mol\s?%)",
    "mass_conc": r"(\d+(?:\.\d+)?)\s?(µg|ug|ng|mg)\s?(L[−-]1|/L|mL[−-]1|/mL)",
    "amount": r"(\d+(?:\.\d+)?)\s?(nmol|µmol|umol|pmol)\b",
    "volume": r"(\d+(?:\.\d+)?)\s?mL\b(?!\s?min)",
}


def pdf_text(data: bytes) -> list[dict]:
    if pymupdf is None:
        raise ExtractionError("PyMuPDF is not installed.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:  # noqa: BLE001 - PyMuPDF raises several types
        raise ExtractionError(f"Could not open PDF: {e}")
    pages = [{"page": i + 1, "text": p.get_text()} for i, p in enumerate(doc)]
    if sum(len(p["text"].strip()) for p in pages) < 200:
        raise ExtractionError("No extractable text; scanned PDFs are not supported.")
    return pages


def passages(pages: list[dict], window: int = 1) -> list[dict]:
    out = []
    for p in pages:
        sentences = re.split(r"(?<=[.;])\s+", re.sub(r"\s+", " ", p["text"]))
        for i, s in enumerate(sentences):
            topics = [t for t, rx in TOPICS.items() if re.search(rx, s, re.I)]
            if not topics or len(topics) < 2 and topics != ["isotope"]:
                continue
            text = " ".join(sentences[max(0, i - window + 1): i + window]).strip()
            vals = {k: [" ".join(m.groups()) for m in re.finditer(rx, s)] for k, rx in VALUE_PATTERNS.items()}
            vals = {k: v for k, v in vals.items() if v}
            out.append({"page": p["page"], "topics": topics, "text": text[:800], "regex_values": vals})
    return out


CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates"],
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "value", "unit", "species", "page", "quote"],
                "properties": {
                    "field": {"type": "string", "enum": [
                        "observed_product", "electrolyte_volume", "collection_duration", "gas_flow",
                        "gas_temperature", "gas_pressure", "gas_impurity", "blank_result", "component_release",
                        "detection_limit", "isotope_evidence"]},
                    "value": {"type": ["number", "null"]},
                    "unit": {"type": ["string", "null"]},
                    "species": {"type": ["string", "null"]},
                    "page": {"type": ["integer", "null"]},
                    "quote": {"type": "string"},
                },
            },
        },
    },
}

SYSTEM_PROMPT = (
    "You extract candidate measurement values from passages of an electrochemical ammonia-synthesis paper. "
    "Only report values that are explicitly written in the given passages, with the exact supporting quote "
    "and page. If a passage does not state a value, use null - never supply a customary flow rate, electrode "
    "area, gas purity, temperature, pressure or control result. Keep the species (NH3, NH4+, N, NOx...) "
    "exactly as written; if it is not stated, use null. The output is reviewed by a scientist before use."
)


def llm_candidates(found: list[dict]) -> dict:
    try:
        import anthropic
    except ImportError:
        return {"error": "The `anthropic` package is not installed (pip install anthropic)."}
    text = "\n\n".join(f"[page {p['page']}] {p['text']}" for p in found[:80])
    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": CANDIDATE_SCHEMA}},
            messages=[{"role": "user", "content": f"Passages:\n\n{text}"}],
        )
    except anthropic.AuthenticationError:
        return {"error": "No valid Anthropic credentials (set ANTHROPIC_API_KEY or run `ant auth login`)."}
    except anthropic.RateLimitError:
        return {"error": "Rate limited by the Anthropic API; try again shortly."}
    except anthropic.APIStatusError as e:
        return {"error": f"Anthropic API error {e.status_code}: {e.message}"}
    except anthropic.APIConnectionError:
        return {"error": "Could not reach the Anthropic API."}
    if response.stop_reason == "refusal":
        return {"error": "The model declined this request."}
    raw = "".join(b.text for b in response.content if b.type == "text")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Model output was not valid JSON."}
    # Keep only candidates whose quote really occurs in the extracted passages.
    haystack = re.sub(r"\s+", " ", text)
    kept, dropped = [], 0
    for c in parsed.get("candidates", []):
        if c.get("quote") and re.sub(r"\s+", " ", c["quote"]) in haystack:
            kept.append(c | {"status": "unconfirmed", "provenance": "model_extracted"})
        else:
            dropped += 1
    return {"model": response.model, "candidates": kept, "dropped_unsupported_quotes": dropped}


def extract(data: bytes, use_llm: bool = False) -> dict:
    pages = pdf_text(data)
    found = passages(pages)
    result = {
        "n_pages": len(pages),
        "passages": found,
        "note": ("Candidates are unconfirmed and stored separately from the experiment record. "
                 "Enter or confirm values in the Experiment tab before running an audit."),
    }
    if use_llm:
        result["llm"] = llm_candidates(found)
    return result
