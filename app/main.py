"""NitroTrace FastAPI application: API endpoints plus the single-page dashboard.

Run:  uvicorn app.main:app --reload    (then open http://127.0.0.1:8000)
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import extraction, reports
from .audit import ENGINE_VERSION, run_audit
from .controls import DATA_DIR, rules, sources
from .schemas import AuditResult, ExperimentRecord

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
CASE_DIRS = {"published": DATA_DIR / "cases", "synthetic": DATA_DIR / "synthetic"}

app = FastAPI(title="NitroTrace", version=ENGINE_VERSION,
              description="Evidence-backed contamination accounting for electrochemical ammonia experiments.")


def _all_cases() -> dict[str, dict]:
    out = {}
    for group, folder in CASE_DIRS.items():
        for p in sorted(folder.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            out[d["case_id"]] = {"group": group, "data": d}
    return out


@app.get("/health")
def health():
    return {"status": "ok", "engine_version": ENGINE_VERSION, "extraction_available": extraction.available()}


@app.get("/cases")
def list_cases():
    featured = ["synthetic_gas_budget_01", "synthetic_gas_budget_02", "synthetic_gas_budget_03"]
    items = []
    for cid, c in _all_cases().items():
        d = c["data"]
        items.append({"case_id": cid, "title": d["title"], "record_type": d["record_type"], "group": c["group"],
                      "featured": cid in featured or c["group"] == "published",
                      "purpose": (d.get("expected") or {}).get("purpose")})
    order = {cid: i for i, cid in enumerate(featured)}
    items.sort(key=lambda x: (x["group"] != "published" and x["case_id"] not in order,
                              x["group"] != "published", order.get(x["case_id"], 99), x["case_id"]))
    return items


@app.get("/cases/{case_id}")
def get_case(case_id: str):
    c = _all_cases().get(case_id)
    if c is None:
        raise HTTPException(404, f"Unknown case '{case_id}'")
    return c["data"]


@app.post("/audit", response_model=AuditResult)
def audit(record: ExperimentRecord):
    return run_audit(record)


@app.post("/report/{fmt}")
def report(fmt: str, record: ExperimentRecord):
    res = run_audit(record)
    if fmt == "json":
        body, media, ext = reports.to_json(record, res), "application/json", "json"
    elif fmt == "csv":
        body, media, ext = reports.to_csv(record, res), "text/csv", "csv"
    elif fmt in ("md", "markdown"):
        body, media, ext = reports.to_markdown(record, res), "text/markdown", "md"
    else:
        raise HTTPException(400, "Format must be json, csv or md")
    name = f"nitrotrace_{record.case_id}.{ext}"
    return Response(body, media_type=f"{media}; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/sources")
def get_sources():
    return {"sources": sources(), "rules": list(rules().values())}


@app.get("/benchmark")
def benchmark():
    p = ROOT / "results" / "benchmark.json"
    if not p.exists():
        raise HTTPException(404, "Run `python scripts/evaluate.py` first.")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.post("/extract")
async def extract(file: UploadFile = File(...), use_llm: bool = False):
    if not extraction.available():
        raise HTTPException(501, "PDF extraction needs PyMuPDF: pip install pymupdf")
    data = await file.read()
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(413, "PDF larger than 30 MB")
    try:
        return extraction.extract(data, use_llm=use_llm)
    except extraction.ExtractionError as e:
        raise HTTPException(422, str(e))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
