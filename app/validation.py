"""Benchmark against baselines, plus machine-checked properties run on every record.

Synthetic records test software and reasoning only; published records carry no ground-truth label
for whether N2 reduction occurred and are reported, not scored.
"""
from __future__ import annotations

import copy
import json
import math
import time

from .audit import run_audit
from .controls import DATA_DIR
from .inversion import evidence_requirements
from .schemas import ExperimentRecord
from .units import UnitError, gas_fraction, gas_moles, gas_moles_from_volume, to_nmol_n


def load_records(data: dict | None = None) -> dict[str, dict]:
    """{"cases/x.json" | "synthetic/y.json": record dict}. ``data`` may hold an embedded copy."""
    if data is not None:
        return {k: v for k, v in data.items() if k.startswith(("cases/", "synthetic/"))}
    out = {}
    for folder in ("cases", "synthetic"):
        for p in sorted((DATA_DIR / folder).glob("*.json")):
            out[f"{folder}/{p.name}"] = json.loads(p.read_text(encoding="utf-8"))
    return out


def naive_status(d: dict) -> str:
    """Baseline: add every number entered, unknown = 0, ignore matching, coverage and unit checks."""
    try:
        op = d.get("observed_product")
        if not op:
            return "insufficient_information"
        lo = op.get("lower") if op.get("lower") is not None else 0.0
        obs_lo = to_nmol_n(lo, op["unit"], op.get("species") or "NH3", op.get("basis") or "amount",
                           op.get("volume_mL") or 1000, op.get("dilution_factor", 1))[0]
        total = 0.0
        for g in d.get("gas_sources", []):
            imp = g.get("impurity")
            if not imp or imp.get("upper") is None:
                continue
            if g.get("total_volume_L"):
                n = gas_moles_from_volume(g["total_volume_L"], g.get("total_volume_temperature_K") or 298.15,
                                          g.get("total_volume_pressure_Pa") or 101325)[0]
            elif g.get("flow"):
                f = g["flow"]
                dur = g.get("exposure") or op.get("interval") or {"end_min": 0}
                dur = dur["end_min"] - dur.get("start_min", 0)
                n = gas_moles(f["value"], f["unit"], "standard" if f["unit"] in ("sccm", "slpm") else "actual", dur,
                              f.get("temperature_K") or 298.15, f.get("pressure_Pa") or 101325,
                              f.get("reference_temperature_K") or f.get("temperature_K") or 298.15,
                              f.get("reference_pressure_Pa") or 101325)[0]
            else:
                continue
            total += n * gas_fraction(imp["upper"], imp["unit"]) * imp.get("nitrogen_atoms", 1) * 1e9
        for c in d.get("controls", []):
            m = c["measurement"]
            total += to_nmol_n(m["upper"], m["unit"], m.get("species") or "NH3", m.get("basis") or "amount",
                               m.get("volume_mL") or 1000)[0]
        return "contamination_not_excluded" if total >= obs_lo else "exceeds_quantified_budget"
    except (UnitError, KeyError, TypeError):
        return "input_inconsistency"


def checklist_status(d: dict) -> str:
    """Protocol-checklist baseline: 'gas analysis and a blank are present' -> trust the claim; no arithmetic."""
    if not d.get("observed_product"):
        return "insufficient_information"
    has_gas = any((g.get("impurity") or {}).get("upper") is not None for g in d.get("gas_sources", []))
    has_blank = any(c["kind"] in ("system_blank", "electrolyte_blank") for c in d.get("controls", []))
    return "exceeds_quantified_budget" if (has_gas and has_blank) else "insufficient_information"


def prop_unit_invariance(d, res):
    if res.status == "input_inconsistency" or not res.observed_nmol:
        return None
    e = copy.deepcopy(d)
    e["observed_product"].update(species="N", basis="amount", unit="nmol", lower=res.observed_nmol["lower"],
                                 upper=res.observed_nmol["upper"], nondetect=False, volume_mL=None, dilution_factor=1.0)
    r2 = run_audit(ExperimentRecord(**e))
    return r2.status == res.status and math.isclose(r2.budget_valid_nmol["upper"], res.budget_valid_nmol["upper"],
                                                    rel_tol=1e-12, abs_tol=1e-12)


def prop_unknown_never_zero(d, res):
    """Deleting any counted piece of evidence must never *create* an 'exceeds' verdict.

    Measurements set aside because they overlap are skipped: deleting one of two conflicting
    measurements removes the conflict itself, which legitimately changes the coverage."""
    set_aside = {c.source_id for c in res.contributions if c.role == "excluded_overlap"}
    checks = []
    for i, g in enumerate(d.get("gas_sources", [])):
        if g["id"] not in set_aside:
            e = copy.deepcopy(d); e["gas_sources"][i]["impurity"] = None; checks.append(e)
    for i, c in enumerate(d.get("controls", [])):
        if c["id"] not in set_aside and not set_aside:
            e = copy.deepcopy(d); e["controls"].pop(i); checks.append(e)
    if not checks:
        return None
    for e in checks:
        r2 = run_audit(ExperimentRecord(**e))
        if r2.status == "exceeds_quantified_budget" and res.status != "exceeds_quantified_budget":
            return False
    return True


def prop_monotone(d, res):
    """Adding a matched component-release bound of k nmol raises the valid budget by exactly k (no system blank)."""
    if res.status == "input_inconsistency" or any(c["kind"] == "system_blank" for c in d.get("controls", [])):
        return None
    e = copy.deepcopy(d)
    iv = (d.get("observed_product") or {}).get("interval")
    e.setdefault("controls", []).append({"id": "probe", "label": "probe", "kind": "component_release",
                                         "covers": ["components"], "conditions_match": "matched",
                                         "measurement": {"species": "N", "basis": "amount", "unit": "nmol",
                                                         "lower": 0, "upper": 7.0, "interval": iv}})
    r2 = run_audit(ExperimentRecord(**e))
    return math.isclose(r2.budget_valid_nmol["upper"], res.budget_valid_nmol["upper"] + 7.0, rel_tol=1e-12)


def prop_inversion_shrink(d, rec, res):
    """Apply each 'tighten' requirement at (1 ± 1e-6) and check the valid budget lands on the right side."""
    req = evidence_requirements(rec, res)
    if not req or not req["shrink"]:
        return None
    ok = True
    for row in req["shrink"]:
        if not row["feasible"] or "required_value" not in row:
            continue
        for f, want_below in ((1 - 1e-6, True), (1 + 1e-6, False)):
            e = copy.deepcopy(d)
            g = next((g for g in e.get("gas_sources", []) if g["id"] == row["source_id"]), None)
            if g is not None:
                g["impurity"]["upper"] = row["required_value"] * f
                g["impurity"]["lower"] = min(g["impurity"].get("lower") or 0, g["impurity"]["upper"])
            else:
                c = next(c for c in e["controls"] if c["id"] == row["source_id"])
                c["measurement"]["upper"] = row["required_value"] * f
                if c["measurement"].get("lower") is not None:
                    c["measurement"]["lower"] = min(c["measurement"]["lower"], c["measurement"]["upper"])
            r2 = run_audit(ExperimentRecord(**e))
            ok &= (r2.budget_valid_nmol["upper"] < r2.observed_nmol["lower"]) == want_below
    return ok


def prop_inversion_gas(d, rec, res):
    """Fill an unknown gas source at the 'alone' ppb requirement (1 ± 1e-6) and re-audit."""
    req = evidence_requirements(rec, res)
    if not req or len(req["unknown"]) != 1 or req["unknown"][0]["category"] != "gas":
        return None
    spec = next((s for s in req["unknown"][0]["specs"] if "ppb_alone" in s), None)
    if spec is None or spec["ppb_alone"] <= 0:
        return None
    if any(g.get("impurity") for g in d["gas_sources"] if g["id"] != spec["source_id"]):
        return None
    counted = next((c.upper_nmol for c in res.contributions if c.source_id == spec["source_id"] and c.included), 0.0)
    ok = True
    for f, want in ((1 - 1e-6, "exceeds_quantified_budget"), (1 + 1e-6, "contamination_not_excluded")):
        e = copy.deepcopy(d)
        g = next(g for g in e["gas_sources"] if g["id"] == spec["source_id"])
        g["impurity"] = {"species": "NOx+NH3 (as N)", "unit": "ppb", "lower": 0,
                         "upper": counted / spec["n_gas_mol"] + spec["ppb_alone"] * f,
                         "measured_where": "delivered_to_cell", "covers_all_reactive_nitrogen": True}
        g["eta_upper"], g["eta_basis"] = 1.0, "conservative_upper_bound_1"
        ok &= run_audit(ExperimentRecord(**e)).status == want
    return ok


PROPERTIES = {
    "Unit invariance (any unit → same verdict & budget)": lambda d, rec, res: prop_unit_invariance(d, res),
    "Unknown is never zero (removing evidence never creates 'exceeds')": lambda d, rec, res: prop_unknown_never_zero(d, res),
    "Monotone budget (adding a bound adds exactly its amount)": lambda d, rec, res: prop_monotone(d, res),
    "Inversion round-trip: tighten requirements flip at the threshold": prop_inversion_shrink,
    "Inversion round-trip: gas ppb requirement flips the verdict": prop_inversion_gas,
}


def run_benchmark(records: dict[str, dict] | None = None) -> tuple[list[dict], dict]:
    records = records or load_records()
    rows, props = [], {k: [] for k in PROPERTIES}
    t0 = time.perf_counter()
    for k, d in records.items():
        rec = ExperimentRecord(**d)
        res = run_audit(rec)
        exp = d.get("expected") or {}
        ref = exp.get("budget_upper_nmol")
        rows.append({"case": d["case_id"], "type": d["record_type"], "expected": exp.get("status"),
                     "nitrotrace": res.status, "naive_sum": naive_status(d), "checklist": checklist_status(d),
                     "budget_upper_nmol": res.budget_valid_nmol["upper"],
                     "rel_error": abs(res.budget_valid_nmol["upper"] - ref) / ref if ref else None})
        for name, fn in PROPERTIES.items():
            v = fn(d, rec, res)
            if v is not None:
                props[name].append((d["case_id"], v))
    syn = [r for r in rows if r["type"] == "synthetic"]
    false_exceeds = lambda col: sum(r[col] == "exceeds_quantified_budget" != r["expected"] for r in syn)
    errs = [r["rel_error"] for r in syn if r["rel_error"] is not None]
    summary = {
        "n_synthetic": len(syn),
        "nitrotrace_correct": sum(r["nitrotrace"] == r["expected"] for r in syn),
        "naive_correct": sum(r["naive_sum"] == r["expected"] for r in syn),
        "checklist_correct": sum(r["checklist"] == r["expected"] for r in syn),
        "false_exceeds_nitrotrace": false_exceeds("nitrotrace"),
        "false_exceeds_naive": false_exceeds("naive_sum"),
        "false_exceeds_checklist": false_exceeds("checklist"),
        "max_rel_error": max(errs) if errs else None, "n_numeric": len(errs),
        "elapsed_s": time.perf_counter() - t0,
        "properties": {k: (sum(v for _, v in vals), len(vals), [c for c, v in vals if not v]) for k, vals in props.items()},
    }
    return rows, summary
