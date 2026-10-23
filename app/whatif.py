"""Hypothetical measurements: show how one new measurement would change the budget and the verdict.

Every value added here carries provenance "hypothetical", so the audit marks its result as hypothetical
and reports never present it as evidence.
"""
from __future__ import annotations

import copy
import math

from .audit import run_audit
from .schemas import ExperimentRecord

KINDS = ("gas", "electrolyte", "electrolyte_nmol", "components")

_NOTE = "HYPOTHETICAL measurement entered to explore its effect; not evidence."


def add_hypothetical(d: dict, kind: str, value: float, source_id: str | None = None) -> dict:
    """Return a copy of record ``d`` with one hypothetical measurement added.

    gas:         total reactive N (as N) in the gas delivered to the cell, ``value`` in ppb, for ``source_id``
                 (default: the first source without a full reactive-N bound). Replaces that source's bound.
    electrolyte: matched electrolyte blank, ``value`` in µg/L NH4+, same volume and interval as the sample.
    electrolyte_nmol: matched electrolyte blank as an amount, ``value`` in nmol N (no volume needed).
    components:  matched component release, ``value`` in nmol N, same interval as the sample.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if not math.isfinite(value) or value < 0:
        raise ValueError("value must be a finite, non-negative number")
    e = copy.deepcopy(d)
    ev = {"provenance": "hypothetical", "confirmed": False, "note": _NOTE}
    op = e.get("observed_product") or {}
    if kind == "gas":
        sources = e.get("gas_sources") or []
        if not sources:
            raise ValueError("the record has no gas source to bound")
        unbounded = [s for s in sources if not (s.get("impurity") or {}).get("covers_all_reactive_nitrogen")]
        g = next((s for s in sources if s["id"] == source_id), None) if source_id else (unbounded or sources)[0]
        if g is None:
            raise ValueError(f"no gas source with id '{source_id}'")
        g["impurity"] = {"species": "total reactive N (as N)", "nitrogen_atoms": 1, "unit": "ppb", "lower": 0,
                         "upper": value, "bound_type": "hard_limit", "measured_where": "delivered_to_cell",
                         "covers_all_reactive_nitrogen": True, "evidence": ev}
    elif kind == "electrolyte":
        if not op.get("volume_mL"):
            raise ValueError("the observed product has no volume; a blank concentration cannot become an amount")
        e.setdefault("controls", []).append({
            "id": "hypothetical_blank", "label": "Hypothetical matched electrolyte blank", "kind": "electrolyte_blank",
            "covers": ["electrolyte"], "conditions_match": "matched", "evidence": ev,
            "measurement": {"species": "NH4+", "basis": "concentration_mass", "unit": "µg/L", "lower": 0,
                            "upper": value, "bound_type": "hard_limit", "volume_mL": op["volume_mL"],
                            "interval": op.get("interval"), "evidence": ev}})
    elif kind == "electrolyte_nmol":
        e.setdefault("controls", []).append({
            "id": "hypothetical_blank", "label": "Hypothetical matched electrolyte blank", "kind": "electrolyte_blank",
            "covers": ["electrolyte"], "conditions_match": "matched", "evidence": ev,
            "measurement": {"species": "N", "basis": "amount", "unit": "nmol", "lower": 0, "upper": value,
                            "bound_type": "hard_limit", "interval": op.get("interval"), "evidence": ev}})
    else:
        e.setdefault("controls", []).append({
            "id": "hypothetical_components", "label": "Hypothetical matched component release",
            "kind": "component_release", "covers": ["components"], "conditions_match": "matched", "evidence": ev,
            "measurement": {"species": "N", "basis": "amount", "unit": "nmol", "lower": 0, "upper": value,
                            "bound_type": "hard_limit", "interval": op.get("interval"), "evidence": ev}})
    e["assumptions"] = list(e.get("assumptions", [])) + [f"{_NOTE} ({kind}: {value:g})"]
    return e


def compare(d: dict, changes: list[tuple[str, float, str | None]]) -> dict:
    """Audit before and after one or more hypothetical measurements: [(kind, value, source_id), ...]."""
    before = run_audit(ExperimentRecord(**d))
    hyp = d
    for kind, value, source_id in changes:
        hyp = add_hypothetical(hyp, kind, value, source_id)
    after = run_audit(ExperimentRecord(**hyp))

    def brief(r):
        return {"status": r.status, "status_label": r.status_label, "coverage": r.coverage,
                "valid_upper_nmol": r.budget_valid_nmol["upper"],
                "observed_lower_nmol": r.observed_nmol["lower"] if r.observed_nmol else None,
                "unquantified": [u.model_dump() for u in r.unquantified]}

    return {"changes": [list(c) for c in changes], "before": brief(before), "after": brief(after),
            "changed": before.status != after.status, "record": hyp, "after_result": after,
            "note": ("Hypothetical: shows what such measurements would imply. They are not evidence, and an "
                     "'exceeds' verdict would still not by itself demonstrate nitrogen reduction.")}


def suggest(rec: ExperimentRecord, req: dict | None) -> list[tuple[str, float, str | None]]:
    """Hypothetical measurements at half of each 'alone' threshold: one per unbounded category."""
    if not req:
        return []
    r3 = lambda x: float(f"{x:.3g}")
    out = []
    for u in req["unknown"]:
        s = u["specs"][0]
        if u["category"] == "gas":
            # one hypothetical bound per unbounded feed, sharing the gas share of the headroom between them
            feeds = [x for x in u["specs"] if "ppb_split" in x]
            out += [("gas", r3(x["ppb_split"] / len(feeds) * 0.5), x["source_id"]) for x in feeds]
        elif u["category"] == "electrolyte" and "µg/L" in s.get("big_unit", ""):
            out.append(("electrolyte", r3(s["big"] / max(len(req["unknown"]), 1) * 0.5), None))
        elif u["category"] == "electrolyte":
            out.append(("electrolyte_nmol", r3(u["max_nmol_equal_split"] * 0.5), None))
        elif u["category"] == "components":
            out.append(("components", r3(u["max_nmol_equal_split"] * 0.5), None))
    return out
