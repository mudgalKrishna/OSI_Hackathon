"""Protocol-based next-control suggestions.

Each rule in data/protocols/rules.json has a transparent trigger evaluated
here, and quotes the source passage that motivates it. These are suggestions,
not an optimized experimental design.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .schemas import Contribution, ExperimentRecord, Issue, Recommendation

# Repository checkout or editable install: <repo>/data. Regular install: the same files ship as app/data.
_PKG = Path(__file__).resolve().parent
DATA_DIR = _PKG.parent / "data" if (_PKG.parent / "data" / "protocols").is_dir() else _PKG / "data"
RULES_PATH = DATA_DIR / "protocols" / "rules.json"
SOURCES_PATH = RULES_PATH.with_name("sources.json")


@lru_cache
def rules() -> dict[str, dict]:
    return {r["rule_id"]: r for r in json.loads(RULES_PATH.read_text(encoding="utf-8"))}


@lru_cache
def sources() -> dict[str, dict]:
    return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))


def triggered_rules(rec: ExperimentRecord, coverage: dict[str, str], contribs: list[Contribution],
                    issues: list[Issue]) -> list[tuple[str, str]]:
    """Returns (rule_id, concrete trigger description) pairs."""
    out: list[tuple[str, str]] = []
    synthetic = rec.record_type == "synthetic"
    where = {g.impurity.measured_where for g in rec.gas_sources if g.impurity is not None}

    if coverage.get("gas") in ("unknown", "partial_species", "scenario_only", "ambiguous_overlap"):
        out.append(("R1_gas_unquantified", f"Gas coverage: {coverage['gas']}."))
    if "supplier_specification" in where:
        out.append(("R2_supplier_spec_only", "A gas impurity value is a supplier specification."))
    if where & {"upstream_of_purifier", "unknown"}:
        out.append(("R3_upstream_of_purifier", "A gas impurity was measured upstream of the purifier or at an unknown point."))
    if coverage.get("electrolyte") not in ("quantified", "not_applicable", "specified_synthetic"):
        out.append(("R4_background_unresolved", f"Electrolyte/system background coverage: {coverage.get('electrolyte')}."))
    if (coverage.get("components") not in ("quantified", "not_applicable", "specified_synthetic")
            or rec.context.electrode_contains_nitrogen):
        why = f"Component coverage: {coverage.get('components')}."
        if rec.context.electrode_contains_nitrogen:
            why += " The electrode/catalyst contains nitrogen."
        out.append(("R5_component_unresolved", why))
    if rec.context.gde_contains_nafion or any(c.contains_nafion for c in rec.controls):
        out.append(("R6_nafion_gde", "A Nafion-containing electrode or component is part of the setup."))
    iso = rec.isotope
    if not synthetic and (not iso.performed or not iso.quantitative or not iso.time_series):
        state = ("not performed" if not iso.performed else "qualitative only" if not iso.quantitative
                 else "single condition only")
        out.append(("R7_isotope_incomplete", f"Isotope evidence: {state}."))
    if iso.performed and not iso.labeled_impurities_quantified:
        out.append(("R8_labeled_impurities", "15N2 experiment recorded without quantified labeled impurities."))
    mismatch = [i for i in issues if i.code in ("interval_mismatch", "volume_mismatch")]
    unmatched = [c for c in contribs if c.role in ("excluded_unmatched", "scenario") and c.category != "gas"]
    if mismatch or unmatched:
        names = [i.message.split(":")[0] for i in mismatch] + [c.label for c in unmatched]
        out.append(("R9_condition_mismatch", "Non-aligned measurements: " + "; ".join(dict.fromkeys(names)) + "."))
    if rec.context.flowing_gas:
        out.append(("R10_fixed_gas_volume", "Continuous gas flow during the measurement."))
    if not synthetic and rec.context.nox_measured_alongside is not True:
        out.append(("R11_nox_alongside", "NOx alongside ammonia: "
                    + ("not reported." if rec.context.nox_measured_alongside is None else "not measured.")))
    op = rec.observed_product
    if not synthetic and op is not None and (op.n_replicates is None or op.n_replicates < 3):
        out.append(("R12_replicates", f"Independent replicates: {op.n_replicates if op.n_replicates else 'not stated'}."))
    return out


def recommend(rec: ExperimentRecord, coverage: dict[str, str], contribs: list[Contribution],
              issues: list[Issue]) -> list[Recommendation]:
    rs, src = rules(), sources()
    recs = []
    for rule_id, trig in triggered_rules(rec, coverage, contribs, issues):
        r = rs[rule_id]
        refs = [dict(ref, citation=src.get(ref["source_id"], {}).get("citation"),
                     url=src.get(ref["source_id"], {}).get("url")) for ref in r["references"]]
        recs.append(Recommendation(rule_id=rule_id, title=r["title"], text=r["text"], trigger=trig, references=refs))
    return recs
