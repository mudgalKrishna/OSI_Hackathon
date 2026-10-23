"""Evaluate NitroTrace on the synthetic benchmark and the published records.

Writes results/benchmark.json and results/benchmark.csv.

Reports, separately for synthetic and published records:
  * status accuracy and numerical error against independent references;
  * unit-conversion failures detected; inconclusive handling; duplicate-source handling;
  * a naive baseline (sum every entered number, treat missing as zero) and a
    protocol-checklist baseline, to show what the coverage rules change;
  * runtime.
The synthetic suite tests software and reasoning only; it does not validate
attribution on laboratory experiments.

Run:  python scripts/evaluate.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.audit import ENGINE_VERSION, run_audit  # noqa: E402
from app.schemas import ExperimentRecord  # noqa: E402
from app.units import UnitError, gas_fraction, gas_moles, gas_moles_from_volume, to_nmol_n  # noqa: E402

UNIT_CODES = {"missing_species", "missing_basis", "ambiguous_unit", "basis_unit_conflict", "unsupported_unit",
              "missing_volume", "flow_basis_conflict", "missing_reference_conditions", "missing_gas_conditions"}


def naive_status(d: dict) -> str:
    """Baseline: add every number entered, unknown = 0, ignore matching, coverage and unit checks where possible."""
    try:
        op = d.get("observed_product")
        if not op:
            return "insufficient_information"
        lo = op.get("lower") if op.get("lower") is not None else 0.0
        species = op.get("species") or "NH3"
        basis = op.get("basis") or "amount"
        obs_lo = to_nmol_n(lo, op["unit"], species, basis, op.get("volume_mL") or 1000, op.get("dilution_factor", 1))[0]
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
                dur = (g.get("exposure") or op.get("interval") or {"end_min": 0})
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


def checklist(d: dict) -> dict:
    """Protocol-checklist baseline: which controls are present at all (no arithmetic)."""
    iso = d.get("isotope") or {}
    present = {
        "gas_impurity_entered": any((g.get("impurity") or {}).get("upper") is not None for g in d.get("gas_sources", [])),
        "blank_or_background": any(c["kind"] in ("system_blank", "electrolyte_blank") for c in d.get("controls", [])),
        "component_release": any(c["kind"] == "component_release" for c in d.get("controls", [])),
        "isotope_quantitative": bool(iso.get("performed") and iso.get("quantitative")),
        "nox_alongside": bool((d.get("context") or {}).get("nox_measured_alongside")),
    }
    present["n_present"] = sum(present.values())
    return present


def time_audit(rec: ExperimentRecord, repeats: int = 50) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        run_audit(rec)
    return (time.perf_counter() - t0) / repeats * 1000


def evaluate_folder(folder: Path, synthetic: bool) -> list[dict]:
    rows = []
    for p in sorted(folder.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rec = ExperimentRecord(**d)
        res = run_audit(rec)
        exp = d.get("expected") or {}
        row = {
            "case_id": d["case_id"], "record_type": d["record_type"], "status": res.status,
            "expected_status": exp.get("status"), "coverage": res.coverage,
            "observed_lower_nmol": res.observed_nmol["lower"] if res.observed_nmol else None,
            "observed_upper_nmol": res.observed_nmol["upper"] if res.observed_nmol else None,
            "budget_upper_nmol": res.budget_valid_nmol["upper"],
            "budget_with_scenarios_upper_nmol": res.budget_with_scenarios_nmol["upper"],
            "error_codes": sorted({i.code for i in res.issues if i.severity == "error"}),
            "recommendations": [r.rule_id for r in res.recommendations],
            "runtime_ms": round(time_audit(rec), 3),
            "checklist": checklist(d),
        }
        if synthetic:
            row["status_correct"] = res.status == exp.get("status")
            row["naive_status"] = naive_status(d)
            row["naive_correct"] = row["naive_status"] == exp.get("status")
            if "budget_upper_nmol" in exp:
                ref = exp["budget_upper_nmol"]
                row["reference_budget_nmol"] = ref
                row["abs_error_nmol"] = abs(res.budget_valid_nmol["upper"] - ref)
                row["rel_error"] = row["abs_error_nmol"] / ref if ref else row["abs_error_nmol"]
            row["purpose"] = exp.get("purpose")
        if d["case_id"] == "choi2020_table1_flowing_gas":
            got = res.contributions[0].upper_nmol
            row["published_reference"] = {"source": "S2 Table 1 row 1", "reference_nmol": exp["reference_gas_nmol"],
                                          "engine_nmol": got, "rel_error": abs(got - exp["reference_gas_nmol"]) / exp["reference_gas_nmol"],
                                          "rate_nmol_per_s": got / 14400, "reference_rate": exp["reference_rate_nmol_per_s"]}
        rows.append(row)
    return rows


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    syn = evaluate_folder(ROOT / "data" / "synthetic", synthetic=True)
    pub = evaluate_folder(ROOT / "data" / "cases", synthetic=False)

    numeric = [r for r in syn if "rel_error" in r]
    unit_cases = [r for r in syn if (r["expected_status"] == "input_inconsistency")]
    inconclusive = [r for r in syn if r["expected_status"] == "insufficient_information"]
    dup = [r for r in syn if "overlap" in r["case_id"] or "includes_gas" in r["case_id"]]
    failures = [r for r in syn if not r["status_correct"]]
    summary = {
        "engine_version": ENGINE_VERSION,
        "generated_unix": int(time.time()),
        "disclaimer": ("Synthetic records test arithmetic and reasoning only. Published records have no ground-truth "
                       "label for whether N2 reduction occurred; they are reported separately and not scored."),
        "synthetic": {
            "n_records": len(syn),
            "status_correct": sum(r["status_correct"] for r in syn),
            "failures": [{"case_id": r["case_id"], "got": r["status"], "expected": r["expected_status"]} for r in failures],
            "numeric_checks": len(numeric),
            "max_rel_error": max((r["rel_error"] for r in numeric), default=None),
            "unit_or_basis_problems_detected": f"{sum(r['status']=='input_inconsistency' for r in unit_cases)}/{len(unit_cases)}",
            "unit_error_codes_seen": sorted({c for r in unit_cases for c in r["error_codes"] if c in UNIT_CODES}),
            "inconclusive_correct": f"{sum(r['status']=='insufficient_information' for r in inconclusive)}/{len(inconclusive)}",
            "duplicate_source_cases_correct": f"{sum(r['status_correct'] for r in dup)}/{len(dup)}",
            "naive_baseline_correct": f"{sum(r['naive_correct'] for r in syn)}/{len(syn)}",
            "naive_baseline_failures": [r["case_id"] for r in syn if not r["naive_correct"]],
            "median_runtime_ms": statistics.median(r["runtime_ms"] for r in syn),
        },
        "published": {
            "n_records": len(pub),
            "records": [{k: r[k] for k in ("case_id", "status", "coverage", "observed_upper_nmol", "budget_upper_nmol",
                                           "budget_with_scenarios_upper_nmol")} | ({"published_reference": r["published_reference"]}
                                                                                   if "published_reference" in r else {})
                        for r in pub],
        },
        "rows": syn + pub,
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    cols = ["case_id", "record_type", "expected_status", "status", "status_correct", "naive_status", "naive_correct",
            "observed_lower_nmol", "observed_upper_nmol", "budget_upper_nmol", "reference_budget_nmol", "rel_error",
            "budget_with_scenarios_upper_nmol", "error_codes", "runtime_ms", "purpose"]
    with (out / "benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in syn + pub:
            w.writerow({**r, "error_codes": ";".join(r["error_codes"])})

    s = summary["synthetic"]
    print(f"Synthetic: {s['status_correct']}/{s['n_records']} statuses correct; max relative numeric error "
          f"{s['max_rel_error']:.2e} over {s['numeric_checks']} checks")
    print(f"  unit/basis problems detected {s['unit_or_basis_problems_detected']}; inconclusive correct "
          f"{s['inconclusive_correct']}; duplicate-source cases {s['duplicate_source_cases_correct']}")
    print(f"  naive sum-everything baseline: {s['naive_baseline_correct']} correct")
    print(f"  median audit runtime {s['median_runtime_ms']:.2f} ms")
    for f_ in s["failures"]:
        print(f"  FAIL {f_['case_id']}: got {f_['got']}, expected {f_['expected']}")
    print("Published (not scored):")
    for r in summary["published"]["records"]:
        print(f"  {r['case_id']}: {r['status']}  budget≤{r['budget_upper_nmol']:.4g} nmol  coverage={r['coverage']}")
        if "published_reference" in r:
            pr = r["published_reference"]
            print(f"    reproduces S2 Table 1: {pr['engine_nmol']:.4g} vs {pr['reference_nmol']:.4g} nmol "
                  f"(rel. err {pr['rel_error']:.1e}); {pr['rate_nmol_per_s']:.3f} vs {pr['reference_rate']} nmol/s")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
