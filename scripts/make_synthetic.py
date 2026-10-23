"""Generate the synthetic benchmark records in data/synthetic/.

Each record carries an ``expected`` block: the status, and reference numbers
computed here by hand-written arithmetic that does not import the engine.
These are software and reasoning tests; they do not validate performance on
laboratory experiments.

Run:  python scripts/make_synthetic.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "synthetic"
R = 8.314462618
ATM = 101325.0


def ref_gas_nmol(P_Pa: float, V_L: float, T_K: float, x: float, atoms: int = 1, eta: float = 1.0) -> float:
    """Independent reference: n = PV/RT, nmol N = n * x * atoms * eta * 1e9."""
    return P_Pa * (V_L / 1000.0) / (R * T_K) * x * atoms * eta * 1e9


DEMO_1PPM = ref_gas_nmol(ATM, 50 * 120 / 1000, 298.15, 1e-6)  # ≈ 245.25 nmol
DEMO_001PPM = DEMO_1PPM / 100  # ≈ 2.4525 nmol

SYN_EV = {"provenance": "synthetic", "confirmed": True, "note": "Illustrative synthetic value; not a measurement."}


def base(case_id: str, title: str) -> dict:
    return {
        "case_id": case_id,
        "title": title,
        "record_type": "synthetic",
        "description": "Synthetic benchmark record. Values are illustrative, not measurements from a published study.",
        "observed_product": {
            "species": "NH3", "basis": "amount", "unit": "nmol", "lower": 190, "upper": 210,
            "bound_type": "stated_interval", "interval": {"start_min": 0, "end_min": 120}, "n_replicates": 3,
            "evidence": SYN_EV,
        },
        "gas_sources": [{
            "id": "feed", "label": "N2 feed gas",
            "impurity": {"species": "generic N-impurity (1 N atom)", "nitrogen_atoms": 1, "unit": "ppm", "lower": 0,
                         "upper": 1.0, "bound_type": "hard_limit", "measured_where": "delivered_to_cell",
                         "covers_all_reactive_nitrogen": True, "evidence": SYN_EV},
            "flow": {"value": 50, "unit": "mL/min", "volume_basis": "actual", "temperature_K": 298.15,
                     "pressure_Pa": ATM, "evidence": SYN_EV},
            "exposure": {"start_min": 0, "end_min": 120},
            "eta_upper": 1.0, "eta_basis": "conservative_upper_bound_1", "evidence": SYN_EV,
        }],
        "controls": [],
        "declared_coverage": {"electrolyte": "specified_synthetic", "components": "specified_synthetic"},
        "isotope": {"performed": False},
        "context": {"flowing_gas": True},
        "assumptions": ["Illustrative ideal-gas scenario",
                        "For this synthetic case only, all other modeled contamination sources are fully specified."],
        "sources": [],
    }


def ctl(id_, label, kind, covers, lo, hi, match="matched", end=120, **kw):
    m = {"species": "NH3", "basis": "amount", "unit": "nmol", "lower": lo, "upper": hi,
         "bound_type": "stated_interval", "interval": {"start_min": 0, "end_min": end}, "evidence": SYN_EV}
    d = {"id": id_, "label": label, "kind": kind, "covers": covers, "measurement": m, "conditions_match": match}
    d.update(kw)
    return d


def imp(rec, **kw):
    rec["gas_sources"][0]["impurity"].update(kw)
    return rec


records: list[dict] = []


def add(rec: dict, status: str, purpose: str, **expected):
    rec["expected"] = {"status": status, "purpose": purpose, **expected}
    records.append(rec)


# --- the three-step demonstration ------------------------------------------------------------
add(base("synthetic_gas_budget_01", "Demo step 1 — 1 ppm delivered impurity"),
    "contamination_not_excluded", "Reference gas calculation (≈245 nmol) covers a 190–210 nmol observation.",
    budget_upper_nmol=DEMO_1PPM)
add(imp(base("synthetic_gas_budget_02", "Demo step 2 — 0.01 ppm delivered impurity"), upper=0.01),
    "exceeds_quantified_budget", "Tightened impurity bound (≈2.45 nmol) is below the observation.",
    budget_upper_nmol=DEMO_001PPM)
r = base("synthetic_gas_budget_03", "Demo step 3 — impurity measurement removed")
r["gas_sources"][0]["impurity"] = None
add(r, "insufficient_information", "Missing gas measurement must be unknown, not zero.",
    budget_upper_nmol=0.0, coverage={"gas": "unknown"})

# --- nondetects -----------------------------------------------------------------------------------
r = imp(base("syn_nondetect_interval", "Nondetect product kept as an interval"), upper=0.01)
r["observed_product"].update(lower=None, upper=5, nondetect=True, bound_type="detection_limit")
add(r, "contamination_not_excluded", "A nondetect is [0, LOD]; its lower bound 0 lies within any budget.",
    observed_nmol=[0.0, 5.0], budget_upper_nmol=DEMO_001PPM)
r = base("syn_nondetect_no_limit", "Nondetect without a detection limit")
r["observed_product"].update(lower=None, upper=None, nondetect=True)
add(r, "input_inconsistency", "A nondetect without its limit cannot be bounded.", issue_code="nondetect_without_limit")

# --- reporting basis and units --------------------------------------------------------------------
r = base("syn_ambiguous_species", "Product species not declared")
r["observed_product"]["species"] = None
add(r, "input_inconsistency", "NH3 vs NH4+ vs N must be declared explicitly.", issue_code="missing_species")
r = base("syn_basis_unit_conflict", "Mass unit with amount basis")
r["observed_product"].update(unit="µg", lower=3.4, upper=3.6)
add(r, "input_inconsistency", "A mass unit cannot be read as an amount.", issue_code="basis_unit_conflict")
r = base("syn_ppm_liquid", "ppm used for a liquid product")
r["observed_product"].update(unit="ppm", basis="concentration_mass", volume_mL=50, lower=0.06, upper=0.07)
add(r, "input_inconsistency", "ppm is ambiguous for a liquid-phase product.", issue_code="ambiguous_unit")
r = imp(base("syn_nh4_mass", "NH4+ mass basis converted with 18.038 g/mol"), upper=0.01)
r["observed_product"].update(species="NH4+", basis="mass", unit="µg", lower=200 * 18.038e-3, upper=200 * 18.038e-3,
                             bound_type="point")
add(r, "exceeds_quantified_budget", "3.6076 µg NH4+ = 200 nmol N.", observed_nmol=[200.0, 200.0])
r = imp(base("syn_n_mass", "Nitrogen mass basis converted with 14.007 g/mol"), upper=0.01)
r["observed_product"].update(species="N", basis="mass", unit="µg", lower=200 * 14.007e-3, upper=200 * 14.007e-3,
                             bound_type="point")
add(r, "exceeds_quantified_budget", "2.8014 µg N = 200 nmol N; the same mass of NH3 would be 164.5 nmol.",
    observed_nmol=[200.0, 200.0])
r = imp(base("syn_molar_dilution", "Molar concentration with dilution factor"), upper=0.01)
r["observed_product"].update(basis="concentration_molar", unit="µM", lower=1.9, upper=2.1, volume_mL=50,
                             dilution_factor=2)
add(r, "exceeds_quantified_budget", "1.9–2.1 µM × DF 2 × 50 mL = 190–210 nmol.", observed_nmol=[190.0, 210.0])
r = base("syn_conc_no_volume", "Concentration without a volume")
r["observed_product"].update(basis="concentration_molar", unit="µM", lower=3.8, upper=4.2)
add(r, "input_inconsistency", "A concentration needs a volume to become an amount.", issue_code="missing_volume")

# --- gas flow basis --------------------------------------------------------------------------------
r = base("syn_sccm_reference", "Standardized flow (sccm) with controller reference conditions")
r["gas_sources"][0]["flow"] = {"value": 50, "unit": "sccm", "volume_basis": "standard", "temperature_K": 323.15,
                               "pressure_Pa": ATM, "reference_temperature_K": 273.15, "reference_pressure_Pa": ATM,
                               "evidence": SYN_EV}
add(r, "contamination_not_excluded", "sccm uses 273.15 K reference, not the 323.15 K gas temperature.",
    budget_upper_nmol=ref_gas_nmol(ATM, 6.0, 273.15, 1e-6))
r = base("syn_sccm_declared_actual", "sccm declared as an actual volume")
r["gas_sources"][0]["flow"].update(unit="sccm", volume_basis="actual")
add(r, "input_inconsistency", "A standardized unit with an actual basis is contradictory.", issue_code="flow_basis_conflict")
r = base("syn_sccm_no_reference", "sccm without reference conditions")
r["gas_sources"][0]["flow"].update(unit="sccm", volume_basis="standard")
add(r, "input_inconsistency", "Cell temperature must not be applied to a standardized volume.",
    issue_code="missing_reference_conditions")
r = base("syn_actual_no_TP", "Actual flow without temperature and pressure")
r["gas_sources"][0]["flow"].update(temperature_K=None, pressure_Pa=None)
add(r, "input_inconsistency", "Ideal-gas conversion of an actual volume needs T and P.", issue_code="missing_gas_conditions")
r = base("syn_total_volume", "Total delivered gas volume instead of a flow")
r["gas_sources"][0].update(flow=None, total_volume_L=6.0, total_volume_temperature_K=298.15,
                           total_volume_pressure_Pa=ATM)
add(r, "contamination_not_excluded", "6 L at 298.15 K and 1 atm gives the same budget as 50 mL/min × 120 min.",
    budget_upper_nmol=DEMO_1PPM)

# --- intervals --------------------------------------------------------------------------------------
r = base("syn_gas_interval_mismatch", "Gas exposure window differs from the collection window")
r["gas_sources"][0]["exposure"] = {"start_min": 0, "end_min": 60}
add(r, "input_inconsistency", "Exposure 60 min vs collection 120 min.", issue_code="interval_mismatch")
r = imp(base("syn_control_interval_mismatch", "Matched blank with a different interval"), upper=0.01)
r["declared_coverage"] = {}
r["controls"] = [ctl("blank", "System blank (60 min)", "system_blank", ["electrolyte", "components"], 5, 10, end=60)]
add(r, "input_inconsistency", "A matched blank over 60 min cannot be compared with 120 min without a model.",
    issue_code="interval_mismatch")
r = imp(base("syn_control_linear_scaling", "Blank scaled linearly in time (declared model)"), upper=0.01)
r["declared_coverage"] = {}
r["controls"] = [ctl("blank", "System blank (60 min)", "system_blank", ["electrolyte", "components"], 5, 10, end=60,
                     time_scaling="linear")]
add(r, "exceeds_quantified_budget", "10 nmol over 60 min → 20 nmol over 120 min under a stated linear model.",
    budget_upper_nmol=DEMO_001PPM + 20)

# --- overlapping coverage -------------------------------------------------------------------------
def overlap_base(cid, title, model):
    r = imp(base(cid, title), upper=0.01)
    r["declared_coverage"] = {}
    r["coverage_model"] = model
    r["controls"] = [ctl("sysblank", "Matched system blank (Ar, same cell)", "system_blank", ["electrolyte", "components"], 10, 20),
                     ctl("membrane", "Membrane release test", "component_release", ["components"], 5, 15)]
    return r


add(overlap_base("syn_overlap_unresolved", "System blank overlaps a component test, no model", None),
    "insufficient_information", "Overlapping measurements are not summed without a selected coverage model.",
    budget_upper_nmol=DEMO_001PPM, coverage={"electrolyte": "ambiguous_overlap", "components": "ambiguous_overlap"})
add(overlap_base("syn_overlap_system_blank", "Overlap resolved with the system-blank model", "system_blank"),
    "exceeds_quantified_budget", "Budget = gas + system blank (22.45), not gas + blank + membrane (37.45).",
    budget_upper_nmol=DEMO_001PPM + 20)
add(overlap_base("syn_overlap_sum_components", "Overlap resolved by summing components", "sum_components"),
    "insufficient_information", "Dropping the system blank leaves the electrolyte unbounded.",
    budget_upper_nmol=DEMO_001PPM + 15, coverage={"electrolyte": "unknown", "components": "quantified"})
r = base("syn_blank_includes_gas", "Whole-system blank already contains the gas contribution")
r["declared_coverage"] = {}
r["coverage_model"] = "system_blank"
r["controls"] = [ctl("sysblank", "Matched whole-system blank with N2 flowing", "system_blank",
                     ["gas", "electrolyte", "components"], 100, 150)]
add(r, "exceeds_quantified_budget", "Gas bound (245) is inside the blank (150); summing would give 395 and flip the result.",
    budget_upper_nmol=150.0)

# --- control matching ------------------------------------------------------------------------------
r = imp(base("syn_blank_different_day", "Blank measured on a different day"), upper=0.01)
r["declared_coverage"] = {"components": "specified_synthetic"}
r["controls"] = [ctl("blank", "Electrolyte blank (other day)", "electrolyte_blank", ["electrolyte"], 1, 2, match="different",
                     conditions_note="different day and electrolyte batch")]
add(r, "insufficient_information", "A non-matched blank is not interchangeable; electrolyte stays unknown.",
    budget_upper_nmol=DEMO_001PPM, coverage={"electrolyte": "unknown"})
r = imp(base("syn_blank_scenario", "Non-matched blank used as an explicit what-if"), upper=0.01)
r["declared_coverage"] = {"components": "specified_synthetic"}
r["controls"] = [ctl("blank", "Electrolyte blank (other cell)", "electrolyte_blank", ["electrolyte"], 250, 300,
                     match="different", include_as_scenario=True)]
add(r, "insufficient_information", "What-if values appear separately and never in the valid budget.",
    budget_upper_nmol=DEMO_001PPM, scenario_upper_nmol=DEMO_001PPM + 300, coverage={"electrolyte": "scenario_only"})

# --- gas evidence quality ---------------------------------------------------------------------------
add(imp(base("syn_supplier_spec", "Supplier purity used as the gas bound"), measured_where="supplier_specification"),
    "insufficient_information", "Supplier purity does not establish a measured delivered-gas concentration.",
    budget_upper_nmol=0.0, coverage={"gas": "unknown"}, rule="R2_supplier_spec_only")
add(imp(base("syn_upstream_purifier", "Impurity measured upstream of the purifier"), measured_where="upstream_of_purifier"),
    "insufficient_information", "Upstream measurement is a what-if, not a delivered bound.",
    budget_upper_nmol=0.0, scenario_upper_nmol=DEMO_1PPM, coverage={"gas": "scenario_only"}, rule="R3_upstream_of_purifier")
add(imp(base("syn_partial_species", "Only NH3 measured in the feed gas"), upper=0.01, species="NH3",
        covers_all_reactive_nitrogen=False),
    "insufficient_information", "A single-species bound leaves NOx unknown, so the result stays inconclusive.",
    budget_upper_nmol=DEMO_001PPM, coverage={"gas": "partial_species"})
r = imp(base("syn_second_feed_unmeasured", "Second gas feed without an impurity measurement"), upper=0.01)
r["gas_sources"].append({**copy.deepcopy(r["gas_sources"][0]), "id": "makeup", "label": "Ar make-up line",
                         "impurity": None})
add(r, "insufficient_information", "Bounding one feed says nothing about another; the unmeasured feed stays unknown.",
    budget_upper_nmol=DEMO_001PPM, coverage={"gas": "partial_feeds"})
r = imp(base("syn_partial_but_covered", "Partial coverage that already explains the observation"), species="NH3",
        covers_all_reactive_nitrogen=False)
r["declared_coverage"] = {"electrolyte": "specified_synthetic"}
add(r, "contamination_not_excluded", "Unknown sources can only enlarge the budget, so 'not excluded' is robust.",
    budget_upper_nmol=DEMO_1PPM)
r = imp(base("syn_isotope_qualitative", "Qualitative 15N2 signal recorded"), upper=0.01)
r["isotope"] = {"performed": True, "quantitative": False, "labeled_impurities_quantified": False}
add(r, "exceeds_quantified_budget", "Isotope evidence is kept separate and does not change the budget.",
    budget_upper_nmol=DEMO_001PPM, rule="R8_labeled_impurities")
r = base("syn_two_nitrogen_atoms", "N2O-type impurity with two nitrogen atoms")
imp(r, upper=0.5, species="N2O", nitrogen_atoms=2)
add(r, "contamination_not_excluded", "a_i = 2 doubles the nitrogen per impurity molecule.",
    budget_upper_nmol=ref_gas_nmol(ATM, 6.0, 298.15, 0.5e-6, atoms=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    for rec in records:
        (OUT / f"{rec['case_id']}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(records)} synthetic records to {OUT}")


if __name__ == "__main__":
    main()
