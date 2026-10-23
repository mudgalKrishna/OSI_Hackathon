"""v0.3: evidence-requirement inversion, hypothetical measurements, validation suite, case files and CLI."""
import math

import pytest

from app import reports
from app.audit import run_audit
from app.cli import main
from app.inversion import evidence_requirements
from app.planner import REF_CONDITIONS, planner, stress_row
from app.schemas import ExperimentRecord
from app.validation import run_benchmark
from app.whatif import add_hypothetical, compare, suggest


def test_benchmark_and_properties_all_pass():
    _, s = run_benchmark()
    assert s["nitrotrace_correct"] == s["n_synthetic"] == 33
    assert s["false_exceeds_nitrotrace"] == 0
    assert s["naive_correct"] < s["nitrotrace_correct"] and s["checklist_correct"] < s["nitrotrace_correct"]
    for name, (passed, n, failed) in s["properties"].items():
        assert passed == n and n > 0, (name, failed)


def test_ocv_requirements_have_assumptions_and_caveat(case):
    rec = ExperimentRecord(**case("bragulla2026_pristine_gde_ocv"))
    res = run_audit(rec)
    req = evidence_requirements(rec, res)
    assert req["headroom_nmol"] > 0 and req["floor_nmol"] == 0
    assert {u["category"] for u in req["unknown"]} == {"gas", "electrolyte", "components"}
    assert all(s["assumptions"] for u in req["unknown"] for s in u["specs"])
    assert any("does not demonstrate nitrogen reduction" in c for c in req["caveats"])


def test_hypothetical_is_flagged_and_never_silent(case):
    d = case("bragulla2026_pristine_gde_ocv")
    r = compare(d, [("electrolyte_nmol", 25, None), ("components", 25, None)])
    assert r["after"]["valid_upper_nmol"] == pytest.approx(50)
    assert r["after_result"].hypothetical and not run_audit(ExperimentRecord(**d)).hypothetical
    assert any("hypothetical" in e.lower() for e in r["after_result"].explanation)
    # The gas flow was never stated, so no hypothetical blank or component test can close the gap.
    assert r["after"]["status"] == "insufficient_information"
    assert r["after"]["coverage"]["gas"] == "unknown"


def test_gas_threshold_flips_verdict_exactly(case):
    d = case("syn_partial_species")
    rec = ExperimentRecord(**d)
    res = run_audit(rec)
    spec = evidence_requirements(rec, res)["unknown"][0]["specs"][0]
    counted = next(c.upper_nmol for c in res.contributions if c.included)
    total = counted / spec["n_gas_mol"] + spec["ppb_alone"]
    below = compare(d, [("gas", total * (1 - 1e-6), spec["source_id"])])["after"]["status"]
    above = compare(d, [("gas", total * (1 + 1e-6), spec["source_id"])])["after"]["status"]
    assert below == "exceeds_quantified_budget" and above == "contamination_not_excluded"


def test_suggestions_are_below_thresholds(case):
    d = case("syn_partial_species")
    rec = ExperimentRecord(**d)
    req = evidence_requirements(rec, run_audit(rec))
    (kind, value, _), = suggest(rec, req)
    assert kind == "gas" and 0 < value < req["unknown"][0]["specs"][0]["ppb_alone"]


def test_bad_hypotheticals_rejected(case):
    d = case("bragulla2026_pristine_gde_ocv")
    with pytest.raises(ValueError):
        add_hypothetical(d, "electrolyte", 5)  # no sample volume -> concentration cannot become an amount
    with pytest.raises(ValueError):
        add_hypothetical(d, "components", -1)


def test_casefile_and_review_pack(case):
    rec = ExperimentRecord(**case("bragulla2026_pristine_gde_ocv"))
    res = run_audit(rec)
    req = evidence_requirements(rec, res)
    cf = reports.to_casefile(rec, res, req)
    for section in ("Measurements as reported", "Counted and set aside", "Still unbounded", "Qualified conclusion",
                    "What would close the gap", "does not show", "Internal review only"):
        assert section in cf
    assert "85.2" in cf  # the reported quote is carried through
    rp = reports.to_review_pack(rec, res, req)
    assert rp.count("- [ ] correct") == 1 + len(rec.gas_sources) + len(rec.controls)


def test_planner_closed_form():
    p = planner(1e-11, 1.0, 100, REF_CONDITIONS["0 °C, 1 atm"], 300, 1.0, 4, 2, 30, 3)
    assert p["verdict"] == "never" and p["s"] < 3 * p["g"]
    q = planner(1e-11, 1.0, 20, REF_CONDITIONS["0 °C, 1 atm"], 100, 1.0, 4, 2, 30, 3)
    t = q["t_star_s"]
    assert math.isclose(q["s"] * t, 3 * (q["g"] * t + q["b"]), rel_tol=1e-12)
    f = planner(1e-11, 1.0, 20, REF_CONDITIONS["0 °C, 1 atm"], 100, 1.0, 4, 2, 30, 3, fixed_volume_L=0.5)
    assert f["g"] == 0 and f["verdict"] == "after"


def test_stress_matches_planner_ratio():
    o = stress_row(1e-11, "mol/s/cm2", 1.0, 0, 20, REF_CONDITIONS["0 °C, 1 atm"])
    p = planner(1e-11, 1.0, 20, REF_CONDITIONS["0 °C, 1 atm"], o["equivalent_ppb"], 1.0, 4, 0, 30, 1)
    assert p["asymptotic_ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize("argv", [["list"], ["audit", "ocv"], ["audit", "ocv", "--format", "json"], ["settle", "ocv"],
                                  ["whatif", "ocv", "--components", "10"], ["casefile", "ocv"], ["review-pack", "ocv"],
                                  ["plan"], ["stress", "--rate", "1e-11", "--flow", "20", "--lod", "10"], ["bench"]])
def test_cli_commands_run(argv, capsys):
    main(argv)
    assert capsys.readouterr().out.strip()


def test_cli_unknown_case_exits():
    with pytest.raises(SystemExit):
        main(["audit", "no_such_case_anywhere"])
