"""Regressions for issues found in external review (v0.3.1): multi-feed gas coverage, non-finite inputs,
exclusive thresholds, and single-topic evidence passages."""
import copy

import pytest

from app.audit import run_audit
from app.cli import main
from app.extraction import keep_passage, passages
from app.inversion import evidence_requirements
from app.schemas import ExperimentRecord
from app.whatif import add_hypothetical


def _audit(d):
    return run_audit(ExperimentRecord(**d))


def test_unmeasured_second_feed_is_not_ignored(case):
    d = case("synthetic_gas_budget_02")
    assert _audit(d).status == "exceeds_quantified_budget"
    d["gas_sources"].append({**copy.deepcopy(d["gas_sources"][0]), "id": "feed2", "label": "second feed",
                             "impurity": None})
    res = _audit(d)
    assert res.status == "insufficient_information"
    assert res.coverage["gas"] == "partial_feeds"
    assert any("second feed" in u.reason for u in res.unquantified)


def test_requirement_targets_only_the_unbounded_feed(case):
    rec = ExperimentRecord(**case("syn_second_feed_unmeasured"))
    req = evidence_requirements(rec, run_audit(rec))
    gas = next(u for u in req["unknown"] if u["category"] == "gas")
    assert [s["source_id"] for s in gas["specs"]] == ["makeup"]
    # bounding that feed (well below the threshold) is what closes the gap
    after = _audit(add_hypothetical(case("syn_second_feed_unmeasured"), "gas", 382, None))
    assert after.status == "exceeds_quantified_budget" and after.coverage["gas"] == "quantified"


def test_matched_system_blank_covering_gas_is_an_alternative(case):
    d = case("syn_second_feed_unmeasured")
    d["controls"].append({"id": "sb", "label": "whole-system blank", "kind": "system_blank",
                          "covers": ["gas", "electrolyte", "components"], "conditions_match": "matched",
                          "measurement": {"species": "N", "basis": "amount", "unit": "nmol", "lower": 0,
                                          "upper": 20, "interval": {"start_min": 0, "end_min": 120}}})
    d["coverage_model"] = "system_blank"
    res = _audit(d)
    assert res.coverage["gas"] == "quantified" and res.status == "exceeds_quantified_budget"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_observation_is_an_input_error(case, bad):
    d = case("synthetic_gas_budget_02")
    d["observed_product"]["lower"] = d["observed_product"]["upper"] = bad
    res = _audit(d)
    assert res.status == "input_inconsistency"
    assert any(i.code == "non_finite_value" and i.field == "observed_product.lower" for i in res.issues)


def test_non_finite_control_and_hypothetical_rejected(case):
    d = case("synthetic_gas_budget_02")
    d["gas_sources"][0]["impurity"]["upper"] = float("nan")
    assert _audit(d).status == "input_inconsistency"
    with pytest.raises(ValueError):
        add_hypothetical(case("synthetic_gas_budget_02"), "components", float("nan"))


def test_cli_rejects_nan_argument():
    with pytest.raises(SystemExit) as e:
        main(["whatif", "ocv", "--components", "nan"])
    assert e.value.code == 2


def test_budget_equal_to_observation_does_not_exceed(case):
    d = case("synthetic_gas_budget_02")
    d["gas_sources"] = []
    d["declared_coverage"] = {"gas": "not_applicable", "electrolyte": "specified_synthetic",
                              "components": "specified_synthetic"}
    d["controls"] = [{"id": "c", "label": "component test", "kind": "component_release", "covers": ["components"],
                      "conditions_match": "matched",
                      "measurement": {"species": "N", "basis": "amount", "unit": "nmol", "lower": 0, "upper": 190.0,
                                      "interval": {"start_min": 0, "end_min": 120}}}]
    rec = ExperimentRecord(**d)
    res = run_audit(rec)
    assert res.observed_nmol["lower"] == res.budget_valid_nmol["upper"] == 190.0
    assert res.status == "contamination_not_excluded"
    # so the inversion states requirements as strict bounds, and says so
    rec01 = ExperimentRecord(**case("synthetic_gas_budget_01"))
    req = evidence_requirements(rec01, run_audit(rec01))
    assert req["shrink"] and all("<" in r["text"] and "≤" not in r["text"] for r in req["shrink"])
    assert any("exclusive" in c for c in req["caveats"])


def test_single_topic_measurement_passages_are_kept():
    text = ("NOx impurities in the N2 supply were below 5 ppb. "
            "An Ar control at open circuit gave a background of 0.8 µg L-1 NH4+. "
            "The cell was then left for 2 h.")
    found = passages([{"page": 1, "text": text}])
    topics = {t for p in found for t in p["topics"]}
    assert {"impurity", "blank"} <= topics
    assert not any(p["text"].startswith("The cell was then left") for p in found)
    assert keep_passage(["impurity"], {}) and not keep_passage(["duration"], {"duration": ["2 h"]})


def test_interactive_ui_runs_commands_and_remembers_the_case(capsys):
    from app.tui import Shell
    sh = Shell()
    sh.welcome()
    assert sh.handle("ocv") and sh.case == "bragulla2026_pristine_gde_ocv"
    assert sh.handle("/settle") and sh.handle("/whatif --blank-nmol 25.5 --components 25.5")
    assert sh.handle("/audit nosuch") and sh.handle("/whatif --components nan") and sh.handle("/nonsense")
    assert sh.case == "bragulla2026_pristine_gde_ocv"
    assert not sh.handle("/exit")
    out = capsys.readouterr().out
    assert "NitroTrace" in out and "What would close the gap" in out and "Before → after" in out
    assert "No record matches 'nosuch'" in out and "Unknown command" in out


def test_no_arguments_without_a_terminal_prints_help(capsys):
    main([])
    assert "usage: nitrotrace" in capsys.readouterr().out
