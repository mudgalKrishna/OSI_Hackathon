"""Every synthetic benchmark record reaches its expected status and reference numbers."""
import json
from pathlib import Path

import pytest

from app import reports
from app.schemas import ExperimentRecord

ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC = sorted((ROOT / "data" / "synthetic").glob("*.json"))
CASES = sorted((ROOT / "data" / "cases").glob("*.json"))


def test_benchmark_size():
    assert 20 <= len(SYNTHETIC) <= 40


@pytest.mark.parametrize("path", SYNTHETIC, ids=lambda p: p.stem)
def test_synthetic_expectations(path, audit):
    d = json.loads(path.read_text(encoding="utf-8"))
    exp = d["expected"]
    res = audit(d)
    assert res.status == exp["status"], res.explanation
    if "budget_upper_nmol" in exp:
        assert res.budget_valid_nmol["upper"] == pytest.approx(exp["budget_upper_nmol"], rel=1e-6, abs=1e-9)
    if "scenario_upper_nmol" in exp:
        assert res.budget_with_scenarios_nmol["upper"] == pytest.approx(exp["scenario_upper_nmol"], rel=1e-6)
    if "observed_nmol" in exp:
        assert [res.observed_nmol["lower"], res.observed_nmol["upper"]] == pytest.approx(exp["observed_nmol"], rel=1e-6)
    for cat, state in exp.get("coverage", {}).items():
        assert res.coverage[cat] == state
    if "issue_code" in exp:
        assert exp["issue_code"] in {i.code for i in res.issues}
    if "rule" in exp:
        assert exp["rule"] in {r.rule_id for r in res.recommendations}


@pytest.mark.parametrize("path", CASES + SYNTHETIC[:3], ids=lambda p: p.stem)
def test_exports_reproduce_result(path, audit):
    d = json.loads(path.read_text(encoding="utf-8"))
    rec = ExperimentRecord(**d)
    res = audit(d)
    j = json.loads(reports.to_json(rec, res))
    assert j["audit"]["status"] == res.status
    assert j["audit"]["budget_valid_nmol"] == res.budget_valid_nmol
    # Re-auditing the exported record gives the identical result.
    again = audit(j["record"])
    assert again.model_dump() == res.model_dump()
    md = reports.to_markdown(rec, res)
    assert res.status_label in md and "prototype" in md
    csv_text = reports.to_csv(rec, res)
    assert res.status_label in csv_text


def test_every_recommendation_cites_a_source(case, audit):
    res = audit(case("bragulla2026_pristine_gde_ocv"))
    assert res.recommendations
    for r in res.recommendations:
        assert r.references and all(ref["quote"] and ref["source_id"] for ref in r.references)


def test_status_vocabulary_has_no_truth_score(case, audit):
    res = audit(case("synthetic_gas_budget_02"))
    text = " ".join([res.headline, *res.explanation]).lower()
    assert "does not prove" in text
    for banned in ("genuine catalyst", "trust score", "proven"):
        assert banned not in text
