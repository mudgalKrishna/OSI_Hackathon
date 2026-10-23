import pytest


def test_overlap_not_silently_summed(case, audit):
    res = audit(case("syn_overlap_unresolved"))
    roles = {c.source_id: c.role for c in res.contributions}
    assert roles["sysblank"] == roles["membrane"] == "excluded_overlap"
    assert res.coverage["components"] == "ambiguous_overlap"
    assert res.coverage["electrolyte"] == "ambiguous_overlap"
    assert any(i.code == "overlap_unresolved" for i in res.issues)


def test_system_blank_model(case, audit):
    res = audit(case("syn_overlap_system_blank"))
    assert res.budget_valid_nmol["upper"] == pytest.approx(2.4524 + 20, abs=1e-3)
    assert {c.source_id: c.included for c in res.contributions}["membrane"] is False


def test_sum_components_model_leaves_gap(case, audit):
    res = audit(case("syn_overlap_sum_components"))
    assert res.coverage["electrolyte"] == "unknown"
    assert res.budget_valid_nmol["upper"] == pytest.approx(2.4524 + 15, abs=1e-3)


def test_blank_that_contains_gas(case, audit):
    res = audit(case("syn_blank_includes_gas"))
    gas = next(c for c in res.contributions if c.category == "gas")
    assert gas.role == "excluded_overlap"
    assert res.budget_valid_nmol["upper"] == pytest.approx(150)
    assert res.status == "exceeds_quantified_budget"


def test_unmatched_and_scenario_controls(case, audit):
    res = audit(case("syn_blank_different_day"))
    assert res.contributions[-1].role == "excluded_unmatched"
    res = audit(case("syn_blank_scenario"))
    assert res.budget_valid_nmol["upper"] == pytest.approx(2.4524, abs=1e-3)
    assert res.budget_with_scenarios_nmol["upper"] == pytest.approx(302.4524, abs=1e-3)
    assert res.status == "insufficient_information"


def test_gas_evidence_quality(case, audit):
    assert audit(case("syn_supplier_spec")).coverage["gas"] == "unknown"
    assert audit(case("syn_upstream_purifier")).coverage["gas"] == "scenario_only"
    assert audit(case("syn_partial_species")).coverage["gas"] == "partial_species"


def test_synthetic_declaration_rejected_on_real_record(case, audit):
    d = case("bragulla2026_pristine_gde_ocv")
    d["declared_coverage"] = {"gas": "specified_synthetic"}
    res = audit(d)
    assert res.status == "input_inconsistency"
    assert any(i.code == "synthetic_coverage_on_real_record" for i in res.issues)


def test_unknown_never_zero(case, audit):
    """Removing a measurement can only move a result toward 'insufficient', never toward 'exceeds'."""
    d = case("synthetic_gas_budget_02")
    d["declared_coverage"] = {}
    res = audit(d)
    assert res.status == "insufficient_information"
    assert {u.category for u in res.unquantified} == {"electrolyte", "components"}
