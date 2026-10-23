"""Reference calculations checked against independent arithmetic and published tables."""
import pytest


def test_demo_three_steps(case, audit):
    r1 = audit(case("synthetic_gas_budget_01"))
    assert r1.budget_valid_nmol["upper"] == pytest.approx(245.24, abs=0.01)
    assert r1.status == "contamination_not_excluded"
    assert r1.feasible_allocation is not None
    assert "Illustrative" in r1.feasible_allocation["label"]

    r2 = audit(case("synthetic_gas_budget_02"))
    assert r2.budget_valid_nmol["upper"] == pytest.approx(2.4524, abs=1e-3)
    assert r2.status == "exceeds_quantified_budget"

    r3 = audit(case("synthetic_gas_budget_03"))
    assert r3.status == "insufficient_information"
    assert r3.coverage["gas"] == "unknown"
    assert any(u.category == "gas" for u in r3.unquantified)


def test_choi2020_table1_reproduced(case, audit):
    """Choi et al. 2020 Table 1, row 1: 100 mL/min, 240 min, 298 K, 1 atm, 10 ppm → 9.82e3 nmol, 0.68 nmol/s."""
    d = case("choi2020_table1_flowing_gas")
    res = audit(d)
    (c,) = res.contributions
    assert c.upper_nmol == pytest.approx(9.82e3, rel=5e-3)
    assert c.upper_nmol / (240 * 60) == pytest.approx(0.68, rel=5e-3)
    assert c.role == "excluded_invalid"  # supplier grade, not a measured delivered concentration
    assert res.budget_valid_nmol["upper"] == 0
    assert res.status == "insufficient_information"  # no product measurement attached


@pytest.mark.parametrize("flow,ppm,minutes,moles_n2,moles_nox", [
    (100, 100, 240, 9.82e8, 9.82e4),
    (20, 10, 240, 1.96e8, 1.96e3),
    (20, 1000, 15, 1.23e7, 1.23e4),   # 15 min pre-purge at 20 mL/min (the '0 mL/min' rows)
])
def test_choi2020_table1_other_rows(case, audit, flow, ppm, minutes, moles_n2, moles_nox):
    d = case("choi2020_table1_flowing_gas")
    g = d["gas_sources"][0]
    g["flow"]["value"], g["impurity"]["upper"] = flow, ppm
    g["exposure"]["end_min"] = minutes
    g["impurity"]["measured_where"] = "delivered_to_cell"
    res = audit(d)
    assert res.contributions[0].upper_nmol == pytest.approx(moles_nox, rel=1e-2)
    assert res.contributions[0].upper_nmol / (ppm * 1e-6) == pytest.approx(moles_n2, rel=1e-2)


def test_bragulla_gas_washing_consistency(case, audit):
    """Reported 80 ppb over 12.92 L should reproduce the 14.8 µg/L × 50 mL trap amount within a few %."""
    res = audit(case("bragulla2026_n2_gas_washing"))
    obs = res.observed_nmol["upper"]
    assert obs == pytest.approx(14.8e-6 * 0.05 / 18.038 * 1e9)
    gas = next(c for c in res.contributions if c.category == "gas")
    assert gas.upper_nmol == pytest.approx(obs, rel=0.05)
    assert res.status == "contamination_not_excluded"


def test_bragulla_purified_nondetect(case, audit):
    res = audit(case("bragulla2026_n2_purified_nondetect"))
    assert res.observed_nmol["lower"] == 0
    assert res.observed_nmol["upper"] == pytest.approx(1.8e-6 * 0.05 / 18.038 * 1e9)
    gas = res.contributions[0]
    assert gas.upper_nmol == pytest.approx(res.observed_nmol["upper"], rel=0.1)  # '<3 ppb' ↔ MLOD


def test_bragulla_gde_ocv(case, audit):
    res = audit(case("bragulla2026_pristine_gde_ocv"))
    assert res.observed_nmol["upper"] == pytest.approx(2.76e-6 / 18.038 * 1e9)
    assert res.budget_valid_nmol["upper"] == 0
    scen = res.budget_with_scenarios_nmol["upper"]
    assert scen == pytest.approx((1.55e-6 + 6.6e-6 * 0.05) / 18.038 * 1e9)
    assert scen < res.observed_nmol["lower"]  # what-if components do not explain the in-cell value
    assert res.status == "insufficient_information"
    assert "R6_nafion_gde" in {r.rule_id for r in res.recommendations}


def test_two_nitrogen_atoms(case, audit):
    res = audit(case("syn_two_nitrogen_atoms"))
    assert res.budget_valid_nmol["upper"] == pytest.approx(245.24, abs=0.01)  # 0.5 ppm × 2 N


def test_linear_time_scaling_declared(case, audit):
    res = audit(case("syn_control_linear_scaling"))
    blank = next(c for c in res.contributions if c.source_id == "blank")
    assert blank.upper_nmol == pytest.approx(20)
    assert any("scaled linearly" in a for a in res.assumptions)
