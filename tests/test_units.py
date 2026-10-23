import math

import pytest

from app.units import UnitError, gas_moles, gas_moles_from_volume, to_nmol_n


def test_amount_units():
    assert to_nmol_n(1, "µmol", "NH3", "amount")[0] == pytest.approx(1000)
    assert to_nmol_n(0.2, "umol", "NH4+", "amount")[0] == pytest.approx(200)


def test_mass_uses_declared_species():
    nh3 = to_nmol_n(17.031, "µg", "NH3", "mass")[0]
    nh4 = to_nmol_n(17.031, "µg", "NH4+", "mass")[0]
    n = to_nmol_n(17.031, "µg", "N", "mass")[0]
    assert nh3 == pytest.approx(1000)
    assert nh4 == pytest.approx(1000 * 17.031 / 18.038)
    assert n == pytest.approx(1000 * 17.031 / 14.007)
    assert len({round(nh3), round(nh4), round(n)}) == 3  # bases must never be mixed silently


def test_concentrations_and_dilution():
    assert to_nmol_n(4, "µM", "NH3", "concentration_molar", volume_mL=50)[0] == pytest.approx(200)
    assert to_nmol_n(2, "µM", "NH3", "concentration_molar", volume_mL=50, dilution_factor=2)[0] == pytest.approx(200)
    # 85.2 µg/L NH4+ in 32.4 mL ≈ 2.76 µg (Bragulla et al. 2026, §3.6)
    assert to_nmol_n(85.2, "µg/L", "NH4+", "concentration_mass", volume_mL=32.4)[0] == pytest.approx(
        2.76e-6 / 18.038 * 1e9, rel=2e-3)


@pytest.mark.parametrize("args,code", [
    ((1, "nmol", None, "amount"), "missing_species"),
    ((1, "nmol", "NH3", None), "missing_basis"),
    ((1, "ppm", "NH3", "concentration_mass"), "ambiguous_unit"),
    ((1, "µg", "NH3", "amount"), "basis_unit_conflict"),
    ((1, "furlong", "NH3", "amount"), "unsupported_unit"),
    ((1, "µM", "NH3", "concentration_molar"), "missing_volume"),
])
def test_rejected_inputs(args, code):
    with pytest.raises(UnitError) as e:
        to_nmol_n(*args)
    assert e.value.code == code


def test_ideal_gas_reference_6L():
    n, _ = gas_moles(50, "mL/min", "actual", 120, 298.15, 101325, None, None)
    assert n == pytest.approx(101325 * 0.006 / (8.314462618 * 298.15))
    assert n * 1e-6 * 1e9 == pytest.approx(245.24, abs=0.01)  # plan: ≈245 nmol at 1 ppm


def test_sccm_uses_reference_not_cell_temperature():
    n_std, _ = gas_moles(50, "sccm", "standard", 120, 353.15, 101325, 273.15, 101325)
    n_act, _ = gas_moles(50, "mL/min", "actual", 120, 353.15, 101325, None, None)
    assert n_std == pytest.approx(101325 * 0.006 / (8.314462618 * 273.15))
    assert n_std > n_act
    with pytest.raises(UnitError, match="reference"):
        gas_moles(50, "sccm", "standard", 120, 298.15, 101325, None, None)
    with pytest.raises(UnitError) as e:
        gas_moles(50, "sccm", "actual", 120, 298.15, 101325, None, None)
    assert e.value.code == "flow_basis_conflict"


def test_total_volume():
    n, _ = gas_moles_from_volume(6.0, 298.15, 101325)
    assert math.isclose(n, gas_moles(50, "mL/min", "actual", 120, 298.15, 101325, None, None)[0])
