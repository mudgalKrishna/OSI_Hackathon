"""Explicit, tested unit conversions.

Internal comparison unit: nanomoles of ammonia-equivalent nitrogen (nmol N).
NH3, NH4+ and N each carry one nitrogen atom, so amounts are interchangeable,
but mass -> amount conversion needs the molar mass of the *declared* species.
Only the unit strings listed here are accepted; anything else is rejected.
"""
from __future__ import annotations

R = 8.314462618  # J mol^-1 K^-1
ATM_PA = 101325.0

MOLAR_MASS_G_PER_MOL = {"NH3": 17.031, "NH4+": 18.038, "N": 14.007}

AMOUNT_UNITS_TO_MOL = {"mol": 1.0, "mmol": 1e-3, "umol": 1e-6, "µmol": 1e-6, "nmol": 1e-9, "pmol": 1e-12}
MASS_UNITS_TO_G = {"g": 1.0, "mg": 1e-3, "ug": 1e-6, "µg": 1e-6, "ng": 1e-9}
MOLAR_CONC_TO_MOL_PER_L = {"M": 1.0, "mol/L": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "nM": 1e-9}
MASS_CONC_TO_G_PER_L = {"g/L": 1.0, "mg/L": 1e-3, "ug/L": 1e-6, "µg/L": 1e-6, "ng/mL": 1e-6, "ug/mL": 1e-3, "µg/mL": 1e-3}

BASIS_UNITS = {
    "amount": AMOUNT_UNITS_TO_MOL,
    "mass": MASS_UNITS_TO_G,
    "concentration_molar": MOLAR_CONC_TO_MOL_PER_L,
    "concentration_mass": MASS_CONC_TO_G_PER_L,
}

# Strings that look like units but are ambiguous for liquid samples.
AMBIGUOUS_UNITS = {"ppm", "ppb", "ppt", "%", "wt%"}

GAS_FRACTION = {"mole_fraction": 1.0, "mol_percent": 1e-2, "ppm": 1e-6, "ppb": 1e-9}
FLOW_TO_L_PER_MIN = {"mL/min": 1e-3, "L/min": 1.0, "sccm": 1e-3, "slpm": 1.0}


class UnitError(ValueError):
    """Raised for an ambiguous, unsupported or inconsistent unit."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def basis_for_unit(unit: str) -> str | None:
    for basis, table in BASIS_UNITS.items():
        if unit in table:
            return basis
    return None


def to_nmol_n(value: float, unit: str, species: str | None, basis: str | None,
              volume_mL: float | None = None, dilution_factor: float = 1.0) -> tuple[float, str]:
    """Convert one product value to nmol N. Returns (value, formula text)."""
    if species is None:
        raise UnitError("missing_species", "Species (NH3, NH4+ or N) must be declared explicitly.")
    if species not in MOLAR_MASS_G_PER_MOL:
        raise UnitError("unknown_species", f"Unsupported species '{species}'.")
    if basis is None:
        raise UnitError("missing_basis", "Reporting basis (amount, mass or concentration) must be declared explicitly.")
    if unit in AMBIGUOUS_UNITS and basis_for_unit(unit) is None:
        raise UnitError("ambiguous_unit", f"Unit '{unit}' is ambiguous for a liquid-phase product; use mol, g or mol/L forms.")
    unit_basis = basis_for_unit(unit)
    if unit_basis is None:
        raise UnitError("unsupported_unit", f"Unit '{unit}' is not supported.")
    if unit_basis != basis:
        raise UnitError("basis_unit_conflict", f"Unit '{unit}' implies basis '{unit_basis}' but basis '{basis}' was declared.")
    if dilution_factor <= 0:
        raise UnitError("bad_dilution", "Dilution factor must be positive.")

    M = MOLAR_MASS_G_PER_MOL[species]
    if basis == "amount":
        mol = value * AMOUNT_UNITS_TO_MOL[unit]
        text = f"{value:g} {unit} {species} = {mol*1e9:.6g} nmol N"
    elif basis == "mass":
        mol = value * MASS_UNITS_TO_G[unit] / M
        text = f"{value:g} {unit} {species} / {M} g/mol = {mol*1e9:.6g} nmol N"
    else:
        if volume_mL is None or volume_mL <= 0:
            raise UnitError("missing_volume", "A concentration needs the sample/electrolyte volume to give an amount.")
        V = volume_mL / 1000.0
        if basis == "concentration_molar":
            mol = value * MOLAR_CONC_TO_MOL_PER_L[unit] * dilution_factor * V
            text = f"{value:g} {unit} x DF {dilution_factor:g} x {volume_mL:g} mL = {mol*1e9:.6g} nmol N"
        else:
            mol = value * MASS_CONC_TO_G_PER_L[unit] * dilution_factor * V / M
            text = (f"{value:g} {unit} {species} x DF {dilution_factor:g} x {volume_mL:g} mL / {M} g/mol"
                    f" = {mol*1e9:.6g} nmol N")
    return mol * 1e9, text


def gas_moles(flow_value: float, flow_unit: str, volume_basis: str, duration_min: float,
              T_K: float | None, P_Pa: float | None,
              ref_T_K: float | None, ref_P_Pa: float | None) -> tuple[float, str]:
    """Moles of gas delivered: n = P Q t / (R T).

    Actual flows use the gas temperature and pressure at the flow measurement.
    Standardized flows (sccm/slpm) use the flow controller's reference
    conditions; the cell temperature must never be applied to them.
    """
    if flow_unit not in FLOW_TO_L_PER_MIN:
        raise UnitError("unsupported_unit", f"Flow unit '{flow_unit}' is not supported.")
    standardized_unit = flow_unit in ("sccm", "slpm")
    if standardized_unit and volume_basis != "standard":
        raise UnitError("flow_basis_conflict", f"'{flow_unit}' is a standardized flow but the volume basis was declared 'actual'.")
    if volume_basis == "standard":
        if ref_T_K is None or ref_P_Pa is None:
            raise UnitError("missing_reference_conditions",
                            "A standardized flow needs the flow controller's reference temperature and pressure.")
        T, P, cond = ref_T_K, ref_P_Pa, "controller reference"
    else:
        if T_K is None or P_Pa is None:
            raise UnitError("missing_gas_conditions", "An actual volumetric flow needs gas temperature and pressure.")
        T, P, cond = T_K, P_Pa, "actual gas"
    if T <= 0 or P <= 0 or duration_min < 0:
        raise UnitError("bad_value", "Temperature, pressure and duration must be positive.")
    V_m3 = flow_value * FLOW_TO_L_PER_MIN[flow_unit] * duration_min / 1000.0
    n = P * V_m3 / (R * T)
    text = (f"n_gas = P·Q·t/(R·T) = {P:g} Pa x {flow_value:g} {flow_unit} x {duration_min:g} min "
            f"/ ({R} x {T:g} K) [{cond} conditions] = {V_m3*1000:.6g} L -> {n:.6g} mol")
    return n, text


def gas_moles_from_volume(volume_L: float, T_K: float | None, P_Pa: float | None) -> tuple[float, str]:
    if T_K is None or P_Pa is None:
        raise UnitError("missing_gas_conditions", "A total gas volume needs its temperature and pressure.")
    n = P_Pa * volume_L / 1000.0 / (R * T_K)
    return n, f"n_gas = P·V/(R·T) = {P_Pa:g} Pa x {volume_L:g} L / ({R} x {T_K:g} K) = {n:.6g} mol"


def gas_fraction(value: float, unit: str) -> float:
    if unit not in GAS_FRACTION:
        raise UnitError("unsupported_unit", f"Gas impurity unit '{unit}' is not supported.")
    return value * GAS_FRACTION[unit]
