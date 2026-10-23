"""Forward planning with the audit's conservative arithmetic: signal vs. contamination budget before a run,
and the feed-gas impurity level that alone would supply a reported rate.

Assumptions (stated wherever results are shown): constant true rate, constant flow, constant delivered
impurity level, capture fraction η (η = 1 is the conservative default), and a time-independent matched blank.
"""
from __future__ import annotations

import math

from .units import MOLAR_MASS_G_PER_MOL, R

REF_CONDITIONS = {"0 °C, 1 atm": (273.15, 101325.0), "20 °C, 1 atm": (293.15, 101325.0),
                  "25 °C, 1 atm": (298.15, 101325.0)}

PLAN_ASSUMPTIONS = (
    "Constant true NRR rate, gas flow, delivered impurity level and capture fraction η over the run.",
    "η = 1 means every delivered reactive-N molecule is counted as product (conservative).",
    "The matched blank is a fixed amount, independent of run time.",
)


def planner(rate_mol_s_cm2: float, area_cm2: float, flow_sccm: float, ref_TP: tuple[float, float], ppb_N: float,
            eta: float, run_h: float, blank_ug_L: float, volume_mL: float, margin: float,
            fixed_volume_L: float | None = None, fixed_T_K: float = 298.15, fixed_P_Pa: float = 101325.0,
            n: int = 240) -> dict:
    """Signal and conservative contamination (nmol N) versus time, and the closed-form break-even time."""
    T_ref, P_ref = ref_TP
    s = rate_mol_s_cm2 * area_cm2 * 1e9                                   # signal, nmol N / s
    blank = blank_ug_L * (volume_mL / 1000) * 1e-6 / MOLAR_MASS_G_PER_MOL["NH4+"] * 1e9   # nmol N, fixed
    if fixed_volume_L:
        n_fixed = fixed_P_Pa * fixed_volume_L / 1000 / (R * fixed_T_K)
        g, gas_const = 0.0, n_fixed * ppb_N * eta                         # nmol N, capped
    else:
        ndot = P_ref * flow_sccm * 1e-6 / 60 / (R * T_ref)                # mol gas / s
        g, gas_const = ndot * ppb_N * eta, 0.0                            # nmol N / s
    b = blank + gas_const
    t = [run_h * 3600 * i / (n - 1) for i in range(n)]
    sig = [s * x for x in t]
    con = [g * x + b for x in t]
    # S(t) >= margin * C(t)  <=>  t (s - margin g) >= margin b
    if s <= margin * g:
        t_star, verdict = None, "never"
    else:
        t_star, verdict = margin * b / (s - margin * g), "after"
    return {"t_s": t, "signal": sig, "contamination": con, "s": s, "g": g, "b": b,
            "t_star_s": t_star, "verdict": verdict, "asymptotic_ratio": (s / g) if g > 0 else math.inf,
            "min_rate_mol_s_cm2": margin * (g + b / (run_h * 3600)) / (area_cm2 * 1e9),
            "end_ratio": (sig[-1] / con[-1]) if con[-1] > 0 else math.inf,
            "assumptions": list(PLAN_ASSUMPTIONS)}


RATE_UNITS = {
    "mol s⁻¹ cm⁻²": lambda v, area, load: v * area,
    "nmol s⁻¹ cm⁻²": lambda v, area, load: v * 1e-9 * area,
    "µg(NH3) h⁻¹ cm⁻²": lambda v, area, load: v * 1e-6 / MOLAR_MASS_G_PER_MOL["NH3"] / 3600 * area,
    "µg(NH3) h⁻¹ mg_cat⁻¹": lambda v, area, load: v * 1e-6 / MOLAR_MASS_G_PER_MOL["NH3"] / 3600 * load * area,
}
RATE_UNIT_ALIASES = {"mol/s/cm2": "mol s⁻¹ cm⁻²", "nmol/s/cm2": "nmol s⁻¹ cm⁻²",
                     "ug/h/cm2": "µg(NH3) h⁻¹ cm⁻²", "ug/h/mg": "µg(NH3) h⁻¹ mg_cat⁻¹"}


def stress_row(rate: float, unit: str, area_cm2: float, loading_mg_cm2: float, flow_sccm: float,
               ref_TP: tuple[float, float]) -> dict:
    """Delivered-gas reactive-N mole fraction (ppb, η = 1) that alone supplies the reported rate."""
    T_ref, P_ref = ref_TP
    unit = RATE_UNIT_ALIASES.get(unit, unit)
    mol_s = RATE_UNITS[unit](rate, area_cm2, loading_mg_cm2 or 0.0)
    ndot = P_ref * flow_sccm * 1e-6 / 60 / (R * T_ref)
    return {"product_nmol_s": mol_s * 1e9, "gas_mol_s": ndot,
            "equivalent_ppb": mol_s / ndot * 1e9 if ndot else math.inf}
