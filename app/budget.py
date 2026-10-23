"""Contamination budget arithmetic: observed product, gas-delivered and control contributions.

All values are nmol of ammonia-equivalent nitrogen (nmol N). Bounds are
propagated with plain interval arithmetic; no distributions are invented.
"""
from __future__ import annotations

from .schemas import (
    Contribution, ControlMeasurement, ExperimentRecord, GasSource, Issue,
    ProductMeasurement, TrailStep,
)
from .units import UnitError, gas_fraction, gas_moles, gas_moles_from_volume, to_nmol_n

INTERVAL_TOLERANCE = 0.01  # 1 % relative difference in durations counts as a mismatch


def _durations_differ(a: float, b: float) -> bool:
    return abs(a - b) > INTERVAL_TOLERANCE * max(abs(a), abs(b), 1e-12)


def measurement_bounds(m: ProductMeasurement, label: str, field: str,
                       issues: list[Issue]) -> tuple[float, float, list[TrailStep]] | None:
    """Normalize a product measurement to a (lower, upper) interval in nmol N."""
    trail: list[TrailStep] = []
    if m.nondetect:
        if m.upper is None:
            issues.append(Issue(code="nondetect_without_limit", severity="error", field=field,
                                message=f"{label}: a nondetect needs its detection/quantification limit as the upper bound."))
            return None
        lo_raw, hi_raw = 0.0, m.upper
    else:
        lo_raw = m.lower if m.lower is not None else m.upper
        hi_raw = m.upper if m.upper is not None else m.lower
        if lo_raw is None:
            issues.append(Issue(code="missing_value", severity="error", field=field,
                                message=f"{label}: no value was entered."))
            return None
    if lo_raw > hi_raw:
        issues.append(Issue(code="inverted_interval", severity="error", field=field,
                            message=f"{label}: lower bound exceeds upper bound."))
        return None
    try:
        lo, lo_text = to_nmol_n(lo_raw, m.unit, m.species, m.basis, m.volume_mL, m.dilution_factor)
        hi, hi_text = to_nmol_n(hi_raw, m.unit, m.species, m.basis, m.volume_mL, m.dilution_factor)
    except UnitError as e:
        issues.append(Issue(code=e.code, severity="error", field=field, message=f"{label}: {e}"))
        return None
    kind = "nondetect (0 to limit)" if m.nondetect else m.bound_type
    trail.append(TrailStep(label=f"{label} — normalize to nmol N ({kind})",
                           values={"lower": lo_text, "upper": hi_text},
                           result=f"[{lo:.4g}, {hi:.4g}] nmol N"))
    return lo, hi, trail


def observed_bounds(rec: ExperimentRecord, issues: list[Issue]):
    if rec.observed_product is None:
        return None
    return measurement_bounds(rec.observed_product, "Observed product", "observed_product", issues)


def observation_duration(rec: ExperimentRecord) -> float | None:
    op = rec.observed_product
    if op is not None and op.interval is not None:
        return op.interval.duration_min
    return None


def gas_contribution(src: GasSource, rec: ExperimentRecord, issues: list[Issue],
                     assumptions: list[str]) -> tuple[Contribution | None, str | None]:
    """Upper/lower bound on product nitrogen from impurities in one gas stream.

    Returns (None, reason) when the stream cannot be bounded because information
    is missing (unknown, not zero), and (None, None) after an input error.
    """
    field = f"gas_sources.{src.id}"
    imp = src.impurity
    if imp is None or imp.upper is None:
        return None, f"{src.label}: impurity not measured"
    if src.total_volume_L is None and src.flow is None:
        return None, f"{src.label}: gas flow / delivered volume during the measurement not stated"

    trail: list[TrailStep] = []
    obs_dur = observation_duration(rec)
    if src.exposure is not None:
        duration = src.exposure.duration_min
        if obs_dur is not None and _durations_differ(duration, obs_dur):
            issues.append(Issue(code="interval_mismatch", severity="error", field=field,
                                message=(f"{src.label}: gas exposure ({duration:g} min) does not match the product "
                                         f"collection interval ({obs_dur:g} min).")))
            return None, None
    elif obs_dur is not None:
        duration = obs_dur
        assumptions.append(f"{src.label}: gas exposure taken equal to the product collection interval ({obs_dur:g} min).")
    else:
        duration = None
    if src.total_volume_L is None and duration is None:
        return None, f"{src.label}: exposure duration not stated"

    try:
        if src.total_volume_L is not None:
            n_gas, text = gas_moles_from_volume(src.total_volume_L, src.total_volume_temperature_K,
                                                src.total_volume_pressure_Pa)
        else:
            f = src.flow
            n_gas, text = gas_moles(f.value, f.unit, f.volume_basis, duration, f.temperature_K, f.pressure_Pa,
                                    f.reference_temperature_K, f.reference_pressure_Pa)
        x_hi = gas_fraction(imp.upper, imp.unit)
        x_lo = gas_fraction(imp.lower, imp.unit) if imp.lower is not None else 0.0
    except UnitError as e:
        issues.append(Issue(code=e.code, severity="error", field=field, message=f"{src.label}: {e}"))
        return None, None
    trail.append(TrailStep(label="Moles of gas delivered", formula="n_gas = P·Q·t / (R·T)",
                           values={"calculation": text}, result=f"{n_gas:.6g} mol"))

    eta_hi = src.eta_upper if src.eta_upper is not None else 1.0
    if src.eta_upper is None or src.eta_basis == "conservative_upper_bound_1":
        assumptions.append(f"{src.label}: delivery/conversion/collection fraction bounded above by 1 "
                           "(explicit conservative scenario, not a fitted efficiency).")
    a = imp.nitrogen_atoms
    upper = n_gas * x_hi * a * eta_hi * 1e9
    lower = n_gas * x_lo * a * src.eta_lower * 1e9
    trail.append(TrailStep(
        label=f"Impurity-derived nitrogen ({imp.species})", formula="U_i = n_gas · x_i,upper · a_i · η_i,upper",
        values={"n_gas": f"{n_gas:.6g} mol", "x_upper": f"{x_hi:.4g} ({imp.upper:g} {imp.unit})",
                "x_lower": f"{x_lo:.4g}", "a_i": str(a), "η_upper": f"{eta_hi:g} ({src.eta_basis})",
                "η_lower": f"{src.eta_lower:g}"},
        result=f"[{lower:.4g}, {upper:.4g}] nmol N"))

    role, reason = "valid_bound", None
    if imp.measured_where == "supplier_specification":
        role = "excluded_invalid"
        reason = ("Supplier purity specification, not a measured delivered-gas impurity concentration; "
                  "shown for reference only.")
    elif imp.measured_where == "upstream_of_purifier":
        role = "scenario"
        reason = ("Measured upstream of a purifier: not a measurement of what reached the cell. Used only as a "
                  "what-if (conservative only if the purifier and downstream lines add no nitrogen species).")
    elif imp.measured_where == "unknown":
        role = "scenario"
        reason = "Measurement location relative to the cell is unknown; used only as a what-if."
    return Contribution(source_id=src.id, label=src.label, category="gas", lower_nmol=lower, upper_nmol=upper,
                        included=role == "valid_bound", role=role, reason=reason, trail=trail), None


def control_contribution(c: ControlMeasurement, rec: ExperimentRecord, issues: list[Issue],
                         assumptions: list[str]) -> Contribution | None:
    field = f"controls.{c.id}"
    b = measurement_bounds(c.measurement, c.label, field, issues)
    if b is None:
        return None
    lo, hi, trail = b

    obs_dur = observation_duration(rec)
    ctl_dur = c.measurement.interval.duration_min if c.measurement.interval else None
    if obs_dur is not None and ctl_dur is not None and _durations_differ(obs_dur, ctl_dur):
        if c.time_scaling == "linear":
            k = obs_dur / ctl_dur
            lo, hi = lo * k, hi * k
            assumptions.append(f"{c.label}: scaled linearly in time by {obs_dur:g}/{ctl_dur:g} min (assumed constant release rate).")
            trail.append(TrailStep(label="Linear time scaling", formula="amount × t_obs / t_control",
                                   values={"factor": f"{k:.4g}"}, result=f"[{lo:.4g}, {hi:.4g}] nmol N"))
        elif c.conditions_match == "matched":
            issues.append(Issue(code="interval_mismatch", severity="error", field=field,
                                message=(f"{c.label}: control interval ({ctl_dur:g} min) does not match the product "
                                         f"collection interval ({obs_dur:g} min); declare a time-scaling model or "
                                         "mark the control as measured under different conditions.")))
            return None
    elif c.conditions_match == "matched" and obs_dur is not None and ctl_dur is None:
        issues.append(Issue(code="control_interval_missing", severity="warning", field=field,
                            message=f"{c.label}: control interval not stated; it is assumed to match the observation."))

    op = rec.observed_product
    if (c.conditions_match == "matched" and op is not None and op.volume_mL and c.measurement.volume_mL
            and _durations_differ(op.volume_mL, c.measurement.volume_mL)):
        issues.append(Issue(code="volume_mismatch", severity="warning", field=field,
                            message=(f"{c.label}: control volume ({c.measurement.volume_mL:g} mL) differs from the "
                                     f"sample volume ({op.volume_mL:g} mL); the amount, not the concentration, is compared.")))

    if c.conditions_match == "matched":
        role, reason = "valid_bound", None
    else:
        why = c.conditions_note or ("measured under different conditions" if c.conditions_match == "different"
                                    else "matching to the experiment is not established")
        if c.include_as_scenario:
            role, reason = "scenario", f"Not interchangeable with the experiment ({why}); used only as an explicit what-if."
        else:
            role, reason = "excluded_unmatched", f"Not interchangeable with the experiment ({why})."
    return Contribution(source_id=c.id, label=c.label, category="+".join(c.covers), lower_nmol=lo, upper_nmol=hi,
                        included=role == "valid_bound", role=role, reason=reason, trail=trail)


def feasible_allocation(contribs: list[Contribution], obs_lo: float, obs_hi: float) -> dict | None:
    """One illustrative allocation inside the independent source ranges reaching the observed interval."""
    inc = [c for c in contribs if c.included]
    lo_sum = sum(c.lower_nmol for c in inc)
    hi_sum = sum(c.upper_nmol for c in inc)
    target = max(obs_lo, lo_sum)
    if not inc or target > min(obs_hi, hi_sum):
        return None
    slack = hi_sum - lo_sum
    frac = 0.0 if slack == 0 else (target - lo_sum) / slack
    parts = {c.label: c.lower_nmol + frac * (c.upper_nmol - c.lower_nmol) for c in inc}
    return {"target_nmol": target, "allocation_nmol": parts,
            "label": "Illustrative feasible explanation — not a reconstruction of the actual experiment."}
