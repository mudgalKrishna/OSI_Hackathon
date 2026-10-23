"""Evidence-requirement inversion: what contamination remains unbounded, and how stringent would new
measurements have to be to close the gap?

The audit is solved backwards. Every threshold carries the assumptions it rests on; meeting a
threshold can move the verdict to "exceeds the quantified budget", which by itself still does not
demonstrate nitrogen reduction.
"""
from __future__ import annotations

from .budget import observation_duration
from .coverage import gas_source_states
from .schemas import AuditResult, ExperimentRecord
from .units import MOLAR_MASS_G_PER_MOL, UnitError, gas_moles, gas_moles_from_volume, to_nmol_n

N_OK = ("quantified", "not_applicable", "specified_synthetic")

ALONE_SPLIT = ("'Alone' assumes every other unbounded category contributes nothing; 'equal split' shares the "
               "headroom equally. Neither is a measurement.")
BOUNDARY = ("Thresholds are exclusive: the audit reports 'exceeds' only when the budget is strictly below the "
            "observation, so a bound exactly equal to a threshold leaves the verdict unchanged. Aim below it.")
NOT_PROOF = ("Meeting these thresholds can move the verdict to 'exceeds the quantified budget'. That alone does not "
             "demonstrate nitrogen reduction: isotope-labelled and other controls are still required.")


def obs_duration_s(rec: ExperimentRecord) -> float | None:
    op = rec.observed_product
    return op.interval.duration_min * 60 if op is not None and op.interval is not None else None


def gas_amount_mol(src, rec: ExperimentRecord) -> float | None:
    """Moles of gas delivered through one source during the observation, or None if not determinable."""
    try:
        if src.total_volume_L is not None:
            return gas_moles_from_volume(src.total_volume_L, src.total_volume_temperature_K,
                                         src.total_volume_pressure_Pa)[0]
        if src.flow is None:
            return None
        dur = src.exposure.duration_min if src.exposure else observation_duration(rec)
        if dur is None:
            return None
        f = src.flow
        return gas_moles(f.value, f.unit, f.volume_basis, dur, f.temperature_K, f.pressure_Pa,
                         f.reference_temperature_K, f.reference_pressure_Pa)[0]
    except UnitError:
        return None


def per_unit_nmol(m) -> float | None:
    """nmol N represented by 1 unit of a product measurement (the conversion is linear), or None."""
    try:
        return to_nmol_n(1.0, m.unit, m.species, m.basis, m.volume_mL, m.dilution_factor)[0]
    except UnitError:
        return None


def _nh4_ug_per_L(nmol: float, volume_mL: float) -> float:
    return nmol * 1e-9 * MOLAR_MASS_G_PER_MOL["NH4+"] * 1e6 / (volume_mL / 1000)


def evidence_requirements(rec: ExperimentRecord, res: AuditResult) -> dict | None:
    """Invert the audit: which measurements, at what level, would change the verdict?"""
    if res.status == "input_inconsistency" or res.observed_nmol is None:
        return None
    obs_lo = res.observed_nmol["lower"]
    U = res.budget_valid_nmol["upper"]
    headroom = obs_lo - U
    op = rec.observed_product
    unit_n = per_unit_nmol(op)
    t_s = obs_duration_s(rec)
    area = rec.context.electrode_area_cm2
    out = {
        "observed_lower_nmol": obs_lo, "valid_upper_nmol": U, "headroom_nmol": headroom,
        "floor_nmol": U, "floor_in_record_units": (U / unit_n) if unit_n else None,
        "record_unit": f"{op.unit} {op.species or ''}".strip(),
        "floor_rate_mol_s_cm2": (U * 1e-9 / (t_s * area)) if (t_s and area) else None,
        "unknown": [], "shrink": [], "notes": [], "caveats": [BOUNDARY, NOT_PROOF],
    }
    unknown_cats = [k for k, v in res.coverage.items() if v not in N_OK]

    if headroom > 0:
        share = headroom / max(len(unknown_cats), 1)
        if len(unknown_cats) > 1:
            out["caveats"].append(ALONE_SPLIT)
        for cat in unknown_cats:
            item = {"category": cat, "state": res.coverage[cat], "max_nmol_alone": headroom,
                    "max_nmol_equal_split": share, "specs": []}
            if cat == "gas":
                states = gas_source_states(rec, [c for c in res.contributions if c.included and "gas" in c.category])
                for g in rec.gas_sources:
                    if states.get(g.id) == "bounded":
                        continue  # this feed is already fully bounded; the requirement is on the others
                    n = gas_amount_mol(g, rec)
                    partial = states.get(g.id) == "partial_species"
                    what = (f"Reactive-N species not yet bounded (beyond the measured {g.impurity.species}), as N,"
                            if partial else "Total reactive-N impurity (NOx + NH3 + N2O, as N)")
                    if n:
                        item["specs"].append({
                            "source": g.label, "source_id": g.id, "additional": partial, "kind": "gas",
                            "text": (f"{what} in the gas delivered to the cell < {headroom / n:.3g} ppb (alone) or "
                                     f"< {share / n:.3g} ppb (equal split), for {n:.4g} mol of delivered gas."),
                            "ppb_alone": headroom / n, "ppb_split": share / n, "n_gas_mol": n,
                            "big": headroom / n, "big_unit": "ppb N in delivered gas (alone)",
                            "assumptions": [
                                "η = 1: every delivered reactive-N molecule is assumed to end up as measured "
                                "ammonia (conservative upper bound, not a measured efficiency).",
                                "Delivered gas amount from the record's stated flow, reference conditions and "
                                "exposure interval.",
                                "Measured downstream of any purifier, under the experiment's own conditions."]})
                    else:
                        item["specs"].append({
                            "source": g.label, "source_id": g.id, "kind": "gas", "big": headroom,
                            "big_unit": "nmol N, x_N · n_gas (alone)",
                            "text": (f"Gas amount not determinable: state flow, reference conditions and exposure. "
                                     f"Requirement: x_N · n_gas < {headroom:.3g} nmol."),
                            "assumptions": ["η = 1 (conservative)."]})
            elif cat == "electrolyte":
                v = op.volume_mL
                spec = {"source": "electrolyte", "kind": "electrolyte", "big": headroom,
                        "big_unit": "nmol N (alone)",
                        "text": f"Matched electrolyte / whole-system blank < {headroom:.3g} nmol N (alone)",
                        "assumptions": ["Blank run with the same electrolyte batch, cell, volume and interval as "
                                        "the experiment (matched conditions)."]}
                if v:
                    spec["big"], spec["big_unit"] = _nh4_ug_per_L(headroom, v), f"µg/L NH4+ in {v:g} mL (alone)"
                    spec["text"] += (f" = {_nh4_ug_per_L(headroom, v):.3g} µg/L NH4+ in {v:g} mL; "
                                     f"equal split: {_nh4_ug_per_L(share, v):.3g} µg/L.")
                item["specs"].append(spec)
            else:
                item["specs"].append({
                    "source": "components", "kind": "components", "big": headroom, "big_unit": "nmol N (alone)",
                    "text": (f"Matched release from cell components, electrode, membrane and ionomer < "
                             f"{headroom:.3g} nmol N (alone) / {share:.3g} nmol N (equal split)."),
                    "assumptions": ["Measured under the experiment's own conditions and interval (same electrode, "
                                    "potential history and electrolyte)."]})
            out["unknown"].append(item)
        if not unknown_cats:
            out["notes"].append(f"Every modelled category is bounded; the observation exceeds the valid budget by "
                                f"{headroom:.3g} nmol N (×{obs_lo / U:.3g})." if U > 0 else
                                "Every modelled category is bounded and the valid budget is zero.")
    else:
        deficit = -headroom
        out["notes"].append(f"The valid budget already reaches the observation (deficit {deficit:.3g} nmol N). "
                            "Unknown sources cannot change this; only tighter bounds on counted sources can.")
        for c in [c for c in res.contributions if c.included]:
            new_upper = c.upper_nmol - deficit
            row = {"source": c.label, "source_id": c.source_id, "category": c.category,
                   "current_upper_nmol": c.upper_nmol, "required_upper_nmol": new_upper, "feasible": new_upper > 0,
                   "assumptions": ["All other counted bounds stay as they are."]}
            if new_upper <= 0:
                row["text"] = "Even a zero bound here is not enough: the other counted sources alone reach the observation."
            else:
                g = next((g for g in rec.gas_sources if g.id == c.source_id), None)
                ctl = next((k for k in rec.controls if k.id == c.source_id), None)
                k = new_upper / c.upper_nmol
                if g is not None and g.impurity is not None:
                    row["required_value"] = g.impurity.upper * k
                    row["big"], row["big_unit"] = row["required_value"], f"{g.impurity.unit} measured, delivered"
                    row["text"] = (f"Measured delivered-gas impurity < {g.impurity.upper * k:.3g} {g.impurity.unit} "
                                   f"(now {g.impurity.upper:g} {g.impurity.unit}).")
                elif ctl is not None and ctl.measurement.upper is not None:
                    row["required_value"] = ctl.measurement.upper * k
                    row["big"], row["big_unit"] = row["required_value"], ctl.measurement.unit
                    row["text"] = (f"Matched control < {ctl.measurement.upper * k:.3g} {ctl.measurement.unit} "
                                   f"(now {ctl.measurement.upper:g} {ctl.measurement.unit}"
                                   f"{', i.e. a lower detection limit' if ctl.measurement.nondetect else ''}).")
                else:
                    row["big"], row["big_unit"] = new_upper, "nmol N"
                    row["text"] = f"Upper bound < {new_upper:.3g} nmol N."
            out["shrink"].append(row)
        if unknown_cats:
            out["notes"].append("Even after that, the verdict can only reach 'exceeds the quantified budget' once the "
                                f"unbounded categories ({', '.join(unknown_cats)}) are also measured.")
        if unit_n:
            out["notes"].append(f"Alternatively the observation would need to be > {U / unit_n:.3g} {out['record_unit']} "
                                f"({U:.3g} nmol N) with today's controls.")
    return out
