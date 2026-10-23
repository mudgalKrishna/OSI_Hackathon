"""Run a full evidence audit on one experiment record.

Status logic (Section 7 of the build plan):
  1. Any input error                                  -> input_inconsistency
  2. No observed product                              -> insufficient_information
  3. Valid budget upper >= observed lower             -> contamination_not_excluded
     (unknown sources can only raise the budget, so this holds regardless of coverage)
  4. Observation above the valid budget, and any
     category unknown / ambiguous / scenario-only     -> insufficient_information
  5. Otherwise                                        -> exceeds_quantified_budget
"""
from __future__ import annotations

import math

from . import budget, controls, coverage
from .schemas import STATUS_LABELS, AuditResult, ExperimentRecord, Issue

ENGINE_VERSION = "0.3.1"


def _bounds(contribs, roles):
    sel = [c for c in contribs if c.role in roles]
    return {"lower": sum(c.lower_nmol for c in sel), "upper": sum(c.upper_nmol for c in sel)}


def _non_finite(obj, path: str = "") -> list[str]:
    """Paths of NaN / infinite numbers anywhere in a dumped record."""
    if isinstance(obj, float):
        return [] if math.isfinite(obj) else [path or "value"]
    if isinstance(obj, dict):
        return [p for k, v in obj.items() if k != "expected" for p in _non_finite(v, f"{path}.{k}" if path else k)]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in _non_finite(v, f"{path}[{i}]")]
    return []


def run_audit(rec: ExperimentRecord) -> AuditResult:
    issues: list[Issue] = []
    assumptions: list[str] = list(rec.assumptions)
    # NaN compares False with everything, so it would silently fall through to "exceeds"; reject it first.
    for p in _non_finite(rec.model_dump()):
        issues.append(Issue(code="non_finite_value", severity="error", field=p,
                            message=f"{p} is not a finite number (NaN or infinity); enter a real value or leave it empty."))

    obs = budget.observed_bounds(rec, issues)
    contribs, gas_unmeasured = [], []
    for g in rec.gas_sources:
        c, unknown_reason = budget.gas_contribution(g, rec, issues, assumptions)
        if c is not None:
            contribs.append(c)
        elif unknown_reason:
            gas_unmeasured.append(unknown_reason)
    for ctl in rec.controls:
        c = budget.control_contribution(ctl, rec, issues, assumptions)
        if c is not None:
            contribs.append(c)

    cov, unq = coverage.resolve(rec, contribs, gas_unmeasured, issues)
    is_complete = coverage.complete(cov)
    valid = _bounds(contribs, {"valid_bound"})
    with_scen = _bounds(contribs, {"valid_bound", "scenario"})
    errors = [i for i in issues if i.severity == "error"]

    explanation: list[str] = []
    observed = None
    allocation = None
    if obs is not None:
        observed = {"lower": obs[0], "upper": obs[1]}

    if errors:
        status = "input_inconsistency"
        headline = "The inputs cannot be compared on a single valid basis; fix the listed issues first."
        explanation += [f"{e.message}" for e in errors]
        explanation.append("This concerns the entered record, not the validity of the underlying study.")
    elif observed is None:
        status = "insufficient_information"
        headline = "No observed product amount was entered, so no comparison is possible."
        explanation.append("The contamination budget below is shown for reference only.")
    elif valid["upper"] >= observed["lower"]:
        status = "contamination_not_excluded"
        headline = ("Under these assumptions, the measured ammonia can be accounted for by the allowed contamination "
                    "budget. This does not demonstrate that contamination actually caused it.")
        explanation.append(f"Valid contamination upper bound {valid['upper']:.4g} nmol N ≥ observed lower bound "
                           f"{observed['lower']:.4g} nmol N.")
        if not is_complete:
            explanation.append("Coverage is partial, but unquantified sources could only enlarge the budget, so this "
                               "conclusion does not depend on them.")
        allocation = budget.feasible_allocation(contribs, observed["lower"], observed["upper"])
    elif not is_complete:
        status = "insufficient_information"
        missing = [k for k, v in cov.items() if v not in ("quantified", "not_applicable", "specified_synthetic")]
        headline = ("The observation is above the quantified part of the budget, but important sources are not "
                    f"bounded ({', '.join(missing)}); no conclusion about the nitrogen source follows.")
        explanation.append(f"Observed lower bound {observed['lower']:.4g} nmol N > valid partial budget upper bound "
                           f"{valid['upper']:.4g} nmol N.")
        explanation.append("A gap above quantified contamination does not establish that all other sources have been excluded.")
        if with_scen["upper"] != valid["upper"]:
            rel = "≥" if with_scen["upper"] >= observed["lower"] else "<"
            explanation.append(f"Including the what-if (non-interchangeable) measurements gives an upper bound of "
                               f"{with_scen['upper']:.4g} nmol N ({rel} observed lower bound); this is not a valid bound.")
    else:
        status = "exceeds_quantified_budget"
        headline = ("The measured ammonia exceeds the quantified contamination budget. Attribution still depends on "
                    "the other relevant sources and controls; this does not prove nitrogen reduction.")
        explanation.append(f"Observed lower bound {observed['lower']:.4g} nmol N > budget upper bound "
                           f"{valid['upper']:.4g} nmol N (margin ×{observed['lower']/valid['upper']:.3g})."
                           if valid["upper"] > 0 else
                           f"Observed lower bound {observed['lower']:.4g} nmol N > budget upper bound 0 nmol N.")
        if any(v == "specified_synthetic" for v in cov.values()):
            explanation.append("Some categories are fully specified only because this is a synthetic record.")

    evidence = [rec.observed_product.evidence] if rec.observed_product else []
    evidence += [g.impurity.evidence for g in rec.gas_sources if g.impurity] + [g.evidence for g in rec.gas_sources]
    evidence += [c.measurement.evidence for c in rec.controls] + [c.evidence for c in rec.controls]
    hypothetical = any(e.provenance == "hypothetical" for e in evidence)
    if hypothetical:
        explanation.append("This result includes hypothetical measurements. It shows what they would imply; it is "
                           "not based on evidence that exists.")

    if rec.isotope.performed:
        explanation.append("Isotope evidence is recorded as a control check and does not change the contamination "
                           "budget; an isotope signal alone does not exclude labeled impurities [S2].")

    recs = controls.recommend(rec, cov, contribs, issues)
    if is_complete:
        label = "Budget covers all modeled source categories (under stated assumptions)"
    else:
        label = "Partial budget — incomplete source coverage"

    return AuditResult(
        case_id=rec.case_id, status=status, status_label=STATUS_LABELS[status], headline=headline,
        explanation=explanation, coverage=cov, budget_complete=is_complete, budget_label=label,
        observed_nmol=observed, budget_valid_nmol=valid, budget_with_scenarios_nmol=with_scen,
        contributions=contribs, unquantified=unq, issues=issues, feasible_allocation=allocation,
        recommendations=recs, observed_trail=obs[2] if obs else [],
        assumptions=list(dict.fromkeys(assumptions)), hypothetical=hypothetical, engine_version=ENGINE_VERSION,
    )
