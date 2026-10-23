"""Source-coverage model: which contamination categories are bounded, and by what.

A matched whole-system blank may already contain gas, electrolyte and
component contributions. When it overlaps other valid measurements of the same
category, the record must select a coverage model; otherwise the overlapping
sources are not aggregated and the budget is reported as partial.
"""
from __future__ import annotations

from .schemas import Contribution, ExperimentRecord, Issue, UnquantifiedSource

CATEGORIES = ("gas", "electrolyte", "components")

UNKNOWN_REASONS = {
    "gas": "No measured upper bound on reactive-nitrogen impurities in the gas delivered to the cell.",
    "electrolyte": "No matched electrolyte or whole-system background measurement.",
    "components": "No matched measurement of release from nitrogen-containing or contaminated components.",
}


def _cats(c: Contribution) -> set[str]:
    return set(c.category.split("+"))


def resolve(rec: ExperimentRecord, contribs: list[Contribution], gas_unmeasured: list[str],
            issues: list[Issue]) -> tuple[dict[str, str], list[UnquantifiedSource]]:
    """Mutates contribution roles for overlap handling; returns coverage map and unquantified list."""
    system_blanks = [c for c in contribs if c.role == "valid_bound" and c.source_id in
                     {ctl.id for ctl in rec.controls if ctl.kind == "system_blank"}]
    others = [c for c in contribs if c.role == "valid_bound" and c not in system_blanks]
    ambiguous: set[str] = set()

    if len(system_blanks) > 1:
        # Several matched whole-system blanks are alternatives, not additive sources.
        keep = max(system_blanks, key=lambda c: c.upper_nmol)
        for c in system_blanks:
            if c is not keep:
                c.role, c.included = "excluded_overlap", False
                c.reason = f"Alternative whole-system blank; the largest upper bound ({keep.label}) is used instead of summing."
        issues.append(Issue(code="multiple_system_blanks", severity="warning",
                            message="Several matched system blanks were entered; only the most conservative one is used."))
        system_blanks = [keep]

    for sb in system_blanks:
        overlap = [o for o in others if _cats(o) & _cats(sb)]
        if not overlap:
            continue
        model = rec.coverage_model
        if model == "system_blank":
            for o in overlap:
                o.role, o.included = "excluded_overlap", False
                o.reason = f"Already contained in the selected whole-system blank ({sb.label}); not added again."
        elif model == "sum_components":
            sb.role, sb.included = "excluded_overlap", False
            sb.reason = "Coverage model 'sum_components' selected; the whole-system blank is shown but not added."
        else:
            shared = set().union(*[_cats(o) & _cats(sb) for o in overlap])
            # Every category the set-aside blank covered is now ambiguous, not only the shared ones.
            ambiguous |= _cats(sb) | set().union(*[_cats(o) for o in overlap])
            for c in [sb, *overlap]:
                c.role, c.included = "excluded_overlap", False
                c.reason = ("Overlaps another measurement of the same source category and no coverage model was "
                            "selected; not aggregated.")
            issues.append(Issue(code="overlap_unresolved", severity="warning", field="coverage_model",
                                message=(f"{sb.label} overlaps {', '.join(o.label for o in overlap)} for "
                                         f"{', '.join(sorted(shared))}. Select a coverage model to aggregate them.")))

    coverage: dict[str, str] = {}
    unq: list[UnquantifiedSource] = []
    for cat in CATEGORIES:
        valid = [c for c in contribs if c.included and cat in _cats(c)]
        scen = [c for c in contribs if c.role == "scenario" and cat in _cats(c)]
        declared = rec.declared_coverage.get(cat)
        gas_states = gas_source_states(rec, valid) if cat == "gas" and valid else {}
        if cat in ambiguous and not valid:
            coverage[cat] = "ambiguous_overlap"
            unq.append(UnquantifiedSource(category=cat, reason="Overlapping measurements without a selected coverage model."))
        elif "unbounded" in gas_states.values():
            # One bounded feed says nothing about another: every feed delivered to the cell must be bounded.
            coverage[cat] = "partial_feeds"
            missing = [g.label for g in rec.gas_sources if gas_states.get(g.id) == "unbounded"]
            unq.append(UnquantifiedSource(category=cat, reason=(
                f"No valid delivered-gas impurity bound for {', '.join(missing)}; the bounded feed(s) do not cover it.")))
            if "partial_species" in gas_states.values():
                unq.append(UnquantifiedSource(category=cat, reason=_partial_species_reason(rec, gas_states)))
        elif "partial_species" in gas_states.values():
            coverage[cat] = "partial_species"
            unq.append(UnquantifiedSource(category=cat, reason=_partial_species_reason(rec, gas_states)))
        elif valid:
            coverage[cat] = "quantified"
        elif declared == "specified_synthetic":
            if rec.record_type == "synthetic":
                coverage[cat] = "specified_synthetic"
            else:
                issues.append(Issue(code="synthetic_coverage_on_real_record", severity="error",
                                    field=f"declared_coverage.{cat}",
                                    message=f"'{cat}' is declared synthetic-specified on a non-synthetic record."))
                coverage[cat] = "unknown"
        elif declared == "not_applicable":
            coverage[cat] = "not_applicable"
            issues.append(Issue(code="declared_not_applicable", severity="info", field=f"declared_coverage.{cat}",
                                message=f"'{cat}' declared not applicable by the user; this is an assumption, not a measurement."))
        elif scen:
            coverage[cat] = "scenario_only"
            unq.append(UnquantifiedSource(category=cat, reason="Only non-interchangeable (what-if) measurements are available: "
                                          + "; ".join(c.label for c in scen)))
        else:
            coverage[cat] = "unknown"
            reason = UNKNOWN_REASONS[cat]
            if cat == "gas" and gas_unmeasured:
                reason += " Details: " + ", ".join(gas_unmeasured) + "."
            unq.append(UnquantifiedSource(category=cat, reason=reason))
    return coverage, unq


def gas_source_states(rec: ExperimentRecord, valid: list[Contribution]) -> dict[str, str]:
    """Per-feed gas coverage: 'bounded', 'partial_species' or 'unbounded'. Empty when a whole-system
    blank that covers gas is counted, because it bounds every feed and species together."""
    if any(c.source_id not in {g.id for g in rec.gas_sources} for c in valid):
        return {}
    ids = {c.source_id for c in valid}
    return {g.id: ("unbounded" if g.id not in ids else
                   "bounded" if g.impurity is not None and g.impurity.covers_all_reactive_nitrogen else
                   "partial_species")
            for g in rec.gas_sources}


def _partial_species_reason(rec: ExperimentRecord, states: dict[str, str]) -> str:
    measured = sorted({g.impurity.species for g in rec.gas_sources if g.impurity and states.get(g.id) == "partial_species"})
    return (f"Only {', '.join(measured)} bounded in the delivered gas; other reactive-nitrogen species "
            "(e.g. NOx, N2O, NH3) are not bounded.")


def complete(coverage: dict[str, str]) -> bool:
    return all(v in ("quantified", "not_applicable", "specified_synthetic") for v in coverage.values())
