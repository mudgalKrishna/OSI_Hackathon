"""JSON, CSV and Markdown exports that reproduce the displayed audit."""
from __future__ import annotations

import csv
import io
import json

from .controls import sources
from .schemas import AuditResult, ExperimentRecord

DISCLAIMER = ("NitroTrace is a software prototype. Results describe what the entered measurements and stated "
              "assumptions permit; they are not a laboratory validation or a judgement of the original study.")


def to_json(rec: ExperimentRecord, res: AuditResult) -> str:
    return json.dumps({"record": rec.model_dump(mode="json"), "audit": res.model_dump(mode="json"),
                       "disclaimer": DISCLAIMER}, indent=2, ensure_ascii=False)


def to_csv(rec: ExperimentRecord, res: AuditResult) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["case_id", "row_type", "source_id", "label", "category", "role", "included",
                "lower_nmol_N", "upper_nmol_N", "note"])
    if res.observed_nmol:
        w.writerow([rec.case_id, "observed", "observed_product", "Observed product", "", "", "",
                    f"{res.observed_nmol['lower']:.6g}", f"{res.observed_nmol['upper']:.6g}", ""])
    for c in res.contributions:
        w.writerow([rec.case_id, "contribution", c.source_id, c.label, c.category, c.role, c.included,
                    f"{c.lower_nmol:.6g}", f"{c.upper_nmol:.6g}", c.reason or ""])
    for u in res.unquantified:
        w.writerow([rec.case_id, "unquantified", "", "", u.category, "unknown", False, "", "", u.reason])
    w.writerow([rec.case_id, "budget_valid", "", res.budget_label, "", "", "",
                f"{res.budget_valid_nmol['lower']:.6g}", f"{res.budget_valid_nmol['upper']:.6g}", ""])
    w.writerow([rec.case_id, "status", "", res.status_label, "", "", "", "", "", res.headline])
    return buf.getvalue()


def _fmt(x: float) -> str:
    return f"{x:.4g}"


NOT_SHOWN = [
    "Whether nitrogen reduction occurred. An 'exceeds' verdict means only that the measured ammonia is larger than "
    "the contamination that was bounded; isotope-labelled and other controls are still required.",
    "Whether the original study is right or wrong. The audit describes what the entered evidence permits.",
    "Anything about sources that were never measured, other than that they remain unbounded.",
]


def _review_line(rec: ExperimentRecord) -> str:
    r = rec.review
    if r is None or r.status == "internal":
        return ("**Internal review only.** The inputs and expected outcome have been checked by the NitroTrace "
                "authors against the cited source, but not yet by an independent domain researcher. "
                "Use `nitrotrace review-pack` to produce a checklist for one.")
    who = r.reviewer or "an independent reviewer"
    return f"**Independently reviewed** by {who}{f' on {r.date}' if r.date else ''}: {r.outcome or 'no outcome recorded'}."


def _value(m) -> str:
    sp = f" {m.species}" if getattr(m, "species", None) else ""
    if getattr(m, "nondetect", False):
        return f"not detected (< {m.upper:g} {m.unit}{sp})"
    lo, hi = m.lower, m.upper
    if lo is not None and hi is not None and lo != hi:
        return f"{lo:g}–{hi:g} {m.unit}{sp}"
    v = hi if hi is not None else lo
    return f"{v:g} {m.unit}{sp}" if v is not None else "not stated"


def _ev_row(name: str, value: str, ev) -> str:
    q = (ev.quote or "").replace("|", "/")
    loc = f"[{ev.source_id}] {ev.location or ''}".strip() if ev.source_id else (ev.location or "—")
    return (f"| {name} | {value} | {loc} | {('“' + q + '”') if q else '—'} | {ev.provenance}"
            f"{' ✓' if ev.confirmed else ''} |")


def to_casefile(rec: ExperimentRecord, res: AuditResult, req: dict | None) -> str:
    """A complete, source-linked account of one case: reported values, what was counted or set aside,
    the calculation, the qualified conclusion, what would close the gap, and the review status."""
    src = sources()
    L = [f"# Case file — {rec.title}", "",
         f"> **Question.** Given this ammonia measurement and these controls, what contamination remains unbounded, "
         f"and how stringent would additional measurements have to be to close that gap?", "",
         f"- **Case:** `{rec.case_id}` ({rec.record_type})" + (" — **contains hypothetical inputs**" if res.hypothetical else ""),
         f"- **Verdict:** {res.status_label}",
         f"- **Review status:** {_review_line(rec)}",
         f"- **Engine:** v{res.engine_version}", ""]
    if rec.description:
        L += [rec.description, ""]
    L += ["## 1. Sources", ""]
    L += [f"- [{s}] {src[s]['citation']} {src[s].get('open_full_text') or src[s].get('url', '')}" for s in rec.sources if s in src] or ["- None (synthetic record)."]
    L += ["", "## 2. Measurements as reported", "",
          "| Quantity | Value as reported | Where | Supporting quote | Provenance |", "|---|---|---|---|---|"]
    if rec.observed_product:
        L.append(_ev_row("Observed ammonia", _value(rec.observed_product), rec.observed_product.evidence))
    for g in rec.gas_sources:
        if g.impurity:
            L.append(_ev_row(f"{g.label} — {g.impurity.species} ({g.impurity.measured_where.replace('_', ' ')})",
                             _value(g.impurity), g.impurity.evidence))
        else:
            L.append(_ev_row(f"{g.label} — impurity", "not measured", g.evidence))
        if g.flow:
            L.append(_ev_row(f"{g.label} — flow", f"{g.flow.value:g} {g.flow.unit} ({g.flow.volume_basis})", g.flow.evidence))
    for c in rec.controls:
        L.append(_ev_row(f"{c.label} ({c.kind.replace('_', ' ')}, {c.conditions_match})", _value(c.measurement),
                         c.measurement.evidence))
    L += ["", "## 3. Calculation (all amounts in nmol of ammonia-equivalent N)", ""]
    for step in res.observed_trail + [s for c in res.contributions for s in c.trail]:
        L.append(f"- **{step.label}**" + (f" — `{step.formula}`" if step.formula else ""))
        L += [f"  - {k}: {v}" for k, v in step.values.items()]
        if step.result:
            L.append(f"  - → {step.result}")
    L += ["", "## 4. Counted and set aside", "", "| Source | Role | Lower | Upper | Why |", "|---|---|---:|---:|---|"]
    for c in res.contributions:
        L.append(f"| {c.label} | {c.role} | {_fmt(c.lower_nmol)} | {_fmt(c.upper_nmol)} | {c.reason or 'matched, valid bound'} |")
    if not res.contributions:
        L.append("| — | — | | | No source could be bounded. |")
    L += ["", f"**Valid contamination budget:** [{_fmt(res.budget_valid_nmol['lower'])}, "
               f"{_fmt(res.budget_valid_nmol['upper'])}] nmol N ({res.budget_label})."]
    if res.observed_nmol:
        L.append(f"**Observed:** [{_fmt(res.observed_nmol['lower'])}, {_fmt(res.observed_nmol['upper'])}] nmol N.")
    L += ["", "## 5. Still unbounded (never treated as zero)", ""]
    L += [f"- **{u.category}:** {u.reason}" for u in res.unquantified] or ["- None."]
    L += ["", "## 6. Assumptions", ""] + ([f"- {a}" for a in res.assumptions] or ["- None."])
    L += ["", "## 7. Qualified conclusion", "", f"**{res.status_label}.** {res.headline}", ""]
    L += [f"- {e}" for e in res.explanation]
    if req:
        L += ["", "## 8. What would close the gap", "",
              f"- Headroom for unbounded sources: **{req['headroom_nmol']:+.4g} nmol N** "
              f"(observed lower bound − valid budget upper bound).",
              f"- Contamination floor of this setup: **{req['floor_nmol']:.4g} nmol N** — an observation "
              "must be strictly above this before today's controls could call it 'above budget'."]
        for u in req["unknown"]:
            for s in u["specs"]:
                L.append(f"- **{u['category']}:** {s['text']}")
                L += [f"  - Assumes: {a}" for a in s.get("assumptions", [])]
        for s in req["shrink"]:
            L.append(f"- **Tighten {s['source']}:** {s['text']}")
            L += [f"  - Assumes: {a}" for a in s.get("assumptions", [])]
        L += [f"- {n}" for n in req["notes"]]
        L += [f"- _{c}_" for c in req["caveats"]]
    L += ["", "## 9. What this case file does not show", ""] + [f"- {x}" for x in NOT_SHOWN]
    L += ["", "---", "", f"_{DISCLAIMER}_", ""]
    return "\n".join(L)


def to_review_pack(rec: ExperimentRecord, res: AuditResult, req: dict | None) -> str:
    """A checklist for an independent domain researcher to confirm or dispute each input and the outcome."""
    L = [f"# Independent review — {rec.title}", "",
         "Thank you for reviewing. Please check each item against the cited source and your own judgement, tick one "
         "box, and add a note where you disagree. Return this file; the outcome is recorded in the record's "
         "`review` field and shown in every report.", "",
         f"- Case: `{rec.case_id}` · NitroTrace v{res.engine_version}",
         "- Reviewer (role / affiliation): ______________________   Date: __________", "",
         "## A. Inputs", ""]
    items = []
    if rec.observed_product:
        items.append(("Observed ammonia", _value(rec.observed_product), rec.observed_product.evidence))
    for g in rec.gas_sources:
        items.append((f"{g.label} — impurity", _value(g.impurity) if g.impurity else "not measured",
                      g.impurity.evidence if g.impurity else g.evidence))
        if g.flow:
            items.append((f"{g.label} — flow", f"{g.flow.value:g} {g.flow.unit}", g.flow.evidence))
    for c in rec.controls:
        items.append((f"{c.label} (conditions: {c.conditions_match})", _value(c.measurement), c.measurement.evidence))
    for i, (name, val, ev) in enumerate(items, 1):
        L += [f"**A{i}. {name}:** {val}",
              f"  Source: [{ev.source_id or '-'}] {ev.location or ''}" + (f" — “{ev.quote}”" if ev.quote else ""),
              "  - [ ] correct   - [ ] incorrect   - [ ] cannot tell      Note: ______________________", ""]
    L += ["## B. Judgements NitroTrace made", ""]
    for i, c in enumerate(res.contributions, 1):
        L += [f"**B{i}. {c.label}** was treated as **{c.role.replace('_', ' ')}**"
              + (f" because: {c.reason}" if c.reason else " (matched, valid bound)."),
              "  - [ ] agree   - [ ] disagree      Note: ______________________", ""]
    for j, a in enumerate(res.assumptions, len(res.contributions) + 1):
        L += [f"**B{j}. Assumption:** {a}", "  - [ ] reasonable   - [ ] not reasonable      Note: ______________________", ""]
    L += ["## C. Outcome", "",
          f"NitroTrace's verdict: **{res.status_label}** — {res.headline}", "",
          "- [ ] I agree with this verdict given the evidence", "- [ ] I would reach a different verdict: ______________________", ""]
    if req and (req["unknown"] or req["shrink"]):
        L += ["NitroTrace's 'what would close the gap' thresholds:", ""]
        L += [f"- {s['text']}" for u in req["unknown"] for s in u["specs"]] + [f"- {s['text']}" for s in req["shrink"]]
        L += ["", "- [ ] these are sensible next measurements   - [ ] not sensible: ______________________", ""]
    L += ["## D. Anything NitroTrace missed?", "", "______________________________________________", ""]
    return "\n".join(L)


def to_markdown(rec: ExperimentRecord, res: AuditResult) -> str:
    src = sources()
    L = [f"# NitroTrace audit — {rec.title}", "",
         f"- **Case:** `{rec.case_id}` ({rec.record_type})",
         f"- **Status:** {res.status_label}",
         f"- **Coverage:** " + ", ".join(f"{k}: {v}" for k, v in res.coverage.items()),
         f"- **Budget:** {res.budget_label}",
         f"- **Engine:** v{res.engine_version}", "",
         f"> {res.headline}", ""]
    if rec.description:
        L += [rec.description, ""]
    L += ["## Comparison (nmol N)", "", "| Quantity | Lower | Upper |", "|---|---:|---:|"]
    if res.observed_nmol:
        L.append(f"| Observed product | {_fmt(res.observed_nmol['lower'])} | {_fmt(res.observed_nmol['upper'])} |")
    L.append(f"| Valid contamination budget | {_fmt(res.budget_valid_nmol['lower'])} | {_fmt(res.budget_valid_nmol['upper'])} |")
    if res.budget_with_scenarios_nmol != res.budget_valid_nmol:
        L.append(f"| Budget incl. what-if measurements (not valid) | {_fmt(res.budget_with_scenarios_nmol['lower'])} | "
                 f"{_fmt(res.budget_with_scenarios_nmol['upper'])} |")
    L += ["", "## Explanation", ""] + [f"- {e}" for e in res.explanation] + [""]
    L += ["## Contributions", "", "| Source | Category | Role | Lower | Upper | Note |", "|---|---|---|---:|---:|---|"]
    for c in res.contributions:
        L.append(f"| {c.label} | {c.category} | {c.role} | {_fmt(c.lower_nmol)} | {_fmt(c.upper_nmol)} | {c.reason or ''} |")
    L += ["", "## Unquantified sources", ""]
    L += [f"- **{u.category}:** {u.reason}" for u in res.unquantified] or ["- None"]
    if res.feasible_allocation:
        fa = res.feasible_allocation
        L += ["", "## Illustrative feasible explanation", "", f"_{fa['label']}_", ""]
        L += [f"- {k}: {_fmt(v)} nmol N" for k, v in fa["allocation_nmol"].items()]
    L += ["", "## Calculation trail", ""]
    for step in res.observed_trail + [s for c in res.contributions for s in c.trail]:
        L.append(f"- **{step.label}**" + (f" — `{step.formula}`" if step.formula else ""))
        L += [f"  - {k}: {v}" for k, v in step.values.items()]
        if step.result:
            L.append(f"  - → {step.result}")
    L += ["", "## Assumptions", ""] + ([f"- {a}" for a in res.assumptions] or ["- None"])
    if res.issues:
        L += ["", "## Issues", ""] + [f"- [{i.severity}] `{i.code}` {i.message}" for i in res.issues]
    L += ["", "## Suggested next controls (protocol-based)", ""]
    for r in res.recommendations:
        L.append(f"### {r.title}")
        L += [f"_Trigger:_ {r.trigger}", "", r.text, ""]
        for ref in r.references:
            L.append(f"> “{ref['quote']}” — [{ref['source_id']}] {ref['location']}")
        L.append("")
    L += ["## Evidence", ""]
    ev_items = []
    if rec.observed_product:
        ev_items.append(("Observed product", rec.observed_product.evidence))
    for g in rec.gas_sources:
        ev_items.append((g.label, g.evidence))
        if g.impurity:
            ev_items.append((f"{g.label} — impurity", g.impurity.evidence))
        if g.flow:
            ev_items.append((f"{g.label} — flow", g.flow.evidence))
    for c in rec.controls:
        ev_items.append((c.label, c.measurement.evidence))
    for name, ev in ev_items:
        if not (ev.source_id or ev.quote or ev.location or ev.note):
            continue
        conf = "confirmed" if ev.confirmed else "unconfirmed"
        L.append(f"- **{name}** ({ev.provenance}, {conf}): [{ev.source_id or '-'}] {ev.location or ''} "
                 + (f"“{ev.quote}”" if ev.quote else "") + (f" — {ev.note}" if ev.note else ""))
    used = sorted({s for s in rec.sources} | {r['source_id'] for x in res.recommendations for r in x.references})
    L += ["", "## References", ""] + [f"- [{s}] {src[s]['citation']} {src[s].get('url','')}" for s in used if s in src]
    L += ["", "---", "", f"_{DISCLAIMER}_", ""]
    return "\n".join(L)
