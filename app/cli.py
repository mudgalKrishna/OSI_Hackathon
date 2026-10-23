"""NitroTrace command-line interface.

    nitrotrace list
    nitrotrace audit CASE [--scale K] [--format text|json|md]
    nitrotrace settle CASE
    nitrotrace whatif CASE [--gas PPB [--source ID]] [--blank UG_PER_L | --blank-nmol NMOL] [--components NMOL]
    nitrotrace casefile CASE [-o FILE]
    nitrotrace review-pack CASE [-o FILE]
    nitrotrace plan --rate R --flow SCCM --ppb PPB [...]
    nitrotrace stress --rate R --unit UNIT --flow SCCM [...]
    nitrotrace bench [--json]
    nitrotrace extract FILE [--llm]

CASE is a path to a record JSON file, a case id (e.g. bragulla2026_pristine_gde_ocv) or a unique part of one (e.g. ocv).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from . import reports
from .audit import run_audit
from .inversion import evidence_requirements
from .planner import RATE_UNIT_ALIASES, RATE_UNITS, REF_CONDITIONS, planner, stress_row
from .schemas import ExperimentRecord
from .validation import load_records, run_benchmark
from .whatif import compare, suggest

# ------------------------------------------------------------------ output ----
_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
STATUS_COLOR = {"contamination_not_excluded": "31", "exceeds_quantified_budget": "32",
                "insufficient_information": "33", "input_inconsistency": "35"}


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(t: str) -> str:
    return c(t, "1")


def dim(t: str) -> str:
    return c(t, "2")


def rule(title: str = "") -> None:
    print("\n" + bold(title) + "\n" + dim("─" * min(88, max(len(title), 40))))


def num(x: float | None) -> str:
    if x is None:
        return "—"
    if x == 0:
        return "0"
    return f"{x:.3g}" if 1e-3 <= abs(x) < 1e5 else f"{x:.2e}"


def wrap(text: str, indent: str = "  ", width: int = 88) -> str:
    words, lines, cur = text.split(), [], indent
    for w in words:
        if len(cur) + len(w) + 1 > width and cur.strip():
            lines.append(cur)
            cur = indent
        cur += ("" if cur == indent else " ") + w
    lines.append(cur)
    return "\n".join(lines)


def ruler(rows: list[tuple[str, float, float, str]], width: int = 46) -> None:
    """ASCII log-scale interval plot. rows: (label, lower, upper, kind)."""
    pos = [v for _, lo, hi, _ in rows for v in (lo, hi) if v > 0]
    if not pos:
        return
    lo_d, hi_d = math.log10(min(pos) / 3), math.log10(max(pos) * 3)
    if hi_d - lo_d < 2:
        mid = (hi_d + lo_d) / 2
        lo_d, hi_d = mid - 1, mid + 1
    x = lambda v: int(round((math.log10(max(v, 10 ** lo_d)) - lo_d) / (hi_d - lo_d) * (width - 1)))
    mark = {"obs": ("█", "32"), "valid_bound": ("■", "0"), "scenario": ("▒", "33"), "budget": ("█", "36"),
            "budget_scen": ("▒", "33")}
    for label, lo, hi, kind in rows:
        ch, col = mark.get(kind, ("┄", "2"))
        line = [" "] * width
        if hi > 0:
            a, b = (x(lo) if lo > 0 else 0), x(hi)
            for i in range(a, b + 1):
                line[i] = ch
            if lo <= 0:
                line[0] = "◁"
        name = (label[:30] + "…") if len(label) > 31 else label
        print(f"  {name:<32}{c(''.join(line), col)}  {dim(f'[{num(lo)}, {num(hi)}]')}")
    ticks = [10 ** e for e in range(math.ceil(lo_d), math.floor(hi_d) + 1)]
    axis = [" "] * (width + 8)
    for t in ticks:
        s = num(t)
        i = x(t)
        for j, ch in enumerate(s):
            if i + j < len(axis):
                axis[i + j] = ch
    print(f"  {'nmol N (log scale)':<32}{dim(''.join(axis))}")


# ------------------------------------------------------------------ records ---
def resolve(case: str) -> tuple[str, dict]:
    p = Path(case)
    if p.suffix == ".json" and p.exists():
        return str(p), json.loads(p.read_text(encoding="utf-8"))
    recs = load_records()
    exact = [k for k, d in recs.items() if d["case_id"] == case or Path(k).stem == case]
    hits = (exact or [k for k, d in recs.items() if d["case_id"].startswith(case)]
            or [k for k, d in recs.items() if case.lower() in d["case_id"].lower()])
    if len(hits) == 1:
        return hits[0], recs[hits[0]]
    if not hits:
        sys.exit(f"No record matches '{case}'. Run `nitrotrace list`.")
    sys.exit(f"'{case}' is ambiguous: " + ", ".join(recs[h]["case_id"] for h in hits))


def audited(case: str, scale: float = 1.0):
    _, d = resolve(case)
    if scale != 1.0:
        op = d.get("observed_product")
        if op and not op.get("nondetect"):
            for b in ("lower", "upper"):
                if op.get(b) is not None:
                    op[b] *= scale
    rec = ExperimentRecord(**d)
    res = run_audit(rec)
    return d, rec, res


def print_verdict(res) -> None:
    col = STATUS_COLOR[res.status]
    tag = " (HYPOTHETICAL INPUTS)" if res.hypothetical else ""
    print("\n" + c(bold(f"■ {res.status_label.upper()}{tag}"), col))
    print(wrap(res.headline))


# ------------------------------------------------------------------ commands --
def cmd_list(a) -> None:
    for group, prefix in (("Published records", "cases/"), ("Synthetic benchmark", "synthetic/")):
        rule(group)
        for k, d in load_records().items():
            if k.startswith(prefix):
                st = run_audit(ExperimentRecord(**d)).status
                print(f"  {c('■', STATUS_COLOR[st])} {d['case_id']:<40} {dim(d['title'][:60])}")


def cmd_audit(a) -> None:
    d, rec, res = audited(a.case, a.scale)
    if a.format == "json":
        print(reports.to_json(rec, res))
        return
    if a.format == "md":
        print(reports.to_markdown(rec, res))
        return
    print(bold(rec.title) + dim(f"  [{rec.case_id} · {rec.record_type}]"))
    print_verdict(res)
    rule("Comparison")
    rows = []
    if res.observed_nmol:
        rows.append(("Measured ammonia", res.observed_nmol["lower"], res.observed_nmol["upper"], "obs"))
    rows += [(f"{x.label} [{x.role}]", x.lower_nmol, x.upper_nmol, x.role) for x in res.contributions]
    rows.append(("VALID BUDGET", res.budget_valid_nmol["lower"], res.budget_valid_nmol["upper"], "budget"))
    if res.budget_with_scenarios_nmol != res.budget_valid_nmol:
        rows.append(("incl. what-ifs (not valid)", res.budget_with_scenarios_nmol["lower"],
                     res.budget_with_scenarios_nmol["upper"], "budget_scen"))
    ruler(rows)
    rule("Coverage")
    for k, v in res.coverage.items():
        ok = v in ("quantified", "not_applicable", "specified_synthetic")
        print(f"  {k:<12} {c(v.replace('_', ' '), '32' if ok else '33')}")
    if res.unquantified:
        rule("Never treated as zero")
        for u in res.unquantified:
            print(wrap(f"{u.category}: {u.reason}"))
    rule("Why")
    for e in res.explanation:
        print(wrap("• " + e))
    errs = [i for i in res.issues if i.severity != "info"]
    if errs:
        rule("Issues")
        for i in errs:
            print(wrap(f"[{i.severity}] {i.code}: {i.message}"))
    if res.recommendations:
        rule("Next controls (protocol-based, each cites its source)")
        for r in res.recommendations:
            print(f"  • {r.title} " + dim(f"[{', '.join(dict.fromkeys(x['source_id'] for x in r.references))}]"))
    print(dim("\n  Next: `nitrotrace settle " + rec.case_id + "` · `nitrotrace casefile " + rec.case_id + "`"))


def cmd_settle(a) -> None:
    d, rec, res = audited(a.case)
    req = evidence_requirements(rec, res)
    print(bold(rec.title))
    print_verdict(res)
    if req is None:
        print(wrap("\nNothing to invert: the record needs an observed product and no input errors."))
        return
    rule("The gap (nmol N)")
    h = req["headroom_nmol"]
    print(f"  measured lower bound   {num(req['observed_lower_nmol'])}")
    print(f"  valid budget upper     {num(req['valid_upper_nmol'])}")
    print(f"  headroom               {c(('+' if h > 0 else '') + num(h), '32' if h > 0 else '31')}"
          + dim("   room left for every source not yet measured"))
    print(f"  contamination floor    {num(req['floor_nmol'])}" + dim("   smallest observation today's controls could call 'above budget'"))
    rule("What would close the gap")
    i = 0
    for u in req["unknown"]:
        for s in u["specs"]:
            i += 1
            print(f"  {i:02d}  {bold(u['category'])} " + c(f"≤ {num(s['big'])} {s['big_unit']}", "36"))
            print(wrap(s["text"], "      "))
            for x in s.get("assumptions", []):
                print(dim(wrap("assumes: " + x, "      ")))
    for s in req["shrink"]:
        i += 1
        head = c(f"≤ {num(s['big'])} {s['big_unit']}", "36") if s["feasible"] else c("not possible alone", "31")
        print(f"  {i:02d}  {bold('tighten ' + s['source'])} {head}")
        print(wrap(s["text"], "      "))
    for n in req["notes"]:
        print(wrap("• " + n))
    for cv in req["caveats"]:
        print(dim(wrap("! " + cv)))
    flag = {"gas": "--gas", "electrolyte": "--blank", "electrolyte_nmol": "--blank-nmol", "components": "--components"}
    sug = suggest(rec, req)
    if sug:
        flags = " ".join(f"{flag[k]} {v:g}" for k, v, _ in sug)
        print(dim(f"\n  Try: `nitrotrace whatif {rec.case_id} {flags}`  (each at half its equal-split threshold)"))


def cmd_whatif(a) -> None:
    _, d = resolve(a.case)
    changes = [(k, v, a.source if k == "gas" else None) for k, v in
               (("gas", a.gas), ("electrolyte", a.blank), ("electrolyte_nmol", a.blank_nmol),
                ("components", a.components)) if v is not None]
    if not changes:
        sys.exit("Give at least one of --gas PPB, --blank UG_PER_L, --blank-nmol NMOL, --components NMOL.")
    try:
        r = compare(d, changes)
    except ValueError as e:
        sys.exit(f"Cannot add that measurement: {e}")
    unit = {"gas": "ppb total reactive N in delivered gas", "electrolyte": "µg/L NH4+ matched blank",
            "electrolyte_nmol": "nmol N matched blank", "components": "nmol N matched component release"}
    print(bold(d["title"]))
    for kind, value, _ in changes:
        print(c(f"  HYPOTHETICAL: {kind} measured ≤ {value:g} {unit[kind]}", "35"))
    rule("Before → after")
    b, f = r["before"], r["after"]
    print(f"  verdict        {c(b['status_label'], STATUS_COLOR[b['status']])}  →  {c(f['status_label'], STATUS_COLOR[f['status']])}")
    print(f"  valid budget   {num(b['valid_upper_nmol'])}  →  {num(f['valid_upper_nmol'])} nmol N")
    print(f"  observed       {num(f['observed_lower_nmol'])} nmol N (lower bound)")
    for k in b["coverage"]:
        if b["coverage"][k] != f["coverage"][k]:
            print(f"  {k:<14} {b['coverage'][k]}  →  {f['coverage'][k]}")
    for u in r["after"]["unquantified"]:
        print(c(wrap(f"still unbounded — {u['category']}: {u['reason']}"), "33"))
    print(dim("\n" + wrap(r["note"])))


def _write(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(text)


def cmd_casefile(a) -> None:
    d, rec, res = audited(a.case)
    _write(reports.to_casefile(rec, res, evidence_requirements(rec, res)), a.output)


def cmd_review(a) -> None:
    d, rec, res = audited(a.case)
    _write(reports.to_review_pack(rec, res, evidence_requirements(rec, res)), a.output)


def cmd_plan(a) -> None:
    p = planner(a.rate, a.area, a.flow, REF_CONDITIONS[a.ref], a.ppb, a.eta, a.hours, a.blank, a.volume, a.margin,
                fixed_volume_L=a.fixed_volume)
    mode = f"fixed gas volume {a.fixed_volume:g} L" if a.fixed_volume else f"continuous purge {a.flow:g} sccm ({a.ref})"
    print(bold("Design before you run") + dim(f"  [{mode}, {a.ppb:g} ppb N, η={a.eta:g}, {a.hours:g} h, margin ×{a.margin:g}]"))
    rule("Answer")
    if p["verdict"] == "never":
        print(c(wrap(f"Not under these assumptions. The feed gas supplies N {num(p['g'] / p['s'])}× as fast as a "
                     f"{a.rate:.2g} mol s⁻¹ cm⁻² catalyst would make it; with constant flow, purity and η the ratio "
                     "stays below the margin however long the run."), "31"))
    elif p["t_star_s"] / 3600 > a.hours:
        print(c(wrap(f"Not within {a.hours:g} h: the signal reaches ×{a.margin:g} the budget after "
                     f"{p['t_star_s'] / 3600:.3g} h."), "33"))
    else:
        print(c(wrap(f"Defensible after {p['t_star_s'] / 3600:.3g} h; at {a.hours:g} h the signal is "
                     f"{p['end_ratio']:.3g}× the contamination budget."), "32"))
    print(wrap(f"Smallest true rate this design can defend in {a.hours:g} h: {p['min_rate_mol_s_cm2']:.3g} mol s⁻¹ cm⁻²."))
    rule("Over the run (nmol N)")
    for i in range(0, len(p["t_s"]), max(1, len(p["t_s"]) // 6)):
        t = p["t_s"][i] / 3600
        print(f"  t = {t:5.2f} h   signal {num(p['signal'][i]):>8}   budget {num(p['contamination'][i]):>8}")
    for x in p["assumptions"]:
        print(dim(wrap("assumes: " + x)))


def cmd_stress(a) -> None:
    unit = RATE_UNIT_ALIASES.get(a.unit, a.unit)
    o = stress_row(a.rate, unit, a.area, a.loading, a.flow, REF_CONDITIONS[a.ref])
    ppb = o["equivalent_ppb"]
    print(bold("Claim stress-test") + dim(f"  [{a.rate:g} {unit}, {a.area:g} cm², {a.flow:g} sccm, {a.ref}]"))
    print(wrap(f"\nProduct rate {num(o['product_nmol_s'])} nmol N/s. Feed gas delivers {num(o['gas_mol_s'])} mol/s."))
    print("  Reactive N in the feed that alone supplies this rate (η = 1): " + bold(f"{num(ppb)} ppb"))
    if a.lod is not None:
        ok = ppb > a.lod
        msg = (f"Above your gas-analysis bound of {a.lod:g} ppb: a clean delivered-gas measurement at that sensitivity "
               "could rule out feed gas as the sole source (other sources still need bounding)." if ok else
               f"Below your gas-analysis bound of {a.lod:g} ppb: even a clean measurement at that sensitivity could not "
               "rule out feed gas as the sole source of this rate.")
        print(c(wrap(msg), "32" if ok else "31"))


def cmd_bench(a) -> None:
    rows, s = run_benchmark()
    if a.json:
        print(json.dumps({"summary": s, "rows": rows}, indent=2, default=str))
        return
    rule(f"Benchmark — {s['n_synthetic']} synthetic records with independent expected values")
    print(f"  NitroTrace          {bold(str(s['nitrotrace_correct']))}/{s['n_synthetic']} correct   "
          f"false 'exceeds': {s['false_exceeds_nitrotrace']}")
    print(f"  naive sum           {s['naive_correct']}/{s['n_synthetic']} correct   false 'exceeds': {s['false_exceeds_naive']}")
    print(f"  protocol checklist  {s['checklist_correct']}/{s['n_synthetic']} correct   false 'exceeds': {s['false_exceeds_checklist']}")
    print(f"  max relative error  {num(s['max_rel_error'])} over {s['n_numeric']} non-zero hand-computed references")
    rule("Machine-checked properties")
    for k, (p, n, fails) in s["properties"].items():
        print(f"  {c('✔', '32') if p == n else c('✘', '31')} {p:>3}/{n:<3} {k}" + (f"  FAILED: {', '.join(fails)}" if fails else ""))
    print(dim(f"\n  {s['elapsed_s'] * 1000:.0f} ms. Synthetic records test software and reasoning, not attribution in real labs."))
    if s["nitrotrace_correct"] != s["n_synthetic"] or any(p != n for p, n, _ in s["properties"].values()):
        sys.exit(1)


def cmd_extract(a) -> None:
    from . import extraction
    p = Path(a.file)
    if p.suffix.lower() == ".pdf":
        try:
            pages = extraction.pdf_text(p.read_bytes())
        except extraction.ExtractionError as e:
            sys.exit(str(e))
    else:
        pages = [{"page": 1, "text": p.read_text(encoding="utf-8", errors="replace")}]
    found = extraction.passages(pages)
    rule(f"{len(found)} relevant passages")
    for f in found:
        print(f"  p{f['page']} [{', '.join(f['topics'])}]")
        print(dim(wrap(f["text"][:400], "    ")))
        if f["regex_values"]:
            print("    " + c("; ".join(f"{k}: {', '.join(v)}" for k, v in f["regex_values"].items()), "36"))
    if a.llm:
        r = extraction.llm_candidates(found)
        rule("Claude candidates (unconfirmed; quotes verified verbatim)")
        if "error" in r:
            sys.exit(r["error"])
        for x in r["candidates"]:
            print(f"  {x['field']:<20} {x['value']} {x.get('unit') or ''}  " + dim(f"“{x['quote'][:90]}”"))
        print(dim(f"  {r['dropped_unsupported_quotes']} candidates dropped for non-verbatim quotes."))


# ------------------------------------------------------------------ parser ----
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nitrotrace", description=(
        "Given an ammonia measurement and the controls actually run, what contamination remains unbounded — "
        "and how stringent must additional measurements be to close that gap?"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list bundled published and synthetic records").set_defaults(fn=cmd_list)
    p = sub.add_parser("audit", help="audit a record")
    p.add_argument("case"); p.add_argument("--scale", type=float, default=1.0, help="what-if: multiply the observed amount")
    p.add_argument("--format", choices=["text", "json", "md"], default="text"); p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("settle", help="what would close the gap (evidence-requirement inversion)")
    p.add_argument("case"); p.set_defaults(fn=cmd_settle)
    p = sub.add_parser("whatif", help="add hypothetical measurements (any combination) and compare before/after")
    p.add_argument("case")
    p.add_argument("--gas", type=float, metavar="PPB", help="total reactive N in the delivered gas, ppb")
    p.add_argument("--blank", type=float, metavar="UG_PER_L", help="matched electrolyte blank, µg/L NH4+")
    p.add_argument("--blank-nmol", type=float, metavar="NMOL", help="matched electrolyte blank as an amount, nmol N")
    p.add_argument("--components", type=float, metavar="NMOL", help="matched component release, nmol N")
    p.add_argument("--source", help="gas source id (default: first)"); p.set_defaults(fn=cmd_whatif)
    for name, fn, hlp in (("casefile", cmd_casefile, "complete source-linked case file (Markdown)"),
                          ("review-pack", cmd_review, "checklist for an independent domain reviewer (Markdown)")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("case"); p.add_argument("-o", "--output"); p.set_defaults(fn=fn)
    p = sub.add_parser("plan", help="design before you run: signal vs. contamination budget")
    for arg, typ, dflt, hlp in (("--rate", float, 1e-11, "expected true rate, mol s-1 cm-2"),
                                ("--area", float, 1.0, "electrode area, cm2"), ("--flow", float, 20.0, "gas flow, sccm"),
                                ("--ppb", float, 100.0, "reactive N in delivered gas, ppb"),
                                ("--eta", float, 1.0, "capture fraction (1 = conservative)"),
                                ("--hours", float, 4.0, "run time, h"), ("--blank", float, 2.0, "matched blank, µg/L NH4+"),
                                ("--volume", float, 30.0, "electrolyte volume, mL"),
                                ("--margin", float, 3.0, "required signal / budget"),
                                ("--fixed-volume", float, None, "closed loop: fixed gas volume in L (instead of purge)")):
        p.add_argument(arg, type=typ, default=dflt, help=hlp)
    p.add_argument("--ref", choices=list(REF_CONDITIONS), default="0 °C, 1 atm"); p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("stress", help="feed-gas impurity level that alone supplies a reported rate")
    p.add_argument("--rate", type=float, required=True)
    p.add_argument("--unit", default="mol/s/cm2", help="one of: " + ", ".join(list(RATE_UNIT_ALIASES) + list(RATE_UNITS)))
    p.add_argument("--area", type=float, default=1.0); p.add_argument("--loading", type=float, default=0.0, help="mg cm-2")
    p.add_argument("--flow", type=float, required=True); p.add_argument("--lod", type=float, help="your gas-analysis bound, ppb")
    p.add_argument("--ref", choices=list(REF_CONDITIONS), default="0 °C, 1 atm"); p.set_defaults(fn=cmd_stress)
    p = sub.add_parser("bench", help="benchmark + machine-checked properties (exit 1 on any failure)")
    p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_bench)
    p = sub.add_parser("extract", help="find evidence passages in a .txt or .pdf; --llm asks Claude")
    p.add_argument("file"); p.add_argument("--llm", action="store_true"); p.set_defaults(fn=cmd_extract)
    return ap


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    a = build_parser().parse_args(argv)
    try:
        a.fn(a)
    except BrokenPipeError:  # output piped into e.g. `head`, which closed early
        sys.stdout = open(os.devnull, "w")


if __name__ == "__main__":
    main()
