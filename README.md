# NitroTrace

**Given an ammonia measurement and the controls actually run: what contamination remains unbounded, and how stringent must additional measurements be to close that gap?**

NitroTrace is a scientific instrument in software for electrochemical N₂ → NH₃ studies, built for the Open Scientific Intelligence (OSI) Hackathon 2026. It does three things:
- puts the measured ammonia and every contamination measurement on one basis;
- shows which contamination sources are bounded and which remain unbounded;
- works backwards to the measurements, and the sensitivity they would need, that could close the gap. Each threshold states its assumptions.

It ships as a **command-line tool** (`nitrotrace`), which opens an **interactive terminal UI** when run on its own, and a **web app** (a marimo notebook with a custom interface). All three run the same engine.

> NitroTrace does not determine whether nitrogen reduction occurred, and it does not judge the original studies. An "exceeds the quantified budget" verdict means only that the measured ammonia is larger than the contamination that was bounded. Isotope-labelled and other controls are still required.

---

## TL;DR

- **The gap it fills.** Protocols (Andersen 2019, Choi 2020) list which controls to run. NitroTrace computes whether the controls that *were* run bound the contamination, and if not, which measurement at what level would.
- **Deterministic core.** Interval arithmetic plus explicit, source-quoted rules. Every number has a calculation trail and is reproducible. No AI model is involved in any verdict.
- **The workflow.** Experimental inputs → defensible contamination budget → qualified verdict → quantitative evidence requirements → hypothetical-measurement check.
- **Real data.** A complete, source-linked case file for each published record, including Bragulla et al. 2026's open-circuit measurement. In that case, ammonia appears with no electrochemistry at all. The naive sum and the checklist both call it "exceeds"; NitroTrace calls it "insufficient", and shows that the missing gas flow is what blocks a conclusion.
- **Validation.** 33/33 synthetic verdicts correct, against 15/33 for a naive sum and 8/33 for a protocol checklist. 0 false "exceeds", against 9 and 5. 96/96 property checks pass on the bundled records (a scoped test result, not a general correctness guarantee). Choi 2020 Table 1 is reproduced within 0.05 %. **Review status: internal only.** An independent-review pack exists for every record, but no external domain researcher has signed off yet.

---

## 1. Install and run

```bash
cd nitrotrace
python -m venv .venv && .venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -e .                                        # CLI only needs pydantic
pip install -e ".[web]"                                 # + marimo web app
pip install -e ".[dev]" && pytest -q                    # 104 passed, 3 skipped; with ".[dev,api,pdf]": 111 passed
```

Without installing, `python -m app <command>` works the same as `nitrotrace <command>`.

### 1.1 Interactive terminal UI

Run `nitrotrace` with no arguments (or `nitrotrace ui`) in a terminal:

```text
╭─ NitroTrace ─────────────────────────────────────────────────────────────────────────────╮
│   █▄ █ █ ▀█▀ █▀█ █▀█ ▀█▀ █▀█ ▄▀█ █▀▀ █▀▀                                                 │
│   █ ▀█ █  █  █▀▄ █▄█  █  █▀▄ █▀█ █▄▄ ██▄   v0.3.1                                        │
│   What contamination remains unbounded, and what would close the gap?                    │
│   /cases            pick a record            /settle           what closes the gap       │
│   /audit ocv        audit by part of an id   /whatif --gas 382 try a measurement         │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────────────────╮
│ ❯ /settle                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
  ● syn_second_feed_unmeasured  insufficient information   / for commands · ? help
```

- `/` opens a command palette and `/cases` opens a record picker. In both, type to filter, use ↑↓ to move and Enter to choose.
- The open case is remembered, so `/settle`, `/whatif --gas 382` and `/casefile -o case.md` need no id.
- Every slash command runs exactly the same code as `nitrotrace <command>`, and plain CLI syntax works too.
- Errors (unknown case, invalid number) are reported and the session continues. Ctrl+C twice or `/exit` quits.
- It uses only the Python standard library and works in Windows Terminal, macOS and Linux terminals. Piped or non-terminal input falls back to a plain prompt.

### 1.2 CLI

| Command | What it does |
|---|---|
| `nitrotrace` | Interactive terminal UI (see 1.1) |
| `nitrotrace list` | Bundled records: 4 published, 33 synthetic |
| `nitrotrace audit CASE [--scale K] [--format text\|json\|md]` | Verdict, log-scale interval plot, coverage, still-unbounded sources, reasoning, next controls. `--scale` multiplies the observed amount (what-if). |
| `nitrotrace settle CASE` | The gap (measured / budget / headroom), contamination floor, and the measurements that would close the gap, each with its assumptions and caveats |
| `nitrotrace whatif CASE [--gas [SOURCE=]PPB ...] [--blank UG_PER_L \| --blank-nmol NMOL] [--components NMOL]` | Adds hypothetical measurements, labelled as such, and shows before → after. `--gas` can be repeated, one per feed. |
| `nitrotrace casefile CASE [-o FILE]` | Complete source-linked case file (Markdown) |
| `nitrotrace review-pack CASE [-o FILE]` | Checklist for an independent domain reviewer |
| `nitrotrace plan --rate R --flow SCCM --ppb PPB [--hours H --blank B --volume V --margin M --fixed-volume L]` | Design before you run: signal vs. conservative contamination, break-even time, smallest defensible rate |
| `nitrotrace stress --rate R --unit U --flow SCCM [--lod PPB]` | Feed-gas reactive-N level that alone would supply a reported rate |
| `nitrotrace bench [--json]` | Benchmark + properties; exits 1 on any failure (CI-ready) |
| `nitrotrace extract FILE [--llm]` | Evidence passages from a .txt/.pdf; `--llm` asks Claude for quote-verified candidates |

`CASE` is a record JSON path, a case id, or a unique part of one (e.g. `ocv`).

```text
$ nitrotrace whatif ocv --blank-nmol 25.5 --components 25.5
  HYPOTHETICAL: electrolyte_nmol measured ≤ 25.5 nmol N matched blank
  HYPOTHETICAL: components measured ≤ 25.5 nmol N matched component release
Before → after
  verdict        Insufficient information  →  Insufficient information
  valid budget   0  →  51 nmol N
  electrolyte    scenario_only  →  quantified
  components     scenario_only  →  quantified
  still unbounded — gas: … gas flow / delivered volume during the measurement not stated.
```

### 1.3 Web app (marimo)

```bash
marimo run marimo/nitrotrace_app.py       # app view
marimo edit marimo/nitrotrace_app.py      # the hidden cells
```

Live sandbox: https://molab.marimo.io/notebooks/nb_L8ugRvggBgaM68bfb9vzuK/app. After notebook changes, save in the editor and update the share, or the link may serve an older version.

The notebook is self-contained, so it runs where the package isn't installed. Its engine cell is **generated from `app/`** by `scripts/build_marimo_engine.py`, so the CLI and web app can't drift apart. The interface is one custom anywidget (paper-and-ink editorial design, hand-built SVG and canvas charts). The browser only draws; every number comes from the Python engine. Pages:

1. **Audit:**
   - verdict and log-scale interval ruler (counted, what-if and excluded sources drawn differently);
   - draggable verdict map from ×10⁻³ to ×10³;
   - coverage, contributions, and next controls with quotes;
   - calculation trail and record editor.
2. **What would settle it:**
   - the gap and the contamination floor;
   - specifications, each with its assumptions and caveats;
   - **Try hypothetical measurements:** prefilled, labelled re-run with before → after.
3. **Design before you run:** live chart, break-even time and a clickable design-space map. Assumptions are stated.
4. **Claim stress-test:** editable claims and a detection-bound slider.
5. **Evidence agent:** passage finder; optional Claude with verbatim-quote filtering.
6. **Report:** case file (rendered), independent-review pack, audit report, JSON, CSV.
7. **Method & proof:** approach, benchmark, properties, review status, limits, sources.

The optional HTTP API from v0.1 (`uvicorn app.main:app`, extras `api` and `pdf`) is still in the repo.

## 2. How the audit works

1. **Typed evidence.** Every value carries species (NH₃ / NH₄⁺ / N), basis, unit, bound type, interval and matching status. It also carries an evidence block: source, location, quote, provenance (`reported`, `manual`, `derived`, `assumed`, `synthetic` or `hypothetical`) and confirmation state.
2. **Strict normalisation to nmol N.** These are rejected: "ppm" for a liquid, a concentration without a volume, sccm without flow-controller reference conditions, and an actual flow without gas T and P.
3. **Gas-delivered contamination.** `n_gas = P·Q·t/(R·T)` and `U_i = n_gas · x_i,upper · a_i · η_i,upper`. Standardised flows use controller reference conditions. η = 1 is a labelled conservative scenario.
4. **Coverage.** Each category (gas, electrolyte, components) is bounded, partly bounded, what-if only, ambiguous or unknown. **Unknown is never zero.**
5. **Ordered verdict rules:**
   1. input error → *Input inconsistency*;
   2. no product → *Insufficient information*;
   3. valid budget upper ≥ observed lower → *Contamination-only explanation not excluded*;
   4. above the budget with any category unbounded → *Insufficient information*;
   5. otherwise → *Observation exceeds the quantified budget*, which is not proof of N₂ reduction.
6. **Next controls.** 12 rules, each with an explicit trigger and a quoted protocol passage.

**Rules that prevent misleading results:**
- Supplier purity is not a measurement.
- Upstream-of-purifier and non-matched controls are what-ifs: shown, never counted.
- A one-species gas bound leaves the other reactive-N species unknown.
- Overlapping blanks need an explicit coverage model (`system_blank` or `sum_components`).
- Nondetects stay `[0, LOD]`.
- Interval mismatches are errors unless a time-scaling model is declared.
- Isotope evidence never changes the budget.

## 3. From verdict to next measurement

- **Evidence-requirement inversion (`settle`).** Headroom = observed lower bound − valid budget upper bound.
  - **If the headroom is positive,** every unbounded category gets a threshold: ppb of total reactive N in the delivered gas, µg/L or nmol for a matched blank, nmol for matched component release. Each is given "alone" and as an "equal split", with its assumptions (η = 1, gas amount from the stated flow and interval, matched conditions).
  - **If the headroom is negative,** it says which counted bound must drop, and to what value.
  - **Every output carries a caveat:** meeting the thresholds can move the verdict to "exceeds", which by itself does not demonstrate nitrogen reduction.
- **Contamination floor.** An observation must be strictly above it before today's controls could call it "above budget". When it is 0, the app says why that comparison proves nothing.
- **Hypothetical measurements (`whatif`).** Adds measurements with provenance `hypothetical`. The audit flags its result as hypothetical, and reports never present it as evidence.
- **Design before you run (`plan`).** Assumes a constant true rate, flow, delivered impurity level and η, and a fixed matched blank.
  - **Continuous purge:** the signal-to-contamination ratio tends to `r·A·R·T / (P·Q·x·η)`, so *under these assumptions* running longer doesn't change it; only lower flow, cleaner gas or a closed gas volume does.
  - **Break-even time:** `t* = m·b / (s − m·g)`.
- **Claim stress-test (`stress`).** Converts a reported rate into the feed-gas reactive-N level that alone would supply it, and compares that with your gas-analysis bound.

## 4. Evidence and validation

### 4.1 Published records (source-linked; internal review only)

| Record | Source | What it shows |
|---|---|---|
| `bragulla2026_pristine_gde_ocv` | Bragulla et al. 2026 [S3], §3.5–3.6, Table 1, Fig. 8 | Ammonium measured at open circuit exceeds every separately quantified release. Naive and checklist methods say "exceeds"; NitroTrace says "insufficient". Even hypothetical matched blank and component tests leave the gas unbounded, because the flow was never stated. |
| `choi2020_table1_flowing_gas` | Choi et al. 2020 [S2], Table 1 | Reproduces their worst-case NOx calculation: 9815 vs 9820 nmol (0.05 %). The value is a supplier grade, so it is not counted. |
| `bragulla2026_n2_gas_washing` | [S3] Table 1, footnote f | Reported 80 ppb NH₃ and the trap amount agree (42.2 vs 41.0 nmol N). |
| `bragulla2026_n2_purified_nondetect` | [S3] Table 1, footnote g, §3.3 | Real nondetect kept as an interval; reproduces "< 3 ppb". |

- Full texts were read from open-access Europe PMC (PMC7641139, PMC12864167); Supporting Information was not read.
- Andersen et al. 2019 [S1] is paywalled; no values come from it.
- Full provenance: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).
- `nitrotrace casefile <case>` gives, for each record:
  - values as reported, with quotes;
  - what was counted or set aside, and why;
  - the calculation and assumptions;
  - the qualified conclusion and what would close the gap;
  - what the file does not show, and the review status.

**Review status.** All records are *internal review only*: checked against the sources by the authors, but not by an independent domain researcher. `nitrotrace review-pack <case>` produces a checklist for that review. Once a reviewer completes one, it is recorded in the record's `review` field and shown in every case file.

### 4.2 Synthetic benchmark (software and reasoning tests)

Published papers carry no ground truth for whether N₂ reduction occurred, so they can't score verdicts. The 33 synthetic records (`scripts/make_synthetic.py`) each target one trap. Their expected values come from separate hand-written arithmetic.

| | NitroTrace | Naive sum (unknown = 0) | Protocol checklist |
|---|---|---|---|
| Correct verdicts | **33/33** | 15/33 | 8/33 |
| False "exceeds" | **0** | 9 | 5 |

- Max relative error against 17 non-zero hand-computed references: 1.9 × 10⁻¹⁶.
- These are software tests; they don't validate attribution in real laboratories.

### 4.3 Machine-checked properties (`nitrotrace bench`, 96/96)

These are checks run on the bundled records. They are not a proof of correctness for arbitrary inputs; v0.3.1 fixed two bugs that the earlier 93/93 run did not catch (section 4.4).

| Property | Result | Scope |
|---|---|---|
| Unit invariance: any unit gives the same verdict and budget | 26/26 | skips input errors |
| Unknown is never zero: removing counted evidence never creates "exceeds" | 36/36 | skips overlap-set-aside measurements |
| Monotone budget: adding a matched bound of k raises the budget by exactly k | 22/22 | skips whole-system-blank records |
| Inversion round trip: "tighten" thresholds flip at ±10⁻⁶ | 8/8 | |
| Inversion round trip: gas ppb threshold flips the verdict at ±10⁻⁶ | 4/4 | |

`tests/` adds 111 pytest cases. They cover:
- unit handling and the published reproductions;
- the hypothetical-flag behaviour;
- the threshold flip through `whatif`;
- case-file contents, the planner's closed form and every CLI command;
- the interactive UI and the v0.3.1 review regressions (`tests/test_review_fixes.py`).

### 4.4 Fixed in v0.3.1 (found by external review)

An external reviewer installed v0.3.0, ran it, and reproduced four problems. Each now has a regression test in `tests/test_review_fixes.py`.

| Problem in v0.3.0 | Fix |
|---|---|
| **An unmeasured second gas feed was ignored.** Gas counted as "quantified" if *any* bounded feed covered all species, so one bounded feed plus an unmeasured one gave "exceeds". | Coverage is now assessed per feed. Every feed delivered to the cell must be bounded; otherwise gas is `partial_feeds` and the unmeasured feed is listed as unbounded. A matched whole-system blank that covers gas remains an explicit alternative. `settle` targets only the unbounded feed. New benchmark record `syn_second_feed_unmeasured`. |
| **NaN observations returned "exceeds".** NaN compares false with everything. | Any NaN or infinite number anywhere in a record is an input error (`non_finite_value` → `input_inconsistency`). The CLI rejects `nan` and `inf` arguments, and hypothetical measurements reject them too. |
| **Thresholds said ≤, but the audit's exceedance rule is strict.** A bound exactly at the threshold makes the budget equal the observation, which is "not excluded". | Requirements are now stated as strict (`<`), with an explicit caveat: a bound equal to the threshold leaves the verdict unchanged. |
| **PDF extraction dropped single-topic evidence.** A sentence needed two topic matches, so a standalone NOx impurity statement or a blank result was lost (the PDF test failed). | A sentence is kept if it has two topics, is an impurity or isotope statement, or states a measured value. Bare durations are still dropped. The PDF tests pass. |

## 5. The AI component (optional)

`nitrotrace extract FILE --llm` and the Evidence agent page ask `claude-opus-5-5` for structured candidates from a paper's text. The guard-rails:
- any candidate whose quote is not found verbatim is discarded;
- candidates never enter an audit until a person enters or confirms them.

The deterministic passage finder works without a key. **The Claude path has not yet been run against the live API**; test it once with a key before relying on it.

## 6. Limitations

- Aqueous electrochemical N₂ → NH₃ only; independent interval arithmetic; no kinetics or NMR inference.
- η = 1 and "alone / equal split" are labelled scenarios, not measurements.
- Published inputs carry stated assumptions. For example, the Bragulla gas T and P are not reported; 298.15 K and 1 atm are assumed and labelled. Of the three MLOD values Bragulla et al. report, the §3.3 one is used.
- Planner and stress-test results hold only under their stated constant-rate, constant-flow assumptions.
- Stress-test example rows are illustrative inputs, not data from specific papers.
- No independent domain review yet. The control rulebook has not been externally reviewed.

## 7. Repository layout

```text
app/                    engine (single source of truth) + CLI
  schemas units budget coverage controls audit reports     core audit
  inversion planner whatif validation                      v0.3 analysis
  cli.py  __main__.py                                      `nitrotrace` command
  extraction.py  main.py  static/                          optional PDF/LLM extraction, optional HTTP API
marimo/nitrotrace_app.py  web app (self-contained notebook, custom UI)
scripts/                make_synthetic.py, evaluate.py, build_marimo_engine.py
data/                   cases/ (published), synthetic/ (33), protocols/ (rules.json, sources.json)
tests/                  pytest suite (111 passed with all extras)
docs/                   DATA_SOURCES.md, DEMO_SCRIPT.md
```

## References

- [S1] Andersen, S. Z. et al. *Nature* 570, 504–508 (2019). doi:10.1038/s41586-019-1260-x
- [S2] Choi, J. et al. Identification and elimination of false positives in electrochemical nitrogen reduction studies. *Nat. Commun.* 11, 5546 (2020). doi:10.1038/s41467-020-19130-z
- [S3] Bragulla, S. et al. *ChemSusChem* 19, e202501974 (2026). doi:10.1002/cssc.202501974
- [S4] Hu et al. *Natl. Sci. Rev.* (2025). doi:10.1093/nsr/nwaf372 (adjacent work, not a data source)
- [S5] Open Scientific Intelligence Hackathon, https://llmhackathon.github.io/
