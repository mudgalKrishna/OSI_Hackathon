# Three-minute demonstration script

**Setup:** `uvicorn app.main:app` → http://127.0.0.1:8000. As a backup, keep `results/benchmark.json` and a downloaded Markdown report open.

**0:00–0:25 — Motivation (Start tab).**
"Detecting ammonia doesn't tell you where its nitrogen came from. Gas impurities, the electrolyte and the electrode can all supply it. NitroTrace asks one question: could the contamination that was *measured* account for the ammonia that was *measured*?"

**0:25–1:05 — Quantitative example.** Click *Step 1*.
The observation is 190–210 nmol. 50 mL/min for 120 min is 6 L of gas, and at ≤ 1 ppm of a one-N impurity the gas could deliver ≈ 245 nmol. Status: *Contamination-only explanation not excluded*. Open the calculation trail: n_gas = P·Q·t/(R·T), U = n·x·a·η, with η = 1 labelled as a conservative scenario. Point at the illustrative allocation and its "not a reconstruction" label.

**1:05–1:35 — Evidence changes the result.** Scroll to *Change the evidence*.
- *Tighten to ≤ 0.01 ppm* → ≈ 2.45 nmol → *Observation exceeds the quantified budget*. Read the caveat: this does not prove reduction.
- *Remove impurity measurement* → *Insufficient information*. The Unquantified panel highlights the gas source. It is unknown, not zero.

**1:35–2:15 — Published case.** Start → *Bragulla 2026 — pristine ZrN-GDE at OCV*.
No potential was applied, so every nmol here is background. The separately measured GDE and electrolyte releases (24 h aging tests) are shown only as what-ifs, because they were measured under different conditions. Even those don't reach the 2.76 µg measured in the cell. The authors trace the gap to Nafion ionomer in the GDE. Lesson: a gap above the quantified contamination does not mean the other sources are excluded. Open Controls: the Nafion/GDE rule quotes the paper's cleaning recommendation.

Optionally open *Choi 2020 Table 1*: the engine reproduces their 9.82 × 10³ nmol NOx figure. Because it is a supplier-grade illustration, it is not counted as a measured bound.

**2:15–2:45 — Validation (Report tab → Benchmark).**
32/32 synthetic statuses correct, numeric error ~1e-16. All unit/basis traps were caught, and overlap is never double-counted. A naive "sum everything, unknown = 0" baseline gets 15/32. "These are software tests, not laboratory validation."

**2:45–3:00 — Deliverable.** Download the Markdown report. It holds the full trail, evidence quotes and the next controls to run.

> NitroTrace turns contamination measurements and controls into an auditable ammonia budget, helping researchers see what their experiment supports and what still needs measuring.
