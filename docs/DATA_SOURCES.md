# Data sources and curation log

Curated 22 October 2026. Every published value in `data/cases/*.json` carries the source id, location and a verbatim supporting passage in its `evidence` block. Values marked `provenance: "assumed"` or with an "ASSUMED" note are not in the source.

## Access

| Id | Paper | Access used | Status |
|---|---|---|---|
| S1 | Andersen et al. 2019, *Nature*, doi:10.1038/s41586-019-1260-x | nature.com (login redirect), Europe PMC (not open access) | **Not accessed.** No values used. Cited only via S2's protocol summary. |
| S2 | Choi et al. 2020, *Nat. Commun.*, doi:10.1038/s41467-020-19130-z | Europe PMC full-text XML, PMC7641139 (CC BY 4.0) | Main text read. Supplementary PDF not parsed. |
| S3 | Bragulla et al. 2026, *ChemSusChem*, doi:10.1002/cssc.202501974 | Europe PMC full-text XML, PMC12864167 | Main text read. Supporting Information (Tables S1–S10) not read. |

## Values taken, and checks performed

### S2 — Choi et al. 2020, Table 1 ("Amount of N2 and potential NOx contaminants supplied in 4-h NRR experiments")

| Row | Flow | Time | Purity → NOx | N2 (nmol) | NOx (nmol) | NH3 rate | NitroTrace |
|---|---|---|---|---|---|---|---|
| 1 | 100 mL/min | 240 min | 99.999% → 10 ppm | 9.82 × 10⁸ | 9.82 × 10³ | 0.68 nmol/s | 9815 nmol, 0.682 nmol/s |
| 2 | 100 | 240 | 99.99% → 100 ppm | | 9.82 × 10⁴ | 6.8 | tested |
| 4 | 20 | 240 | 10 ppm | 1.96 × 10⁸ | 1.96 × 10³ | 0.14 | tested |
| 9 | 0 (15 min pre-purge at 20) | 15 | 1000 ppm | 1.23 × 10⁷ | 1.23 × 10⁴ | 0.9 | tested |

Footnotes: (a) arbitrary 4 h duration; (b) 298 K, 1 atm; (c) "Assumes for illustration purposes that all of the impurities are NOx compounds." The 10 ppm is therefore encoded as `measured_where: supplier_specification`, and the engine reports it without counting it.

Protocol passages used in `rules.json`: §"Analysis and elimination of NOx", §"Undertaking an NRR experiment", Criteria (II) and (III), and the CNx/VN case studies.

### S3 — Bragulla et al. 2026

| Value | Location | Used in |
|---|---|---|
| Pristine GDE at OCV: NH4+ time-weighted average 85.2 µg/L (2.76 µg) | §3.6, Fig. 8 | `bragulla2026_pristine_gde_ocv` observed product |
| GDE release net of GDL: 1.55 µg NH4+ (24 h, 50 mL; Table 1: 35.9 − 4.9 µg/L) | §3.5, Table 1 | component, *what-if* (different conditions) |
| Electrolyte 24 h blank: 6.6 µg/L NH4+ | Table 1 | electrolyte, *what-if* |
| N2 5.0: 10 mL/min, 21.53 h, 12.92 L, 50 mL trap, 14.8 µg/L, 80 ppb_mol | Table 1 footnote f, §4 | `bragulla2026_n2_gas_washing` |
| Purified N2 (Agilent OT3-4): 64.32 h, 38.57 L, < MLOD, 3 ppb_mol | Table 1 footnote g, §4 | `bragulla2026_n2_purified_nondetect` |
| MLOD 1.8 µg/L (§3.3); also 2 µg/L (abstract) and 5.4 µg/L (§4) | | nondetect upper bound; conflict noted in the record |
| Authors' attribution to Nafion ionomer; cleaning recommendation | §4 | rule R6 |

Arithmetic checks:

- 2.76 µg ÷ 85.2 µg/L = 32.4 mL, consistent with the 32 mL in SI Figs S2/S3 (§3.6 says "about 35 mL").
- 10 mL/min × 21.53 h = 12.918 L ✓; 80 ppb × 12.92 L at an assumed 298.15 K and 1 atm = 42.25 nmol N vs. trap 14.8 µg/L × 50 mL ÷ 18.038 g/mol = 41.03 nmol N (3 % apart).
- 3 ppb × 38.57 L = 4.73 nmol N vs. MLOD 1.8 µg/L × 50 mL = 4.99 nmol N.

Not found in the S3 main text, so left unknown: NOx measurements, Ar controls, gas flow during the in-situ OCV measurement, total duration of the Fig. 8 run.

## Why only these records

The plan targeted three verified experiment records and said to "ship two complete records rather than inventing a third". S1 could not be accessed. Choi's Table 1 is a published worst-case calculation, not an experiment, and has no product measurement. The Bragulla OCV measurement is the one published experimental case. The two Bragulla gas-washing measurements are reproduction checks of published contamination numbers. Choi's case-study figures (Bi, Au/C, CNx, VN) give their values only graphically, so they were not digitized.

## Licensing

S2 is CC BY 4.0. The short quotations from S3 are used for attribution and evidence linking. Before publishing a dataset, check S3's licence on Europe PMC.
