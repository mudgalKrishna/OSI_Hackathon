import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup(hide_code=True):
    import base64, copy, csv, io, json, math, re, statistics, time, zlib
    from typing import Literal, Optional

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from pydantic import BaseModel, Field



@app.cell(hide_code=True)
def data():
    # Curated published records (Choi 2020, Bragulla 2026), 33 synthetic benchmark records,
    # the protocol rulebook and the source registry, compressed. Provenance: docs/DATA_SOURCES.md.
    _BLOB = (
        "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"
        "Aog="
    )
    DATA = json.loads(zlib.decompress(base64.b64decode(_BLOB)).decode())
    CASE_KEYS = sorted(k for k in DATA if k.startswith(('cases/', 'synthetic/')))

    return CASE_KEYS, DATA


@app.cell(hide_code=True)
def engine(DATA):
    # NitroTrace engine, generated from the app/ package by scripts/build_marimo_engine.py — edit the package,
    # not this cell. All amounts are nmol of ammonia-equivalent N; missing evidence is unknown, never zero.

    # ======================= schemas =======================
    # Pydantic schemas for NitroTrace experiment records, evidence and audit results.
    # 
    # Every scientifically meaningful value carries its original unit, bounds, bound
    # type and provenance. Missing values are ``None`` and mean *unknown*, never zero.

    Provenance = Literal["reported", "manual", "derived", "assumed", "synthetic", "hypothetical"]
    BoundType = Literal["hard_limit", "detection_limit", "stated_interval", "assumed_range", "point"]
    Species = Literal["NH3", "NH4+", "N"]
    Basis = Literal["amount", "mass", "concentration_molar", "concentration_mass"]
    Category = Literal["gas", "electrolyte", "components"]
    CoverageState = Literal["quantified", "unknown", "not_applicable", "specified_synthetic"]

    class Evidence(BaseModel):
        """Where a value came from and whether a human has confirmed it."""

        source_id: Optional[str] = None  # key into data/protocols/sources.json, e.g. "S2"
        url: Optional[str] = None
        location: Optional[str] = None  # page / table / figure / section
        quote: Optional[str] = None  # short supporting passage or table-cell reference
        provenance: Provenance = "manual"
        confirmed: bool = False
        note: Optional[str] = None

    class Interval(BaseModel):
        """A collection / exposure window in minutes relative to an experiment start."""

        start_min: float = 0.0
        end_min: float

        @property
        def duration_min(self) -> float:
            return self.end_min - self.start_min

    class ProductMeasurement(BaseModel):
        """An ammonia/ammonium amount, as measured or as a blank/control result.

        ``lower``/``upper`` are in ``unit``; a nondetect has ``nondetect=True`` and
        ``upper`` equal to the detection limit (lower defaults to 0).
        """

        species: Optional[Species] = None
        basis: Optional[Basis] = None
        unit: str
        lower: Optional[float] = None
        upper: Optional[float] = None
        bound_type: BoundType = "stated_interval"
        nondetect: bool = False
        volume_mL: Optional[float] = None  # required for concentrations
        dilution_factor: float = 1.0  # measured concentration x factor = original sample concentration
        interval: Optional[Interval] = None
        n_replicates: Optional[int] = None
        evidence: Evidence = Field(default_factory=Evidence)

    class GasFlow(BaseModel):
        value: float
        unit: Literal["mL/min", "L/min", "sccm", "slpm"]
        volume_basis: Literal["actual", "standard"]
        temperature_K: Optional[float] = None  # gas temperature at the flow measurement (actual volume)
        pressure_Pa: Optional[float] = None
        reference_temperature_K: Optional[float] = None  # flow-controller standard conditions
        reference_pressure_Pa: Optional[float] = None
        evidence: Evidence = Field(default_factory=Evidence)

    class GasImpurity(BaseModel):
        species: str  # e.g. "NH3", "NO", "NOx (as NO)", "N2O"
        nitrogen_atoms: int = 1
        unit: Literal["ppm", "ppb", "mole_fraction", "mol_percent"]
        lower: Optional[float] = None
        upper: Optional[float] = None
        bound_type: BoundType = "stated_interval"
        measured_where: Literal["delivered_to_cell", "upstream_of_purifier", "supplier_specification", "unknown"]
        # True only if this bound covers every reactive-nitrogen species of concern (e.g. a total NOx+NH3 bound,
        # or a synthetic record that defines a single impurity). A bound on one species leaves the others unknown.
        covers_all_reactive_nitrogen: bool = False
        evidence: Evidence = Field(default_factory=Evidence)

    class GasSource(BaseModel):
        id: str
        label: str
        impurity: Optional[GasImpurity] = None  # None = impurity not measured (unknown)
        flow: Optional[GasFlow] = None
        total_volume_L: Optional[float] = None  # alternative to flow x time, actual volume at T, P
        total_volume_temperature_K: Optional[float] = None
        total_volume_pressure_Pa: Optional[float] = None
        exposure: Optional[Interval] = None
        eta_upper: Optional[float] = None  # upper bound on fraction reaching the measured product
        eta_lower: float = 0.0
        eta_basis: Literal["measured", "conservative_upper_bound_1", "assumed"] = "conservative_upper_bound_1"
        evidence: Evidence = Field(default_factory=Evidence)

    class ControlMeasurement(BaseModel):
        """A blank, background or component-release measurement."""

        id: str
        label: str
        kind: Literal["system_blank", "electrolyte_blank", "component_release"]
        covers: list[Category]
        measurement: ProductMeasurement
        conditions_match: Literal["matched", "different", "unknown"] = "unknown"
        conditions_note: Optional[str] = None
        include_as_scenario: bool = False  # use a non-matched control as an explicit what-if
        time_scaling: Literal["none", "linear"] = "none"  # linear: scale amount by observation/ control duration
        contains_nafion: bool = False
        evidence: Evidence = Field(default_factory=Evidence)

    class IsotopeEvidence(BaseModel):
        performed: bool = False
        quantitative: bool = False
        time_series: bool = False
        labeled_impurities_quantified: bool = False  # 15NOx / 15NH3 in 15N2 supply
        note: Optional[str] = None
        evidence: Evidence = Field(default_factory=Evidence)

    class ExperimentContext(BaseModel):
        flowing_gas: Optional[bool] = None
        nox_measured_alongside: Optional[bool] = None
        argon_control_same_electrode: Optional[bool] = None
        electrode_contains_nitrogen: Optional[bool] = None
        gde_contains_nafion: Optional[bool] = None
        electrode_area_cm2: Optional[float] = None
        electrolyte: Optional[str] = None
        note: Optional[str] = None

    class Review(BaseModel):
        """Review state of a record. 'internal' means checked only by the people who built it."""

        status: Literal["internal", "independent"] = "internal"
        reviewer: Optional[str] = None  # role or affiliation, e.g. "electrochemistry postdoc, external"
        date: Optional[str] = None
        outcome: Optional[str] = None  # e.g. "agrees with inputs and verdict", "disputes gas flow"
        note: Optional[str] = None

    class ExperimentRecord(BaseModel):
        case_id: str
        title: str
        record_type: Literal["synthetic", "published", "published_scenario", "user"] = "user"
        description: Optional[str] = None
        observed_product: Optional[ProductMeasurement] = None
        gas_sources: list[GasSource] = Field(default_factory=list)
        controls: list[ControlMeasurement] = Field(default_factory=list)
        declared_coverage: dict[Category, CoverageState] = Field(default_factory=dict)
        coverage_model: Optional[Literal["system_blank", "sum_components"]] = None
        isotope: IsotopeEvidence = Field(default_factory=IsotopeEvidence)
        context: ExperimentContext = Field(default_factory=ExperimentContext)
        assumptions: list[str] = Field(default_factory=list)
        sources: list[str] = Field(default_factory=list)  # source ids used by this record
        expected: Optional[dict] = None  # benchmark expectations (synthetic records only)
        review: Optional[Review] = None  # who has checked this record's inputs and expected outcome

    # ---------------------------------------------------------------- results ----

    Status = Literal[
        "contamination_not_excluded",
        "exceeds_quantified_budget",
        "insufficient_information",
        "input_inconsistency",
    ]

    STATUS_LABELS: dict[str, str] = {
        "contamination_not_excluded": "Contamination-only explanation not excluded",
        "exceeds_quantified_budget": "Observation exceeds the quantified budget",
        "insufficient_information": "Insufficient information",
        "input_inconsistency": "Input inconsistency",
    }

    class TrailStep(BaseModel):
        label: str
        formula: Optional[str] = None
        values: dict[str, str] = Field(default_factory=dict)
        result: Optional[str] = None

    class Contribution(BaseModel):
        source_id: str
        label: str
        category: str
        lower_nmol: float
        upper_nmol: float
        included: bool
        role: Literal["valid_bound", "scenario", "excluded_overlap", "excluded_unmatched", "excluded_invalid"]
        reason: Optional[str] = None
        trail: list[TrailStep] = Field(default_factory=list)

    class Issue(BaseModel):
        code: str
        severity: Literal["error", "warning", "info"]
        message: str
        field: Optional[str] = None

    class UnquantifiedSource(BaseModel):
        category: str
        reason: str

    class Recommendation(BaseModel):
        rule_id: str
        title: str
        text: str
        trigger: str
        references: list[dict]

    class AuditResult(BaseModel):
        case_id: str
        status: Status
        status_label: str
        headline: str
        explanation: list[str]
        coverage: dict[str, str]
        budget_complete: bool
        budget_label: str
        observed_nmol: Optional[dict[str, float]] = None
        budget_valid_nmol: dict[str, float]
        budget_with_scenarios_nmol: dict[str, float]
        contributions: list[Contribution]
        unquantified: list[UnquantifiedSource]
        issues: list[Issue]
        feasible_allocation: Optional[dict] = None
        recommendations: list[Recommendation] = Field(default_factory=list)
        observed_trail: list[TrailStep] = Field(default_factory=list)
        assumptions: list[str] = Field(default_factory=list)
        hypothetical: bool = False  # True when any input has provenance "hypothetical"
        engine_version: str

    # ======================= units =======================
    # Explicit, tested unit conversions.
    # 
    # Internal comparison unit: nanomoles of ammonia-equivalent nitrogen (nmol N).
    # NH3, NH4+ and N each carry one nitrogen atom, so amounts are interchangeable,
    # but mass -> amount conversion needs the molar mass of the *declared* species.
    # Only the unit strings listed here are accepted; anything else is rejected.

    R = 8.314462618  # J mol^-1 K^-1
    ATM_PA = 101325.0

    MOLAR_MASS_G_PER_MOL = {"NH3": 17.031, "NH4+": 18.038, "N": 14.007}

    AMOUNT_UNITS_TO_MOL = {"mol": 1.0, "mmol": 1e-3, "umol": 1e-6, "µmol": 1e-6, "nmol": 1e-9, "pmol": 1e-12}
    MASS_UNITS_TO_G = {"g": 1.0, "mg": 1e-3, "ug": 1e-6, "µg": 1e-6, "ng": 1e-9}
    MOLAR_CONC_TO_MOL_PER_L = {"M": 1.0, "mol/L": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "nM": 1e-9}
    MASS_CONC_TO_G_PER_L = {"g/L": 1.0, "mg/L": 1e-3, "ug/L": 1e-6, "µg/L": 1e-6, "ng/mL": 1e-6, "ug/mL": 1e-3, "µg/mL": 1e-3}

    BASIS_UNITS = {
        "amount": AMOUNT_UNITS_TO_MOL,
        "mass": MASS_UNITS_TO_G,
        "concentration_molar": MOLAR_CONC_TO_MOL_PER_L,
        "concentration_mass": MASS_CONC_TO_G_PER_L,
    }

    # Strings that look like units but are ambiguous for liquid samples.
    AMBIGUOUS_UNITS = {"ppm", "ppb", "ppt", "%", "wt%"}

    GAS_FRACTION = {"mole_fraction": 1.0, "mol_percent": 1e-2, "ppm": 1e-6, "ppb": 1e-9}
    FLOW_TO_L_PER_MIN = {"mL/min": 1e-3, "L/min": 1.0, "sccm": 1e-3, "slpm": 1.0}

    class UnitError(ValueError):
        """Raised for an ambiguous, unsupported or inconsistent unit."""

        def __init__(self, code: str, message: str):
            super().__init__(message)
            self.code = code

    def basis_for_unit(unit: str) -> str | None:
        for basis, table in BASIS_UNITS.items():
            if unit in table:
                return basis
        return None

    def to_nmol_n(value: float, unit: str, species: str | None, basis: str | None,
                  volume_mL: float | None = None, dilution_factor: float = 1.0) -> tuple[float, str]:
        """Convert one product value to nmol N. Returns (value, formula text)."""
        if species is None:
            raise UnitError("missing_species", "Species (NH3, NH4+ or N) must be declared explicitly.")
        if species not in MOLAR_MASS_G_PER_MOL:
            raise UnitError("unknown_species", f"Unsupported species '{species}'.")
        if basis is None:
            raise UnitError("missing_basis", "Reporting basis (amount, mass or concentration) must be declared explicitly.")
        if unit in AMBIGUOUS_UNITS and basis_for_unit(unit) is None:
            raise UnitError("ambiguous_unit", f"Unit '{unit}' is ambiguous for a liquid-phase product; use mol, g or mol/L forms.")
        unit_basis = basis_for_unit(unit)
        if unit_basis is None:
            raise UnitError("unsupported_unit", f"Unit '{unit}' is not supported.")
        if unit_basis != basis:
            raise UnitError("basis_unit_conflict", f"Unit '{unit}' implies basis '{unit_basis}' but basis '{basis}' was declared.")
        if dilution_factor <= 0:
            raise UnitError("bad_dilution", "Dilution factor must be positive.")

        M = MOLAR_MASS_G_PER_MOL[species]
        if basis == "amount":
            mol = value * AMOUNT_UNITS_TO_MOL[unit]
            text = f"{value:g} {unit} {species} = {mol*1e9:.6g} nmol N"
        elif basis == "mass":
            mol = value * MASS_UNITS_TO_G[unit] / M
            text = f"{value:g} {unit} {species} / {M} g/mol = {mol*1e9:.6g} nmol N"
        else:
            if volume_mL is None or volume_mL <= 0:
                raise UnitError("missing_volume", "A concentration needs the sample/electrolyte volume to give an amount.")
            V = volume_mL / 1000.0
            if basis == "concentration_molar":
                mol = value * MOLAR_CONC_TO_MOL_PER_L[unit] * dilution_factor * V
                text = f"{value:g} {unit} x DF {dilution_factor:g} x {volume_mL:g} mL = {mol*1e9:.6g} nmol N"
            else:
                mol = value * MASS_CONC_TO_G_PER_L[unit] * dilution_factor * V / M
                text = (f"{value:g} {unit} {species} x DF {dilution_factor:g} x {volume_mL:g} mL / {M} g/mol"
                        f" = {mol*1e9:.6g} nmol N")
        return mol * 1e9, text

    def gas_moles(flow_value: float, flow_unit: str, volume_basis: str, duration_min: float,
                  T_K: float | None, P_Pa: float | None,
                  ref_T_K: float | None, ref_P_Pa: float | None) -> tuple[float, str]:
        """Moles of gas delivered: n = P Q t / (R T).

        Actual flows use the gas temperature and pressure at the flow measurement.
        Standardized flows (sccm/slpm) use the flow controller's reference
        conditions; the cell temperature must never be applied to them.
        """
        if flow_unit not in FLOW_TO_L_PER_MIN:
            raise UnitError("unsupported_unit", f"Flow unit '{flow_unit}' is not supported.")
        standardized_unit = flow_unit in ("sccm", "slpm")
        if standardized_unit and volume_basis != "standard":
            raise UnitError("flow_basis_conflict", f"'{flow_unit}' is a standardized flow but the volume basis was declared 'actual'.")
        if volume_basis == "standard":
            if ref_T_K is None or ref_P_Pa is None:
                raise UnitError("missing_reference_conditions",
                                "A standardized flow needs the flow controller's reference temperature and pressure.")
            T, P, cond = ref_T_K, ref_P_Pa, "controller reference"
        else:
            if T_K is None or P_Pa is None:
                raise UnitError("missing_gas_conditions", "An actual volumetric flow needs gas temperature and pressure.")
            T, P, cond = T_K, P_Pa, "actual gas"
        if T <= 0 or P <= 0 or duration_min < 0:
            raise UnitError("bad_value", "Temperature, pressure and duration must be positive.")
        V_m3 = flow_value * FLOW_TO_L_PER_MIN[flow_unit] * duration_min / 1000.0
        n = P * V_m3 / (R * T)
        text = (f"n_gas = P·Q·t/(R·T) = {P:g} Pa x {flow_value:g} {flow_unit} x {duration_min:g} min "
                f"/ ({R} x {T:g} K) [{cond} conditions] = {V_m3*1000:.6g} L -> {n:.6g} mol")
        return n, text

    def gas_moles_from_volume(volume_L: float, T_K: float | None, P_Pa: float | None) -> tuple[float, str]:
        if T_K is None or P_Pa is None:
            raise UnitError("missing_gas_conditions", "A total gas volume needs its temperature and pressure.")
        n = P_Pa * volume_L / 1000.0 / (R * T_K)
        return n, f"n_gas = P·V/(R·T) = {P_Pa:g} Pa x {volume_L:g} L / ({R} x {T_K:g} K) = {n:.6g} mol"

    def gas_fraction(value: float, unit: str) -> float:
        if unit not in GAS_FRACTION:
            raise UnitError("unsupported_unit", f"Gas impurity unit '{unit}' is not supported.")
        return value * GAS_FRACTION[unit]

    # ======================= budget =======================
    # Contamination budget arithmetic: observed product, gas-delivered and control contributions.
    # 
    # All values are nmol of ammonia-equivalent nitrogen (nmol N). Bounds are
    # propagated with plain interval arithmetic; no distributions are invented.

    INTERVAL_TOLERANCE = 0.01  # 1 % relative difference in durations counts as a mismatch

    def nt_durations_differ(a: float, b: float) -> bool:
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
            if obs_dur is not None and nt_durations_differ(duration, obs_dur):
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
        if obs_dur is not None and ctl_dur is not None and nt_durations_differ(obs_dur, ctl_dur):
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
                and nt_durations_differ(op.volume_mL, c.measurement.volume_mL)):
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

    # ======================= coverage =======================
    # Source-coverage model: which contamination categories are bounded, and by what.
    # 
    # A matched whole-system blank may already contain gas, electrolyte and
    # component contributions. When it overlaps other valid measurements of the same
    # category, the record must select a coverage model; otherwise the overlapping
    # sources are not aggregated and the budget is reported as partial.

    CATEGORIES = ("gas", "electrolyte", "components")

    UNKNOWN_REASONS = {
        "gas": "No measured upper bound on reactive-nitrogen impurities in the gas delivered to the cell.",
        "electrolyte": "No matched electrolyte or whole-system background measurement.",
        "components": "No matched measurement of release from nitrogen-containing or contaminated components.",
    }

    def nt_cats(c: Contribution) -> set[str]:
        return set(c.category.split("+"))

    def coverage_resolve(rec: ExperimentRecord, contribs: list[Contribution], gas_unmeasured: list[str],
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
            overlap = [o for o in others if nt_cats(o) & nt_cats(sb)]
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
                shared = set().union(*[nt_cats(o) & nt_cats(sb) for o in overlap])
                # Every category the set-aside blank covered is now ambiguous, not only the shared ones.
                ambiguous |= nt_cats(sb) | set().union(*[nt_cats(o) for o in overlap])
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
            valid = [c for c in contribs if c.included and cat in nt_cats(c)]
            scen = [c for c in contribs if c.role == "scenario" and cat in nt_cats(c)]
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
                    unq.append(UnquantifiedSource(category=cat, reason=nt_partial_species_reason(rec, gas_states)))
            elif "partial_species" in gas_states.values():
                coverage[cat] = "partial_species"
                unq.append(UnquantifiedSource(category=cat, reason=nt_partial_species_reason(rec, gas_states)))
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

    def nt_partial_species_reason(rec: ExperimentRecord, states: dict[str, str]) -> str:
        measured = sorted({g.impurity.species for g in rec.gas_sources if g.impurity and states.get(g.id) == "partial_species"})
        return (f"Only {', '.join(measured)} bounded in the delivered gas; other reactive-nitrogen species "
                "(e.g. NOx, N2O, NH3) are not bounded.")

    def coverage_complete(coverage: dict[str, str]) -> bool:
        return all(v in ("quantified", "not_applicable", "specified_synthetic") for v in coverage.values())

    # ======================= controls =======================
    # Protocol-based next-control suggestions.
    # 
    # Each rule in data/protocols/rules.json has a transparent trigger evaluated
    # here, and quotes the source passage that motivates it. These are suggestions,
    # not an optimized experimental design.

    DATA_DIR = None  # notebook: records come from the embedded DATA

    def rules() -> dict[str, dict]:
        return {r["rule_id"]: r for r in DATA["protocols/rules.json"]}

    def sources() -> dict[str, dict]:
        return DATA["protocols/sources.json"]

    def triggered_rules(rec: ExperimentRecord, coverage: dict[str, str], contribs: list[Contribution],
                        issues: list[Issue]) -> list[tuple[str, str]]:
        """Returns (rule_id, concrete trigger description) pairs."""
        out: list[tuple[str, str]] = []
        synthetic = rec.record_type == "synthetic"
        where = {g.impurity.measured_where for g in rec.gas_sources if g.impurity is not None}

        if coverage.get("gas") not in ("quantified", "not_applicable", "specified_synthetic"):
            out.append(("R1_gas_unquantified", f"Gas coverage: {coverage['gas']}."))
        if "supplier_specification" in where:
            out.append(("R2_supplier_spec_only", "A gas impurity value is a supplier specification."))
        if where & {"upstream_of_purifier", "unknown"}:
            out.append(("R3_upstream_of_purifier", "A gas impurity was measured upstream of the purifier or at an unknown point."))
        if coverage.get("electrolyte") not in ("quantified", "not_applicable", "specified_synthetic"):
            out.append(("R4_background_unresolved", f"Electrolyte/system background coverage: {coverage.get('electrolyte')}."))
        if (coverage.get("components") not in ("quantified", "not_applicable", "specified_synthetic")
                or rec.context.electrode_contains_nitrogen):
            why = f"Component coverage: {coverage.get('components')}."
            if rec.context.electrode_contains_nitrogen:
                why += " The electrode/catalyst contains nitrogen."
            out.append(("R5_component_unresolved", why))
        if rec.context.gde_contains_nafion or any(c.contains_nafion for c in rec.controls):
            out.append(("R6_nafion_gde", "A Nafion-containing electrode or component is part of the setup."))
        iso = rec.isotope
        if not synthetic and (not iso.performed or not iso.quantitative or not iso.time_series):
            state = ("not performed" if not iso.performed else "qualitative only" if not iso.quantitative
                     else "single condition only")
            out.append(("R7_isotope_incomplete", f"Isotope evidence: {state}."))
        if iso.performed and not iso.labeled_impurities_quantified:
            out.append(("R8_labeled_impurities", "15N2 experiment recorded without quantified labeled impurities."))
        mismatch = [i for i in issues if i.code in ("interval_mismatch", "volume_mismatch")]
        unmatched = [c for c in contribs if c.role in ("excluded_unmatched", "scenario") and c.category != "gas"]
        if mismatch or unmatched:
            names = [i.message.split(":")[0] for i in mismatch] + [c.label for c in unmatched]
            out.append(("R9_condition_mismatch", "Non-aligned measurements: " + "; ".join(dict.fromkeys(names)) + "."))
        if rec.context.flowing_gas:
            out.append(("R10_fixed_gas_volume", "Continuous gas flow during the measurement."))
        if not synthetic and rec.context.nox_measured_alongside is not True:
            out.append(("R11_nox_alongside", "NOx alongside ammonia: "
                        + ("not reported." if rec.context.nox_measured_alongside is None else "not measured.")))
        op = rec.observed_product
        if not synthetic and op is not None and (op.n_replicates is None or op.n_replicates < 3):
            out.append(("R12_replicates", f"Independent replicates: {op.n_replicates if op.n_replicates else 'not stated'}."))
        return out

    def recommend(rec: ExperimentRecord, coverage: dict[str, str], contribs: list[Contribution],
                  issues: list[Issue]) -> list[Recommendation]:
        rs, src = rules(), sources()
        recs = []
        for rule_id, trig in triggered_rules(rec, coverage, contribs, issues):
            r = rs[rule_id]
            refs = [dict(ref, citation=src.get(ref["source_id"], {}).get("citation"),
                         url=src.get(ref["source_id"], {}).get("url")) for ref in r["references"]]
            recs.append(Recommendation(rule_id=rule_id, title=r["title"], text=r["text"], trigger=trig, references=refs))
        return recs

    # ======================= audit =======================
    # Run a full evidence audit on one experiment record.
    # 
    # Status logic (Section 7 of the build plan):
    #   1. Any input error                                  -> input_inconsistency
    #   2. No observed product                              -> insufficient_information
    #   3. Valid budget upper >= observed lower             -> contamination_not_excluded
    #      (unknown sources can only raise the budget, so this holds regardless of coverage)
    #   4. Observation above the valid budget, and any
    #      category unknown / ambiguous / scenario-only     -> insufficient_information
    #   5. Otherwise                                        -> exceeds_quantified_budget

    ENGINE_VERSION = "0.3.1"

    def nt_bounds(contribs, roles):
        sel = [c for c in contribs if c.role in roles]
        return {"lower": sum(c.lower_nmol for c in sel), "upper": sum(c.upper_nmol for c in sel)}

    def nt_non_finite(obj, path: str = "") -> list[str]:
        """Paths of NaN / infinite numbers anywhere in a dumped record."""
        if isinstance(obj, float):
            return [] if math.isfinite(obj) else [path or "value"]
        if isinstance(obj, dict):
            return [p for k, v in obj.items() if k != "expected" for p in nt_non_finite(v, f"{path}.{k}" if path else k)]
        if isinstance(obj, list):
            return [p for i, v in enumerate(obj) for p in nt_non_finite(v, f"{path}[{i}]")]
        return []

    def run_audit(rec: ExperimentRecord) -> AuditResult:
        issues: list[Issue] = []
        assumptions: list[str] = list(rec.assumptions)
        # NaN compares False with everything, so it would silently fall through to "exceeds"; reject it first.
        for p in nt_non_finite(rec.model_dump()):
            issues.append(Issue(code="non_finite_value", severity="error", field=p,
                                message=f"{p} is not a finite number (NaN or infinity); enter a real value or leave it empty."))

        obs = observed_bounds(rec, issues)
        contribs, gas_unmeasured = [], []
        for g in rec.gas_sources:
            c, unknown_reason = gas_contribution(g, rec, issues, assumptions)
            if c is not None:
                contribs.append(c)
            elif unknown_reason:
                gas_unmeasured.append(unknown_reason)
        for ctl in rec.controls:
            c = control_contribution(ctl, rec, issues, assumptions)
            if c is not None:
                contribs.append(c)

        cov, unq = coverage_resolve(rec, contribs, gas_unmeasured, issues)
        is_complete = coverage_complete(cov)
        valid = nt_bounds(contribs, {"valid_bound"})
        with_scen = nt_bounds(contribs, {"valid_bound", "scenario"})
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
            allocation = feasible_allocation(contribs, observed["lower"], observed["upper"])
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

        recs = recommend(rec, cov, contribs, issues)
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

    # ======================= reports =======================
    # JSON, CSV and Markdown exports that reproduce the displayed audit.

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

    def nt_fmt(x: float) -> str:
        return f"{x:.4g}"

    NOT_SHOWN = [
        "Whether nitrogen reduction occurred. An 'exceeds' verdict means only that the measured ammonia is larger than "
        "the contamination that was bounded; isotope-labelled and other controls are still required.",
        "Whether the original study is right or wrong. The audit describes what the entered evidence permits.",
        "Anything about sources that were never measured, other than that they remain unbounded.",
    ]

    def nt_review_line(rec: ExperimentRecord) -> str:
        r = rec.review
        if r is None or r.status == "internal":
            return ("**Internal review only.** The inputs and expected outcome have been checked by the NitroTrace "
                    "authors against the cited source, but not yet by an independent domain researcher. "
                    "Use `nitrotrace review-pack` to produce a checklist for one.")
        who = r.reviewer or "an independent reviewer"
        return f"**Independently reviewed** by {who}{f' on {r.date}' if r.date else ''}: {r.outcome or 'no outcome recorded'}."

    def nt_value(m) -> str:
        sp = f" {m.species}" if getattr(m, "species", None) else ""
        if getattr(m, "nondetect", False):
            return f"not detected (< {m.upper:g} {m.unit}{sp})"
        lo, hi = m.lower, m.upper
        if lo is not None and hi is not None and lo != hi:
            return f"{lo:g}–{hi:g} {m.unit}{sp}"
        v = hi if hi is not None else lo
        return f"{v:g} {m.unit}{sp}" if v is not None else "not stated"

    def nt_ev_row(name: str, value: str, ev) -> str:
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
             f"- **Review status:** {nt_review_line(rec)}",
             f"- **Engine:** v{res.engine_version}", ""]
        if rec.description:
            L += [rec.description, ""]
        L += ["## 1. Sources", ""]
        L += [f"- [{s}] {src[s]['citation']} {src[s].get('open_full_text') or src[s].get('url', '')}" for s in rec.sources if s in src] or ["- None (synthetic record)."]
        L += ["", "## 2. Measurements as reported", "",
              "| Quantity | Value as reported | Where | Supporting quote | Provenance |", "|---|---|---|---|---|"]
        if rec.observed_product:
            L.append(nt_ev_row("Observed ammonia", nt_value(rec.observed_product), rec.observed_product.evidence))
        for g in rec.gas_sources:
            if g.impurity:
                L.append(nt_ev_row(f"{g.label} — {g.impurity.species} ({g.impurity.measured_where.replace('_', ' ')})",
                                 nt_value(g.impurity), g.impurity.evidence))
            else:
                L.append(nt_ev_row(f"{g.label} — impurity", "not measured", g.evidence))
            if g.flow:
                L.append(nt_ev_row(f"{g.label} — flow", f"{g.flow.value:g} {g.flow.unit} ({g.flow.volume_basis})", g.flow.evidence))
        for c in rec.controls:
            L.append(nt_ev_row(f"{c.label} ({c.kind.replace('_', ' ')}, {c.conditions_match})", nt_value(c.measurement),
                             c.measurement.evidence))
        L += ["", "## 3. Calculation (all amounts in nmol of ammonia-equivalent N)", ""]
        for step in res.observed_trail + [s for c in res.contributions for s in c.trail]:
            L.append(f"- **{step.label}**" + (f" — `{step.formula}`" if step.formula else ""))
            L += [f"  - {k}: {v}" for k, v in step.values.items()]
            if step.result:
                L.append(f"  - → {step.result}")
        L += ["", "## 4. Counted and set aside", "", "| Source | Role | Lower | Upper | Why |", "|---|---|---:|---:|---|"]
        for c in res.contributions:
            L.append(f"| {c.label} | {c.role} | {nt_fmt(c.lower_nmol)} | {nt_fmt(c.upper_nmol)} | {c.reason or 'matched, valid bound'} |")
        if not res.contributions:
            L.append("| — | — | | | No source could be bounded. |")
        L += ["", f"**Valid contamination budget:** [{nt_fmt(res.budget_valid_nmol['lower'])}, "
                   f"{nt_fmt(res.budget_valid_nmol['upper'])}] nmol N ({res.budget_label})."]
        if res.observed_nmol:
            L.append(f"**Observed:** [{nt_fmt(res.observed_nmol['lower'])}, {nt_fmt(res.observed_nmol['upper'])}] nmol N.")
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
            items.append(("Observed ammonia", nt_value(rec.observed_product), rec.observed_product.evidence))
        for g in rec.gas_sources:
            items.append((f"{g.label} — impurity", nt_value(g.impurity) if g.impurity else "not measured",
                          g.impurity.evidence if g.impurity else g.evidence))
            if g.flow:
                items.append((f"{g.label} — flow", f"{g.flow.value:g} {g.flow.unit}", g.flow.evidence))
        for c in rec.controls:
            items.append((f"{c.label} (conditions: {c.conditions_match})", nt_value(c.measurement), c.measurement.evidence))
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
            L.append(f"| Observed product | {nt_fmt(res.observed_nmol['lower'])} | {nt_fmt(res.observed_nmol['upper'])} |")
        L.append(f"| Valid contamination budget | {nt_fmt(res.budget_valid_nmol['lower'])} | {nt_fmt(res.budget_valid_nmol['upper'])} |")
        if res.budget_with_scenarios_nmol != res.budget_valid_nmol:
            L.append(f"| Budget incl. what-if measurements (not valid) | {nt_fmt(res.budget_with_scenarios_nmol['lower'])} | "
                     f"{nt_fmt(res.budget_with_scenarios_nmol['upper'])} |")
        L += ["", "## Explanation", ""] + [f"- {e}" for e in res.explanation] + [""]
        L += ["## Contributions", "", "| Source | Category | Role | Lower | Upper | Note |", "|---|---|---|---:|---:|---|"]
        for c in res.contributions:
            L.append(f"| {c.label} | {c.category} | {c.role} | {nt_fmt(c.lower_nmol)} | {nt_fmt(c.upper_nmol)} | {c.reason or ''} |")
        L += ["", "## Unquantified sources", ""]
        L += [f"- **{u.category}:** {u.reason}" for u in res.unquantified] or ["- None"]
        if res.feasible_allocation:
            fa = res.feasible_allocation
            L += ["", "## Illustrative feasible explanation", "", f"_{fa['label']}_", ""]
            L += [f"- {k}: {nt_fmt(v)} nmol N" for k, v in fa["allocation_nmol"].items()]
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

    # ======================= inversion =======================
    # Evidence-requirement inversion: what contamination remains unbounded, and how stringent would new
    # measurements have to be to close the gap?
    # 
    # The audit is solved backwards. Every threshold carries the assumptions it rests on; meeting a
    # threshold can move the verdict to "exceeds the quantified budget", which by itself still does not
    # demonstrate nitrogen reduction.

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

    def nt_nh4_ug_per_L(nmol: float, volume_mL: float) -> float:
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
                        spec["big"], spec["big_unit"] = nt_nh4_ug_per_L(headroom, v), f"µg/L NH4+ in {v:g} mL (alone)"
                        spec["text"] += (f" = {nt_nh4_ug_per_L(headroom, v):.3g} µg/L NH4+ in {v:g} mL; "
                                         f"equal split: {nt_nh4_ug_per_L(share, v):.3g} µg/L.")
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

    # ======================= planner =======================
    # Forward planning with the audit's conservative arithmetic: signal vs. contamination budget before a run,
    # and the feed-gas impurity level that alone would supply a reported rate.
    # 
    # Assumptions (stated wherever results are shown): constant true rate, constant flow, constant delivered
    # impurity level, capture fraction η (η = 1 is the conservative default), and a time-independent matched blank.

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

    # ======================= whatif =======================
    # Hypothetical measurements: show how one new measurement would change the budget and the verdict.
    # 
    # Every value added here carries provenance "hypothetical", so the audit marks its result as hypothetical
    # and reports never present it as evidence.

    KINDS = ("gas", "electrolyte", "electrolyte_nmol", "components")

    nt_NOTE = "HYPOTHETICAL measurement entered to explore its effect; not evidence."

    def add_hypothetical(d: dict, kind: str, value: float, source_id: str | None = None) -> dict:
        """Return a copy of record ``d`` with one hypothetical measurement added.

        gas:         total reactive N (as N) in the gas delivered to the cell, ``value`` in ppb, for ``source_id``
                     (default: the first source without a full reactive-N bound). Replaces that source's bound.
        electrolyte: matched electrolyte blank, ``value`` in µg/L NH4+, same volume and interval as the sample.
        electrolyte_nmol: matched electrolyte blank as an amount, ``value`` in nmol N (no volume needed).
        components:  matched component release, ``value`` in nmol N, same interval as the sample.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not math.isfinite(value) or value < 0:
            raise ValueError("value must be a finite, non-negative number")
        e = copy.deepcopy(d)
        ev = {"provenance": "hypothetical", "confirmed": False, "note": nt_NOTE}
        op = e.get("observed_product") or {}
        if kind == "gas":
            sources = e.get("gas_sources") or []
            if not sources:
                raise ValueError("the record has no gas source to bound")
            unbounded = [s for s in sources if not (s.get("impurity") or {}).get("covers_all_reactive_nitrogen")]
            g = next((s for s in sources if s["id"] == source_id), None) if source_id else (unbounded or sources)[0]
            if g is None:
                raise ValueError(f"no gas source with id '{source_id}'")
            g["impurity"] = {"species": "total reactive N (as N)", "nitrogen_atoms": 1, "unit": "ppb", "lower": 0,
                             "upper": value, "bound_type": "hard_limit", "measured_where": "delivered_to_cell",
                             "covers_all_reactive_nitrogen": True, "evidence": ev}
        elif kind == "electrolyte":
            if not op.get("volume_mL"):
                raise ValueError("the observed product has no volume; a blank concentration cannot become an amount")
            e.setdefault("controls", []).append({
                "id": "hypothetical_blank", "label": "Hypothetical matched electrolyte blank", "kind": "electrolyte_blank",
                "covers": ["electrolyte"], "conditions_match": "matched", "evidence": ev,
                "measurement": {"species": "NH4+", "basis": "concentration_mass", "unit": "µg/L", "lower": 0,
                                "upper": value, "bound_type": "hard_limit", "volume_mL": op["volume_mL"],
                                "interval": op.get("interval"), "evidence": ev}})
        elif kind == "electrolyte_nmol":
            e.setdefault("controls", []).append({
                "id": "hypothetical_blank", "label": "Hypothetical matched electrolyte blank", "kind": "electrolyte_blank",
                "covers": ["electrolyte"], "conditions_match": "matched", "evidence": ev,
                "measurement": {"species": "N", "basis": "amount", "unit": "nmol", "lower": 0, "upper": value,
                                "bound_type": "hard_limit", "interval": op.get("interval"), "evidence": ev}})
        else:
            e.setdefault("controls", []).append({
                "id": "hypothetical_components", "label": "Hypothetical matched component release",
                "kind": "component_release", "covers": ["components"], "conditions_match": "matched", "evidence": ev,
                "measurement": {"species": "N", "basis": "amount", "unit": "nmol", "lower": 0, "upper": value,
                                "bound_type": "hard_limit", "interval": op.get("interval"), "evidence": ev}})
        e["assumptions"] = list(e.get("assumptions", [])) + [f"{nt_NOTE} ({kind}: {value:g})"]
        return e

    def compare(d: dict, changes: list[tuple[str, float, str | None]]) -> dict:
        """Audit before and after one or more hypothetical measurements: [(kind, value, source_id), ...]."""
        before = run_audit(ExperimentRecord(**d))
        hyp = d
        for kind, value, source_id in changes:
            hyp = add_hypothetical(hyp, kind, value, source_id)
        after = run_audit(ExperimentRecord(**hyp))

        def brief(r):
            return {"status": r.status, "status_label": r.status_label, "coverage": r.coverage,
                    "valid_upper_nmol": r.budget_valid_nmol["upper"],
                    "observed_lower_nmol": r.observed_nmol["lower"] if r.observed_nmol else None,
                    "unquantified": [u.model_dump() for u in r.unquantified]}

        return {"changes": [list(c) for c in changes], "before": brief(before), "after": brief(after),
                "changed": before.status != after.status, "record": hyp, "after_result": after,
                "note": ("Hypothetical: shows what such measurements would imply. They are not evidence, and an "
                         "'exceeds' verdict would still not by itself demonstrate nitrogen reduction.")}

    def suggest(rec: ExperimentRecord, req: dict | None) -> list[tuple[str, float, str | None]]:
        """Hypothetical measurements at half of each 'alone' threshold: one per unbounded category."""
        if not req:
            return []
        r3 = lambda x: float(f"{x:.3g}")
        out = []
        for u in req["unknown"]:
            s = u["specs"][0]
            if u["category"] == "gas":
                # one hypothetical bound per unbounded feed, sharing the gas share of the headroom between them
                feeds = [x for x in u["specs"] if "ppb_split" in x]
                out += [("gas", r3(x["ppb_split"] / len(feeds) * 0.5), x["source_id"]) for x in feeds]
            elif u["category"] == "electrolyte" and "µg/L" in s.get("big_unit", ""):
                out.append(("electrolyte", r3(s["big"] / max(len(req["unknown"]), 1) * 0.5), None))
            elif u["category"] == "electrolyte":
                out.append(("electrolyte_nmol", r3(u["max_nmol_equal_split"] * 0.5), None))
            elif u["category"] == "components":
                out.append(("components", r3(u["max_nmol_equal_split"] * 0.5), None))
        return out

    # ======================= validation =======================
    # Benchmark against baselines, plus machine-checked properties run on every record.
    # 
    # Synthetic records test software and reasoning only; published records carry no ground-truth label
    # for whether N2 reduction occurred and are reported, not scored.

    def load_records(data: dict | None = None) -> dict[str, dict]:
        """{"cases/x.json" | "synthetic/y.json": record dict}. ``data`` may hold an embedded copy."""
        if data is not None:
            return {k: v for k, v in data.items() if k.startswith(("cases/", "synthetic/"))}
        out = {}
        for folder in ("cases", "synthetic"):
            for p in sorted((DATA_DIR / folder).glob("*.json")):
                out[f"{folder}/{p.name}"] = json.loads(p.read_text(encoding="utf-8"))
        return out

    def naive_status(d: dict) -> str:
        """Baseline: add every number entered, unknown = 0, ignore matching, coverage and unit checks."""
        try:
            op = d.get("observed_product")
            if not op:
                return "insufficient_information"
            lo = op.get("lower") if op.get("lower") is not None else 0.0
            obs_lo = to_nmol_n(lo, op["unit"], op.get("species") or "NH3", op.get("basis") or "amount",
                               op.get("volume_mL") or 1000, op.get("dilution_factor", 1))[0]
            total = 0.0
            for g in d.get("gas_sources", []):
                imp = g.get("impurity")
                if not imp or imp.get("upper") is None:
                    continue
                if g.get("total_volume_L"):
                    n = gas_moles_from_volume(g["total_volume_L"], g.get("total_volume_temperature_K") or 298.15,
                                              g.get("total_volume_pressure_Pa") or 101325)[0]
                elif g.get("flow"):
                    f = g["flow"]
                    dur = g.get("exposure") or op.get("interval") or {"end_min": 0}
                    dur = dur["end_min"] - dur.get("start_min", 0)
                    n = gas_moles(f["value"], f["unit"], "standard" if f["unit"] in ("sccm", "slpm") else "actual", dur,
                                  f.get("temperature_K") or 298.15, f.get("pressure_Pa") or 101325,
                                  f.get("reference_temperature_K") or f.get("temperature_K") or 298.15,
                                  f.get("reference_pressure_Pa") or 101325)[0]
                else:
                    continue
                total += n * gas_fraction(imp["upper"], imp["unit"]) * imp.get("nitrogen_atoms", 1) * 1e9
            for c in d.get("controls", []):
                m = c["measurement"]
                total += to_nmol_n(m["upper"], m["unit"], m.get("species") or "NH3", m.get("basis") or "amount",
                                   m.get("volume_mL") or 1000)[0]
            return "contamination_not_excluded" if total >= obs_lo else "exceeds_quantified_budget"
        except (UnitError, KeyError, TypeError):
            return "input_inconsistency"

    def checklist_status(d: dict) -> str:
        """Protocol-checklist baseline: 'gas analysis and a blank are present' -> trust the claim; no arithmetic."""
        if not d.get("observed_product"):
            return "insufficient_information"
        has_gas = any((g.get("impurity") or {}).get("upper") is not None for g in d.get("gas_sources", []))
        has_blank = any(c["kind"] in ("system_blank", "electrolyte_blank") for c in d.get("controls", []))
        return "exceeds_quantified_budget" if (has_gas and has_blank) else "insufficient_information"

    def prop_unit_invariance(d, res):
        if res.status == "input_inconsistency" or not res.observed_nmol:
            return None
        e = copy.deepcopy(d)
        e["observed_product"].update(species="N", basis="amount", unit="nmol", lower=res.observed_nmol["lower"],
                                     upper=res.observed_nmol["upper"], nondetect=False, volume_mL=None, dilution_factor=1.0)
        r2 = run_audit(ExperimentRecord(**e))
        return r2.status == res.status and math.isclose(r2.budget_valid_nmol["upper"], res.budget_valid_nmol["upper"],
                                                        rel_tol=1e-12, abs_tol=1e-12)

    def prop_unknown_never_zero(d, res):
        """Deleting any counted piece of evidence must never *create* an 'exceeds' verdict.

        Measurements set aside because they overlap are skipped: deleting one of two conflicting
        measurements removes the conflict itself, which legitimately changes the coverage."""
        set_aside = {c.source_id for c in res.contributions if c.role == "excluded_overlap"}
        checks = []
        for i, g in enumerate(d.get("gas_sources", [])):
            if g["id"] not in set_aside:
                e = copy.deepcopy(d); e["gas_sources"][i]["impurity"] = None; checks.append(e)
        for i, c in enumerate(d.get("controls", [])):
            if c["id"] not in set_aside and not set_aside:
                e = copy.deepcopy(d); e["controls"].pop(i); checks.append(e)
        if not checks:
            return None
        for e in checks:
            r2 = run_audit(ExperimentRecord(**e))
            if r2.status == "exceeds_quantified_budget" and res.status != "exceeds_quantified_budget":
                return False
        return True

    def prop_monotone(d, res):
        """Adding a matched component-release bound of k nmol raises the valid budget by exactly k (no system blank)."""
        if res.status == "input_inconsistency" or any(c["kind"] == "system_blank" for c in d.get("controls", [])):
            return None
        e = copy.deepcopy(d)
        iv = (d.get("observed_product") or {}).get("interval")
        e.setdefault("controls", []).append({"id": "probe", "label": "probe", "kind": "component_release",
                                             "covers": ["components"], "conditions_match": "matched",
                                             "measurement": {"species": "N", "basis": "amount", "unit": "nmol",
                                                             "lower": 0, "upper": 7.0, "interval": iv}})
        r2 = run_audit(ExperimentRecord(**e))
        return math.isclose(r2.budget_valid_nmol["upper"], res.budget_valid_nmol["upper"] + 7.0, rel_tol=1e-12)

    def prop_inversion_shrink(d, rec, res):
        """Apply each 'tighten' requirement at (1 ± 1e-6) and check the valid budget lands on the right side."""
        req = evidence_requirements(rec, res)
        if not req or not req["shrink"]:
            return None
        ok = True
        for row in req["shrink"]:
            if not row["feasible"] or "required_value" not in row:
                continue
            for f, want_below in ((1 - 1e-6, True), (1 + 1e-6, False)):
                e = copy.deepcopy(d)
                g = next((g for g in e.get("gas_sources", []) if g["id"] == row["source_id"]), None)
                if g is not None:
                    g["impurity"]["upper"] = row["required_value"] * f
                    g["impurity"]["lower"] = min(g["impurity"].get("lower") or 0, g["impurity"]["upper"])
                else:
                    c = next(c for c in e["controls"] if c["id"] == row["source_id"])
                    c["measurement"]["upper"] = row["required_value"] * f
                    if c["measurement"].get("lower") is not None:
                        c["measurement"]["lower"] = min(c["measurement"]["lower"], c["measurement"]["upper"])
                r2 = run_audit(ExperimentRecord(**e))
                ok &= (r2.budget_valid_nmol["upper"] < r2.observed_nmol["lower"]) == want_below
        return ok

    def prop_inversion_gas(d, rec, res):
        """Fill an unknown gas source at the 'alone' ppb requirement (1 ± 1e-6) and re-audit."""
        req = evidence_requirements(rec, res)
        if not req or len(req["unknown"]) != 1 or req["unknown"][0]["category"] != "gas":
            return None
        spec = next((s for s in req["unknown"][0]["specs"] if "ppb_alone" in s), None)
        if spec is None or spec["ppb_alone"] <= 0:
            return None
        if any(g.get("impurity") for g in d["gas_sources"] if g["id"] != spec["source_id"]):
            return None
        counted = next((c.upper_nmol for c in res.contributions if c.source_id == spec["source_id"] and c.included), 0.0)
        ok = True
        for f, want in ((1 - 1e-6, "exceeds_quantified_budget"), (1 + 1e-6, "contamination_not_excluded")):
            e = copy.deepcopy(d)
            g = next(g for g in e["gas_sources"] if g["id"] == spec["source_id"])
            g["impurity"] = {"species": "NOx+NH3 (as N)", "unit": "ppb", "lower": 0,
                             "upper": counted / spec["n_gas_mol"] + spec["ppb_alone"] * f,
                             "measured_where": "delivered_to_cell", "covers_all_reactive_nitrogen": True}
            g["eta_upper"], g["eta_basis"] = 1.0, "conservative_upper_bound_1"
            ok &= run_audit(ExperimentRecord(**e)).status == want
        return ok

    PROPERTIES = {
        "Unit invariance (any unit → same verdict & budget)": lambda d, rec, res: prop_unit_invariance(d, res),
        "Unknown is never zero (removing evidence never creates 'exceeds')": lambda d, rec, res: prop_unknown_never_zero(d, res),
        "Monotone budget (adding a bound adds exactly its amount)": lambda d, rec, res: prop_monotone(d, res),
        "Inversion round-trip: tighten requirements flip at the threshold": prop_inversion_shrink,
        "Inversion round-trip: gas ppb requirement flips the verdict": prop_inversion_gas,
    }

    def run_benchmark(records: dict[str, dict] | None = None) -> tuple[list[dict], dict]:
        records = records or load_records()
        rows, props = [], {k: [] for k in PROPERTIES}
        t0 = time.perf_counter()
        for k, d in records.items():
            rec = ExperimentRecord(**d)
            res = run_audit(rec)
            exp = d.get("expected") or {}
            ref = exp.get("budget_upper_nmol")
            rows.append({"case": d["case_id"], "type": d["record_type"], "expected": exp.get("status"),
                         "nitrotrace": res.status, "naive_sum": naive_status(d), "checklist": checklist_status(d),
                         "budget_upper_nmol": res.budget_valid_nmol["upper"],
                         "rel_error": abs(res.budget_valid_nmol["upper"] - ref) / ref if ref else None})
            for name, fn in PROPERTIES.items():
                v = fn(d, rec, res)
                if v is not None:
                    props[name].append((d["case_id"], v))
        syn = [r for r in rows if r["type"] == "synthetic"]
        false_exceeds = lambda col: sum(r[col] == "exceeds_quantified_budget" != r["expected"] for r in syn)
        errs = [r["rel_error"] for r in syn if r["rel_error"] is not None]
        summary = {
            "n_synthetic": len(syn),
            "nitrotrace_correct": sum(r["nitrotrace"] == r["expected"] for r in syn),
            "naive_correct": sum(r["naive_sum"] == r["expected"] for r in syn),
            "checklist_correct": sum(r["checklist"] == r["expected"] for r in syn),
            "false_exceeds_nitrotrace": false_exceeds("nitrotrace"),
            "false_exceeds_naive": false_exceeds("naive_sum"),
            "false_exceeds_checklist": false_exceeds("checklist"),
            "max_rel_error": max(errs) if errs else None, "n_numeric": len(errs),
            "elapsed_s": time.perf_counter() - t0,
            "properties": {k: (sum(v for _, v in vals), len(vals), [c for c, v in vals if not v]) for k, vals in props.items()},
        }
        return rows, summary


    return (
        ExperimentRecord,
        N_OK,
        STATUS_LABELS,
        compare,
        evidence_requirements,
        gas_amount_mol,
        gas_fraction,
        load_records,
        run_audit,
        run_benchmark,
        suggest,
        to_casefile,
        to_csv,
        to_json,
        to_markdown,
        to_review_pack,
    )


@app.cell(hide_code=True)
def evidence_agent():
    # Evidence agent: pasted methods/results text -> passages -> (optional) Claude candidates.
    # Candidates are never trusted: any quote that does not occur verbatim in the text is dropped,
    # and nothing enters an audit until a person types it into the record.

    CLAUDE_MODEL = "claude-opus-5-5"

    TOPICS = {
        "product": r"\b(NH3|NH4\+?|ammonia|ammonium)\b.{0,80}\b(yield|amount|concentration|produced|nmol|µg|ug|mg L|µM)\b",
        "gas_flow": r"\b(flow(?: rate)?|purg\w*|bubbl\w*|sccm|mL\s?min)\b",
        "duration": r"\b(\d+(?:\.\d+)?)\s?(h|hours?|min|minutes?)\b",
        "blank": r"\b(blank|background|control|open[- ]circuit|OCV|argon|Ar\b)",
        "impurity": r"\b(NOx|NO2?|N2O|nitrate|nitrite|ppm|ppb|impurit\w*|contaminat\w*)\b",
        "isotope": r"(15N|¹⁵N|isotop\w*|labell?ed)",
    }
    VALUE_PATTERNS = {
        "flow": r"(\d+(?:\.\d+)?)\s?(mL\s?min[−-]1|mL/min|sccm|L\s?min[−-]1)",
        "duration": r"(\d+(?:\.\d+)?)\s?(h|min)\b",
        "gas_fraction": r"(\d+(?:\.\d+)?)\s?(ppm|ppb|ppbmol|ppmv|mol\s?%)",
        "mass_conc": r"(\d+(?:\.\d+)?)\s?(µg|ug|ng|mg)\s?(L[−-]1|/L|mL[−-]1|/mL)",
        "amount": r"(\d+(?:\.\d+)?)\s?(nmol|µmol|umol|pmol)\b",
        "volume": r"(\d+(?:\.\d+)?)\s?mL\b(?!\s?min)",
    }


    def find_passages(text):
        out = []
        sentences = re.split(r"(?<=[.;])\s+", re.sub(r"\s+", " ", text))
        for i, sent in enumerate(sentences):
            topics = [t for t, rx in TOPICS.items() if re.search(rx, sent, re.I)]
            vals = {k: [" ".join(m.groups()) for m in re.finditer(rx, sent)] for k, rx in VALUE_PATTERNS.items()}
            # Two topics, a standalone impurity/isotope statement, or one topic with a measured value.
            if not (len(topics) >= 2 or set(topics) & {"isotope", "impurity"}
                    or (topics and any(k != "duration" and v for k, v in vals.items()))):
                continue
            out.append({"sentence": i + 1, "topics": ", ".join(topics), "text": sent[:600],
                        "regex_values": "; ".join(f"{k}: {', '.join(v)}" for k, v in vals.items() if v)})
        return out


    CANDIDATE_SCHEMA = {
        "type": "object", "additionalProperties": False, "required": ["candidates"],
        "properties": {"candidates": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["field", "value", "unit", "species", "quote"],
            "properties": {
                "field": {"type": "string", "enum": [
                    "observed_product", "electrolyte_volume", "collection_duration", "gas_flow", "gas_temperature",
                    "gas_pressure", "gas_impurity", "blank_result", "component_release", "detection_limit",
                    "isotope_evidence", "electrode_area"]},
                "value": {"type": ["number", "null"]}, "unit": {"type": ["string", "null"]},
                "species": {"type": ["string", "null"]}, "quote": {"type": "string"}}}}},
    }
    AGENT_SYSTEM = (
        "You extract candidate measurement values from an electrochemical ammonia-synthesis paper. Only report values "
        "explicitly written in the text, each with an exact verbatim supporting quote. If a value is not stated, use null "
        "- never supply a customary flow rate, electrode area, gas purity, temperature, pressure or control result. Keep "
        "the species exactly as written. A scientist reviews every candidate before use."
    )


    def claude_candidates(text, api_key):
        try:
            import anthropic
        except ImportError:
            return {"error": "The `anthropic` package is not installed in this kernel."}
        client = anthropic.Anthropic(api_key=api_key or None)
        msgs = [{"role": "user", "content": f"Paper text:\n\n{text[:150000]}"}]
        try:
            try:
                response = client.beta.messages.create(
                    model=CLAUDE_MODEL, max_tokens=16000, system=AGENT_SYSTEM, messages=msgs,
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                    output_config={"effort": "medium", "format": {"type": "json_schema", "schema": CANDIDATE_SCHEMA}})
            except (TypeError, anthropic.BadRequestError):
                response = client.messages.create(
                    model=CLAUDE_MODEL, max_tokens=16000, system=AGENT_SYSTEM, messages=msgs,
                    output_config={"format": {"type": "json_schema", "schema": CANDIDATE_SCHEMA}})
        except anthropic.AuthenticationError:
            return {"error": "No valid Anthropic credentials: paste an API key above."}
        except anthropic.RateLimitError:
            return {"error": "Rate limited by the Anthropic API; try again shortly."}
        except anthropic.APIStatusError as e:
            return {"error": f"Anthropic API error {e.status_code}: {e.message}"}
        except anthropic.APIConnectionError:
            return {"error": "Could not reach the Anthropic API from this sandbox."}
        if response.stop_reason == "refusal":
            return {"error": "The model declined this request."}
        raw = "".join(b.text for b in response.content if b.type == "text")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"error": "Model output was not valid JSON."}
        hay = re.sub(r"\s+", " ", text)
        kept, dropped = [], 0
        for c in parsed.get("candidates", []):
            if c.get("quote") and re.sub(r"\s+", " ", c["quote"]) in hay:
                kept.append(c | {"quote_verified": "✓ verbatim", "status": "unconfirmed"})
            else:
                dropped += 1
        return {"model": response.model, "candidates": kept, "dropped": dropped}


    return claude_candidates, find_passages


@app.cell(hide_code=True)
def bench_results(DATA, load_records, run_benchmark):
    # Benchmark + machine-checked properties, computed once at startup.
    bench_rows, bench_summary = run_benchmark(load_records(DATA))

    return (bench_summary,)


@app.cell(hide_code=True)
def ui_assets():
    # Front-end source for the NitroTrace interface (ES module + stylesheet), kept as data.
    NT_ESM = r'''// NitroTrace front end. Python (the marimo kernel) owns the engine; this file only draws and asks.
    const S = {
      contamination_not_excluded: { hex: "#c8401f", label: "Contamination not excluded",
        say: "The measured ammonia <em>fits inside</em> the contamination that was actually measured." },
      exceeds_quantified_budget: { hex: "#1d6b4f", label: "Exceeds the quantified budget",
        say: "The measured ammonia is <em>larger</em> than every measured contamination source combined." },
      insufficient_information: { hex: "#a86e12", label: "Insufficient information",
        say: "There is <em>not enough evidence</em> to say where this ammonia came from." },
      input_inconsistency: { hex: "#4f46a5", label: "Input inconsistency",
        say: "These inputs <em>cannot yet be put on one basis</em>." },
    };
    const COV = { quantified: ["Bounded", "#1d6b4f"], not_applicable: ["Not applicable", "#8a8474"],
      specified_synthetic: ["Specified (synthetic)", "#20507a"], partial_species: ["Partly bounded", "#a86e12"], partial_feeds: ["A feed unbounded", "#c8401f"],
      scenario_only: ["What-if only", "#a86e12"], ambiguous_overlap: ["Ambiguous overlap", "#4f46a5"], unknown: ["Unknown", "#c8401f"] };
    const ROLE = { valid_bound: ["counted", "#17150f"], scenario: ["what-if", "#a86e12"], excluded_overlap: ["overlap", "#8a8474"],
      excluded_unmatched: ["not matched", "#8a8474"], excluded_invalid: ["not a measurement", "#8a8474"] };
    const TABS = [["audit", "Audit"], ["settle", "What would settle it"], ["design", "Design before you run"],
      ["stress", "Claim stress-test"], ["evidence", "Evidence agent"], ["report", "Report"], ["method", "Method & checks"]];
    let CW = 760;
    const R = 8.314462618, REF = { "0 °C": 273.15, "20 °C": 293.15, "25 °C": 298.15 };

    const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
    function num(x, p = 3) {
      if (x == null || Number.isNaN(x)) return "—";
      if (!Number.isFinite(x)) return "∞";
      if (x === 0) return "0";
      const a = Math.abs(x);
      if (a >= 1e5 || a < 1e-3) {
        const [m, e] = x.toExponential(p - 1).split("e");
        return `${m}×10${String(+e).split("").map((c) => SUP[c]).join("")}`;
      }
      return String(+x.toPrecision(p));
    }
    const hrs = (h) => (h < 1 ? `${num(h * 60, 2)} min` : `${num(h, 3)} h`);
    const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
    const pill = (st) => `<span class="nt-pill" style="color:${S[st].hex}"><i></i>${S[st].label}</span>`;
    function download(name, text, type) {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([text], { type }));
      a.download = name; document.body.appendChild(a); a.click(); a.remove();
    }
    function logTicks(lo, hi) {
      const out = [];
      for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
        for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo && v <= hi) out.push([v, m === 1]); }
      return out;
    }
    const tickLabel = (v) => {
      const e = Math.round(Math.log10(v));
      if (Math.abs(v / 10 ** e - 1) < 1e-9 && (e > 3 || e < -2)) return `10${String(e).split("").map((c) => SUP[c]).join("")}`;
      return num(v, 2);
    };

    function md(src) {
      const lines = src.split("\n"); let out = "", list = false, table = null;
      const inline = (s) => s.split(/(`[^`]+`)/).map((part) => part.startsWith("`") && part.endsWith("`") && part.length > 1
        ? `<code>${esc(part.slice(1, -1))}</code>`
        : esc(part).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|\s)_([^_]+?)_(?=$|[\s.,;:)])/g, "$1<i>$2</i>")).join("");
      const flush = () => {
        if (list) { out += "</ul>"; list = false; }
        if (table) {
          out += `<table class="nt-tbl"><tr>${table[0].map((c) => `<th>${inline(c)}</th>`).join("")}</tr>` +
            table.slice(2).map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("") + "</table>";
          table = null;
        }
      };
      for (const raw of lines) {
        const l = raw.trimEnd();
        if (l.startsWith("|")) { const cells = l.slice(1, -1).split("|").map((c) => c.trim()); (table ||= []).push(cells); continue; }
        if (/^\s*- /.test(l)) { if (table) flush(); if (!list) { out += "<ul>"; list = true; } out += `<li>${inline(l.replace(/^\s*- /, ""))}</li>`; continue; }
        flush();
        if (l.startsWith("### ")) out += `<h3>${inline(l.slice(4))}</h3>`;
        else if (l.startsWith("## ")) out += `<h2>${inline(l.slice(3))}</h2>`;
        else if (l.startsWith("# ")) out += `<h1>${inline(l.slice(2))}</h1>`;
        else if (l.startsWith("> ")) out += `<blockquote>${inline(l.slice(2))}</blockquote>`;
        else if (l === "---") out += "<hr>";
        else if (l) out += `<p>${inline(l)}</p>`;
      }
      flush(); return out;
    }

    /* ---------------------------------------------------------------- audit ---- */
    function ruler(a) {
      const rows = [];
      if (a.observed) rows.push({ k: "obs", label: "Measured ammonia", lo: a.observed.lower, hi: a.observed.upper, kind: "obs" });
      for (const c of a.contributions) rows.push({ label: c.label, lo: c.lower, hi: c.upper, kind: c.role, sub: ROLE[c.role][0] });
      rows.push({ label: "Valid contamination budget", lo: a.valid.lower, hi: a.valid.upper, kind: "budget", rule: true });
      if (a.with_scen.upper !== a.valid.upper) rows.push({ label: "…including what-ifs (not valid)", lo: a.with_scen.lower, hi: a.with_scen.upper, kind: "budget_scen" });
      const pos = rows.flatMap((r) => [r.lo, r.hi]).filter((v) => v > 0);
      let lo = pos.length ? Math.min(...pos) / 5 : 1e-2, hi = pos.length ? Math.max(...pos) * 5 : 10;
      if (hi / lo < 100) { const m = Math.sqrt(hi * lo); lo = m / 10; hi = m * 10; }
      const W = Math.max(520, CW), L = Math.min(300, W * 0.36), PR = W - 12, RH = 40, TOP = 34, H = TOP + rows.length * RH + 36;
      const x = (v) => L + (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * (PR - L);
      const color = { obs: "#1d6b4f", valid_bound: "#17150f", scenario: "#a86e12", excluded_overlap: "#8a8474",
        excluded_unmatched: "#8a8474", excluded_invalid: "#8a8474", budget: "#20507a", budget_scen: "#a86e12" };
      let g = `<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke="#a86e12" stroke-width="2.2"/></pattern>`;
      for (const [k, c] of Object.entries(color))
        g += `<linearGradient id="z-${k}" x1="0" x2="1"><stop offset="0" stop-color="${c}" stop-opacity="0"/><stop offset=".12" stop-color="${c}"/></linearGradient>`;
      g += "</defs>";
      for (const [v, major] of logTicks(lo, hi)) {
        g += `<line x1="${x(v)}" x2="${x(v)}" y1="${TOP - 6}" y2="${H - 30}" stroke="${major ? "#d8d1c1" : "#e6e0d2"}"/>`;
        if (major || hi / lo < 3e3) g += `<text class="tick" x="${x(v)}" y="${H - 14}" text-anchor="middle">${tickLabel(v)}</text>`;
      }
      rows.forEach((r, i) => {
        const y = TOP + i * RH, cy = y + RH / 2, c = color[r.kind];
        if (r.rule) g += `<line x1="0" x2="${PR}" y1="${y + 1}" y2="${y + 1}" stroke="#17150f"/>`;
        g += `<text class="lab" x="0" y="${cy - 2}" ${r.kind === "obs" || r.kind === "budget" ? 'font-weight="600"' : ""}>${esc(r.label.length > L / 6.6 ? r.label.slice(0, Math.floor(L / 6.6) - 1) + "…" : r.label)}</text>`;
        g += `<text class="sub" x="0" y="${cy + 12}">[${num(r.lo)}, ${num(r.hi)}] nmol N${r.sub ? " · " + r.sub : ""}</text>`;
        const x0 = r.lo > 0 ? x(r.lo) : L, x1 = Math.max(x(r.hi), x0 + 3), h = r.kind === "obs" || r.kind === "budget" ? 14 : 10;
        const delay = `style="animation-delay:${i * 45}ms"`;
        if (r.hi <= 0) { g += `<text class="sub" x="${L + 4}" y="${cy + 4}">0 — nothing counted</text>`; return; }
        if (r.kind.startsWith("excluded")) g += `<rect class="nt-grow" ${delay} x="${x0}" y="${cy - h / 2}" width="${x1 - x0}" height="${h}" fill="none" stroke="${c}" stroke-dasharray="3 3"/>`;
        else if (r.kind === "scenario" || r.kind === "budget_scen") g += `<rect class="nt-grow" ${delay} x="${x0}" y="${cy - h / 2}" width="${x1 - x0}" height="${h}" fill="url(#hatch)" stroke="${c}" stroke-width=".8"/>`;
        else g += `<rect class="nt-grow" ${delay} x="${x0}" y="${cy - h / 2}" width="${x1 - x0}" height="${h}" fill="${r.lo > 0 ? c : `url(#z-${r.kind})`}"/>`;
        if (r.lo === r.hi) g += `<path d="M${x1} ${cy - 8} l6 8 l-6 8 l-6 -8z" fill="${r.kind.startsWith("excluded") ? "#f2efe7" : c}" stroke="${c}"/>`;
      });
      if (a.observed) {
        const ox = x(a.observed.lower);
        g += `<line x1="${ox}" x2="${ox}" y1="${TOP - 10}" y2="${H - 30}" stroke="#1d6b4f" stroke-dasharray="4 3"/>
              <text x="${ox}" y="${TOP - 16}" text-anchor="middle" class="sub" style="fill:#1d6b4f">measured lower bound</text>`;
      }
      return `<div class="nt-ruler"><svg viewBox="0 0 ${W} ${H}">${g}</svg></div>`;
    }

    function track(model, a) {
      const wrap = el(`<div class="nt-track" tabindex="0"></div>`);
      const vm = a.vmap || [];
      const pos = (k) => ((Math.log10(k) + 3) / 6) * 100;
      vm.forEach(([k, st], i) => {
        const l = i ? (pos(vm[i - 1][0]) + pos(k)) / 2 : 0, r = i < vm.length - 1 ? (pos(k) + pos(vm[i + 1][0])) / 2 : 100;
        wrap.appendChild(el(`<div class="seg" style="left:${l}%;width:${r - l + 0.05}%;background:${S[st].hex}"></div>`));
      });
      for (let e = -3; e <= 3; e++) wrap.appendChild(el(`<div class="ax" style="left:${pos(10 ** e)}%">×${num(10 ** e)}</div>`));
      const thumb = el(`<div class="thumb"></div>`); wrap.appendChild(thumb);
      const statusAt = (k) => vm.reduce((b, v) => (Math.abs(Math.log10(v[0] / k)) < Math.abs(Math.log10(b[0] / k)) ? v : b), vm[0] || [1, a.status]);
      const place = (k) => { thumb.style.left = pos(k) + "%"; thumb.dataset.v = `×${num(k)} · ${S[statusAt(k)[1]].label.toLowerCase()}`; };
      place(a.scale || 1);
      const kFromX = (cx) => { const r = wrap.getBoundingClientRect(); const f = Math.min(1, Math.max(0, (cx - r.left) / r.width)); return 10 ** (f * 6 - 3); };
      let drag = false, last = a.scale || 1;
      const commit = (k) => { const snap = statusAt(k)[0]; place(snap); model.set("scale", snap); model.save_changes(); };
      wrap.addEventListener("pointerdown", (e) => { drag = true; wrap.setPointerCapture(e.pointerId); last = kFromX(e.clientX); place(last); });
      wrap.addEventListener("pointermove", (e) => { if (drag) { last = kFromX(e.clientX); place(last); } });
      wrap.addEventListener("pointerup", () => { if (drag) { drag = false; commit(last); } });
      wrap.addEventListener("keydown", (e) => {
        const i = vm.findIndex((v) => v[0] === statusAt(last)[0]);
        if (e.key === "ArrowRight" && i < vm.length - 1) { last = vm[i + 1][0]; commit(last); }
        if (e.key === "ArrowLeft" && i > 0) { last = vm[i - 1][0]; commit(last); }
      });
      return wrap;
    }

    function auditPage(model, a) {
      if (!a || !a.status) return el(`<div class="nt-main"><p class="nt-empty">Loading…</p></div>`);
      const m = el(`<div class="nt-main nt-fade"></div>`);
      const st = S[a.status];
      m.innerHTML = `
        <div class="nt-kicker">${pill(a.status)}<span class="caps">${esc(a.record_type)}</span><span class="mono cite">${esc(a.case_id)}</span>
          ${a.scale !== 1 ? `<span class="caps" style="color:#a86e12">what-if: signal ×${num(a.scale)}</span>` : ""}</div>
        <h1 class="nt-verdict">${st.say}</h1>
        <p class="nt-head">${esc(a.headline)}</p>
        <p class="nt-title">${esc(a.title)}${a.sources.length ? ` — <span class="cite">${a.sources.map(esc).join(" · ")}</span>` : ""}</p>
        ${a.hypothetical ? `<div class="nt-hyp">This record contains <b>hypothetical</b> inputs. The result shows what they would imply; it is not evidence.</div>` : ""}
        ${a.error ? `<div class="nt-err">${esc(a.error)}</div>` : ""}`;
      const sec = (title, note, body) => { const s = el(`<section class="nt-sec"><header><h3>${title}</h3><span class="caps">${note}</span></header></section>`); if (typeof body === "string") s.insertAdjacentHTML("beforeend", body); else s.appendChild(body); return s; };
      m.appendChild(sec("The comparison", "nmol of ammonia-equivalent N · log scale", ruler(a)));
      const t = el(`<div></div>`);
      t.appendChild(el(`<p style="margin:0;color:var(--ink-2);font-size:13px">Drag along the band to rescale the measured ammonia. The colour under every point is the verdict NitroTrace would reach there, with all evidence held fixed.</p>`));
      t.appendChild(track(model, a));
      m.appendChild(sec("If the signal were different", "verdict map · ×10⁻³ to ×10³", t));
      m.appendChild(sec("Coverage", a.budget_label, `<div class="nt-cols3">${Object.entries(a.coverage).map(([k, v]) => {
        const [lab, c] = COV[v] || [v, "#8a8474"]; const u = a.unquantified.find((q) => q.category === k);
        return `<div class="nt-cov"><span class="caps">${k}</span><div class="v">${lab}</div><div class="bar" style="background:${c}"></div><p>${esc(u ? u.reason : "Bounded by matched, valid measurements.")}</p></div>`; }).join("")}</div>`));
      m.appendChild(sec("Reasoning", "", `<div class="nt-cols"><div><span class="caps">Why this verdict</span><ol class="nt-list">${a.explanation.map((e) => `<li>${esc(e)}</li>`).join("")}</ol></div>
        <div><span class="caps">Never treated as zero</span><ol class="nt-list">${a.unquantified.map((u) => `<li><b>${esc(u.category)}</b> — ${esc(u.reason)}</li>`).join("") || '<li class="nt-empty">Every category is bounded.</li>'}</ol></div></div>`));
      m.appendChild(sec("Contributions", `${a.contributions.length} sources`, a.contributions.length ? `<table class="nt-tbl"><tr><th>Source</th><th>Category</th><th>Role</th><th style="text-align:right">Lower</th><th style="text-align:right">Upper</th><th>Why</th></tr>
        ${a.contributions.map((c) => `<tr class="${c.included ? "" : "off"}"><td>${esc(c.label)}</td><td>${esc(c.category)}</td><td><span class="nt-role" style="color:${ROLE[c.role][1]}">${ROLE[c.role][0]}</span></td><td class="n">${num(c.lower)}</td><td class="n">${num(c.upper)}</td><td style="font-size:12px;color:var(--ink-2)">${esc(c.reason || "")}</td></tr>`).join("")}</table>` : `<p class="nt-empty">No source could be bounded.</p>`));
      m.appendChild(sec("Next controls", "each quotes the protocol it comes from", a.recs.map((r, i) => `<div class="nt-proto"><div class="k">${String(i + 1).padStart(2, "0")}</div><div>
        <h4>${esc(r.title)}</h4><p><span class="caps">Trigger</span> ${esc(r.trigger)}</p><p>${esc(r.text)}</p>
        ${r.refs.map((q) => `<div class="quote">“${esc(q.quote)}”</div><div class="cite">[${esc(q.source_id)}] ${esc(q.location)}</div>`).join("")}</div></div>`).join("") || `<p class="nt-empty">None triggered.</p>`));
      const det = el(`<section class="nt-sec"></section>`);
      det.innerHTML = `
        <details><summary>Calculation trail</summary><div class="trail">${a.trail.map((s) => `<b>${esc(s.label)}${s.formula ? ` — <span class="mono">${esc(s.formula)}</span>` : ""}</b>${Object.entries(s.values).map(([k, v]) => `<div>${esc(k)}: ${esc(v)}</div>`).join("")}${s.result ? `<div>→ ${esc(s.result)}</div>` : ""}`).join("")}</div></details>
        <details><summary>Assumptions (${a.assumptions.length})</summary><ol class="nt-list">${a.assumptions.map((x) => `<li>${esc(x)}</li>`).join("") || "<li>None</li>"}</ol></details>
        <details><summary>Issues (${a.issues.length})</summary><ol class="nt-list">${a.issues.map((i) => `<li><span class="mono">${esc(i.severity)} · ${esc(i.code)}</span> — ${esc(i.message)}</li>`).join("") || "<li>None</li>"}</ol></details>
        <details class="edit"><summary>Edit the record</summary><p style="font-size:12.5px;color:var(--ink-2);margin:0 0 8px">Every value carries species, basis, unit, bound type and evidence. Missing means unknown.</p>
          <textarea class="nt-field" rows="18"></textarea><div style="margin:10px 0 16px"><button class="nt-btn">Re-run audit</button><button class="nt-btn ghost">Reset</button></div></details>`;
      const ta = det.querySelector("textarea"); ta.value = model.get("record_json");
      const [run, reset] = det.querySelectorAll(".edit button");
      run.onclick = () => { model.set("edit_json", { text: ta.value, n: Date.now() }); model.save_changes(); };
      reset.onclick = () => { model.set("selected", ""); model.save_changes(); model.set("selected", a.key); model.save_changes(); };
      m.appendChild(det);
      return m;
    }

    /* --------------------------------------------------------------- settle ---- */
    function settlePage(model, a) {
      const m = el(`<div class="nt-main nt-fade"></div>`);
      if (!a || !a.status) return m;
      const r = a.req;
      if (!r) { m.innerHTML = `<div class="nt-kicker">${pill(a.status)}</div><h1 class="nt-verdict">Nothing to invert <em>yet</em>.</h1><p class="nt-head">The audit needs an observed product and inputs without errors before it can be solved backwards.</p>${a.error ? `<div class="nt-err">${esc(a.error)}</div>` : ""}`; return m; }
      const h = r.headroom_nmol, unknown = r.unknown.length;
      const say = h > 0 ? (unknown ? `For the signal to exceed the contamination budget, everything still unbounded must stay below <em>${num(h)} nmol N</em> in total.`
        : `Every modelled source is bounded, and the signal clears them by <em>×${num(r.observed_lower_nmol / Math.max(r.valid_upper_nmol, 1e-300))}</em>.`)
        : `The counted contamination already reaches the signal. <em>Something measured has to come down.</em>`;
      m.innerHTML = `<div class="nt-kicker">${pill(a.status)}<span class="caps">evidence-requirement inversion</span></div>
        <h1 class="nt-verdict">${say}</h1>
        <p class="nt-head">The audit solved backwards: which measurement, at what level, would change the verdict, and what each threshold assumes. Each threshold is checked automatically: applied at ±10⁻⁶ it flips the verdict on the right side. Meeting them would not by itself demonstrate nitrogen reduction.</p>
        <section class="nt-sec"><header><h3>The gap</h3><span class="caps">nmol N</span></header>
          <div class="nt-gap"><div><span class="caps">Measured, lower bound</span><b>${num(r.observed_lower_nmol)}</b><span>what the experiment shows at least</span></div>
          <div><span class="caps">Valid budget, upper bound</span><b>${num(r.valid_upper_nmol)}</b><span>counted, matched contamination at most</span></div>
          <div><span class="caps">Headroom</span><b style="color:${h > 0 ? "#1d6b4f" : "#c8401f"}">${h > 0 ? "+" : ""}${num(h)}</b><span>room left for every source not yet measured</span></div></div>
          <div class="nt-note"><b>Contamination floor: ${num(r.floor_nmol)} nmol N${r.floor_in_record_units != null ? ` (${num(r.floor_in_record_units)} ${esc(r.record_unit)})` : ""}.</b>
            An observation must be strictly above this before today's controls could call it "above budget".${r.floor_nmol === 0 ? " It is zero because nothing matched and valid is counted yet: any signal beats an empty budget, which is exactly why that comparison proves nothing." : ""}
            ${r.floor_rate_mol_s_cm2 ? ` As a rate: ${num(r.floor_rate_mol_s_cm2)} mol s⁻¹ cm⁻².` : ""}</div></section>`;
      const specs = [];
      for (const u of r.unknown) for (const s of u.specs)
        specs.push({ cat: u.category, state: u.state, title: { gas: "Measure reactive nitrogen in the gas that reaches the cell", electrolyte: "Run a matched electrolyte / whole-system blank", components: "Measure release from cell components under the experiment's conditions" }[u.category], text: s.text, big: s.big, unit: s.big_unit, as: s.assumptions || [] });
      for (const s of r.shrink) specs.push({ cat: "tighten", title: `Tighten: ${s.source}`, text: s.text, big: s.feasible ? s.big : null, unit: s.feasible ? s.big_unit : "not possible alone", as: s.assumptions || [] });
      const sec = el(`<section class="nt-sec"><header><h3>${specs.length ? "Specifications" : "Nothing left to specify"}</h3><span class="caps">${unknown ? `${unknown} unbounded categor${unknown > 1 ? "ies" : "y"} · alone / equal split` : ""}</span></header></section>`);
      specs.forEach((s, i) => sec.appendChild(el(`<div class="nt-spec"><div class="k">${String(i + 1).padStart(2, "0")}</div><div><span class="caps">${esc(s.cat)}${s.state ? " · " + esc(s.state.replace(/_/g, " ")) : ""}</span><h4>${esc(s.title)}</h4><p>${esc(s.text)}</p>${s.as.length ? `<ul class="nt-as">${s.as.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}</div>
        <div class="big">${s.big != null ? `<b>&lt; ${num(s.big)}</b>` : `<b style="color:#c8401f">—</b>`}<span>${esc(s.unit || "")}</span></div></div>`)));
      m.appendChild(sec);
      if (r.notes.length || r.caveats.length) m.appendChild(el(`<section class="nt-sec">${r.notes.map((n) => `<div class="nt-note">${esc(n)}</div>`).join("")}${r.caveats.map((n) => `<div class="nt-note caveat">${esc(n)}</div>`).join("")}</section>`));
      m.appendChild(whatifPanel(model, a));
      return m;
    }

    function whatifPanel(model, a) {
      const fields = a.whatif_fields || [];
      const sec = el(`<section class="nt-sec"><header><h3>Try hypothetical measurements</h3><span class="caps">not evidence · shows what they would imply</span></header>
        <p style="font-size:13.5px;color:var(--ink-2);margin:0 0 12px;max-width:860px">Enter measurements you could make. NitroTrace adds them as labelled <i>hypothetical</i> inputs and re-runs the audit. Values are prefilled at half of each equal-split threshold.</p>
        <div class="nt-wi"></div><div style="margin-top:12px"><button class="nt-btn run">Re-run with these</button></div><div class="wo"></div></section>`);
      const box = sec.querySelector(".nt-wi");
      fields.forEach((f, i) => {
        const row = el(`<label class="nt-wirow ${f.disabled ? "off" : ""}"><input type="checkbox" ${f.on && !f.disabled ? "checked" : ""} ${f.disabled ? "disabled" : ""}>
          <span>${esc(f.label)}${f.disabled ? `<small>${esc(f.disabled)}</small>` : ""}</span>
          <span class="u">≤</span><input class="nt-field v" type="number" min="0" step="any" value="${f.value ?? ""}" ${f.disabled ? "disabled" : ""}><span class="u">${esc(f.unit)}</span></label>`);
        row.dataset.i = i; box.appendChild(row);
      });
      sec.querySelector(".run").onclick = () => {
        const changes = [];
        box.querySelectorAll(".nt-wirow").forEach((row) => {
          const f = fields[+row.dataset.i], on = row.querySelector("input[type=checkbox]").checked, v = row.querySelector(".v").value;
          if (on && v !== "") changes.push({ kind: f.kind, value: +v, source_id: f.source_id || null });
        });
        model.set("whatif_req", { changes, n: Date.now() }); model.save_changes();
        sec.querySelector(".wo").innerHTML = `<p class="nt-empty">Re-running…</p>`;
      };
      const show = () => {
        const w = model.get("whatif_res") || {}, out = sec.querySelector(".wo");
        if (w.error) { out.innerHTML = `<div class="nt-err" style="margin-top:14px">${esc(w.error)}</div>`; return; }
        if (!w.before) { out.innerHTML = ""; return; }
        const cov = Object.keys(w.before.coverage).filter((k) => w.before.coverage[k] !== w.after.coverage[k])
          .map((k) => `<li><b>${esc(k)}</b>: ${esc(w.before.coverage[k].replace(/_/g, " "))} → ${esc(w.after.coverage[k].replace(/_/g, " "))}</li>`).join("");
        out.innerHTML = `<div class="nt-hyp" style="margin-top:16px"><span class="caps" style="color:inherit">hypothetical result</span>
          <div class="nt-ba">${pill(w.before.status)}<span class="arr">→</span>${pill(w.after.status)}</div>
          <div class="mono" style="font-size:13px">valid budget ${num(w.before.valid_upper_nmol)} → <b>${num(w.after.valid_upper_nmol)}</b> nmol N · measured lower bound ${num(w.after.observed_lower_nmol)} nmol N</div>
          ${cov ? `<ul class="nt-as" style="margin-top:8px">${cov}</ul>` : ""}
          ${w.after.unquantified.map((u) => `<div class="nt-note">Still unbounded — <b>${esc(u.category)}</b>: ${esc(u.reason)}</div>`).join("")}
          <p style="font-size:12.5px;margin:8px 0 0">${esc(w.note)}</p></div>`;
      };
      model.on("change:whatif_res", show); show();
      return sec;
    }

    /* --------------------------------------------------------------- design ---- */
    const D = { mode: "flow", rate: -11, area: 1, flow: Math.log10(20), ppb: 2, eta: 1, hours: 4, blank: 2, vol: 30, margin: 3, fixedL: 0.5, ref: "0 °C" };
    function plan(p, override = {}) {
      const q = { ...p, ...override };
      const T = REF[q.ref], s = 10 ** q.rate * q.area * 1e9, ppb = 10 ** q.ppb, flow = 10 ** q.flow;
      const blank = q.blank * (q.vol / 1000) * 1e-6 / 18.038 * 1e9;
      let g = 0, b = blank;
      if (q.mode === "flow") g = (101325 * flow * 1e-6 / 60 / (R * T)) * ppb * q.eta;
      else b += (101325 * q.fixedL / 1000 / (R * 298.15)) * ppb * q.eta;
      const Ts = q.hours * 3600, tstar = s <= q.margin * g ? null : (q.margin * b) / (s - q.margin * g);
      return { s, g, b, Ts, tstar, end: (s * Ts) / (g * Ts + b), rmin: (q.margin * (g + b / Ts)) / (q.area * 1e9) };
    }
    function designPage(model) {
      const m = el(`<div class="nt-main nt-fade"><div class="nt-kicker"><span class="caps">forward planning · same arithmetic as the audit</span></div>
        <h1 class="nt-answer"></h1><p class="nt-head sub2"></p>
        <section class="nt-sec"><header><h3>Plan the experiment</h3><span class="caps">everything updates as you move</span></header>
        <div class="nt-design"><div class="ctl"></div><div><div class="nt-chart c1"></div><div style="height:18px"></div>
          <span class="caps maplab"></span><div class="nt-map"><canvas width="260" height="150"></canvas><svg></svg></div>
          <p style="font-size:12.5px;color:var(--ink-2)">Click anywhere on the map to move the design there. Green: the signal beats the margin by the end of the run; red: it doesn't. Ink line: exactly the margin.</p></div></div></section>
        <section class="nt-sec"><header><h3>Why run time can't rescue a purged cell</h3><span class="caps">closed form</span></header>
          <div class="nt-cols"><p style="font-size:14px;color:var(--ink-2);margin:0">With gas flowing continuously, both the catalyst's ammonia and the feed gas's reactive nitrogen accumulate linearly in time, so the ratio settles at <span class="mono">S/C → r·A·R·T / (P·Q·x·η)</span>. It doesn't depend on how long you run. Only lower flow, cleaner gas or a closed gas volume moves it. With a fixed gas volume, gas contamination is capped and the ratio grows with time.</p>
          <p style="font-size:14px;color:var(--ink-2);margin:0"><b>Assumes</b> a constant true rate, flow, delivered impurity level and capture fraction η, and a fixed matched blank. Break-even time for a required margin <span class="mono">m</span>: <span class="mono">t* = m·b / (s − m·g)</span>, where <span class="mono">s</span> is the signal rate, <span class="mono">g</span> the gas contamination rate and <span class="mono">b</span> the fixed blank. If <span class="mono">s ≤ m·g</span> there is no t*. η = 1 is a labelled conservative scenario.</p></div></section></div>`);
      const ctl = m.querySelector(".ctl");
      const seg = el(`<div><div class="nt-seg mode"><button data-v="flow">Continuous purge</button><button data-v="fixed">Fixed gas volume</button></div><br><div class="nt-seg ref">${Object.keys(REF).map((k) => `<button data-v="${k}">${k} ref.</button>`).join("")}</div></div>`);
      ctl.appendChild(seg);
      const sliders = [
        ["rate", "Expected true NRR rate", -14, -7, 0.05, (v) => `${num(10 ** v)} mol s⁻¹ cm⁻²`],
        ["area", "Electrode area", 0.1, 25, 0.1, (v) => `${num(v)} cm²`],
        ["flow", "Gas flow", 0, 2.7, 0.01, (v) => `${num(10 ** v)} sccm`, "flow"],
        ["fixedL", "Fixed gas volume", 0.05, 5, 0.05, (v) => `${num(v)} L`, "fixed"],
        ["ppb", "Reactive N in delivered gas", -1, 4, 0.05, (v) => `${num(10 ** v)} ppb`],
        ["eta", "η, fraction captured", 0.01, 1, 0.01, (v) => num(v)],
        ["hours", "Run time", 0.25, 48, 0.25, (v) => hrs(v)],
        ["blank", "Matched blank, upper bound", 0, 50, 0.1, (v) => `${num(v)} µg/L NH₄⁺`],
        ["vol", "Electrolyte volume", 5, 200, 1, (v) => `${num(v)} mL`],
        ["margin", "Required margin", 1, 10, 0.5, (v) => `×${num(v)}`],
      ];
      const rows = {};
      for (const [k, lab, a, b, st, f, only] of sliders) {
        const r = el(`<div class="nt-ctl"><div class="row"><label>${lab}</label><span class="val"></span></div><input type="range" min="${a}" max="${b}" step="${st}" value="${D[k]}"></div>`);
        r.dataset.only = only || ""; rows[k] = r; ctl.appendChild(r);
        r.querySelector("input").addEventListener("input", (e) => { D[k] = +e.target.value; update(); });
      }
      seg.querySelectorAll(".mode button").forEach((b) => b.onclick = () => { D.mode = b.dataset.v; update(); });
      seg.querySelectorAll(".ref button").forEach((b) => b.onclick = () => { D.ref = b.dataset.v; update(); });
      const cv = m.querySelector("canvas"), ov = m.querySelector(".nt-map svg");
      cv.parentElement.addEventListener("click", (e) => {
        const r = cv.getBoundingClientRect(), fx = (e.clientX - r.left) / r.width, fy = 1 - (e.clientY - r.top) / r.height;
        if (D.mode === "flow") D.flow = +(fx * 2.7).toFixed(2); else D.fixedL = +(0.05 + fx * 4.95).toFixed(2);
        D.ppb = +(-1 + fy * 5).toFixed(2);
        rows[D.mode === "flow" ? "flow" : "fixedL"].querySelector("input").value = D.mode === "flow" ? D.flow : D.fixedL;
        rows.ppb.querySelector("input").value = D.ppb; update();
      });
      function update() {
        seg.querySelectorAll(".mode button").forEach((b) => b.classList.toggle("on", b.dataset.v === D.mode));
        seg.querySelectorAll(".ref button").forEach((b) => b.classList.toggle("on", b.dataset.v === D.ref));
        for (const [k, , , , , f] of sliders) { rows[k].querySelector(".val").textContent = f(D[k]); rows[k].style.display = rows[k].dataset.only && rows[k].dataset.only !== D.mode ? "none" : ""; }
        const p = plan(D), H = D.hours;
        const ans = m.querySelector(".nt-answer"), sub = m.querySelector(".sub2");
        if (p.tstar === null) { ans.innerHTML = `<em>Not under these assumptions.</em> The feed gas supplies nitrogen ${num(p.g / p.s)}× as fast as the catalyst would fix it; with constant flow, purity and η, running longer doesn't change that ratio.`; }
        else if (p.tstar / 3600 > H) { ans.innerHTML = `Not within ${hrs(H)}. The signal reaches ×${num(D.margin)} the budget only after <em>${hrs(p.tstar / 3600)}</em>.`; }
        else { ans.innerHTML = `Defensible after <em>${hrs(p.tstar / 3600)}</em>. By the end of the run the signal is ${num(p.end)}× the contamination budget.`; }
        sub.textContent = `Smallest true rate this design can defend in ${hrs(H)}: ${num(p.rmin)} mol s⁻¹ cm⁻² (at margin ×${num(D.margin)}).`;
        // time chart
        const W = Math.max(460, CW - 340), Hh = 300, L = 56, B = 34, T0 = 14, Rr = 16, N = 120;
        const ts = Array.from({ length: N }, (_, i) => (i / (N - 1)) * p.Ts);
        const sig = ts.map((t) => p.s * t), con = ts.map((t) => p.g * t + p.b);
        const raw = Math.max(sig[N - 1], D.margin * con[N - 1]) * 1.05 || 1;
        const step = (() => { const r = raw / 4, p10 = 10 ** Math.floor(Math.log10(r)); return [1, 2, 2.5, 5, 10].map((k) => k * p10).find((k) => k >= r); })();
        const ymax = Math.ceil(raw / step) * step;
        const X = (t) => L + (t / p.Ts) * (W - L - Rr), Y = (v) => Hh - B - (v / ymax) * (Hh - B - T0);
        const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${X(ts[i]).toFixed(1)} ${Y(v).toFixed(1)}`).join("");
        let g = "";
        for (let v = 0; v <= ymax + 1e-9; v += step) { g += `<line x1="${L}" x2="${W - Rr}" y1="${Y(v)}" y2="${Y(v)}" stroke="#e1dacb"/><text class="tick" x="${L - 8}" y="${Y(v) + 3}" text-anchor="end">${num(v, 2)}</text>`; }
        for (let i = 0; i <= 4; i++) { const t = (p.Ts * i) / 4; g += `<text class="tick" x="${X(t)}" y="${Hh - B + 16}" text-anchor="middle">${hrs(t / 3600)}</text>`; }
        g += `<path d="${path(con)}L${X(p.Ts)} ${Y(0)}L${X(0)} ${Y(0)}Z" fill="#c8401f" fill-opacity=".09"/>`;
        g += `<path d="${path(con)}" fill="none" stroke="#c8401f" stroke-width="1.8"/>`;
        g += `<path d="${path(con.map((v) => v * D.margin))}" fill="none" stroke="#c8401f" stroke-width="1" stroke-dasharray="5 4"/>`;
        g += `<path d="${path(sig)}" fill="none" stroke="#1d6b4f" stroke-width="2.6"/>`;
        g += `<text x="${W - Rr}" y="${Y(sig[N - 1]) - 8}" text-anchor="end" class="axl" style="fill:#1d6b4f">true signal</text>`;
        g += `<text x="${W - Rr}" y="${Y(D.margin * con[N - 1]) - 8}" text-anchor="end" class="axl" style="fill:#c8401f">×${num(D.margin)} contamination budget</text>`;
        if (p.tstar !== null && p.tstar <= p.Ts) g += `<line x1="${X(p.tstar)}" x2="${X(p.tstar)}" y1="${T0}" y2="${Hh - B}" stroke="#17150f" stroke-dasharray="3 3"/><text class="axl" x="${X(p.tstar) + 6}" y="${T0 + 10}">t* = ${hrs(p.tstar / 3600)}</text>`;
        g += `<line x1="${L}" x2="${W - Rr}" y1="${Hh - B}" y2="${Hh - B}" stroke="#17150f"/><text class="axl" x="${L}" y="${T0 - 2}">nmol N</text>`;
        m.querySelector(".c1").innerHTML = `<svg viewBox="0 0 ${W} ${Hh}">${g}</svg>`;
        // design map
        const ctx = cv.getContext("2d"), cw = cv.width, ch = cv.height, img = ctx.createImageData(cw, ch);
        const val = [];
        for (let j = 0; j < ch; j++) for (let i = 0; i < cw; i++) {
          const fx = i / (cw - 1), fy = 1 - j / (ch - 1);
          const o = D.mode === "flow" ? { flow: fx * 2.7, ppb: -1 + fy * 5 } : { fixedL: 0.05 + fx * 4.95, ppb: -1 + fy * 5 };
          val.push(Math.log10(plan(D, o).end / D.margin));
        }
        for (let j = 0; j < ch; j++) for (let i = 0; i < cw; i++) {
          const v = val[j * cw + i], k = (j * cw + i) * 4, a = Math.min(1, Math.abs(v) / 2.5) * 0.62 + 0.06;
          const c = v >= 0 ? [29, 107, 79] : [200, 64, 31];
          const col = c.map((x, n) => Math.round([242, 239, 231][n] * (1 - a) + x * a));
          img.data.set([...col, 255], k);
        }
        ctx.putImageData(img, 0, 0);
        const px = D.mode === "flow" ? D.flow / 2.7 : (D.fixedL - 0.05) / 4.95, py = 1 - (D.ppb + 1) / 5;
        m.querySelector(".maplab").textContent = D.mode === "flow" ? "Design space · gas flow × feed purity (log–log)" : "Design space · fixed gas volume × feed purity";
        const xt = D.mode === "flow" ? [1, 10, 100, 500].map((v) => [Math.log10(v) / 2.7, `${v} sccm`]) : [0.05, 1, 2, 3, 4, 5].map((v) => [(v - 0.05) / 4.95, `${v} L`]);
        ov.setAttribute("viewBox", "0 0 100 100"); ov.setAttribute("preserveAspectRatio", "none");
        const at = (fx, fy) => Math.log10(plan(D, D.mode === "flow" ? { flow: fx * 2.7, ppb: -1 + fy * 5 } : { fixedL: 0.05 + fx * 4.95, ppb: -1 + fy * 5 }).end / D.margin);
        const pts = [];
        for (let i = 0; i <= 80; i++) {
          const fx = i / 80; let a0 = 0, b0 = 1;
          if (at(fx, 0) < 0 || at(fx, 1) > 0) continue;
          for (let k = 0; k < 30; k++) { const mid = (a0 + b0) / 2; if (at(fx, mid) >= 0) a0 = mid; else b0 = mid; }
          pts.push(`${(fx * 100).toFixed(2)},${((1 - a0) * 100).toFixed(2)}`);
        }
        ov.innerHTML = (pts.length > 1 ? `<polyline points="${pts.join(" ")}" fill="none" stroke="#17150f" stroke-width="2" vector-effect="non-scaling-stroke"/>` : "") + `<line x1="${px * 100}" x2="${px * 100}" y1="0" y2="100" stroke="#17150f" stroke-width=".25" vector-effect="non-scaling-stroke"/><line y1="${py * 100}" y2="${py * 100}" x1="0" x2="100" stroke="#17150f" stroke-width=".25" vector-effect="non-scaling-stroke"/>`;
        const lab = m.querySelector(".nt-map");
        lab.querySelectorAll(".lbl").forEach((n) => n.remove());
        const put = (html, css) => { const d = el(`<div class="lbl mono" style="position:absolute;font-size:10px;color:#4b473d;${css}">${html}</div>`); lab.appendChild(d); };
        xt.forEach(([f, t]) => put(t, `left:${f * 100}%;bottom:4px;transform:translateX(${f > 0.9 ? -100 : 4}px)`));
        [0.1, 10, 1000].forEach((v) => put(`${v} ppb`, `left:4px;top:${(1 - (Math.log10(v) + 1) / 5) * 100}%;transform:translateY(-50%)`));
        put(`● you`, `left:${px * 100}%;top:${py * 100}%;transform:translate(-4px,-50%);color:#17150f;font-weight:600;font-size:12px`);
      }
      update();
      return m;
    }

    /* --------------------------------------------------------------- stress ---- */
    const ST = { lod: 300, ref: "0 °C", rows: [
      { claim: "Example A", rate: 1e-11, unit: "mol s⁻¹ cm⁻²", area: 1, load: 0, flow: 20 },
      { claim: "Example B", rate: 5, unit: "µg(NH3) h⁻¹ mg_cat⁻¹", area: 1, load: 0.5, flow: 30 },
      { claim: "Example C", rate: 1e-10, unit: "mol s⁻¹ cm⁻²", area: 2, load: 0, flow: 10 },
      { claim: "Example D", rate: 0.5, unit: "µg(NH3) h⁻¹ cm⁻²", area: 1, load: 0, flow: 50 },
    ] };
    const UNITS = { "mol s⁻¹ cm⁻²": (v, a) => v * a, "nmol s⁻¹ cm⁻²": (v, a) => v * 1e-9 * a,
      "µg(NH3) h⁻¹ cm⁻²": (v, a) => (v * 1e-6) / 17.031 / 3600 * a, "µg(NH3) h⁻¹ mg_cat⁻¹": (v, a, l) => (v * 1e-6) / 17.031 / 3600 * l * a };
    function stressPage() {
      const m = el(`<div class="nt-main nt-fade"><div class="nt-kicker"><span class="caps">claim stress-test · continuously purged cells</span></div>
        <h1 class="nt-answer"></h1><p class="nt-head">A reported rate becomes a single number: the reactive-nitrogen level in the feed gas that would supply <i>all</i> of it (η = 1). If that level is below what the gas analysis could bound, the claim can't be separated from feed-gas contamination, whatever else was done. The rows are illustrative inputs; type in any paper's numbers.</p>
        <section class="nt-sec"><header><h3>Claims</h3><span class="caps"><span class="refseg"></span></span></header><div class="tbl"></div>
          <button class="nt-btn ghost add" style="margin-top:10px">+ Add claim</button></section>
        <section class="nt-sec"><header><h3>Equivalent feed-gas contamination</h3><span class="caps">ppb N · log scale</span></header>
          <div class="nt-ctl" style="max-width:420px"><div class="row"><label>Your gas analysis can bound total reactive N down to</label><span class="val lodv"></span></div><input class="lod" type="range" min="-2" max="3" step="0.05"></div>
          <div class="nt-chart plot"></div></section></div>`);
      const tbl = m.querySelector(".tbl"), lod = m.querySelector(".lod");
      lod.value = Math.log10(ST.lod);
      lod.oninput = () => { ST.lod = 10 ** +lod.value; draw(); };
      const refseg = el(`<div class="nt-seg" style="margin:0">${Object.keys(REF).map((k) => `<button data-v="${k}">${k} ref.</button>`).join("")}</div>`);
      m.querySelector(".refseg").appendChild(refseg);
      refseg.querySelectorAll("button").forEach((b) => b.onclick = () => { ST.ref = b.dataset.v; draw(); });
      m.querySelector(".add").onclick = () => { ST.rows.push({ claim: `Claim ${ST.rows.length + 1}`, rate: 1e-11, unit: "mol s⁻¹ cm⁻²", area: 1, load: 0, flow: 20 }); table(); draw(); };
      const eq = (r) => { const T = REF[ST.ref]; const ndot = 101325 * r.flow * 1e-6 / 60 / (R * T); return (UNITS[r.unit](+r.rate, +r.area, +r.load) / ndot) * 1e9; };
      function table() {
        tbl.innerHTML = `<table class="nt-tbl nt-stbl"><tr><th>Claim</th><th>Rate</th><th>Unit</th><th>Area cm²</th><th>Loading mg cm⁻²</th><th>Flow sccm</th><th style="text-align:right">≡ feed ppb</th><th></th></tr>
          ${ST.rows.map((r, i) => `<tr data-i="${i}"><td><input data-k="claim" value="${esc(r.claim)}"></td><td><input data-k="rate" value="${r.rate}"></td>
            <td><select data-k="unit">${Object.keys(UNITS).map((u) => `<option ${u === r.unit ? "selected" : ""}>${u}</option>`).join("")}</select></td>
            <td><input data-k="area" value="${r.area}"></td><td><input data-k="load" value="${r.load}"></td><td><input data-k="flow" value="${r.flow}"></td>
            <td class="n eq"></td><td><button class="nt-x" title="remove">×</button></td></tr>`).join("")}</table>`;
        tbl.querySelectorAll("tr[data-i]").forEach((tr) => {
          const i = +tr.dataset.i;
          tr.querySelectorAll("[data-k]").forEach((inp) => inp.addEventListener("input", () => { ST.rows[i][inp.dataset.k] = inp.dataset.k === "claim" || inp.dataset.k === "unit" ? inp.value : +inp.value; draw(); }));
          tr.querySelector(".nt-x").onclick = () => { ST.rows.splice(i, 1); table(); draw(); };
        });
      }
      function draw() {
        refseg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === ST.ref));
        m.querySelector(".lodv").textContent = `${num(ST.lod)} ppb`;
        const vals = ST.rows.map(eq);
        tbl.querySelectorAll("tr[data-i]").forEach((tr, i) => { const v = vals[i]; tr.querySelector(".eq").innerHTML = `<span style="color:${v > ST.lod ? "#1d6b4f" : "#c8401f"}">${num(v)}</span>`; });
        const bad = vals.filter((v) => !(v > ST.lod)).length;
        m.querySelector(".nt-answer").innerHTML = bad ? `<em>${bad} of ${vals.length}</em> claims sit below your gas-analysis bound: even a clean measurement at that sensitivity couldn't rule out feed gas as their sole source.` : `All ${vals.length} claims sit above your gas-analysis bound. A clean measurement at that sensitivity could rule out feed gas as their sole source; other sources still need bounding.`;
        const W = Math.max(520, CW), L = 150, Rr = 20, RH = 34, H = 40 + ST.rows.length * RH + 30, lo = 1e-3, hi = 1e6;
        const X = (v) => L + (Math.log10(Math.min(Math.max(v, lo), hi)) + 3) / 9 * (W - L - Rr);
        let g = `<rect x="${L}" y="20" width="${X(ST.lod) - L}" height="${H - 50}" fill="#c8401f" fill-opacity=".07"/>`;
        for (let e = -3; e <= 6; e++) g += `<line x1="${X(10 ** e)}" x2="${X(10 ** e)}" y1="20" y2="${H - 30}" stroke="#e1dacb"/><text class="tick" x="${X(10 ** e)}" y="${H - 12}" text-anchor="middle">${tickLabel(10 ** e)}</text>`;
        g += `<line x1="${X(ST.lod)}" x2="${X(ST.lod)}" y1="12" y2="${H - 30}" stroke="#c8401f" stroke-width="1.5"/><text class="axl" x="${X(ST.lod) + 6}" y="16" style="fill:#c8401f">your detection bound</text>`;
        ST.rows.forEach((r, i) => {
          const y = 40 + i * RH, v = vals[i], ok = v > ST.lod;
          g += `<text class="axl" x="0" y="${y + 4}">${esc(r.claim)}</text><line x1="${L}" x2="${X(v)}" y1="${y}" y2="${y}" stroke="#c6bda9" stroke-dasharray="2 3"/>
            <circle cx="${X(v)}" cy="${y}" r="6" fill="${ok ? "#1d6b4f" : "#c8401f"}"/><text class="tick" x="${X(v) + 10}" y="${y + 3}">${num(v)} ppb</text>`;
        });
        m.querySelector(".plot").innerHTML = `<svg viewBox="0 0 ${W} ${H}">${g}</svg>`;
      }
      table(); draw();
      return m;
    }

    /* ------------------------------------------------------------- evidence ---- */
    function evidencePage(model) {
      const res = model.get("agent_res") || {};
      const m = el(`<div class="nt-main nt-fade"><div class="nt-kicker"><span class="caps">the agent proposes · the scientist confirms</span></div>
        <h1 class="nt-answer">Paste a paper's methods. Get evidence with <em>verbatim quotes</em>, or nothing.</h1>
        <p class="nt-head">A deterministic pass finds passages about product amounts, gas flow, duration, blanks, impurities and isotopes. With a key, Claude proposes structured candidates; any candidate whose quote isn't found verbatim in your text is discarded. Nothing enters an audit until you type it into the record.</p>
        <section class="nt-sec"><header><h3>Source text</h3><span class="caps">${esc(res.model || "claude-opus-5-5")}</span></header>
          <textarea class="nt-field txt" rows="12" placeholder="Paste the experimental section and results…"></textarea>
          <div style="display:grid;grid-template-columns:1fr auto;gap:10px;margin-top:10px;align-items:center">
            <input class="nt-field key" type="password" placeholder="Anthropic API key (optional; stays in this kernel)">
            <div><button class="nt-btn ghost find">Find passages</button><button class="nt-btn ask">Ask Claude</button></div></div></section>
        <div class="out"></div></div>`);
      const txt = m.querySelector(".txt"), key = m.querySelector(".key");
      txt.value = model.get("agent_text") || "";
      txt.oninput = () => { model.set("agent_text", txt.value); model.save_changes(); };
      const go = (llm) => { model.set("agent_req", { text: txt.value, key: key.value, llm, n: Date.now() }); model.save_changes(); m.querySelectorAll(".nt-btn").forEach((b) => (b.disabled = true)); m.querySelector(".out").innerHTML = `<p class="nt-empty" style="margin-top:20px">${llm ? "Claude is reading…" : "Searching…"}</p>`; };
      m.querySelector(".find").onclick = () => go(false);
      m.querySelector(".ask").onclick = () => go(true);
      const out = m.querySelector(".out");
      if (res.error) out.appendChild(el(`<div class="nt-err" style="margin-top:20px">${esc(res.error)}</div>`));
      if (res.candidates) out.appendChild(el(`<section class="nt-sec"><header><h3>Candidates</h3><span class="caps">${res.candidates.length} verified · ${res.dropped} dropped</span></header>
        <table class="nt-tbl"><tr><th>Field</th><th style="text-align:right">Value</th><th>Unit</th><th>Species</th><th>Quote</th></tr>${res.candidates.map((c) => `<tr><td>${esc(c.field)}</td><td class="n">${c.value ?? "—"}</td><td>${esc(c.unit ?? "")}</td><td>${esc(c.species ?? "")}</td><td><span class="serif" style="font-style:italic;font-size:14px">“${esc(c.quote)}”</span></td></tr>`).join("")}</table></section>`));
      if (res.passages) out.appendChild(el(`<section class="nt-sec"><header><h3>Passages</h3><span class="caps">${res.passages.length} found</span></header>
        ${res.passages.map((p) => `<div class="nt-proto"><div class="k">§${p.sentence}</div><div><span class="caps">${esc(p.topics)}</span><div class="quote" style="font-size:15px">${esc(p.text)}</div>${p.regex_values ? `<div class="cite">${esc(p.regex_values)}</div>` : ""}</div></div>`).join("") || '<p class="nt-empty">No relevant passages.</p>'}</section>`));
      return m;
    }

    /* --------------------------------------------------------------- report ---- */
    function reportPage(model, a) {
      const m = el(`<div class="nt-main nt-fade"></div>`);
      if (!a || !a.report) { m.innerHTML = `<p class="nt-empty">${esc(a && a.error ? a.error : "No report yet.")}</p>`; return m; }
      const bar = el(`<div style="margin-bottom:22px;display:flex;flex-wrap:wrap;gap:8px"><button class="nt-btn">Case file (.md)</button><button class="nt-btn ghost">Independent-review pack (.md)</button><button class="nt-btn ghost">Audit report (.md)</button><button class="nt-btn ghost">JSON</button><button class="nt-btn ghost">CSV</button></div>`);
      const [b1, b2, b3, b4, b5] = bar.querySelectorAll("button");
      b1.onclick = () => download(`${a.case_id}_casefile.md`, a.report.casefile, "text/markdown");
      b2.onclick = () => download(`${a.case_id}_review_pack.md`, a.report.review, "text/markdown");
      b3.onclick = () => download(`${a.case_id}_audit.md`, a.report.md, "text/markdown");
      b4.onclick = () => download(`${a.case_id}_audit.json`, a.report.json, "application/json");
      b5.onclick = () => download(`${a.case_id}_contributions.csv`, a.report.csv, "text/csv");
      m.appendChild(bar);
      m.appendChild(el(`<p class="nt-head" style="font-size:13px;margin:0 0 14px">The case file is the complete, source-linked account: values as reported with their quotes, what was counted or set aside and why, the calculation, the qualified conclusion, what would close the gap, and the review status. The review pack is a checklist for an independent domain researcher.</p>`));
      m.appendChild(el(`<article class="nt-doc">${md(a.report.casefile)}</article>`));
      return m;
    }

    /* --------------------------------------------------------------- method ---- */
    function methodPage(model) {
      const b = model.get("bench") || {};
      const props = Object.entries(b.properties || {});
      const m = el(`<div class="nt-main nt-fade"><div class="nt-kicker"><span class="caps">method · evidence · limits</span></div>
        <h1 class="nt-verdict">A signal above <em>unmeasured</em> contamination is not yet a result.</h1>
        <div class="nt-essay"><p>Electrochemical N₂-to-NH₃ studies report nanomoles of ammonia, and ammonia and NOx in gases, electrolytes, membranes and ionomers produce the same signal. Protocols say which controls to run. NitroTrace computes whether the controls that were actually run exclude contamination. When they don't, it works out which measurement, at what level, would.</p>
        <p>Every input carries species, basis, unit, bound type, matching status and a quoted source. Missing evidence is <i>unknown</i>, never zero. Supplier purity, upstream-of-purifier gas and non-matched blanks are shown but never counted. Overlapping blanks need an explicit coverage model. Bounds are combined as intervals, nothing is fitted, and the verdict follows five ordered rules.</p></div>
        <section class="nt-sec"><header><h3>Benchmark</h3><span class="caps">${b.n_synthetic} synthetic records with independent expected values</span></header>
          <div class="nt-kpis"><div><b class="nt-big">${b.nitrotrace_correct}/${b.n_synthetic}</b><span>NitroTrace verdicts correct</span></div>
          <div><b class="nt-big" style="color:#8a8474">${b.naive_correct}/${b.n_synthetic}</b><span>naive "sum every number, unknown = 0"</span></div>
          <div><b class="nt-big" style="color:#8a8474">${b.checklist_correct}/${b.n_synthetic}</b><span>protocol checklist, no arithmetic</span></div>
          <div><b class="nt-big" style="color:#c8401f">${b.false_exceeds_nitrotrace}<span style="font-size:.4em;color:#8a8474;display:inline"> vs ${b.false_exceeds_naive} vs ${b.false_exceeds_checklist}</span></b><span>false "exceeds" claims (NitroTrace / naive / checklist)</span></div></div>
          <p style="font-size:13px;color:var(--ink-2)">Max relative error against ${b.n_numeric} hand-computed references: <span class="mono">${num(b.max_rel_error)}</span>. Reproduces Choi et al. 2020 Table 1 within 0.05 %. The whole suite, including the property checks below, runs live in this kernel in <span class="mono">${num(b.elapsed_ms)} ms</span>.</p></section>
        <section class="nt-sec"><header><h3>Machine-checked properties</h3><span class="caps">run on every record, every launch</span></header>
          ${props.map(([k, [p, n, f]]) => `<div class="nt-prop"><span class="${p === n ? "ok" : "bad"}"></span><span>${esc(k)}</span><span class="mono">${p}/${n}</span></div>`).join("")}
          <p style="font-size:12px;color:var(--ink-3)">Scope: unit invariance skips records with input errors. "Unknown is never zero" skips measurements set aside as overlaps, because deleting one of two conflicting measurements removes the conflict itself. Monotonicity skips records with a whole-system blank.</p></section>
        <section class="nt-sec"><header><h3>Review status</h3><span class="caps">stated plainly</span></header>
          <p style="font-size:14.5px;color:var(--ink-2);max-width:820px">The published records have been checked against their sources by the NitroTrace authors only. They have <b>not yet been independently reviewed</b> by a domain researcher. Every case file says so, and the Report page exports an independent-review checklist for each record.</p></section>
        <section class="nt-sec"><header><h3>The trap, on real data</h3><span class="caps">Bragulla et al. 2026 · open circuit</span></header>
          <p style="font-size:14.5px;color:var(--ink-2);max-width:820px">Ammonium measured with <i>no electrochemistry at all</i> exceeds every separately quantified material release. The naive sum and the checklist both call it "exceeds". NitroTrace calls it "insufficient", because the in-cell component release was never measured under matched conditions. The authors trace the difference to the Nafion ionomer.</p></section>
        <section class="nt-sec"><header><h3>Limits</h3><span class="caps">stated, not hidden</span></header><ol class="nt-list">
          <li>Aqueous electrochemical N₂→NH₃ only; independent interval arithmetic; no kinetics.</li>
          <li>η = 1 is a labelled conservative scenario, not a fitted efficiency.</li>
          <li>Synthetic records test software and reasoning, not attribution in real laboratories. Published records carry no ground truth and are not scored.</li>
          <li>The control rulebook has not yet been reviewed by an independent chemist. Outputs describe what the entered evidence permits, not the original studies.</li></ol></section>
        <section class="nt-sec"><header><h3>Sources</h3><span class="caps"></span></header><ol class="nt-list">
          <li>[S1] Andersen et al., <i>Nature</i> 570, 504–508 (2019).</li><li>[S2] Choi et al., <i>Nat. Commun.</i> 11, 5546 (2020).</li>
          <li>[S3] Bragulla et al., <i>ChemSusChem</i> 19, e202501974 (2026).</li><li>[S4] Hu et al., <i>Natl. Sci. Rev.</i> (2025).</li></ol></section></div>`);
      return m;
    }

    /* ----------------------------------------------------------------- shell ---- */
    function render({ model, el: host }) {
      if (!document.getElementById("nt-fonts")) {
        const l = document.createElement("link"); l.id = "nt-fonts"; l.rel = "stylesheet";
        l.href = "https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap";
        document.head.appendChild(l);
      }
      const b = model.get("bench") || {};
      const propOk = Object.values(b.properties || {}).reduce((s, [p]) => s + p, 0), propN = Object.values(b.properties || {}).reduce((s, [, n]) => s + n, 0);
      const root = el(`<div class="nt">
        <div class="nt-mast"><div class="nt-mark">Nitro<i>Trace</i></div>
          <div class="nt-tag">Given an ammonia measurement and the controls actually run: <b>what contamination remains unbounded, and how stringent must new measurements be to close the gap?</b></div>
          <div class="nt-figs"><div class="nt-fig"><b>${b.nitrotrace_correct}/${b.n_synthetic}</b><span>verdicts</span></div><div class="nt-fig"><b>${propOk}/${propN}</b><span>checks</span></div><div class="nt-fig"><b>0.05%</b><span>vs Choi 2020</span></div></div></div>
        <nav class="nt-nav"></nav><div class="nt-body"></div></div>`);
      host.appendChild(root);
      const nav = root.querySelector(".nt-nav"), body = root.querySelector(".nt-body");
      TABS.forEach(([k, lab], i) => { const bt = el(`<button data-k="${k}"><span class="n">0${i + 1}</span>${lab}</button>`); bt.onclick = () => { model.set("tab", k); model.save_changes(); draw(); }; nav.appendChild(bt); });
      let q = "";
      function index() {
        const cat = model.get("catalog") || [], sel = model.get("selected");
        const box = el(`<aside class="nt-index"><input placeholder="Search records…"></aside>`);
        const inp = box.querySelector("input"); inp.value = q;
        const list = el(`<div></div>`); box.appendChild(list);
        const fill = () => {
          list.innerHTML = "";
          for (const [grp, name] of [["published", "Published records"], ["synthetic", "Synthetic benchmark"]]) {
            const items = cat.filter((c) => c.group === grp && (c.title + c.id).toLowerCase().includes(q.toLowerCase()));
            if (!items.length) continue;
            list.appendChild(el(`<h6 class="caps">${name} · ${items.length}</h6>`));
            for (const c of items) {
              const r = el(`<div class="nt-rec ${c.key === sel ? "on" : ""}" title="${esc(c.status_label)}"><div class="sq" style="background:${S[c.status].hex}"></div><div><div class="t">${esc(c.title)}</div><div class="s">${esc(c.sub)}</div></div></div>`);
              r.onclick = () => { model.set("selected", c.key); model.set("scale", 1); model.save_changes(); };
              list.appendChild(r);
            }
          }
        };
        inp.oninput = () => { q = inp.value; fill(); };
        fill(); return box;
      }
      function draw() {
        const tab = model.get("tab") || "audit", a = model.get("audit");
        nav.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x.dataset.k === tab));
        const withIndex = ["audit", "settle", "report"].includes(tab);
        body.className = "nt-body" + (withIndex ? "" : " solo");
        const scroll = body.querySelector(".nt-index")?.scrollTop || 0;
        const full = root.clientWidth || 1000;
        CW = Math.round((withIndex && full > 900 ? full - 250 : full) - 70);
        body.innerHTML = "";
        if (withIndex) { const ix = index(); body.appendChild(ix); ix.scrollTop = scroll; }
        const page = { audit: () => auditPage(model, a), settle: () => settlePage(model, a), design: () => designPage(model),
          stress: () => stressPage(), evidence: () => evidencePage(model), report: () => reportPage(model, a), method: () => methodPage(model) }[tab];
        body.appendChild(page());
      }
      model.on("change:audit", () => { if (["audit", "settle", "report"].includes(model.get("tab") || "audit")) draw(); });
      model.on("change:tab", draw);
      model.on("change:agent_res", () => { if (model.get("tab") === "evidence") draw(); });
      let lastW = 0;
      new ResizeObserver(() => { const w = root.clientWidth; if (w && Math.abs(w - lastW) > 40) { lastW = w; draw(); } }).observe(root);
      draw();
    }
    export default { render };
    '''

    NT_CSS2 = r'''.nt {
      --paper: #f2efe7; --paper-2: #e9e4d8; --card: #faf8f3; --ink: #17150f; --ink-2: #4b473d; --ink-3: #8a8474;
      --rule: #d8d1c1; --rule-2: #c6bda9;
      --red: #c8401f; --green: #1d6b4f; --ochre: #a86e12; --violet: #4f46a5; --blue: #20507a;
      --serif: "Instrument Serif", "Iowan Old Style", Georgia, serif;
      --sans: "Inter Tight", "Inter", system-ui, sans-serif;
      --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", monospace;
      width: 100%; container-type: inline-size; container-name: nt;
      background: var(--paper); color: var(--ink); font-family: var(--sans); font-size: 14px; line-height: 1.5;
      border: 1px solid var(--rule); border-radius: 4px; overflow: hidden;
      font-feature-settings: "ss01", "cv11";
    }
    .nt * { box-sizing: border-box; }
    .nt button, .nt input, .nt textarea, .nt select { font: inherit; color: inherit; }
    .nt .mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }
    .nt .serif { font-family: var(--serif); font-weight: 400; }
    .nt .caps { font-size: 10.5px; letter-spacing: .14em; text-transform: uppercase; font-weight: 600; color: var(--ink-3); }

    /* masthead */
    .nt-mast { display: grid; grid-template-columns: auto 1fr auto; align-items: end; gap: 28px;
      padding: 22px 32px 14px; border-bottom: 1px solid var(--ink); }
    .nt-mark { font-family: var(--serif); font-size: 40px; line-height: .9; letter-spacing: -.01em; }
    .nt-mark i { font-style: italic; }
    .nt-mark sub { font-size: .45em; vertical-align: baseline; position: relative; top: .05em; }
    .nt-tag { color: var(--ink-2); max-width: 520px; font-size: 13px; padding-bottom: 3px; }
    .nt-figs { display: flex; gap: 22px; padding-bottom: 3px; }
    .nt-fig { text-align: right; }
    .nt-fig b { display: block; font-family: var(--mono); font-size: 17px; font-weight: 500; }
    .nt-fig span { font-size: 10.5px; color: var(--ink-3); letter-spacing: .06em; text-transform: uppercase; }
    .nt-nav { display: flex; gap: 0; padding: 0 32px; border-bottom: 1px solid var(--rule); background: var(--paper); }
    .nt-nav button { background: none; border: 0; padding: 12px 0 11px; margin-right: 26px; cursor: pointer;
      color: var(--ink-3); font-weight: 500; border-bottom: 2px solid transparent; transition: color .15s; }
    .nt-nav button:hover { color: var(--ink); }
    .nt-nav button.on { color: var(--ink); border-bottom-color: var(--ink); }
    .nt-nav button .n { font-family: var(--mono); font-size: 11px; margin-right: 6px; color: var(--ink-3); }

    /* layout */
    .nt-body { display: grid; grid-template-columns: 250px minmax(0, 1fr); min-height: 760px; }
    .nt-body.solo { grid-template-columns: 1fr; }
    .nt-index { border-right: 1px solid var(--rule); background: var(--paper-2); padding: 16px 0; max-height: 1400px; overflow: auto; }
    .nt-index input { width: calc(100% - 32px); margin: 0 16px 12px; padding: 7px 10px; border: 1px solid var(--rule-2);
      background: var(--card); border-radius: 3px; outline: none; }
    .nt-index input:focus { border-color: var(--ink); }
    .nt-index h6 { margin: 14px 16px 6px; }
    .nt-rec { display: grid; grid-template-columns: 10px 1fr; gap: 10px; padding: 7px 16px; cursor: pointer; align-items: start;
      border-left: 3px solid transparent; }
    .nt-rec:hover { background: #e2dccd; }
    .nt-rec.on { background: var(--card); border-left-color: var(--ink); }
    .nt-rec .sq { width: 9px; height: 9px; margin-top: 5px; }
    .nt-rec .t { font-size: 12.5px; line-height: 1.35; color: var(--ink); }
    .nt-rec .s { font-size: 11px; color: var(--ink-3); }
    .nt-main { padding: 28px 34px 48px; min-width: 0; }

    /* verdict */
    .nt-kicker { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }
    .nt-pill { display: inline-flex; align-items: center; gap: 7px; padding: 2px 9px 2px 7px; border: 1px solid currentColor;
      border-radius: 2px; font-size: 10.5px; letter-spacing: .12em; text-transform: uppercase; font-weight: 700; }
    .nt-pill i { width: 8px; height: 8px; background: currentColor; display: block; }
    .nt-verdict { font-family: var(--serif); font-size: clamp(32px, 4.2cqi, 50px); line-height: 1.04; letter-spacing: -.012em; margin: 4px 0 12px; max-width: 980px; }
    .nt-verdict em { font-style: italic; }
    .nt-head { color: var(--ink-2); max-width: 860px; font-size: 15px; }
    .nt-title { font-size: 13px; color: var(--ink-2); }

    /* sections */
    .nt-sec { margin-top: 34px; }
    .nt-sec > header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px;
      border-top: 1px solid var(--ink); padding-top: 8px; margin-bottom: 14px; }
    .nt-sec > header h3 { margin: 0; font-family: var(--serif); font-weight: 400; font-size: 24px; }
    .nt-sec > header .caps { white-space: nowrap; }
    .nt-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
    .nt-cols3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; border: 1px solid var(--rule); }
    .nt-cov { padding: 14px 16px; border-right: 1px solid var(--rule); background: var(--card); }
    .nt-cov:last-child { border-right: 0; }
    .nt-cov .v { font-family: var(--serif); font-size: 22px; margin-top: 2px; }
    .nt-cov .bar { height: 3px; margin: 10px 0 8px; }
    .nt-cov p { margin: 0; font-size: 12.5px; color: var(--ink-2); }
    .nt-list { margin: 0; padding: 0; list-style: none; counter-reset: li; }
    .nt-list li { position: relative; padding: 8px 0 8px 30px; border-bottom: 1px dotted var(--rule-2); font-size: 13.5px; }
    .nt-list li::before { counter-increment: li; content: counter(li, decimal-leading-zero); position: absolute; left: 0; top: 9px;
      font-family: var(--mono); font-size: 11px; color: var(--ink-3); }

    /* ruler */
    .nt-ruler svg { width: 100%; display: block; overflow: visible; }
    .nt-ruler .lab { font-size: 12px; fill: var(--ink); }
    .nt-ruler .sub { font-size: 10.5px; fill: var(--ink-3); font-family: var(--mono); }
    .nt-ruler .tick { font-family: var(--mono); font-size: 10px; fill: var(--ink-3); }
    .nt-grow { transform-box: fill-box; transform-origin: left center; animation: nt-grow .7s cubic-bezier(.2,.8,.2,1) both; }
    @keyframes nt-grow { from { transform: scaleX(0); } }
    .nt-fade { animation: nt-fade .5s ease both; }
    @keyframes nt-fade { from { opacity: 0; transform: translateY(4px); } }

    /* scale track */
    .nt-track { position: relative; height: 54px; margin: 6px 0 0; user-select: none; touch-action: none; cursor: ew-resize; }
    .nt-track .seg { position: absolute; top: 14px; height: 14px; }
    .nt-track .thumb { position: absolute; top: 4px; width: 2px; height: 34px; background: var(--ink); transform: translateX(-1px); }
    .nt-track .thumb::after { content: attr(data-v); position: absolute; top: -2px; left: 8px; font-family: var(--mono);
      font-size: 11px; background: var(--ink); color: var(--paper); padding: 1px 6px; white-space: nowrap; }
    .nt-track .ax { position: absolute; top: 33px; font-family: var(--mono); font-size: 10px; color: var(--ink-3); transform: translateX(-50%); }

    /* table */
    .nt-tbl { width: 100%; border-collapse: collapse; font-size: 12.5px; }
    .nt-tbl th { text-align: left; font-weight: 600; font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3);
      border-bottom: 1px solid var(--ink); padding: 6px 10px 6px 0; }
    .nt-tbl td { padding: 7px 10px 7px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
    .nt-tbl td.n { font-family: var(--mono); text-align: right; white-space: nowrap; }
    .nt-tbl tr.off td { color: var(--ink-3); }
    .nt-role { font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; font-weight: 600; white-space: nowrap; }

    /* details */
    .nt details { border-top: 1px solid var(--rule); }
    .nt details:last-child { border-bottom: 1px solid var(--rule); }
    .nt summary { cursor: pointer; padding: 11px 0; list-style: none; display: flex; justify-content: space-between; font-weight: 500; }
    .nt summary::-webkit-details-marker { display: none; }
    .nt summary::after { content: "+"; font-family: var(--mono); color: var(--ink-3); }
    .nt details[open] summary::after { content: "–"; }
    .nt .trail { font-family: var(--mono); font-size: 11.5px; color: var(--ink-2); padding: 0 0 14px; }
    .nt .trail div { padding: 2px 0; }
    .nt .trail b { font-family: var(--sans); font-size: 12.5px; color: var(--ink); display: block; margin-top: 10px; }
    .nt .quote { font-family: var(--serif); font-style: italic; font-size: 16px; line-height: 1.35; color: var(--ink);
      border-left: 2px solid var(--ink); padding: 2px 0 2px 14px; margin: 8px 0; }
    .nt .cite { font-size: 11px; color: var(--ink-3); font-family: var(--mono); }
    .nt-proto { padding: 16px 0; border-bottom: 1px solid var(--rule); display: grid; grid-template-columns: 54px 1fr; gap: 14px; }
    .nt-proto .k { font-family: var(--mono); font-size: 11px; color: var(--ink-3); padding-top: 4px; }
    .nt-proto h4 { margin: 0 0 4px; font-size: 15px; font-weight: 600; }
    .nt-proto p { margin: 2px 0; color: var(--ink-2); font-size: 13px; }

    /* settle */
    .nt-spec { display: grid; grid-template-columns: 54px 1fr 280px; gap: 18px; padding: 20px 0; border-bottom: 1px solid var(--rule); align-items: start; }
    .nt-spec .k { font-family: var(--mono); font-size: 11px; color: var(--ink-3); padding-top: 8px; }
    .nt-spec h4 { margin: 0 0 4px; font-size: 16px; }
    .nt-spec p { margin: 0; color: var(--ink-2); font-size: 13px; max-width: 620px; }
    .nt-spec .big { text-align: right; }
    .nt-spec .big b { font-family: var(--serif); font-size: 44px; line-height: 1; font-weight: 400; display: block; }
    .nt-spec .big span { font-family: var(--mono); font-size: 11px; color: var(--ink-3); }
    .nt-gap { display: grid; grid-template-columns: repeat(3, 1fr); border: 1px solid var(--ink); background: var(--card); }
    .nt-gap > div { padding: 16px 18px; border-right: 1px solid var(--rule); }
    .nt-gap > div:last-child { border-right: 0; }
    .nt-gap b { display: block; font-family: var(--serif); font-size: 36px; font-weight: 400; line-height: 1.1; }
    .nt-gap span { font-size: 12px; color: var(--ink-2); }
    .nt-note { background: var(--card); border: 1px solid var(--rule); border-left: 3px solid var(--ochre); padding: 10px 14px; margin: 10px 0; font-size: 13px; color: var(--ink-2); }

    .nt-as { margin: 8px 0 0; padding: 0 0 0 16px; font-size: 12px; color: var(--ink-3); }
    .nt-as li { margin: 2px 0; }
    .nt-note.caveat { border-left-color: var(--ink); }
    .nt-hyp { border: 1px dashed var(--violet); border-left: 3px solid var(--violet); background: #f3f1fb; color: #2f2a6b;
      padding: 12px 16px; margin: 10px 0; font-size: 13.5px; }
    .nt-ba { display: flex; align-items: center; gap: 10px; margin: 8px 0; flex-wrap: wrap; }
    .nt-ba .arr { font-family: var(--mono); color: var(--ink-3); }
    .nt-wi { display: grid; gap: 8px; }
    .nt-wirow { display: grid; grid-template-columns: 22px 1fr auto 140px 90px; gap: 10px; align-items: center;
      background: var(--card); border: 1px solid var(--rule); padding: 8px 12px; font-size: 13.5px; }
    .nt-wirow.off { opacity: .6; }
    .nt-wirow small { display: block; color: var(--red); font-size: 11.5px; }
    .nt-wirow .u { font-family: var(--mono); font-size: 12px; color: var(--ink-3); }

    /* controls */
    .nt-ctl { margin: 0 0 16px; }
    .nt-ctl .row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
    .nt-ctl label { font-size: 12.5px; color: var(--ink-2); }
    .nt-ctl .val { font-family: var(--mono); font-size: 13px; font-weight: 500; }
    .nt input[type=range] { -webkit-appearance: none; appearance: none; width: 100%; height: 18px; background: transparent; cursor: pointer; }
    .nt input[type=range]::-webkit-slider-runnable-track { height: 2px; background: var(--ink); }
    .nt input[type=range]::-moz-range-track { height: 2px; background: var(--ink); }
    .nt input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 12px; height: 18px; margin-top: -8px; background: var(--paper);
      border: 2px solid var(--ink); border-radius: 0; }
    .nt input[type=range]::-moz-range-thumb { width: 10px; height: 16px; background: var(--paper); border: 2px solid var(--ink); border-radius: 0; }
    .nt-seg { display: inline-flex; border: 1px solid var(--ink); margin-bottom: 18px; }
    .nt-seg button { background: none; border: 0; padding: 6px 14px; cursor: pointer; font-size: 12.5px; }
    .nt-seg button.on { background: var(--ink); color: var(--paper); }
    .nt-design { display: grid; grid-template-columns: 300px 1fr; gap: 40px; }
    .nt-chart svg { width: 100%; display: block; }
    .nt-chart .tick { font-family: var(--mono); font-size: 10px; fill: var(--ink-3); }
    .nt-chart .axl { font-size: 11px; fill: var(--ink-2); }
    .nt-map { position: relative; cursor: crosshair; }
    .nt-map canvas { width: 100%; display: block; image-rendering: auto; }
    .nt-map svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; overflow: visible; }
    .nt-answer { font-family: var(--serif); font-size: 32px; line-height: 1.12; margin: 0 0 6px; max-width: 900px; }

    /* inputs */
    .nt-field { width: 100%; padding: 7px 9px; border: 1px solid var(--rule-2); background: var(--card); border-radius: 2px; outline: none; }
    .nt-field:focus { border-color: var(--ink); }
    .nt textarea.nt-field { font-family: var(--mono); font-size: 12px; line-height: 1.45; resize: vertical; }
    .nt .nt-btn { background: var(--ink); color: var(--paper); border: 1px solid var(--ink); padding: 8px 16px; cursor: pointer; font-weight: 600; border-radius: 2px; }
    .nt .nt-btn.ghost { background: transparent; color: var(--ink); }
    .nt .nt-btn:disabled { opacity: .5; cursor: progress; }
    .nt .nt-btn + .nt-btn { margin-left: 8px; }
    .nt-stbl input, .nt-stbl select { width: 100%; border: 0; border-bottom: 1px solid var(--rule-2); background: transparent; padding: 4px 2px; font-family: var(--mono); font-size: 12px; outline: none; }
    .nt-stbl input:focus, .nt-stbl select:focus { border-bottom-color: var(--ink); }
    .nt-x { background: none; border: 0; color: var(--ink-3); cursor: pointer; font-size: 16px; }

    /* report */
    .nt-doc { max-width: 860px; background: var(--card); border: 1px solid var(--rule); padding: 36px 44px; }
    .nt-doc h1 { font-family: var(--serif); font-weight: 400; font-size: 34px; line-height: 1.1; margin: 0 0 14px; }
    .nt-doc h2 { font-family: var(--serif); font-weight: 400; font-size: 22px; margin: 26px 0 8px; border-top: 1px solid var(--ink); padding-top: 8px; }
    .nt-doc h3 { font-size: 14px; margin: 16px 0 4px; }
    .nt-doc p, .nt-doc li { font-size: 13.5px; color: var(--ink-2); }
    .nt-doc blockquote { font-family: var(--serif); font-style: italic; font-size: 15px; border-left: 2px solid var(--ink); margin: 8px 0; padding-left: 12px; }
    .nt-doc code { font-family: var(--mono); font-size: 12px; background: var(--paper-2); padding: 0 4px; }

    /* method */
    .nt-essay { max-width: 760px; }
    .nt-essay p { font-size: 15.5px; line-height: 1.6; color: var(--ink-2); margin: 0 0 14px; }
    .nt-essay p:first-of-type::first-letter { font-family: var(--serif); float: left; font-size: 64px; line-height: .8; padding: 6px 8px 0 0; color: var(--ink); }
    .nt-big { font-family: var(--serif); font-size: 56px; line-height: 1; }
    .nt-kpis { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--ink); border-bottom: 1px solid var(--ink); }
    .nt-kpis > div { padding: 18px 18px 16px 0; border-right: 1px solid var(--rule); }
    .nt-kpis > div + div { padding-left: 18px; }
    .nt-kpis > div:last-child { border-right: 0; }
    .nt-kpis span { display: block; font-size: 12px; color: var(--ink-2); margin-top: 6px; }
    .nt-prop { display: grid; grid-template-columns: 22px 1fr auto; gap: 10px; padding: 9px 0; border-bottom: 1px solid var(--rule); align-items: baseline; }
    .nt-prop .ok { width: 12px; height: 12px; background: var(--green); display: inline-block; }
    .nt-prop .bad { width: 12px; height: 12px; background: var(--red); display: inline-block; }
    .nt-empty { color: var(--ink-3); font-style: italic; }
    .nt-err { border: 1px solid var(--violet); border-left-width: 3px; background: var(--card); padding: 12px 16px; font-family: var(--mono); font-size: 12px; color: var(--violet); }

    @container nt (max-width: 900px) {
      .nt-body { grid-template-columns: 1fr; }
      .nt-index { max-height: 260px; border-right: 0; border-bottom: 1px solid var(--rule); }
      .nt-cols, .nt-design { grid-template-columns: 1fr; }
      .nt-mast { grid-template-columns: 1fr; }
      .nt-figs { justify-content: flex-start; }
      .nt-verdict { font-size: 34px; }
      .nt-spec { grid-template-columns: 40px 1fr; }
      .nt-spec .big { grid-column: 2; text-align: left; }
      .nt-kpis, .nt-gap, .nt-cols3 { grid-template-columns: 1fr 1fr; }
    }
    '''

    return NT_CSS2, NT_ESM


@app.cell(hide_code=True)
def ui_widget(
    CASE_KEYS,
    DATA,
    ExperimentRecord,
    NT_CSS2,
    NT_ESM,
    N_OK,
    STATUS_LABELS: dict[str, str],
    bench_summary,
    claude_candidates,
    compare,
    evidence_requirements,
    find_passages,
    gas_amount_mol,
    gas_fraction,
    run_audit,
    suggest,
    to_casefile,
    to_csv,
    to_json,
    to_markdown,
    to_review_pack,
):
    # NitroTrace interface: one anywidget. The browser draws; every number comes from the engine cells above.
    import anywidget
    import traitlets


    def scaled_record(d, k):
        """Copy of a record with the observed product multiplied by k (what-if; nondetects untouched)."""
        e = copy.deepcopy(d)
        op = e.get("observed_product")
        if op and k != 1 and not op.get("nondetect"):
            for b in ("lower", "upper"):
                if op.get(b) is not None:
                    op[b] = op[b] * k
        return e


    def audit_payload(key, base, scale):
        out = {"key": key, "scale": scale, "error": None}
        d = DATA.get(key, base) if base is None else base
        out.update(case_id=d.get("case_id", "?"), title=d.get("title", ""), record_type=d.get("record_type", "user"),
                   sources=[f"[{s}] " + DATA["protocols/sources.json"].get(s, {}).get("citation", s).split(",")[0] + " et al."
                            for s in d.get("sources", [])])
        try:
            rec = ExperimentRecord(**scaled_record(d, scale))
            res = run_audit(rec)
        except Exception as e:  # malformed edits are shown, not raised
            out.update(status="input_inconsistency", headline="The edited record could not be read.",
                       error=f"{type(e).__name__}: {e}")
            return out
        vmap = []
        if d.get("observed_product"):
            for i in range(-24, 25):
                k = 10 ** (i / 8)
                try:
                    vmap.append([k, run_audit(ExperimentRecord(**scaled_record(d, k))).status])
                except Exception:
                    vmap.append([k, "input_inconsistency"])
        req = evidence_requirements(rec, res)
        out.update(
            status=res.status, status_label=res.status_label, headline=res.headline, explanation=res.explanation,
            coverage=res.coverage, budget_label=res.budget_label, observed=res.observed_nmol,
            valid=res.budget_valid_nmol, with_scen=res.budget_with_scenarios_nmol,
            contributions=[{"label": c.label, "category": c.category, "role": c.role, "included": c.included,
                            "lower": c.lower_nmol, "upper": c.upper_nmol, "reason": c.reason} for c in res.contributions],
            unquantified=[u.model_dump() for u in res.unquantified], issues=[i.model_dump() for i in res.issues],
            recs=[{"title": r.title, "text": r.text, "trigger": r.trigger,
                   "refs": [{"quote": q["quote"], "source_id": q["source_id"], "location": q["location"]} for q in r.references]}
                  for r in res.recommendations],
            trail=[t.model_dump() for t in res.observed_trail + [s for c in res.contributions for s in c.trail]],
            assumptions=res.assumptions, req=req, vmap=vmap, hypothetical=res.hypothetical,
            whatif_fields=whatif_fields(rec, res, req),
            report={"casefile": to_casefile(rec, res, req), "review": to_review_pack(rec, res, req),
                    "md": to_markdown(rec, res), "json": to_json(rec, res), "csv": to_csv(rec, res)})
        return out


    def whatif_fields(rec, res, req):
        """Form fields for hypothetical measurements, prefilled at half of each equal-split threshold."""
        sug = {k: (v, sid) for k, v, sid in suggest(rec, req)}
        gas_ok = [g for g in rec.gas_sources if gas_amount_mol(g, rec)]
        gas_val = sug.get("gas", (None, None))[0]
        if gas_val is None and req:
            for row in req["shrink"]:
                g = next((g for g in rec.gas_sources if g.id == row["source_id"]), None)
                if g is not None and row.get("feasible") and "required_value" in row:
                    gas_val = float(f"{gas_fraction(row['required_value'], g.impurity.unit) * 1e9 * 0.9:.3g}")
        vol = rec.observed_product.volume_mL if rec.observed_product else None
        blank_kind = "electrolyte" if vol else "electrolyte_nmol"
        blank_val = sug.get("electrolyte", sug.get("electrolyte_nmol", (None, None)))[0]
        open_cat = {k for k, v in res.coverage.items() if v not in N_OK}
        return [
            {"kind": "gas", "label": "Total reactive N in the gas delivered to the cell", "unit": "ppb",
             "value": gas_val, "on": gas_val is not None, "source_id": gas_ok[0].id if gas_ok else None,
             "disabled": None if gas_ok else "Gas flow or exposure not stated in the record: no ppb level can be "
                                             "turned into an amount. State the flow first."},
            {"kind": blank_kind, "label": "Matched electrolyte blank", "unit": "µg/L NH4+" if vol else "nmol N",
             "value": blank_val, "on": "electrolyte" in open_cat and blank_val is not None, "disabled": None},
            {"kind": "components", "label": "Matched component release", "unit": "nmol N",
             "value": sug.get("components", (None, None))[0], "on": "components" in open_cat and "components" in sug,
             "disabled": None},
        ]


    def catalog():
        rows = []
        for k in sorted(CASE_KEYS, key=lambda k: (not k.startswith("cases/"), DATA[k]["title"])):
            d = DATA[k]
            try:
                st = run_audit(ExperimentRecord(**d)).status
            except Exception:
                st = "input_inconsistency"
            pub = k.startswith("cases/")
            sub = (d.get("expected") or {}).get("purpose") if not pub else ", ".join(d.get("sources", []))
            rows.append({"key": k, "id": d["case_id"], "title": d["title"], "group": "published" if pub else "synthetic",
                         "status": st, "status_label": STATUS_LABELS[st], "sub": (sub or d["case_id"])[:90]})
        return rows



    class NitroTraceApp(anywidget.AnyWidget):
        _esm = NT_ESM
        _css = NT_CSS2
        catalog = traitlets.List([]).tag(sync=True)
        bench = traitlets.Dict({}).tag(sync=True)
        selected = traitlets.Unicode("").tag(sync=True)
        scale = traitlets.Float(1.0).tag(sync=True)
        tab = traitlets.Unicode("audit").tag(sync=True)
        audit = traitlets.Dict({}).tag(sync=True)
        record_json = traitlets.Unicode("").tag(sync=True)
        edit_json = traitlets.Dict({}).tag(sync=True)
        agent_text = traitlets.Unicode("").tag(sync=True)
        agent_req = traitlets.Dict({}).tag(sync=True)
        agent_res = traitlets.Dict({}).tag(sync=True)
        whatif_req = traitlets.Dict({}).tag(sync=True)
        whatif_res = traitlets.Dict({}).tag(sync=True)

        def __init__(self, **kw):
            super().__init__(**kw)
            self._base = None
            b = dict(bench_summary)
            b["elapsed_ms"] = b.pop("elapsed_s") * 1000
            b["properties"] = {k: list(v) for k, v in b["properties"].items()}
            self.bench = b
            self.catalog = catalog()
            self.selected = "cases/bragulla2026_pristine_gde_ocv.json"
            self._load()

        def _load(self):
            if self.selected not in DATA:
                return
            self._base = copy.deepcopy(DATA[self.selected])
            self.record_json = json.dumps(self._base, indent=2, ensure_ascii=False)
            self._run()

        def _run(self):
            self.whatif_res = {}
            self.audit = audit_payload(self.selected, self._base, self.scale)

        @traitlets.observe("whatif_req")
        def _on_whatif(self, change):
            r = change["new"] or {}
            changes = [(c["kind"], float(c["value"]), c.get("source_id")) for c in r.get("changes", [])]
            if not changes:
                self.whatif_res = {"error": "Tick at least one measurement."}
                return
            try:
                out = compare(scaled_record(self._base, self.scale), changes)
            except (ValueError, KeyError, TypeError) as e:
                self.whatif_res = {"error": str(e)}
                return
            out.pop("record"), out.pop("after_result")
            self.whatif_res = out

        @traitlets.observe("selected")
        def _on_selected(self, change):
            self._load()

        @traitlets.observe("scale")
        def _on_scale(self, change):
            if self._base is not None:
                self._run()

        @traitlets.observe("edit_json")
        def _on_edit(self, change):
            text = (change["new"] or {}).get("text")
            if not text:
                return
            try:
                self._base = json.loads(text)
                self.record_json = text
                self._run()
            except json.JSONDecodeError as e:
                self.audit = {**self.audit, "error": f"JSON syntax error: {e}"}

        @traitlets.observe("agent_req")
        def _on_agent(self, change):
            r = change["new"] or {}
            text = (r.get("text") or "").strip()
            if not text:
                self.agent_res = {"error": "Paste some text first."}
                return
            out = {"passages": find_passages(text)}
            if r.get("llm"):
                c = claude_candidates(text, (r.get("key") or "").strip())
                out.update(c)
            self.agent_res = out


    return (NitroTraceApp,)


@app.cell(hide_code=True)
def app(NitroTraceApp):
    # The app: everything above is hidden behind this one interface.
    nt_app = NitroTraceApp()
    mo.ui.anywidget(nt_app)

    return


if __name__ == "__main__":
    app.run()
