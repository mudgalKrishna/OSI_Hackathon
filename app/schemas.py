"""Pydantic schemas for NitroTrace experiment records, evidence and audit results.

Every scientifically meaningful value carries its original unit, bounds, bound
type and provenance. Missing values are ``None`` and mean *unknown*, never zero.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

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
