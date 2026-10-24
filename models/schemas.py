"""
models/schemas.py

Domain models for the AI Race Engineer platform.

These models are the single source of truth for data shape across the
entire application (Section 3 / Section 7 of the SRS). They are the only
part of the domain core that every other layer is allowed to depend on:

    analysis/  -> produces these models
    llm/       -> consumes SessionAnalysis.model_dump_json() ONLY
    db/        -> persists/retrieves AnalysisReport
    app/       -> renders these models; never mutates or recomputes them

Design rules enforced here:
    - Immutability: once analysis is computed, it must not be mutated by
      a downstream layer (e.g. the Streamlit UI). All models are frozen.
    - No duplicated fields: aggregate totals (e.g. total time loss, score)
      live in exactly one model (DriverSummary) and are referenced, not
      re-declared, by SessionAnalysis.
    - Enums replace magic strings for issue classification, so the Rule
      Engine (analysis/rules.py) and the LLM prompt layer share one
      controlled vocabulary instead of ad hoc string literals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class IssueType(str, Enum):
    """
    Controlled vocabulary for corner-level driving issues detected by the
    Rule Engine (analysis/rules.py).

    Using a str Enum (rather than a bare string field) means:
      - engine.py and rules.py cannot silently introduce a typo'd issue
        label that the LLM prompt layer or the UI don't recognize.
      - model_dump(mode="json") serializes cleanly to the enum's string
        value, so the JSON handed to Gemini stays human-readable.
    """

    NONE = "none"
    EARLY_BRAKING = "early_braking"
    LATE_BRAKING = "late_braking"
    SLOW_APEX = "slow_apex"
    POOR_EXIT_SPEED = "poor_exit_speed"
    LATE_THROTTLE = "late_throttle"
    CONSISTENCY_ISSUE = "consistency_issue"


class IssueSeverity(str, Enum):
    """
    Severity tier for a detected CornerMetric issue. Kept separate from
    IssueType so future rule tuning (e.g. Version 3 ML-calibrated
    thresholds) can adjust severity scoring without changing the set of
    recognized issue categories.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CornerMetric(BaseModel):
    """
    Deterministic, Python-computed metrics for a single corner, comparing
    one driver's lap against a reference driver's lap (SRS FR-3, FR-4).

    This is the atomic unit of analysis: SessionAnalysis is built by
    aggregating a list of these. It carries both the raw measurements
    (braking point, apex speed, etc.) AND the Rule Engine's classification
    of that corner (issue_type / severity), because the classification is
    a per-corner fact -- keeping it here avoids a separate CornerIssue
    model that would otherwise duplicate corner_number as a join key.

    Instances are immutable: once the Rule Engine has classified a corner,
    no downstream layer (UI, LLM adapter) may alter the result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    corner_number: int = Field(
        ge=1,
        description="1-indexed corner number on the circuit.",
    )
    start_distance_m: float = Field(
        ge=0.0,
        description="Distance (m) along the lap where this corner's analysis window begins.",
    )
    end_distance_m: float = Field(
        gt=0.0,
        description="Distance (m) along the lap where this corner's analysis window ends.",
    )
    braking_point_m: float = Field(
        description="Distance (m) at which the driver first applies brake before this corner.",
    )
    braking_delta_m: float = Field(
        description="Driver braking point minus reference braking point, in meters. "
        "Negative means the driver braked earlier than the reference.",
    )
    apex_speed_kph: float = Field(
        ge=0.0,
        description="Driver's minimum speed (km/h) within the corner window (the apex).",
    )
    apex_speed_delta_kph: float = Field(
        description="Driver apex speed minus reference apex speed, in km/h.",
    )
    exit_speed_kph: float = Field(
        ge=0.0,
        description="Driver's speed (km/h) at the corner exit boundary.",
    )
    exit_speed_delta_kph: float = Field(
        description="Driver exit speed minus reference exit speed, in km/h.",
    )
    throttle_application_m: float = Field(
        description="Distance (m) at which the driver first re-applies throttle after the apex.",
    )
    throttle_delay_m: float = Field(
        description="Driver throttle application point minus reference throttle application "
        "point, in meters. Positive means the driver got back to throttle later.",
    )
    time_loss_s: float = Field(
        description="Cumulative time lost (seconds) to the reference driver across this "
        "corner's window. Negative values indicate time gained.",
    )
    issue_type: IssueType = Field(
        default=IssueType.NONE,
        description="Rule Engine classification for this corner, if any.",
    )
    severity: IssueSeverity = Field(
        default=IssueSeverity.NONE,
        description="Severity tier associated with issue_type.",
    )

    @field_validator("end_distance_m")
    @classmethod
    def _end_after_start(cls, end_distance_m: float, info) -> float:
        """Ensure the corner window is well-formed (end strictly after start)."""
        start = info.data.get("start_distance_m")
        if start is not None and end_distance_m <= start:
            raise ValueError(
                f"end_distance_m ({end_distance_m}) must be greater than "
                f"start_distance_m ({start})."
            )
        return end_distance_m


class DriverSummary(BaseModel):
    """
    Session-level summary comparing a driver to a reference driver
    (SRS Section 4 / FR-4.5). This is the single owner of the aggregate
    time-loss and score figures -- SessionAnalysis embeds this model
    rather than re-declaring total_time_loss_s or performance_score, to
    satisfy the "no duplicated fields" requirement.

    Immutable: computed once by analysis/engine.py and never modified
    afterward.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    driver_code: str = Field(
        pattern=r"^[A-Z]{2,3}$",
        description="Three-letter (or two-letter) FIA driver code, e.g. 'VER'.",
    )
    reference_driver_code: str = Field(
        pattern=r"^[A-Z]{2,3}$",
        description="FIA driver code of the comparison/reference driver.",
    )
    total_time_loss_s: float = Field(
        ge=0.0,
        description="Total lap time lost (seconds) relative to the reference driver, "
        "summed across all corners.",
    )
    performance_score: int = Field(
        ge=0,
        le=100,
        description="Deterministic overall performance score for this lap, 0-100.",
    )

    @model_validator(mode="after")
    def _driver_differs_from_reference(self) -> "DriverSummary":
        """A driver cannot be meaningfully compared against themselves."""
        if self.driver_code == self.reference_driver_code:
            raise ValueError(
                "driver_code and reference_driver_code must differ "
                f"(both were '{self.driver_code}')."
            )
        return self


class SessionAnalysis(BaseModel):
    """
    The complete, deterministic output of the Python analysis pipeline
    for one driver/reference-driver/session combination (SRS FR-4.5).

    This is the ONLY object that is ever serialized and sent to the LLM
    (SRS FR-5.1, design principle 3) -- llm/feedback.py accepts nothing
    else. Keeping this model free of any AI-generated content (that lives
    in AnalysisReport instead) means it stays fully deterministic and
    therefore hashable/comparable for regression testing (NFR-3).

    Immutable: represents a finished computation, not a work-in-progress
    object to be mutated by the UI layer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(
        min_length=1,
        description="Unique identifier for the session, e.g. '2024_Monza_Q'.",
    )
    driver_summary: DriverSummary = Field(
        description="Aggregate comparison summary for this driver vs. the reference driver.",
    )
    corner_metrics: list[CornerMetric] = Field(
        min_length=1,
        description="Per-corner metrics and issue classifications, ordered by corner_number.",
    )
    worst_corner: int = Field(
        ge=1,
        description="corner_number of the corner with the greatest time loss.",
    )
    best_corner: int = Field(
        ge=1,
        description="corner_number of the corner with the least time loss (or greatest gain).",
    )
    biggest_mistake: str = Field(
        min_length=1,
        description="Short, human-readable description of the single biggest driving error "
        "identified in this lap, derived deterministically from corner_metrics.",
    )

    @model_validator(mode="after")
    def _worst_and_best_corner_exist(self) -> "SessionAnalysis":
        """
        worst_corner and best_corner must reference corners that are
        actually present in corner_metrics -- this catches orchestration
        bugs in analysis/engine.py before they reach the UI or the LLM.
        """
        known_corners = {metric.corner_number for metric in self.corner_metrics}

        if self.worst_corner not in known_corners:
            raise ValueError(
                f"worst_corner ({self.worst_corner}) does not match any "
                f"corner_number in corner_metrics {sorted(known_corners)}."
            )
        if self.best_corner not in known_corners:
            raise ValueError(
                f"best_corner ({self.best_corner}) does not match any "
                f"corner_number in corner_metrics {sorted(known_corners)}."
            )
        if len(known_corners) > 1 and self.worst_corner == self.best_corner:
            raise ValueError(
                "worst_corner and best_corner must differ when more than "
                "one corner was analyzed."
            )
        return self


class AnalysisReport(BaseModel):
    """
    Presentation/persistence-facing composite that pairs a deterministic
    SessionAnalysis with the AI-generated coaching text and a generation
    timestamp (SRS Section 7 class diagram).

    Deliberately kept separate from SessionAnalysis rather than merged
    into it: SessionAnalysis must remain pure Python-computed output for
    testing and reproducibility (NFR-3), while ai_feedback is inherently
    non-deterministic (LLM output) and generated_at is wall-clock
    metadata -- neither belongs in the deterministic core.

    This is the object db/repository.py persists and app/main.py renders
    as the final "AI Race Engineer Radio" card.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis: SessionAnalysis = Field(
        description="The deterministic analysis this report explains.",
    )
    ai_feedback: str = Field(
        min_length=1,
        description="Race-engineer-style coaching text produced by the LLM adapter "
        "(llm/feedback.py) from analysis.model_dump_json(). Plain text, "
        "at most three bullet points per FR-5.2.",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp at which this report (including AI feedback) was assembled.",
    )
