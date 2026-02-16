"""
Pydantic models for the report_generator component.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class TrendDirection(str, Enum):
    """Direction of confidence score trend over recent windows."""
    improving = "improving"
    stable = "stable"
    declining = "declining"


class AlertSeverity(str, Enum):
    """Severity level for system alerts."""
    info = "info"
    warning = "warning"
    critical = "critical"


class AlertType(str, Enum):
    """Typed category of system alert."""
    quality_degradation = "quality_degradation"
    budget_warning = "budget_warning"
    budget_exhausted = "budget_exhausted"
    staleness = "staleness"
    local_model_failure = "local_model_failure"
    fine_tuning_failure = "fine_tuning_failure"


class ReportFormat(str, Enum):
    """Supported output format for report rendering."""
    json = "json"
    cli_summary = "cli_summary"


class DeliveryStatus(str, Enum):
    """Outcome status of a webhook delivery attempt."""
    success = "success"
    failure = "failure"
    timeout = "timeout"


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ══════════════════════════════════════════════════════════════════════════════

class TaskConfidenceReport(BaseModel):
    """Confidence and phase data for a single task type."""
    model_config = ConfigDict(frozen=True)

    task_type: str
    phase: str
    confidence_score: float
    trend: TrendDirection
    sampling_frequency: float

    @field_validator('confidence_score')
    @classmethod
    def validate_confidence_score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_score must be in [0.0, 1.0]")
        return v

    @field_validator('sampling_frequency')
    @classmethod
    def validate_sampling_frequency(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("sampling_frequency must be in [0.0, 1.0]")
        return v


class CostBreakdown(BaseModel):
    """Cost accounting data with Decimal precision."""
    model_config = ConfigDict(frozen=True)

    total_remote_spend: str
    estimated_local_savings: str
    cost_per_task: dict[str, str]
    budget_limit: str
    budget_remaining: str
    budget_utilization_pct: float

    @field_validator('budget_utilization_pct')
    @classmethod
    def validate_budget_utilization_pct(cls, v: float) -> float:
        if not 0.0 <= v <= 100.0:
            raise ValueError("budget_utilization_pct must be in [0.0, 100.0]")
        return v


class CorrelationWindow(BaseModel):
    """A single correlation measurement window."""
    model_config = ConfigDict(frozen=True)

    task_type: str
    window_start: str
    window_end: str
    sample_count: int
    correlation_score: float
    local_agreement_rate: float

    @field_validator('sample_count')
    @classmethod
    def validate_sample_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError("sample_count must be >= 0")
        return v

    @field_validator('correlation_score')
    @classmethod
    def validate_correlation_score(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError("correlation_score must be in [-1.0, 1.0]")
        return v

    @field_validator('local_agreement_rate')
    @classmethod
    def validate_local_agreement_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("local_agreement_rate must be in [0.0, 1.0]")
        return v


class CorrelationHistory(BaseModel):
    """Correlation window history for a task type."""
    model_config = ConfigDict(frozen=True)

    task_type: str
    windows_requested: int
    windows_available: int
    windows: list[CorrelationWindow]

    @field_validator('windows_requested')
    @classmethod
    def validate_windows_requested(cls, v: int) -> int:
        if v < 1:
            raise ValueError("windows_requested must be >= 1")
        return v

    @field_validator('windows_available')
    @classmethod
    def validate_windows_available(cls, v: int) -> int:
        if v < 0:
            raise ValueError("windows_available must be >= 0")
        return v


class Alert(BaseModel):
    """A typed system alert with severity."""
    model_config = ConfigDict(frozen=True)

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    task_type: Optional[str] = None
    timestamp: str


class SystemReport(BaseModel):
    """Top-level aggregated system report."""
    model_config = ConfigDict(frozen=True)

    report_id: str
    generated_at: str
    task_confidence_reports: list[TaskConfidenceReport]
    cost_breakdown: CostBreakdown
    correlation_histories: list[CorrelationHistory]
    alerts: list[Alert]
    task_types_count: int

    @field_validator('task_types_count')
    @classmethod
    def validate_task_types_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError("task_types_count must be >= 0")
        return v


class ReportConfig(BaseModel):
    """Configuration for report generation."""
    model_config = ConfigDict(frozen=True, validate_assignment=False)

    correlation_window_count: int
    alerts_lookback_hours: int
    webhook_url: Optional[str] = None
    webhook_timeout_seconds: int = 30


class DeliveryResult(BaseModel):
    """Outcome of a webhook delivery attempt."""
    model_config = ConfigDict(frozen=True)

    status: DeliveryStatus
    http_status_code: int = 0
    url: str
    attempt_timestamp: str
    error_message: Optional[str] = None
    duration_ms: int

    @field_validator('duration_ms')
    @classmethod
    def validate_duration_ms(cls, v: int) -> int:
        if v < 0:
            raise ValueError("duration_ms must be >= 0")
        return v


class FormattedReport(BaseModel):
    """A rendered report with metadata."""
    model_config = ConfigDict(frozen=True)

    content: str
    format: ReportFormat
    report_id: str
    content_length: int

    @field_validator('content_length')
    @classmethod
    def validate_content_length(cls, v: int) -> int:
        if v < 0:
            raise ValueError("content_length must be >= 0")
        return v
