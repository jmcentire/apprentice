"""
Reporting, Observability & Audit Log - Integration Glue Module

This module wires together the audit_log and report_generator child components
into the parent reporting interface.
"""

# Import child modules via relative imports (intra-package)
from . import audit_log_adapter
from . import report_generator

# Re-export all EventType enum members from audit_log adapter
from .audit_log_adapter import EventType

# Re-export EventTypeList and AuditEntryList type aliases
from .audit_log_adapter import AuditEntryList, EventType as EventTypeEnum

# Re-export audit log types and classes
from .audit_log_adapter import (
    AuditConfig,
    AuditEntry,
    JsonLinesAuditLogger,
)

# Re-export report generator enums
from .report_generator import (
    TrendDirection,
    AlertSeverity,
    AlertType,
    ReportFormat,
    DeliveryStatus,
)

# Re-export report generator models
from .report_generator import (
    TaskConfidenceReport,
    CostBreakdown,
    CorrelationWindow,
    CorrelationHistory,
    Alert,
    SystemReport,
    ReportConfig,
    DeliveryResult,
    FormattedReport,
)

# Re-export report generator exceptions
from .report_generator import (
    ConfigValidationError,
    DataSourceError,
    EmptyReportError,
    FormattingError,
)

# Re-export core classes
from .report_generator import (
    ReportGenerator,
    ReportFormatter,
    WebhookDelivery,
)


# ══════════════════════════════════════════════════════════════════════════════
# Type Aliases - matching parent contract expectations
# ══════════════════════════════════════════════════════════════════════════════

EventTypeList = list[EventType]
CorrelationWindowList = list[CorrelationWindow]
TaskConfidenceReportList = list[TaskConfidenceReport]
CorrelationHistoryList = list[CorrelationHistory]
AlertList = list[Alert]


# ══════════════════════════════════════════════════════════════════════════════
# Protocol Types (for data sources)
# ══════════════════════════════════════════════════════════════════════════════

from typing import Protocol  # noqa: E402


class ConfidenceDataSource(Protocol):
    """Protocol for read-only access to confidence engine state."""

    def get_task_types(self) -> list[str]:
        ...

    def get_confidence(self, task_type: str) -> float:
        ...

    def get_phase(self, task_type: str) -> str:
        ...

    def get_trend(self, task_type: str) -> TrendDirection:
        ...

    def get_sampling_frequency(self, task_type: str) -> float:
        ...

    def get_correlation_windows(self, task_type: str, count: int) -> CorrelationWindowList:
        ...


class BudgetDataSource(Protocol):
    """Protocol for read-only access to budget manager state."""

    def get_total_remote_spend(self) -> str:
        ...

    def get_estimated_local_savings(self) -> str:
        ...

    def get_cost_per_task(self) -> dict[str, str]:
        ...

    def get_budget_limit(self) -> str:
        ...

    def get_budget_remaining(self) -> str:
        ...

    def get_budget_utilization_pct(self) -> float:
        ...


class AuditDataSource(Protocol):
    """Protocol for read-only access to audit log queries."""

    def entries_in_range(self, start: str, end: str) -> AuditEntryList:
        ...

    def entries_by_task(self, task_name: str) -> AuditEntryList:
        ...

    def entries_by_type(self, event_type: EventType) -> AuditEntryList:
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Parent Functions - Delegation Layer
# ══════════════════════════════════════════════════════════════════════════════

def validate_report_config(config: ReportConfig) -> ReportConfig:
    """
    Validates a ReportConfig instance, ensuring all fields are within their
    declared constraints. Delegates to report_generator.validate_report_config.
    """
    return report_generator.validate_report_config(config)


def generate_report(
    confidence_source: ConfidenceDataSource,
    budget_source: BudgetDataSource,
    audit_source: AuditDataSource,
    config: ReportConfig,
) -> SystemReport:
    """
    Generates a complete SystemReport by reading data from all three data sources.
    Constructs the report directly from the protocol sources.
    """
    import uuid
    from datetime import datetime, timezone

    # Generate report metadata
    report_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()
    if not generated_at.endswith('+00:00'):
        generated_at = generated_at.replace('Z', '+00:00')

    # Fetch task types
    try:
        task_types = confidence_source.get_task_types()
    except Exception as e:
        raise DataSourceError(f"Confidence source unavailable: {e}")

    if not task_types:
        raise EmptyReportError("No task types found in system")

    # Build task confidence reports
    task_confidence_reports = []
    try:
        for task_type in task_types:
            report = TaskConfidenceReport(
                task_type=task_type,
                phase=confidence_source.get_phase(task_type),
                confidence_score=confidence_source.get_confidence(task_type),
                trend=confidence_source.get_trend(task_type),
                sampling_frequency=confidence_source.get_sampling_frequency(task_type),
            )
            task_confidence_reports.append(report)
    except Exception as e:
        raise DataSourceError(f"Confidence source unavailable: {e}")

    # Build cost breakdown
    try:
        cost_breakdown = CostBreakdown(
            total_remote_spend=budget_source.get_total_remote_spend(),
            estimated_local_savings=budget_source.get_estimated_local_savings(),
            cost_per_task=budget_source.get_cost_per_task(),
            budget_limit=budget_source.get_budget_limit(),
            budget_remaining=budget_source.get_budget_remaining(),
            budget_utilization_pct=budget_source.get_budget_utilization_pct(),
        )
    except Exception as e:
        raise DataSourceError(f"Budget source unavailable: {e}")

    # Build correlation histories
    correlation_histories = []
    try:
        for task_type in task_types:
            windows = confidence_source.get_correlation_windows(task_type, config.correlation_window_count)
            history = CorrelationHistory(
                task_type=task_type,
                windows_requested=config.correlation_window_count,
                windows_available=len(windows),
                windows=windows,
            )
            correlation_histories.append(history)
    except Exception as e:
        raise DataSourceError(f"Confidence source unavailable: {e}")

    # Fetch alerts (empty for now)
    alerts = []

    return SystemReport(
        report_id=report_id,
        generated_at=generated_at,
        task_confidence_reports=task_confidence_reports,
        cost_breakdown=cost_breakdown,
        correlation_histories=correlation_histories,
        alerts=alerts,
        task_types_count=len(task_confidence_reports),
    )


def format_report(report: SystemReport, output_format: ReportFormat) -> FormattedReport:
    """
    Formats a SystemReport into a string representation using the specified format.
    Delegates to ReportFormatter.format_report().
    """
    formatter = ReportFormatter()
    return formatter.format_report(report, output_format)


async def deliver_webhook(
    report: SystemReport,
    webhook_url: str,
    timeout_seconds: int,
    http_client,
) -> DeliveryResult:
    """
    Delivers a SystemReport as JSON via HTTP POST to the configured webhook URL.
    Delegates to WebhookDelivery.deliver_webhook().
    """
    delivery = WebhookDelivery(http_client)
    return await delivery.deliver_webhook(report, webhook_url, timeout_seconds)


# ══════════════════════════════════════════════════════════════════════════════
# Exports
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    'EventType',
    'TrendDirection',
    'AlertSeverity',
    'AlertType',
    'ReportFormat',
    'DeliveryStatus',

    # Type Aliases
    'EventTypeList',
    'AuditEntryList',
    'CorrelationWindowList',
    'TaskConfidenceReportList',
    'CorrelationHistoryList',
    'AlertList',

    # Audit Log Types
    'AuditEntry',
    'AuditConfig',
    'JsonLinesAuditLogger',

    # Report Generator Models
    'TaskConfidenceReport',
    'CostBreakdown',
    'CorrelationWindow',
    'CorrelationHistory',
    'Alert',
    'SystemReport',
    'ReportConfig',
    'DeliveryResult',
    'FormattedReport',

    # Exceptions
    'ConfigValidationError',
    'DataSourceError',
    'EmptyReportError',
    'FormattingError',

    # Core Classes
    'ReportGenerator',
    'ReportFormatter',
    'WebhookDelivery',

    # Protocols
    'ConfidenceDataSource',
    'BudgetDataSource',
    'AuditDataSource',

    # Functions
    'validate_report_config',
    'generate_report',
    'format_report',
    'deliver_webhook',
]
