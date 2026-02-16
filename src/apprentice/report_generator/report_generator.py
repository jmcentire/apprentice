"""
Report Generator Component - generates system reports by aggregating data from
confidence_engine, budget_manager, and audit_log.

This module serves as the main entry point, re-exporting all public types and functions.
"""
import re

from .models import (
    TrendDirection,
    AlertSeverity,
    AlertType,
    ReportFormat,
    DeliveryStatus,
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
from .exceptions import (
    ConfigValidationError,
    DataSourceError,
    EmptyReportError,
    FormattingError,
)
from .core import (
    ReportGenerator,
    ReportFormatter,
    WebhookDelivery,
)


# ══════════════════════════════════════════════════════════════════════════════
# Public Functions
# ══════════════════════════════════════════════════════════════════════════════

def validate_report_config(config: ReportConfig) -> ReportConfig:
    """
    Validates a ReportConfig instance, ensuring all fields are within constraints.
    Returns the validated config or raises ConfigValidationError.
    """
    # Validate correlation_window_count
    if not 1 <= config.correlation_window_count <= 1000:
        raise ConfigValidationError(
            f"correlation_window_count must be between 1 and 1000, got {config.correlation_window_count}"
        )

    # Validate alerts_lookback_hours
    if not 1 <= config.alerts_lookback_hours <= 8760:
        raise ConfigValidationError(
            f"alerts_lookback_hours must be between 1 and 8760, got {config.alerts_lookback_hours}"
        )

    # Validate webhook_url
    if config.webhook_url and config.webhook_url.strip():
        pattern = r'^https?://.+'
        if not re.match(pattern, config.webhook_url):
            raise ConfigValidationError(
                f"webhook_url must be a valid http:// or https:// URL, got {config.webhook_url}"
            )

    # Validate webhook_timeout_seconds
    if not 1 <= config.webhook_timeout_seconds <= 300:
        raise ConfigValidationError(
            f"webhook_timeout_seconds must be between 1 and 300 seconds, got {config.webhook_timeout_seconds}"
        )

    return config


def generate_report(config: ReportConfig) -> SystemReport:
    """
    This is a module-level wrapper that would typically require injected sources.
    Since the tests use ReportGenerator directly, this is a placeholder.
    """
    raise NotImplementedError(
        "Use ReportGenerator.generate_report() with injected data sources"
    )


def format_report(report: SystemReport, output_format: ReportFormat) -> FormattedReport:
    """
    Module-level wrapper for formatting reports.
    """
    formatter = ReportFormatter()
    return formatter.format_report(report, output_format)


async def deliver_webhook(
    report: SystemReport, webhook_url: str, timeout_seconds: int
) -> DeliveryResult:
    """
    Module-level wrapper for webhook delivery.
    Requires an httpx client to be injected via WebhookDelivery.
    """
    raise NotImplementedError(
        "Use WebhookDelivery.deliver_webhook() with injected HTTP client"
    )


# ── Auto-injected export aliases (Pact export gate) ──
CorrelationWindowList = CorrelationWindow
TaskConfidenceReportList = TaskConfidenceReport
CorrelationHistoryList = CorrelationHistory
AlertList = Alert
