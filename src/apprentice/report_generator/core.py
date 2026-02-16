"""
Core classes for report generation, formatting, and webhook delivery.
"""
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from .models import (
    SystemReport,
    ReportConfig,
    FormattedReport,
    ReportFormat,
    DeliveryResult,
    DeliveryStatus,
)
from .exceptions import DataSourceError, EmptyReportError, FormattingError


class ReportGenerator:
    """Generates SystemReports by aggregating data from injected data sources."""

    def __init__(self, confidence_source: Any, budget_source: Any, audit_source: Any):
        self.confidence_source = confidence_source
        self.budget_source = budget_source
        self.audit_source = audit_source

    def generate_report(self, config: ReportConfig) -> SystemReport:
        """Generate a complete SystemReport from all data sources."""
        report_id = str(uuid.uuid4())
        generated_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

        # Fetch task types from confidence source
        try:
            task_types = self.confidence_source.get_task_types()
        except Exception as e:
            raise DataSourceError(f"Confidence source unavailable: {e}")

        if not task_types:
            raise EmptyReportError("No task types found in system")

        # Fetch task confidence reports
        task_confidence_reports = []
        try:
            for task_type in task_types:
                report = self.confidence_source.get_confidence_report(task_type)
                task_confidence_reports.append(report)
        except Exception as e:
            raise DataSourceError(f"Confidence source unavailable: {e}")

        # Fetch cost breakdown
        try:
            cost_breakdown = self.budget_source.get_cost_breakdown()
        except Exception as e:
            raise DataSourceError(f"Budget source unavailable: {e}")

        # Fetch correlation histories
        correlation_histories = []
        try:
            for task_type in task_types:
                history = self.audit_source.get_correlation_history(
                    task_type, config.correlation_window_count
                )
                correlation_histories.append(history)
        except Exception as e:
            raise DataSourceError(f"Audit source unavailable: {e}")

        # Fetch alerts
        try:
            alerts = self.audit_source.get_alerts()
        except Exception as e:
            raise DataSourceError(f"Audit source unavailable: {e}")

        return SystemReport(
            report_id=report_id,
            generated_at=generated_at,
            task_confidence_reports=task_confidence_reports,
            cost_breakdown=cost_breakdown,
            correlation_histories=correlation_histories,
            alerts=alerts,
            task_types_count=len(task_confidence_reports),
        )


class ReportFormatter:
    """Formats SystemReports into various output formats."""

    def format_report(self, report: SystemReport, output_format: str) -> FormattedReport:
        """Format a SystemReport into the requested output format."""
        try:
            if output_format == "json" or output_format == ReportFormat.json:
                content = self._format_json(report)
            elif output_format == "cli_summary" or output_format == ReportFormat.cli_summary:
                content = self._format_cli_summary(report)
            else:
                raise FormattingError(f"Unknown format: {output_format}")

            return FormattedReport(
                content=content,
                format=ReportFormat(output_format),
                report_id=report.report_id,
                content_length=len(content),
            )
        except (ValueError, TypeError, AttributeError) as e:
            raise FormattingError(f"Serialization error: {e}")

    def _format_json(self, report: SystemReport) -> str:
        """Format report as JSON."""
        data = report.model_dump()
        return json.dumps(data, indent=2)

    def _format_cli_summary(self, report: SystemReport) -> str:
        """Format report as human-readable CLI summary."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"SYSTEM REPORT - {report.report_id}")
        lines.append(f"Generated: {report.generated_at}")
        lines.append("=" * 80)
        lines.append("")

        # Task Confidence Section
        lines.append("TASK CONFIDENCE REPORTS")
        lines.append("-" * 80)
        for tcr in report.task_confidence_reports:
            lines.append(f"  Task: {tcr.task_type}")
            lines.append(f"    Phase: {tcr.phase}")
            lines.append(f"    Confidence: {tcr.confidence_score:.2f}")
            lines.append(f"    Trend: {tcr.trend.value}")
            lines.append(f"    Sampling Frequency: {tcr.sampling_frequency:.2f}")
            lines.append("")

        # Cost Breakdown Section
        lines.append("COST BREAKDOWN")
        lines.append("-" * 80)
        cb = report.cost_breakdown
        lines.append(f"  Total Remote Spend: ${cb.total_remote_spend}")
        lines.append(f"  Estimated Local Savings: ${cb.estimated_local_savings}")
        lines.append(f"  Budget Limit: ${cb.budget_limit}")
        lines.append(f"  Budget Remaining: ${cb.budget_remaining}")
        lines.append(f"  Budget Utilization: {cb.budget_utilization_pct:.2f}%")
        lines.append("")

        # Correlation Histories Section
        lines.append("CORRELATION HISTORIES")
        lines.append("-" * 80)
        for ch in report.correlation_histories:
            lines.append(f"  Task: {ch.task_type}")
            lines.append(f"    Windows Requested: {ch.windows_requested}")
            lines.append(f"    Windows Available: {ch.windows_available}")
            lines.append("")

        # Alerts Section
        lines.append("ALERTS")
        lines.append("-" * 80)
        if report.alerts:
            for alert in report.alerts:
                lines.append(f"  [{alert.severity.value.upper()}] {alert.alert_type.value}")
                lines.append(f"    Message: {alert.message}")
                if alert.task_type:
                    lines.append(f"    Task Type: {alert.task_type}")
                lines.append(f"    Timestamp: {alert.timestamp}")
                lines.append("")
        else:
            lines.append("  No alerts")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)


class WebhookDelivery:
    """Handles webhook delivery of reports via HTTP POST."""

    def __init__(self, client: Any):
        self.client = client

    async def deliver_webhook(
        self, report: SystemReport, webhook_url: str, timeout_seconds: int
    ) -> DeliveryResult:
        """Deliver a SystemReport via HTTP POST to webhook URL."""
        start_time = time.time()
        attempt_timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

        try:
            # Serialize report to JSON
            try:
                json_data = report.model_dump()
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                return DeliveryResult(
                    status=DeliveryStatus.failure,
                    http_status_code=0,
                    url=webhook_url,
                    attempt_timestamp=attempt_timestamp,
                    error_message=f"Serialization failure: {e}",
                    duration_ms=duration_ms,
                )

            # Attempt HTTP POST
            try:
                response = await self.client.post(
                    webhook_url,
                    json=json_data,
                    timeout=timeout_seconds,
                )

                duration_ms = int((time.time() - start_time) * 1000)

                # Check response status
                if 200 <= response.status_code <= 299:
                    return DeliveryResult(
                        status=DeliveryStatus.success,
                        http_status_code=response.status_code,
                        url=webhook_url,
                        attempt_timestamp=attempt_timestamp,
                        error_message=None,
                        duration_ms=duration_ms,
                    )
                else:
                    return DeliveryResult(
                        status=DeliveryStatus.failure,
                        http_status_code=response.status_code,
                        url=webhook_url,
                        attempt_timestamp=attempt_timestamp,
                        error_message=f"Webhook returned non-2xx status code: {response.status_code}",
                        duration_ms=duration_ms,
                    )

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)

                # Determine error type
                error_str = str(e).lower()

                # Check for timeout
                if 'timeout' in error_str or 'timed out' in error_str:
                    return DeliveryResult(
                        status=DeliveryStatus.timeout,
                        http_status_code=0,
                        url=webhook_url,
                        attempt_timestamp=attempt_timestamp,
                        error_message=f"Request timeout: {e}",
                        duration_ms=duration_ms,
                    )

                # All other errors are failures
                return DeliveryResult(
                    status=DeliveryStatus.failure,
                    http_status_code=0,
                    url=webhook_url,
                    attempt_timestamp=attempt_timestamp,
                    error_message=str(e),
                    duration_ms=duration_ms,
                )

        except Exception as e:
            # Catch-all for any unexpected errors
            duration_ms = int((time.time() - start_time) * 1000)
            return DeliveryResult(
                status=DeliveryStatus.failure,
                http_status_code=0,
                url=webhook_url,
                attempt_timestamp=attempt_timestamp,
                error_message=f"Unexpected error: {e}",
                duration_ms=duration_ms,
            )
