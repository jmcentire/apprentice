"""
Contract test suite for the report_generator component.

Tests cover all 4 public functions:
  - validate_report_config
  - generate_report
  - format_report
  - deliver_webhook

All dependencies (confidence_engine, budget_manager, audit_log) are mocked via DI.
"""
import json
import uuid
import asyncio
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest

# Attempt to import the component under test
from src.report_generator import (
    ReportGenerator,
    ReportFormatter,
    WebhookDelivery,
    validate_report_config,
    ReportConfig,
    SystemReport,
    TaskConfidenceReport,
    CostBreakdown,
    CorrelationHistory,
    CorrelationWindow,
    Alert,
    FormattedReport,
    DeliveryResult,
)


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def make_report_config(**overrides):
    """Factory for ReportConfig with sensible defaults."""
    defaults = dict(
        correlation_window_count=10,
        alerts_lookback_hours=24,
        webhook_url="https://hooks.example.com/report",
        webhook_timeout_seconds=30,
    )
    defaults.update(overrides)
    return ReportConfig(**defaults)


def make_cost_breakdown(**overrides):
    defaults = dict(
        total_remote_spend="50.00",
        estimated_local_savings="120.00",
        cost_per_task={"classification": "10.00", "extraction": "40.00"},
        budget_limit="200.00",
        budget_remaining="150.00",
        budget_utilization_pct=25.0,
    )
    defaults.update(overrides)
    return CostBreakdown(**defaults)


def make_task_confidence_report(**overrides):
    defaults = dict(
        task_type="classification",
        phase="calibration",
        confidence_score=0.85,
        trend="stable",
        sampling_frequency=0.1,
    )
    defaults.update(overrides)
    return TaskConfidenceReport(**defaults)


def make_correlation_window(**overrides):
    defaults = dict(
        task_type="classification",
        window_start="2024-01-01T00:00:00Z",
        window_end="2024-01-01T01:00:00Z",
        sample_count=100,
        correlation_score=0.92,
        local_agreement_rate=0.88,
    )
    defaults.update(overrides)
    return CorrelationWindow(**defaults)


def make_correlation_history(**overrides):
    defaults = dict(
        task_type="classification",
        windows_requested=10,
        windows_available=5,
        windows=[make_correlation_window()],
    )
    defaults.update(overrides)
    return CorrelationHistory(**defaults)


def make_alert(**overrides):
    defaults = dict(
        alert_id=str(uuid.uuid4()),
        alert_type="quality_degradation",
        severity="warning",
        message="Confidence below threshold",
        task_type="classification",
        timestamp="2024-01-15T12:00:00Z",
    )
    defaults.update(overrides)
    return Alert(**defaults)


def make_system_report(**overrides):
    """Factory for a fully-populated SystemReport."""
    task_reports = [
        make_task_confidence_report(task_type="classification"),
        make_task_confidence_report(task_type="extraction", confidence_score=0.72, trend="declining"),
    ]
    defaults = dict(
        report_id=str(uuid.uuid4()),
        generated_at="2024-01-15T12:00:00Z",
        task_confidence_reports=task_reports,
        cost_breakdown=make_cost_breakdown(),
        correlation_histories=[
            make_correlation_history(task_type="classification"),
            make_correlation_history(task_type="extraction"),
        ],
        alerts=[make_alert()],
        task_types_count=2,
    )
    defaults.update(overrides)
    return SystemReport(**defaults)


def _is_valid_uuid4(s: str) -> bool:
    try:
        val = uuid.UUID(s, version=4)
        return str(val) == s
    except (ValueError, AttributeError):
        return False


def _is_utc_iso8601(s: str) -> bool:
    """Check whether string is a valid ISO-8601 UTC datetime."""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.tzinfo is not None or s.endswith("Z")
    except (ValueError, AttributeError):
        return False


def _is_valid_decimal_string(s: str) -> bool:
    try:
        Decimal(s)
        return True
    except (InvalidOperation, ValueError):
        return False


@pytest.fixture
def valid_config():
    return make_report_config()


@pytest.fixture
def mock_confidence_source():
    source = MagicMock()
    source.get_task_types.return_value = ["classification", "extraction"]
    source.get_confidence_report.side_effect = lambda tt: make_task_confidence_report(task_type=tt)
    return source


@pytest.fixture
def mock_budget_source():
    source = MagicMock()
    source.get_cost_breakdown.return_value = make_cost_breakdown()
    return source


@pytest.fixture
def mock_audit_source():
    source = MagicMock()
    source.get_correlation_history.side_effect = lambda tt, count: make_correlation_history(
        task_type=tt, windows_requested=count, windows_available=min(count, 5)
    )
    source.get_alerts.return_value = [make_alert()]
    return source


@pytest.fixture
def report_generator(mock_confidence_source, mock_budget_source, mock_audit_source):
    return ReportGenerator(
        confidence_source=mock_confidence_source,
        budget_source=mock_budget_source,
        audit_source=mock_audit_source,
    )


@pytest.fixture
def sample_report():
    return make_system_report()


@pytest.fixture
def mock_http_client():
    client = AsyncMock()
    return client


@pytest.fixture
def webhook_delivery(mock_http_client):
    return WebhookDelivery(client=mock_http_client)


# ===========================================================================
# validate_report_config tests
# ===========================================================================

class TestValidateReportConfig:
    """Tests for validate_report_config."""

    def test_valid_config_passes(self, valid_config):
        """Valid ReportConfig passes validation and returns equivalent config."""
        result = validate_report_config(valid_config)
        assert result is not None, "validate_report_config should return a config"
        assert result.correlation_window_count == valid_config.correlation_window_count
        assert result.alerts_lookback_hours == valid_config.alerts_lookback_hours
        assert result.webhook_url == valid_config.webhook_url
        assert result.webhook_timeout_seconds == valid_config.webhook_timeout_seconds

    @pytest.mark.parametrize("window_count", [1, 500, 1000])
    def test_valid_correlation_window_count_boundaries(self, window_count):
        """correlation_window_count within [1, 1000] passes validation."""
        config = make_report_config(correlation_window_count=window_count)
        result = validate_report_config(config)
        assert result.correlation_window_count == window_count

    @pytest.mark.parametrize("hours", [1, 4380, 8760])
    def test_valid_alerts_lookback_hours_boundaries(self, hours):
        """alerts_lookback_hours within [1, 8760] passes validation."""
        config = make_report_config(alerts_lookback_hours=hours)
        result = validate_report_config(config)
        assert result.alerts_lookback_hours == hours

    @pytest.mark.parametrize("timeout", [1, 150, 300])
    def test_valid_webhook_timeout_boundaries(self, timeout):
        """webhook_timeout_seconds within [1, 300] passes validation."""
        config = make_report_config(webhook_timeout_seconds=timeout)
        result = validate_report_config(config)
        assert result.webhook_timeout_seconds == timeout

    def test_empty_webhook_url_is_valid(self):
        """Empty webhook_url is acceptable (webhook delivery is optional)."""
        config = make_report_config(webhook_url="")
        result = validate_report_config(config)
        assert result.webhook_url == ""

    @pytest.mark.parametrize("url", [
        "https://hooks.example.com/report",
        "http://localhost:8080/webhook",
        "https://internal.corp.net/api/hook",
    ])
    def test_valid_http_https_urls(self, url):
        """Valid HTTP and HTTPS URLs pass validation."""
        config = make_report_config(webhook_url=url)
        result = validate_report_config(config)
        assert result.webhook_url == url

    # --- Error cases ---

    @pytest.mark.parametrize("bad_count", [0, -1, -100, 1001, 9999])
    def test_error_invalid_correlation_window_count(self, bad_count):
        """correlation_window_count outside [1, 1000] raises invalid_correlation_window_count."""
        config = make_report_config(correlation_window_count=bad_count)
        with pytest.raises(Exception) as exc_info:
            validate_report_config(config)
        exc_text = str(exc_info.value).lower()
        assert "correlation_window_count" in exc_text or "correlation" in exc_text, (
            f"Error should mention correlation_window_count, got: {exc_info.value}"
        )

    @pytest.mark.parametrize("bad_hours", [0, -1, 8761, 99999])
    def test_error_invalid_alerts_lookback_hours(self, bad_hours):
        """alerts_lookback_hours outside [1, 8760] raises invalid_alerts_lookback_hours."""
        config = make_report_config(alerts_lookback_hours=bad_hours)
        with pytest.raises(Exception) as exc_info:
            validate_report_config(config)
        exc_text = str(exc_info.value).lower()
        assert "alerts_lookback" in exc_text or "lookback" in exc_text or "alerts" in exc_text, (
            f"Error should mention alerts_lookback_hours, got: {exc_info.value}"
        )

    @pytest.mark.parametrize("bad_url", [
        "ftp://example.com/hook",
        "not-a-url",
        "tcp://example.com",
        "file:///etc/passwd",
        "://missing-scheme.com",
    ])
    def test_error_invalid_webhook_url(self, bad_url):
        """Non-empty webhook_url that is not HTTP(S) raises invalid_webhook_url."""
        config = make_report_config(webhook_url=bad_url)
        with pytest.raises(Exception) as exc_info:
            validate_report_config(config)
        exc_text = str(exc_info.value).lower()
        assert "webhook_url" in exc_text or "webhook" in exc_text or "url" in exc_text, (
            f"Error should mention webhook_url, got: {exc_info.value}"
        )

    @pytest.mark.parametrize("bad_timeout", [0, -1, -50, 301, 1000])
    def test_error_invalid_webhook_timeout(self, bad_timeout):
        """webhook_timeout_seconds outside [1, 300] raises invalid_webhook_timeout."""
        config = make_report_config(webhook_timeout_seconds=bad_timeout)
        with pytest.raises(Exception) as exc_info:
            validate_report_config(config)
        exc_text = str(exc_info.value).lower()
        assert "timeout" in exc_text or "webhook_timeout" in exc_text, (
            f"Error should mention webhook_timeout, got: {exc_info.value}"
        )

    def test_idempotent_validation(self, valid_config):
        """Validating an already-valid config twice yields equivalent results."""
        result1 = validate_report_config(valid_config)
        result2 = validate_report_config(result1)
        assert result1.correlation_window_count == result2.correlation_window_count
        assert result1.alerts_lookback_hours == result2.alerts_lookback_hours
        assert result1.webhook_url == result2.webhook_url
        assert result1.webhook_timeout_seconds == result2.webhook_timeout_seconds


# ===========================================================================
# generate_report tests
# ===========================================================================

class TestGenerateReport:
    """Tests for generate_report."""

    def test_happy_path_assembles_complete_report(self, report_generator, valid_config):
        """generate_report returns a complete SystemReport from all three data sources."""
        report = report_generator.generate_report(valid_config)

        assert report is not None, "Should return a SystemReport"
        assert isinstance(report.report_id, str), "report_id should be a string"
        assert len(report.task_confidence_reports) > 0, "Should have task confidence reports"
        assert report.cost_breakdown is not None, "Should have cost breakdown"
        assert report.correlation_histories is not None, "Should have correlation histories"
        assert report.alerts is not None, "Should have alerts list"

    def test_report_id_is_valid_uuid4(self, report_generator, valid_config):
        """Returned report_id is a valid UUID v4 string."""
        report = report_generator.generate_report(valid_config)
        assert _is_valid_uuid4(report.report_id), (
            f"report_id should be a valid UUID v4, got: {report.report_id}"
        )

    def test_generated_at_is_utc_iso8601(self, report_generator, valid_config):
        """generated_at is set to current UTC time in ISO-8601 format."""
        report = report_generator.generate_report(valid_config)
        assert _is_utc_iso8601(report.generated_at), (
            f"generated_at should be UTC ISO-8601, got: {report.generated_at}"
        )

    def test_task_types_count_matches_reports_length(self, report_generator, valid_config):
        """task_types_count equals the length of task_confidence_reports."""
        report = report_generator.generate_report(valid_config)
        assert report.task_types_count == len(report.task_confidence_reports), (
            f"task_types_count ({report.task_types_count}) should equal "
            f"len(task_confidence_reports) ({len(report.task_confidence_reports)})"
        )

    def test_one_report_per_task_type(self, report_generator, valid_config, mock_confidence_source):
        """task_confidence_reports has exactly one entry per task type from ConfidenceDataSource."""
        expected_task_types = mock_confidence_source.get_task_types.return_value
        report = report_generator.generate_report(valid_config)

        report_task_types = [r.task_type for r in report.task_confidence_reports]
        assert sorted(report_task_types) == sorted(expected_task_types), (
            f"Expected task types {expected_task_types}, got {report_task_types}"
        )

    def test_correlation_windows_requested_matches_config(self, report_generator, valid_config):
        """Each CorrelationHistory.windows_requested equals config.correlation_window_count."""
        report = report_generator.generate_report(valid_config)
        for history in report.correlation_histories:
            assert history.windows_requested == valid_config.correlation_window_count, (
                f"windows_requested ({history.windows_requested}) should equal "
                f"config.correlation_window_count ({valid_config.correlation_window_count}) "
                f"for task_type={history.task_type}"
            )

    def test_correlation_windows_available_lte_requested(self, report_generator, valid_config):
        """Each CorrelationHistory.windows_available <= windows_requested."""
        report = report_generator.generate_report(valid_config)
        for history in report.correlation_histories:
            assert history.windows_available <= history.windows_requested, (
                f"windows_available ({history.windows_available}) should be <= "
                f"windows_requested ({history.windows_requested}) "
                f"for task_type={history.task_type}"
            )

    def test_budget_utilization_consistent(self, report_generator, valid_config):
        """budget_utilization_pct is consistent with total_remote_spend and budget_limit."""
        report = report_generator.generate_report(valid_config)
        cb = report.cost_breakdown
        total = Decimal(cb.total_remote_spend)
        limit = Decimal(cb.budget_limit)
        if limit > 0:
            expected_pct = float((total / limit) * 100)
            assert abs(cb.budget_utilization_pct - expected_pct) < 0.01, (
                f"budget_utilization_pct ({cb.budget_utilization_pct}) should be ~"
                f"{expected_pct} based on spend={cb.total_remote_spend}/limit={cb.budget_limit}"
            )

    def test_decimal_string_fields_are_valid(self, report_generator, valid_config):
        """All Decimal string fields in cost_breakdown are valid decimal representations."""
        report = report_generator.generate_report(valid_config)
        cb = report.cost_breakdown
        decimal_fields = ["total_remote_spend", "estimated_local_savings", "budget_limit", "budget_remaining"]
        for field_name in decimal_fields:
            value = getattr(cb, field_name)
            assert _is_valid_decimal_string(value), (
                f"cost_breakdown.{field_name} should be a valid Decimal string, got: {value}"
            )
        for task_type, cost in cb.cost_per_task.items():
            assert _is_valid_decimal_string(cost), (
                f"cost_per_task[{task_type}] should be a valid Decimal string, got: {cost}"
            )

    def test_unique_report_ids_across_calls(self, report_generator, valid_config):
        """Two consecutive generate_report calls produce different report_ids."""
        report1 = report_generator.generate_report(valid_config)
        report2 = report_generator.generate_report(valid_config)
        assert report1.report_id != report2.report_id, (
            "Each report should have a unique report_id"
        )

    def test_all_datetime_fields_utc_iso8601(self, report_generator, valid_config):
        """All datetime strings throughout a generated report are UTC ISO-8601 formatted."""
        report = report_generator.generate_report(valid_config)
        assert _is_utc_iso8601(report.generated_at), f"generated_at: {report.generated_at}"
        for alert in report.alerts:
            assert _is_utc_iso8601(alert.timestamp), f"alert.timestamp: {alert.timestamp}"
        for history in report.correlation_histories:
            for window in history.windows:
                assert _is_utc_iso8601(window.window_start), f"window_start: {window.window_start}"
                assert _is_utc_iso8601(window.window_end), f"window_end: {window.window_end}"

    def test_generation_is_read_only(self, report_generator, valid_config,
                                      mock_confidence_source, mock_budget_source, mock_audit_source):
        """generate_report does not mutate data source state (only reads)."""
        report_generator.generate_report(valid_config)
        # Verify no write/mutate methods were called on sources
        for source in [mock_confidence_source, mock_budget_source, mock_audit_source]:
            for call in source.method_calls:
                method_name = call[0]
                # Only read-like methods should be called
                assert not any(
                    verb in method_name.lower()
                    for verb in ["set", "update", "delete", "write", "save", "put", "post", "remove"]
                ), f"Data source should not be mutated, but '{method_name}' was called"

    # --- Error cases ---

    def test_error_confidence_source_unavailable(self, mock_confidence_source, mock_budget_source,
                                                   mock_audit_source, valid_config):
        """Raises confidence_source_unavailable when ConfidenceDataSource raises."""
        mock_confidence_source.get_task_types.side_effect = ConnectionError("source down")
        generator = ReportGenerator(
            confidence_source=mock_confidence_source,
            budget_source=mock_budget_source,
            audit_source=mock_audit_source,
        )
        with pytest.raises(Exception) as exc_info:
            generator.generate_report(valid_config)
        exc_text = str(exc_info.value).lower()
        assert "confidence" in exc_text or "unavailable" in exc_text, (
            f"Error should indicate confidence source unavailable, got: {exc_info.value}"
        )

    def test_error_budget_source_unavailable(self, mock_confidence_source, mock_budget_source,
                                               mock_audit_source, valid_config):
        """Raises budget_source_unavailable when BudgetDataSource raises."""
        mock_budget_source.get_cost_breakdown.side_effect = ConnectionError("budget source down")
        generator = ReportGenerator(
            confidence_source=mock_confidence_source,
            budget_source=mock_budget_source,
            audit_source=mock_audit_source,
        )
        with pytest.raises(Exception) as exc_info:
            generator.generate_report(valid_config)
        exc_text = str(exc_info.value).lower()
        assert "budget" in exc_text or "unavailable" in exc_text, (
            f"Error should indicate budget source unavailable, got: {exc_info.value}"
        )

    def test_error_audit_source_unavailable(self, mock_confidence_source, mock_budget_source,
                                              mock_audit_source, valid_config):
        """Raises audit_source_unavailable when AuditDataSource raises."""
        mock_audit_source.get_correlation_history.side_effect = ConnectionError("audit source down")
        generator = ReportGenerator(
            confidence_source=mock_confidence_source,
            budget_source=mock_budget_source,
            audit_source=mock_audit_source,
        )
        with pytest.raises(Exception) as exc_info:
            generator.generate_report(valid_config)
        exc_text = str(exc_info.value).lower()
        assert "audit" in exc_text or "unavailable" in exc_text, (
            f"Error should indicate audit source unavailable, got: {exc_info.value}"
        )

    def test_error_no_task_types_found(self, mock_confidence_source, mock_budget_source,
                                        mock_audit_source, valid_config):
        """Raises no_task_types_found when ConfidenceDataSource returns empty task types."""
        mock_confidence_source.get_task_types.return_value = []
        generator = ReportGenerator(
            confidence_source=mock_confidence_source,
            budget_source=mock_budget_source,
            audit_source=mock_audit_source,
        )
        with pytest.raises(Exception) as exc_info:
            generator.generate_report(valid_config)
        exc_text = str(exc_info.value).lower()
        assert "no_task_types" in exc_text or "no task" in exc_text or "empty" in exc_text, (
            f"Error should indicate no task types found, got: {exc_info.value}"
        )


# ===========================================================================
# format_report tests
# ===========================================================================

class TestFormatReport:
    """Tests for format_report."""

    @pytest.fixture
    def formatter(self):
        return ReportFormatter()

    def test_json_format_happy_path(self, formatter, sample_report):
        """format_report with json format returns valid parseable JSON."""
        result = formatter.format_report(sample_report, "json")

        assert isinstance(result, FormattedReport), "Should return a FormattedReport"
        # Verify JSON is parseable
        parsed = json.loads(result.content)
        assert isinstance(parsed, dict), "JSON content should parse to a dict"

    def test_cli_summary_format_happy_path(self, formatter, sample_report):
        """format_report with cli_summary format returns human-readable multi-line string."""
        result = formatter.format_report(sample_report, "cli_summary")

        assert isinstance(result, FormattedReport), "Should return a FormattedReport"
        assert "\n" in result.content, "CLI summary should be multi-line"
        assert len(result.content) > 0, "CLI summary should not be empty"

    def test_report_id_preserved(self, formatter, sample_report):
        """FormattedReport.report_id matches input SystemReport.report_id."""
        result = formatter.format_report(sample_report, "json")
        assert result.report_id == sample_report.report_id, (
            f"FormattedReport.report_id ({result.report_id}) should match "
            f"input report.report_id ({sample_report.report_id})"
        )

    @pytest.mark.parametrize("fmt", ["json", "cli_summary"])
    def test_content_length_matches_content(self, formatter, sample_report, fmt):
        """FormattedReport.content_length equals len(FormattedReport.content)."""
        result = formatter.format_report(sample_report, fmt)
        assert result.content_length == len(result.content), (
            f"content_length ({result.content_length}) should equal "
            f"len(content) ({len(result.content)}) for format={fmt}"
        )

    @pytest.mark.parametrize("fmt", ["json", "cli_summary"])
    def test_format_field_matches_requested(self, formatter, sample_report, fmt):
        """FormattedReport.format matches the requested output_format."""
        result = formatter.format_report(sample_report, fmt)
        assert result.format == fmt, (
            f"FormattedReport.format ({result.format}) should equal requested format ({fmt})"
        )

    def test_json_decimal_precision_preserved(self, formatter):
        """JSON output preserves Decimal string precision (not converted to float)."""
        precise_cost = make_cost_breakdown(
            total_remote_spend="123.456789012345",
            budget_limit="1000.000000000001",
            budget_remaining="876.543210987656",
            budget_utilization_pct=12.3456789,
        )
        report = make_system_report(cost_breakdown=precise_cost)
        result = formatter.format_report(report, "json")
        parsed = json.loads(result.content)

        # Navigate to cost_breakdown in parsed JSON
        cost_data = parsed.get("cost_breakdown", parsed)
        # The precise decimal strings should appear verbatim in the content
        assert "123.456789012345" in result.content, (
            "JSON should preserve full decimal precision for total_remote_spend"
        )
        assert "1000.000000000001" in result.content, (
            "JSON should preserve full decimal precision for budget_limit"
        )

    def test_json_round_trip_contains_report_id(self, formatter, sample_report):
        """JSON output can be parsed back and contains report_id."""
        result = formatter.format_report(sample_report, "json")
        parsed = json.loads(result.content)
        assert "report_id" in parsed, "JSON output should contain report_id field"
        assert parsed["report_id"] == sample_report.report_id

    def test_cli_summary_is_multiline_with_sections(self, formatter, sample_report):
        """CLI summary format produces multi-line content with sections."""
        result = formatter.format_report(sample_report, "cli_summary")
        lines = result.content.strip().split("\n")
        assert len(lines) > 5, (
            f"CLI summary should have multiple lines for sections, got {len(lines)} lines"
        )

    def test_error_serialization_error(self, formatter):
        """Raises serialization_error when report contains unserializable data."""
        # Create a report with problematic data that can't be serialized
        report = make_system_report()
        # Monkey-patch a field to be unserializable (e.g., via mock)
        bad_report = MagicMock(spec=SystemReport)
        bad_report.report_id = sample_report_id = "test-id"
        # Make model_dump or serialization raise
        bad_report.model_dump.side_effect = ValueError("Cannot serialize")
        bad_report.dict.side_effect = ValueError("Cannot serialize")

        with pytest.raises(Exception) as exc_info:
            formatter.format_report(bad_report, "json")
        # Should raise some serialization-related error
        assert exc_info.value is not None, "Should raise on unserializable input"


# ===========================================================================
# deliver_webhook tests (async)
# ===========================================================================

class TestDeliverWebhook:
    """Tests for deliver_webhook (async)."""

    @pytest.fixture
    def mock_response_200(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "OK"
        return resp

    @pytest.fixture
    def mock_response_500(self):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "Internal Server Error"
        return resp

    @pytest.mark.asyncio
    async def test_successful_delivery(self, webhook_delivery, mock_http_client,
                                        sample_report, mock_response_200):
        """Successful webhook delivery returns DeliveryResult with status=success."""
        mock_http_client.post.return_value = mock_response_200
        url = "https://hooks.example.com/report"

        result = await webhook_delivery.deliver_webhook(sample_report, url, timeout_seconds=30)

        assert isinstance(result, DeliveryResult), "Should return a DeliveryResult"
        assert result.status == "success", f"Status should be success, got: {result.status}"
        assert 200 <= result.http_status_code <= 299, (
            f"Success should have 2xx status code, got: {result.http_status_code}"
        )

    @pytest.mark.asyncio
    async def test_url_preserved_in_result(self, webhook_delivery, mock_http_client,
                                            sample_report, mock_response_200):
        """DeliveryResult.url matches the input webhook_url."""
        mock_http_client.post.return_value = mock_response_200
        url = "https://hooks.example.com/my-specific-endpoint"

        result = await webhook_delivery.deliver_webhook(sample_report, url, timeout_seconds=30)
        assert result.url == url, f"result.url ({result.url}) should equal input url ({url})"

    @pytest.mark.asyncio
    async def test_duration_nonnegative(self, webhook_delivery, mock_http_client,
                                         sample_report, mock_response_200):
        """DeliveryResult.duration_ms is always non-negative."""
        mock_http_client.post.return_value = mock_response_200

        result = await webhook_delivery.deliver_webhook(
            sample_report, "https://hooks.example.com/report", timeout_seconds=30
        )
        assert result.duration_ms >= 0, (
            f"duration_ms should be non-negative, got: {result.duration_ms}"
        )

    @pytest.mark.asyncio
    async def test_attempt_timestamp_utc_iso8601(self, webhook_delivery, mock_http_client,
                                                   sample_report, mock_response_200):
        """DeliveryResult.attempt_timestamp is UTC ISO-8601 formatted."""
        mock_http_client.post.return_value = mock_response_200

        result = await webhook_delivery.deliver_webhook(
            sample_report, "https://hooks.example.com/report", timeout_seconds=30
        )
        assert _is_utc_iso8601(result.attempt_timestamp), (
            f"attempt_timestamp should be UTC ISO-8601, got: {result.attempt_timestamp}"
        )

    @pytest.mark.asyncio
    async def test_never_raises_on_failure(self, webhook_delivery, mock_http_client, sample_report):
        """deliver_webhook never raises an exception even on total failure."""
        mock_http_client.post.side_effect = Exception("Catastrophic failure")

        # Should NOT raise
        result = await webhook_delivery.deliver_webhook(
            sample_report, "https://hooks.example.com/report", timeout_seconds=30
        )
        assert isinstance(result, DeliveryResult), (
            "Should always return a DeliveryResult, even on catastrophic failure"
        )

    @pytest.mark.asyncio
    async def test_success_status_code_in_2xx_range(self, webhook_delivery, mock_http_client,
                                                      sample_report):
        """Success status implies http_status_code in [200, 299]."""
        for status_code in [200, 201, 202, 204]:
            resp = MagicMock()
            resp.status_code = status_code
            mock_http_client.post.return_value = resp

            result = await webhook_delivery.deliver_webhook(
                sample_report, "https://hooks.example.com/report", timeout_seconds=30
            )
            if result.status == "success":
                assert 200 <= result.http_status_code <= 299, (
                    f"Success status should have 2xx code, got {result.http_status_code}"
                )

    # --- Error cases (all captured in DeliveryResult, never raised) ---

    @pytest.mark.asyncio
    async def test_connection_refused_returns_failure(self, webhook_delivery, mock_http_client,
                                                       sample_report):
        """Connection refused returns failure DeliveryResult with non-empty error_message."""
        try:
            import httpx
            mock_http_client.post.side_effect = httpx.ConnectError("Connection refused")
        except ImportError:
            mock_http_client.post.side_effect = ConnectionRefusedError("Connection refused")

        result = await webhook_delivery.deliver_webhook(
            sample_report, "https://hooks.example.com/report", timeout_seconds=30
        )
        assert result.status == "failure", f"Expected failure status, got: {result.status}"
        assert result.error_message, "Failure should have non-empty error_message"

    @pytest.mark.asyncio
    async def test_dns_resolution_failure_returns_failure(self, webhook_delivery, mock_http_client,
                                                           sample_report):
        """DNS resolution failure returns failure DeliveryResult."""
        try:
            import httpx
            mock_http_client.post.side_effect = httpx.ConnectError(
                "DNS resolution failed for nonexistent.invalid"
            )
        except ImportError:
            mock_http_client.post.side_effect = OSError("DNS resolution failed")

        result = await webhook_delivery.deliver_webhook(
            sample_report, "https://nonexistent.invalid/hook", timeout_seconds=30
        )
        assert result.status == "failure", f"Expected failure status, got: {result.status}"
        assert result.error_message, "DNS failure should have non-empty error_message"

    @pytest.mark.asyncio
    async def test_timeout_exceeded_returns_timeout_status(self, webhook_delivery, mock_http_client,
                                                            sample_report):
        """Timeout returns DeliveryResult with status=timeout and descriptive error_message."""
        try:
            import httpx
            mock_http_client.post.side_effect = httpx.TimeoutException("Request timed out")
        except ImportError:
            mock_http_client.post.side_effect = TimeoutError("Request timed out")

        result = await webhook_delivery.deliver_webhook(
            sample_report, "https://hooks.example.com/report", timeout_seconds=5
        )
        assert result.status == "timeout", f"Expected timeout status, got: {result.status}"
        assert result.error_message, "Timeout should have non-empty error_message"
        assert "timeout" in result.error_message.lower(), (
            f"Timeout error_message should mention timeout, got: {result.error_message}"
        )

    @pytest.mark.asyncio
    async def test_non_2xx_response_returns_failure(self, webhook_delivery, mock_http_client,
                                                      sample_report, mock_response_500):
        """Non-2xx response returns failure DeliveryResult with the actual status code."""
        mock_http_client.post.return_value = mock_response_500

        result = await webhook_delivery.deliver_webhook(
            sample_report, "https://hooks.example.com/report", timeout_seconds=30
        )
        assert result.status == "failure", f"Expected failure status, got: {result.status}"
        assert result.http_status_code == 500, (
            f"Should report actual status code 500, got: {result.http_status_code}"
        )
        assert result.error_message, "Non-2xx response should have non-empty error_message"

    @pytest.mark.asyncio
    async def test_tls_error_returns_failure(self, webhook_delivery, mock_http_client,
                                              sample_report):
        """TLS handshake failure returns failure DeliveryResult."""
        try:
            import httpx
            mock_http_client.post.side_effect = httpx.ConnectError("SSL: CERTIFICATE_VERIFY_FAILED")
        except ImportError:
            import ssl
            mock_http_client.post.side_effect = ssl.SSLError("CERTIFICATE_VERIFY_FAILED")

        result = await webhook_delivery.deliver_webhook(
            sample_report, "https://bad-cert.example.com/hook", timeout_seconds=30
        )
        assert result.status == "failure", f"Expected failure status, got: {result.status}"
        assert result.error_message, "TLS error should have non-empty error_message"

    @pytest.mark.asyncio
    async def test_serialization_failure_returns_failure(self, webhook_delivery, mock_http_client):
        """Serialization failure returns failure DeliveryResult."""
        bad_report = MagicMock(spec=SystemReport)
        bad_report.model_dump.side_effect = TypeError("Cannot serialize")
        bad_report.model_dump_json.side_effect = TypeError("Cannot serialize")
        bad_report.dict.side_effect = TypeError("Cannot serialize")
        bad_report.json.side_effect = TypeError("Cannot serialize")

        result = await webhook_delivery.deliver_webhook(
            bad_report, "https://hooks.example.com/report", timeout_seconds=30
        )
        assert isinstance(result, DeliveryResult), "Should still return DeliveryResult"
        assert result.status == "failure", f"Expected failure status, got: {result.status}"
        assert result.error_message, "Serialization failure should have non-empty error_message"

    @pytest.mark.asyncio
    async def test_failure_status_always_has_error_message(self, webhook_delivery, mock_http_client,
                                                            sample_report, mock_response_500):
        """Failure status always has non-empty error_message."""
        mock_http_client.post.return_value = mock_response_500

        result = await webhook_delivery.deliver_webhook(
            sample_report, "https://hooks.example.com/report", timeout_seconds=30
        )
        if result.status == "failure":
            assert result.error_message and len(result.error_message.strip()) > 0, (
                "Failure status must have non-empty error_message"
            )

    @pytest.mark.asyncio
    async def test_timeout_status_always_has_descriptive_error_message(
        self, webhook_delivery, mock_http_client, sample_report
    ):
        """Timeout status always has error_message describing the timeout condition."""
        try:
            import httpx
            mock_http_client.post.side_effect = httpx.TimeoutException("timed out")
        except ImportError:
            mock_http_client.post.side_effect = TimeoutError("timed out")

        result = await webhook_delivery.deliver_webhook(
            sample_report, "https://hooks.example.com/report", timeout_seconds=5
        )
        if result.status == "timeout":
            assert result.error_message and len(result.error_message.strip()) > 0, (
                "Timeout status must have non-empty error_message"
            )


# ===========================================================================
# Cross-cutting invariant tests
# ===========================================================================

class TestCrossCuttingInvariants:
    """Tests for invariants that span multiple functions."""

    def test_formatting_and_delivery_are_independent_of_generation(self):
        """Formatting and delivery are separate steps from generation.
        A SystemReport can be formatted without a generator instance."""
        report = make_system_report()
        formatter = ReportFormatter()
        result = formatter.format_report(report, "json")
        assert isinstance(result, FormattedReport), (
            "Formatting should work independently of report generation"
        )

    def test_all_monetary_values_are_string_decimals_not_floats(self):
        """All Decimal/monetary values are string-serialized, not floating point."""
        report = make_system_report()
        cb = report.cost_breakdown
        # Verify types are strings
        assert isinstance(cb.total_remote_spend, str), "total_remote_spend should be a string"
        assert isinstance(cb.estimated_local_savings, str), "estimated_local_savings should be a string"
        assert isinstance(cb.budget_limit, str), "budget_limit should be a string"
        assert isinstance(cb.budget_remaining, str), "budget_remaining should be a string"
        for task_type, cost in cb.cost_per_task.items():
            assert isinstance(cost, str), f"cost_per_task[{task_type}] should be a string"

    def test_frozen_models_are_immutable(self):
        """Frozen Pydantic v2 models should not allow attribute mutation."""
        report = make_system_report()
        with pytest.raises((AttributeError, TypeError, Exception)):
            report.report_id = "modified-id"

    def test_system_report_task_types_count_invariant(self):
        """Number of TaskConfidenceReport entries equals task_types_count."""
        report = make_system_report()
        assert report.task_types_count == len(report.task_confidence_reports), (
            f"task_types_count ({report.task_types_count}) must equal "
            f"len(task_confidence_reports) ({len(report.task_confidence_reports)})"
        )

    def test_correlation_history_windows_invariant(self):
        """Each CorrelationHistory has windows_available <= windows_requested."""
        report = make_system_report()
        for history in report.correlation_histories:
            assert history.windows_available <= history.windows_requested, (
                f"windows_available ({history.windows_available}) must be <= "
                f"windows_requested ({history.windows_requested})"
            )
