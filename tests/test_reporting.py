"""
Contract tests for the Reporting, Observability & Audit Log component.

Organized into logical sections:
1. AuditConfig validation
2. AuditEntry serialization
3. JsonLinesAuditLogger init/log/query
4. Report generation, formatting, validation
5. Webhook delivery

All dependencies are mocked. File I/O tests use pytest tmp_path.
"""

import json
import os
import stat
import uuid
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest

# Import component modules
from src.audit_log_adapter import (
    AuditConfig,
    AuditEntry,
    JsonLinesAuditLogger,
    EventType,
)
from src.reporting import (
    generate_report,
    format_report,
    deliver_webhook,
    validate_report_config,
    ReportConfig,
    ReportFormat,
    SystemReport,
    FormattedReport,
    DeliveryResult,
    DeliveryStatus,
    TrendDirection,
    AlertSeverity,
    AlertType,
    CostBreakdown,
    TaskConfidenceReport,
    CorrelationHistory,
    CorrelationWindow,
    Alert,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def valid_log_path(tmp_path):
    """A valid log path in an existing writable directory."""
    return str(tmp_path / "audit.jsonl")


@pytest.fixture
def sample_audit_config(valid_log_path):
    """A valid AuditConfig instance."""
    return AuditConfig(
        log_path=valid_log_path,
        max_file_size_mb=100,
        rotation_backup_count=5,
        enabled_event_types=[EventType.REQUEST_ROUTED, EventType.TRAINING_EXAMPLE_STORED],
    )


@pytest.fixture
def sample_audit_entry():
    """A valid AuditEntry instance with a fixed timestamp."""
    return AuditEntry(
        timestamp="2024-01-15T10:30:00+00:00",
        task_name="classification",
        event_type=EventType.REQUEST_ROUTED,
        details={"model": "gpt-4", "latency_ms": 120},
        correlation_id="corr-001",
    )


@pytest.fixture
def sample_report_config():
    """A valid ReportConfig instance."""
    return ReportConfig(
        correlation_window_count=10,
        alerts_lookback_hours=24,
        webhook_url="https://example.com/webhook",
        webhook_timeout_seconds=30,
    )


@pytest.fixture
def mock_confidence_source():
    """Mock ConfidenceDataSource with two task types."""
    source = MagicMock()
    source.get_task_types.return_value = ["classification", "extraction"]
    source.get_confidence.side_effect = lambda t: 0.85 if t == "classification" else 0.72
    source.get_phase.side_effect = lambda t: "local_primary" if t == "classification" else "shadow"
    source.get_trend.return_value = TrendDirection.improving
    source.get_sampling_frequency.return_value = 0.1
    source.get_correlation_windows.return_value = [
        CorrelationWindow(
            task_type="classification",
            window_start="2024-01-15T00:00:00+00:00",
            window_end="2024-01-15T01:00:00+00:00",
            sample_count=50,
            correlation_score=0.92,
            local_agreement_rate=0.88,
        )
    ]
    return source


@pytest.fixture
def mock_budget_source():
    """Mock BudgetDataSource."""
    source = MagicMock()
    source.get_total_remote_spend.return_value = "45.50"
    source.get_estimated_local_savings.return_value = "120.00"
    source.get_cost_per_task.return_value = {"classification": "0.05", "extraction": "0.08"}
    source.get_budget_limit.return_value = "500.00"
    source.get_budget_remaining.return_value = "454.50"
    source.get_budget_utilization_pct.return_value = 9.1
    return source


@pytest.fixture
def mock_audit_source():
    """Mock AuditDataSource."""
    source = MagicMock()
    source.entries_in_range.return_value = []
    source.entries_by_task.return_value = []
    source.entries_by_type.return_value = []
    return source


def _make_system_report():
    """Helper to create a minimal valid SystemReport for formatting/delivery tests."""
    return SystemReport(
        report_id=str(uuid.uuid4()),
        generated_at="2024-01-15T12:00:00+00:00",
        task_confidence_reports=[
            TaskConfidenceReport(
                task_type="classification",
                phase="local_primary",
                confidence_score=0.85,
                trend=TrendDirection.improving,
                sampling_frequency=0.1,
            )
        ],
        cost_breakdown=CostBreakdown(
            total_remote_spend="45.50",
            estimated_local_savings="120.00",
            cost_per_task={"classification": "0.05"},
            budget_limit="500.00",
            budget_remaining="454.50",
            budget_utilization_pct=9.1,
        ),
        correlation_histories=[
            CorrelationHistory(
                task_type="classification",
                windows_requested=10,
                windows_available=1,
                windows=[
                    CorrelationWindow(
                        task_type="classification",
                        window_start="2024-01-15T00:00:00+00:00",
                        window_end="2024-01-15T01:00:00+00:00",
                        sample_count=50,
                        correlation_score=0.92,
                        local_agreement_rate=0.88,
                    )
                ],
            )
        ],
        alerts=[],
        task_types_count=1,
    )


# ============================================================
# 1. AuditConfig Validation Tests
# ============================================================

class TestAuditConfig:
    """Tests for AuditConfig construction and validation."""

    def test_happy_valid_config(self, valid_log_path):
        """AuditConfig construction with valid parameters succeeds."""
        config = AuditConfig(
            log_path=valid_log_path,
            max_file_size_mb=100,
            rotation_backup_count=5,
            enabled_event_types=[EventType.REQUEST_ROUTED],
        )
        assert config.log_path == valid_log_path, "log_path should match input"
        assert config.max_file_size_mb == 100
        assert config.rotation_backup_count == 5
        assert EventType.REQUEST_ROUTED in config.enabled_event_types

    def test_error_empty_log_path(self):
        """AuditConfig rejects empty log_path string."""
        with pytest.raises((ValueError, Exception)):
            AuditConfig(
                log_path="",
                max_file_size_mb=100,
                rotation_backup_count=5,
                enabled_event_types=[EventType.REQUEST_ROUTED],
            )

    def test_error_parent_not_exists(self):
        """AuditConfig rejects log_path whose parent directory does not exist."""
        with pytest.raises((ValueError, OSError, Exception)):
            AuditConfig(
                log_path="/nonexistent/directory/audit.jsonl",
                max_file_size_mb=100,
                rotation_backup_count=5,
                enabled_event_types=[EventType.REQUEST_ROUTED],
            )

    def test_error_parent_not_writable(self, tmp_path):
        """AuditConfig rejects log_path whose parent directory is not writable."""
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        os.chmod(str(readonly_dir), stat.S_IRUSR | stat.S_IXUSR)
        try:
            with pytest.raises((ValueError, PermissionError, OSError, Exception)):
                AuditConfig(
                    log_path=str(readonly_dir / "audit.jsonl"),
                    max_file_size_mb=100,
                    rotation_backup_count=5,
                    enabled_event_types=[EventType.REQUEST_ROUTED],
                )
        finally:
            os.chmod(str(readonly_dir), stat.S_IRWXU)

    @pytest.mark.parametrize("bad_size", [0, -1, 10001, -100])
    def test_error_invalid_max_file_size(self, valid_log_path, bad_size):
        """AuditConfig rejects max_file_size_mb outside [1, 10000]."""
        with pytest.raises((ValueError, Exception)):
            AuditConfig(
                log_path=valid_log_path,
                max_file_size_mb=bad_size,
                rotation_backup_count=5,
                enabled_event_types=[EventType.REQUEST_ROUTED],
            )

    @pytest.mark.parametrize("bad_count", [-1, 101, -50])
    def test_error_invalid_rotation_count(self, valid_log_path, bad_count):
        """AuditConfig rejects rotation_backup_count outside [0, 100]."""
        with pytest.raises((ValueError, Exception)):
            AuditConfig(
                log_path=valid_log_path,
                max_file_size_mb=100,
                rotation_backup_count=bad_count,
                enabled_event_types=[EventType.REQUEST_ROUTED],
            )

    @pytest.mark.parametrize("size", [1, 10000])
    def test_edge_boundary_max_file_size(self, valid_log_path, size):
        """AuditConfig accepts boundary values for max_file_size_mb."""
        config = AuditConfig(
            log_path=valid_log_path,
            max_file_size_mb=size,
            rotation_backup_count=5,
            enabled_event_types=[EventType.REQUEST_ROUTED],
        )
        assert config.max_file_size_mb == size

    @pytest.mark.parametrize("count", [0, 100])
    def test_edge_boundary_rotation_count(self, valid_log_path, count):
        """AuditConfig accepts boundary values for rotation_backup_count."""
        config = AuditConfig(
            log_path=valid_log_path,
            max_file_size_mb=100,
            rotation_backup_count=count,
            enabled_event_types=[EventType.REQUEST_ROUTED],
        )
        assert config.rotation_backup_count == count

    def test_invariant_frozen(self, sample_audit_config):
        """AuditConfig instance is immutable after construction."""
        with pytest.raises((AttributeError, TypeError, Exception)):
            sample_audit_config.log_path = "/new/path.jsonl"


# ============================================================
# 2. AuditEntry Serialization Tests
# ============================================================

class TestAuditEntry:
    """Tests for AuditEntry construction and to_json_line()."""

    def test_happy_to_json_line(self, sample_audit_entry):
        """to_json_line() returns valid compact JSON string."""
        result = sample_audit_entry.to_json_line()
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Result should be non-empty"
        parsed = json.loads(result)
        assert isinstance(parsed, dict), "Parsed result should be a dict"

    def test_json_line_roundtrip(self, sample_audit_entry):
        """to_json_line() output can be parsed back and contains expected keys."""
        result = sample_audit_entry.to_json_line()
        parsed = json.loads(result)
        assert "timestamp" in parsed, "JSON should contain 'timestamp' key"
        assert "task_name" in parsed, "JSON should contain 'task_name' key"
        assert "event_type" in parsed, "JSON should contain 'event_type' key"
        assert "details" in parsed, "JSON should contain 'details' key"
        assert parsed["task_name"] == "classification"
        assert parsed["details"]["model"] == "gpt-4"

    def test_json_line_no_newline(self, sample_audit_entry):
        """to_json_line() output contains no newline characters."""
        result = sample_audit_entry.to_json_line()
        assert "\n" not in result, "JSON line should not contain newline"
        assert "\r" not in result, "JSON line should not contain carriage return"

    def test_json_line_deterministic(self, sample_audit_entry):
        """to_json_line() produces identical output on repeated calls."""
        result1 = sample_audit_entry.to_json_line()
        result2 = sample_audit_entry.to_json_line()
        assert result1 == result2, "Repeated calls should produce identical output"

    def test_invariant_frozen(self, sample_audit_entry):
        """AuditEntry is frozen (immutable) after construction."""
        with pytest.raises((AttributeError, TypeError, Exception)):
            sample_audit_entry.task_name = "other_task"

    def test_invariant_valid_event_type(self):
        """AuditEntry rejects invalid event_type values at construction."""
        with pytest.raises((ValueError, Exception)):
            AuditEntry(
                timestamp="2024-01-15T10:30:00+00:00",
                task_name="test",
                event_type="INVALID_EVENT",
                details={},
                correlation_id="corr-001",
            )

    @pytest.mark.parametrize("event_type", list(EventType))
    def test_all_event_types_accepted(self, event_type):
        """AuditEntry accepts all 9 valid EventType variants."""
        entry = AuditEntry(
            timestamp="2024-01-15T10:30:00+00:00",
            task_name="test",
            event_type=event_type,
            details={},
            correlation_id="corr-001",
        )
        assert entry.event_type == event_type


# ============================================================
# 3. JsonLinesAuditLogger Tests
# ============================================================

class TestJsonLinesAuditLogger:
    """Tests for JsonLinesAuditLogger init, log, and query methods."""

    def test_happy_init(self, sample_audit_config):
        """Logger initializes successfully with valid config and creates log file."""
        logger = JsonLinesAuditLogger(sample_audit_config)
        assert os.path.exists(sample_audit_config.log_path), "Log file should be created"

    def test_init_preserves_existing_content(self, valid_log_path):
        """Logger does not truncate existing log file content."""
        # Pre-populate the file
        with open(valid_log_path, "w") as f:
            f.write('{"existing": "line"}\n')
        config = AuditConfig(
            log_path=valid_log_path,
            max_file_size_mb=100,
            rotation_backup_count=5,
            enabled_event_types=[EventType.REQUEST_ROUTED],
        )
        JsonLinesAuditLogger(config)
        with open(valid_log_path, "r") as f:
            content = f.read()
        assert '{"existing": "line"}' in content, "Existing content should be preserved"

    def test_error_parent_not_found(self):
        """Logger raises when parent directory does not exist."""
        config = MagicMock(spec=AuditConfig)
        config.log_path = "/nonexistent/path/audit.jsonl"
        config.max_file_size_mb = 100
        config.rotation_backup_count = 5
        config.enabled_event_types = [EventType.REQUEST_ROUTED]
        with pytest.raises((FileNotFoundError, OSError, ValueError, Exception)):
            JsonLinesAuditLogger(config)


class TestJsonLinesAuditLoggerLog:
    """Tests for JsonLinesAuditLogger.log() method."""

    @pytest.fixture
    def logger_and_path(self, tmp_path):
        """Create a logger with a valid config."""
        log_path = str(tmp_path / "audit.jsonl")
        config = AuditConfig(
            log_path=log_path,
            max_file_size_mb=100,
            rotation_backup_count=5,
            enabled_event_types=[EventType.REQUEST_ROUTED, EventType.TRAINING_EXAMPLE_STORED],
        )
        logger = JsonLinesAuditLogger(config)
        return logger, log_path

    def test_happy_log(self, logger_and_path):
        """log() appends entry to JSONL file when event type is enabled."""
        logger, log_path = logger_and_path
        entry = AuditEntry(
            timestamp="2024-01-15T10:30:00+00:00",
            task_name="classification",
            event_type=EventType.REQUEST_ROUTED,
            details={"key": "value"},
            correlation_id="corr-001",
        )
        # Run async log
        asyncio.get_event_loop().run_until_complete(logger.log(entry)) if asyncio.iscoroutinefunction(logger.log) else logger.log(entry)
        with open(log_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        assert len(lines) >= 1, "At least one line should be written"
        parsed = json.loads(lines[-1])
        assert parsed["task_name"] == "classification"

    def test_log_filtered_event(self, logger_and_path):
        """log() silently drops entry when event_type not in enabled list."""
        logger, log_path = logger_and_path
        entry = AuditEntry(
            timestamp="2024-01-15T10:30:00+00:00",
            task_name="classification",
            event_type=EventType.FINE_TUNE_STARTED,  # Not in enabled list
            details={},
            correlation_id="corr-002",
        )
        # Get file size before
        initial_size = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        asyncio.get_event_loop().run_until_complete(logger.log(entry)) if asyncio.iscoroutinefunction(logger.log) else logger.log(entry)
        final_size = os.path.getsize(log_path)
        assert final_size == initial_size, "No write should occur for filtered event type"

    def test_log_append_only(self, logger_and_path):
        """Multiple log() calls append without modifying existing lines."""
        logger, log_path = logger_and_path
        entries = []
        for i in range(3):
            entry = AuditEntry(
                timestamp=f"2024-01-15T10:3{i}:00+00:00",
                task_name=f"task_{i}",
                event_type=EventType.REQUEST_ROUTED,
                details={"index": i},
                correlation_id=f"corr-{i:03d}",
            )
            entries.append(entry)

        for entry in entries:
            if asyncio.iscoroutinefunction(logger.log):
                asyncio.get_event_loop().run_until_complete(logger.log(entry))
            else:
                logger.log(entry)

        with open(log_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        assert len(lines) >= 3, "Should have at least 3 lines"
        # Verify each line is valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert "task_name" in parsed


class TestJsonLinesAuditLoggerQueries:
    """Tests for query methods: entries_by_task, entries_by_type, entries_in_range."""

    @pytest.fixture
    def populated_logger(self, tmp_path):
        """Create a logger with pre-populated entries."""
        log_path = str(tmp_path / "audit.jsonl")
        all_types = list(EventType)
        config = AuditConfig(
            log_path=log_path,
            max_file_size_mb=100,
            rotation_backup_count=5,
            enabled_event_types=all_types,
        )
        logger = JsonLinesAuditLogger(config)

        # Log some entries
        test_entries = [
            AuditEntry(
                timestamp="2024-01-15T10:00:00+00:00",
                task_name="classification",
                event_type=EventType.REQUEST_ROUTED,
                details={"seq": 1},
                correlation_id="corr-001",
            ),
            AuditEntry(
                timestamp="2024-01-15T11:00:00+00:00",
                task_name="extraction",
                event_type=EventType.TRAINING_EXAMPLE_STORED,
                details={"seq": 2},
                correlation_id="corr-002",
            ),
            AuditEntry(
                timestamp="2024-01-15T12:00:00+00:00",
                task_name="classification",
                event_type=EventType.FINE_TUNE_STARTED,
                details={"seq": 3},
                correlation_id="corr-003",
            ),
        ]
        for entry in test_entries:
            if asyncio.iscoroutinefunction(logger.log):
                asyncio.get_event_loop().run_until_complete(logger.log(entry))
            else:
                logger.log(entry)

        return logger, log_path

    def test_happy_entries_by_task(self, populated_logger):
        """entries_by_task returns only entries matching task_name in chronological order."""
        logger, _ = populated_logger
        results = logger.entries_by_task("classification")
        assert len(results) == 2, "Should find exactly 2 classification entries"
        for entry in results:
            assert entry.task_name == "classification", "All entries should match task_name"
        # Verify chronological order
        assert results[0].timestamp <= results[1].timestamp, "Entries should be chronological"

    def test_entries_by_task_empty(self, populated_logger):
        """entries_by_task returns empty list when no entries match."""
        logger, _ = populated_logger
        results = logger.entries_by_task("nonexistent_task")
        assert results == [], "Should return empty list for non-matching task"

    def test_error_entries_by_task_empty_name(self, populated_logger):
        """entries_by_task raises on empty task_name."""
        logger, _ = populated_logger
        with pytest.raises((ValueError, Exception)):
            logger.entries_by_task("")

    def test_entries_by_task_skips_corrupted(self, tmp_path):
        """entries_by_task skips corrupted lines without failing."""
        log_path = str(tmp_path / "corrupt_audit.jsonl")
        # Write a corrupted line and a valid line
        valid_entry = AuditEntry(
            timestamp="2024-01-15T10:00:00+00:00",
            task_name="classification",
            event_type=EventType.REQUEST_ROUTED,
            details={"seq": 1},
            correlation_id="corr-001",
        )
        valid_json = valid_entry.to_json_line()
        with open(log_path, "w") as f:
            f.write("THIS IS NOT VALID JSON\n")
            f.write(valid_json + "\n")
            f.write("{broken json\n")

        all_types = list(EventType)
        config = AuditConfig(
            log_path=log_path,
            max_file_size_mb=100,
            rotation_backup_count=5,
            enabled_event_types=all_types,
        )
        logger = JsonLinesAuditLogger(config)
        results = logger.entries_by_task("classification")
        assert len(results) == 1, "Should find exactly 1 valid entry, skipping corrupted"
        assert results[0].task_name == "classification"

    def test_happy_entries_by_type(self, populated_logger):
        """entries_by_type returns only entries matching event_type."""
        logger, _ = populated_logger
        results = logger.entries_by_type(EventType.REQUEST_ROUTED)
        assert len(results) >= 1, "Should find at least 1 REQUEST_ROUTED entry"
        for entry in results:
            assert entry.event_type == EventType.REQUEST_ROUTED

    def test_error_entries_by_type_invalid(self, populated_logger):
        """entries_by_type raises on invalid event_type."""
        logger, _ = populated_logger
        with pytest.raises((ValueError, Exception)):
            logger.entries_by_type("INVALID_TYPE")

    def test_happy_entries_in_range(self, populated_logger):
        """entries_in_range returns entries within inclusive time bounds."""
        logger, _ = populated_logger
        results = logger.entries_in_range(
            "2024-01-15T10:00:00+00:00",
            "2024-01-15T11:30:00+00:00",
        )
        assert len(results) == 2, "Should find 2 entries in range"
        for entry in results:
            assert entry.timestamp >= "2024-01-15T10:00:00+00:00"
            assert entry.timestamp <= "2024-01-15T11:30:00+00:00"

    def test_error_start_after_end(self, populated_logger):
        """entries_in_range raises when start > end."""
        logger, _ = populated_logger
        with pytest.raises((ValueError, Exception)):
            logger.entries_in_range(
                "2024-01-15T12:00:00+00:00",
                "2024-01-15T10:00:00+00:00",
            )

    def test_error_invalid_start(self, populated_logger):
        """entries_in_range raises on invalid start datetime."""
        logger, _ = populated_logger
        with pytest.raises((ValueError, Exception)):
            logger.entries_in_range(
                "not-a-date",
                "2024-01-15T12:00:00+00:00",
            )

    def test_error_invalid_end(self, populated_logger):
        """entries_in_range raises on invalid end datetime."""
        logger, _ = populated_logger
        with pytest.raises((ValueError, Exception)):
            logger.entries_in_range(
                "2024-01-15T10:00:00+00:00",
                "not-a-date",
            )

    def test_invariant_query_no_modify(self, populated_logger):
        """Query methods do not modify the audit log file."""
        logger, log_path = populated_logger
        with open(log_path, "r") as f:
            content_before = f.read()

        logger.entries_by_task("classification")
        logger.entries_by_type(EventType.REQUEST_ROUTED)
        logger.entries_in_range(
            "2024-01-15T00:00:00+00:00",
            "2024-01-15T23:59:59+00:00",
        )

        with open(log_path, "r") as f:
            content_after = f.read()
        assert content_before == content_after, "Query methods must not modify the log file"


# ============================================================
# 4. Report Generation, Formatting & Validation Tests
# ============================================================

class TestGenerateReport:
    """Tests for the generate_report function."""

    def test_happy_generate(
        self, mock_confidence_source, mock_budget_source, mock_audit_source, sample_report_config
    ):
        """generate_report produces a valid SystemReport with correct structure."""
        report = generate_report(
            confidence_source=mock_confidence_source,
            budget_source=mock_budget_source,
            audit_source=mock_audit_source,
            config=sample_report_config,
        )
        assert isinstance(report, SystemReport), "Should return a SystemReport"
        assert report.report_id is not None and len(report.report_id) > 0
        assert report.generated_at is not None
        assert len(report.task_confidence_reports) == 2, "Should have 2 task types"
        assert report.cost_breakdown is not None

    def test_generate_uuid(
        self, mock_confidence_source, mock_budget_source, mock_audit_source, sample_report_config
    ):
        """generate_report produces a report with a valid UUID v4 report_id."""
        report = generate_report(
            confidence_source=mock_confidence_source,
            budget_source=mock_budget_source,
            audit_source=mock_audit_source,
            config=sample_report_config,
        )
        # Validate UUID v4 format
        parsed_uuid = uuid.UUID(report.report_id, version=4)
        assert str(parsed_uuid) == report.report_id, "report_id should be a valid UUID v4"

    def test_invariant_task_count(
        self, mock_confidence_source, mock_budget_source, mock_audit_source, sample_report_config
    ):
        """task_types_count equals length of task_confidence_reports."""
        report = generate_report(
            confidence_source=mock_confidence_source,
            budget_source=mock_budget_source,
            audit_source=mock_audit_source,
            config=sample_report_config,
        )
        assert report.task_types_count == len(report.task_confidence_reports), (
            f"task_types_count ({report.task_types_count}) should equal "
            f"len(task_confidence_reports) ({len(report.task_confidence_reports)})"
        )

    def test_invariant_correlation_windows(
        self, mock_confidence_source, mock_budget_source, mock_audit_source, sample_report_config
    ):
        """Each CorrelationHistory has windows_available <= windows_requested."""
        report = generate_report(
            confidence_source=mock_confidence_source,
            budget_source=mock_budget_source,
            audit_source=mock_audit_source,
            config=sample_report_config,
        )
        for history in report.correlation_histories:
            assert history.windows_available <= history.windows_requested, (
                f"windows_available ({history.windows_available}) should be <= "
                f"windows_requested ({history.windows_requested})"
            )
            assert history.windows_requested == sample_report_config.correlation_window_count

    def test_error_confidence_unavailable(
        self, mock_budget_source, mock_audit_source, sample_report_config
    ):
        """generate_report propagates error when confidence source is unavailable."""
        bad_source = MagicMock()
        bad_source.get_task_types.side_effect = RuntimeError("Connection failed")
        with pytest.raises(Exception):
            generate_report(
                confidence_source=bad_source,
                budget_source=mock_budget_source,
                audit_source=mock_audit_source,
                config=sample_report_config,
            )

    def test_error_budget_unavailable(
        self, mock_confidence_source, mock_audit_source, sample_report_config
    ):
        """generate_report propagates error when budget source is unavailable."""
        bad_source = MagicMock()
        bad_source.get_total_remote_spend.side_effect = RuntimeError("Connection failed")
        with pytest.raises(Exception):
            generate_report(
                confidence_source=mock_confidence_source,
                budget_source=bad_source,
                audit_source=mock_audit_source,
                config=sample_report_config,
            )

    def test_error_no_task_types(
        self, mock_budget_source, mock_audit_source, sample_report_config
    ):
        """generate_report raises when confidence source returns empty task list."""
        empty_source = MagicMock()
        empty_source.get_task_types.return_value = []
        with pytest.raises(Exception):
            generate_report(
                confidence_source=empty_source,
                budget_source=mock_budget_source,
                audit_source=mock_audit_source,
                config=sample_report_config,
            )


class TestFormatReport:
    """Tests for the format_report function."""

    def test_happy_json_format(self):
        """format_report with JSON produces valid parseable JSON."""
        report = _make_system_report()
        result = format_report(report, ReportFormat.json)
        assert isinstance(result, FormattedReport)
        assert result.format == ReportFormat.json
        # Verify content is valid JSON
        parsed = json.loads(result.content)
        assert isinstance(parsed, dict), "JSON content should parse to a dict"

    def test_happy_cli_format(self):
        """format_report with CLI_SUMMARY produces multi-line human-readable output."""
        report = _make_system_report()
        result = format_report(report, ReportFormat.cli_summary)
        assert isinstance(result, FormattedReport)
        assert result.format == ReportFormat.cli_summary
        assert "\n" in result.content, "CLI output should be multi-line"

    def test_invariant_report_id_match(self):
        """FormattedReport.report_id matches input report.report_id."""
        report = _make_system_report()
        for fmt in ReportFormat:
            result = format_report(report, fmt)
            assert result.report_id == report.report_id, (
                f"report_id mismatch for format {fmt}: "
                f"{result.report_id} != {report.report_id}"
            )

    def test_invariant_content_length(self):
        """FormattedReport.content_length equals len(content)."""
        report = _make_system_report()
        for fmt in ReportFormat:
            result = format_report(report, fmt)
            assert result.content_length == len(result.content), (
                f"content_length ({result.content_length}) != "
                f"len(content) ({len(result.content)}) for format {fmt}"
            )

    def test_json_decimal_precision(self):
        """JSON format preserves Decimal string values without float conversion."""
        report = _make_system_report()
        result = format_report(report, ReportFormat.json)
        parsed = json.loads(result.content)
        # Budget values should be strings (Decimal representation), not floats
        cost = parsed.get("cost_breakdown", {})
        if "total_remote_spend" in cost:
            assert isinstance(cost["total_remote_spend"], str), (
                "Decimal values should be serialized as strings, not floats"
            )
        if "budget_limit" in cost:
            assert isinstance(cost["budget_limit"], str), (
                "Decimal values should be serialized as strings"
            )


class TestValidateReportConfig:
    """Tests for validate_report_config."""

    def test_happy_valid(self, sample_report_config):
        """validate_report_config accepts valid config and returns it."""
        result = validate_report_config(sample_report_config)
        assert result is not None
        assert result.correlation_window_count == sample_report_config.correlation_window_count

    @pytest.mark.parametrize("bad_count", [0, -1, 1001])
    def test_error_correlation_window_count(self, bad_count):
        """validate_report_config rejects correlation_window_count outside [1, 1000]."""
        config = ReportConfig(
            correlation_window_count=bad_count,
            alerts_lookback_hours=24,
            webhook_url="https://example.com/hook",
            webhook_timeout_seconds=30,
        )
        with pytest.raises((ValueError, Exception)):
            validate_report_config(config)

    @pytest.mark.parametrize("bad_hours", [0, -1, 8761])
    def test_error_alerts_lookback_hours(self, bad_hours):
        """validate_report_config rejects alerts_lookback_hours outside [1, 8760]."""
        config = ReportConfig(
            correlation_window_count=10,
            alerts_lookback_hours=bad_hours,
            webhook_url="https://example.com/hook",
            webhook_timeout_seconds=30,
        )
        with pytest.raises((ValueError, Exception)):
            validate_report_config(config)

    @pytest.mark.parametrize("bad_url", ["not-a-url", "ftp://invalid.com", "just-text"])
    def test_error_invalid_webhook_url(self, bad_url):
        """validate_report_config rejects non-empty invalid webhook URL."""
        config = ReportConfig(
            correlation_window_count=10,
            alerts_lookback_hours=24,
            webhook_url=bad_url,
            webhook_timeout_seconds=30,
        )
        with pytest.raises((ValueError, Exception)):
            validate_report_config(config)

    @pytest.mark.parametrize("bad_timeout", [0, -1, 301])
    def test_error_invalid_webhook_timeout(self, bad_timeout):
        """validate_report_config rejects webhook_timeout_seconds outside [1, 300]."""
        config = ReportConfig(
            correlation_window_count=10,
            alerts_lookback_hours=24,
            webhook_url="https://example.com/hook",
            webhook_timeout_seconds=bad_timeout,
        )
        with pytest.raises((ValueError, Exception)):
            validate_report_config(config)

    def test_edge_empty_webhook_url(self):
        """validate_report_config accepts empty webhook_url (webhook disabled)."""
        config = ReportConfig(
            correlation_window_count=10,
            alerts_lookback_hours=24,
            webhook_url="",
            webhook_timeout_seconds=30,
        )
        result = validate_report_config(config)
        assert result.webhook_url == ""

    def test_edge_boundary_values(self):
        """validate_report_config accepts all boundary values."""
        config = ReportConfig(
            correlation_window_count=1,
            alerts_lookback_hours=1,
            webhook_url="https://example.com/hook",
            webhook_timeout_seconds=1,
        )
        result = validate_report_config(config)
        assert result.correlation_window_count == 1
        assert result.alerts_lookback_hours == 1
        assert result.webhook_timeout_seconds == 1

        config_max = ReportConfig(
            correlation_window_count=1000,
            alerts_lookback_hours=8760,
            webhook_url="https://example.com/hook",
            webhook_timeout_seconds=300,
        )
        result_max = validate_report_config(config_max)
        assert result_max.correlation_window_count == 1000
        assert result_max.alerts_lookback_hours == 8760
        assert result_max.webhook_timeout_seconds == 300


# ============================================================
# 5. Webhook Delivery Tests
# ============================================================

class TestDeliverWebhook:
    """Tests for the deliver_webhook function."""

    @pytest.fixture
    def sample_report(self):
        return _make_system_report()

    @pytest.mark.asyncio
    async def test_happy_success(self, sample_report):
        """deliver_webhook returns SUCCESS DeliveryResult on 200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await deliver_webhook(
            report=sample_report,
            webhook_url="https://example.com/webhook",
            timeout_seconds=30,
            http_client=mock_client,
        )
        assert isinstance(result, DeliveryResult), "Should return a DeliveryResult"
        assert result.status == DeliveryStatus.success, "Status should be SUCCESS"
        assert 200 <= result.http_status_code <= 299, "HTTP status should be 2xx"
        assert result.url == "https://example.com/webhook"

    @pytest.mark.asyncio
    async def test_happy_2xx_range(self, sample_report):
        """deliver_webhook returns SUCCESS for any 2xx status code."""
        for status_code in [200, 201, 202, 204, 299]:
            mock_response = MagicMock()
            mock_response.status_code = status_code
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            result = await deliver_webhook(
                report=sample_report,
                webhook_url="https://example.com/webhook",
                timeout_seconds=30,
                http_client=mock_client,
            )
            assert result.status == DeliveryStatus.success, (
                f"Status should be SUCCESS for HTTP {status_code}"
            )

    @pytest.mark.asyncio
    async def test_error_connection_refused(self, sample_report):
        """deliver_webhook returns FAILURE on connection refused."""
        import httpx
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        result = await deliver_webhook(
            report=sample_report,
            webhook_url="https://example.com/webhook",
            timeout_seconds=30,
            http_client=mock_client,
        )
        assert result.status == DeliveryStatus.failure, "Status should be FAILURE"
        assert result.error_message, "error_message should be non-empty"
        assert result.url == "https://example.com/webhook"

    @pytest.mark.asyncio
    async def test_error_timeout(self, sample_report):
        """deliver_webhook returns TIMEOUT on timeout exceeded."""
        import httpx
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        result = await deliver_webhook(
            report=sample_report,
            webhook_url="https://example.com/webhook",
            timeout_seconds=5,
            http_client=mock_client,
        )
        assert result.status == DeliveryStatus.timeout, "Status should be TIMEOUT"
        assert result.error_message, "error_message should describe timeout"

    @pytest.mark.asyncio
    async def test_error_non_2xx(self, sample_report):
        """deliver_webhook returns FAILURE on non-2xx response."""
        for status_code in [400, 403, 404, 500, 502, 503]:
            mock_response = MagicMock()
            mock_response.status_code = status_code
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            result = await deliver_webhook(
                report=sample_report,
                webhook_url="https://example.com/webhook",
                timeout_seconds=30,
                http_client=mock_client,
            )
            assert result.status == DeliveryStatus.failure, (
                f"Status should be FAILURE for HTTP {status_code}"
            )
            assert result.error_message, f"error_message should be non-empty for HTTP {status_code}"

    @pytest.mark.asyncio
    async def test_error_dns_failure(self, sample_report):
        """deliver_webhook returns FAILURE on DNS resolution failure."""
        import httpx
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("DNS resolution failed")
        )

        result = await deliver_webhook(
            report=sample_report,
            webhook_url="https://nonexistent.example.invalid/webhook",
            timeout_seconds=30,
            http_client=mock_client,
        )
        assert result.status == DeliveryStatus.failure
        assert result.error_message, "error_message should be non-empty"

    @pytest.mark.asyncio
    async def test_error_tls(self, sample_report):
        """deliver_webhook returns FAILURE on TLS error."""
        import httpx
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("TLS handshake failed")
        )

        result = await deliver_webhook(
            report=sample_report,
            webhook_url="https://example.com/webhook",
            timeout_seconds=30,
            http_client=mock_client,
        )
        assert result.status == DeliveryStatus.failure
        assert result.error_message, "error_message should be non-empty"

    @pytest.mark.asyncio
    async def test_invariant_never_raises(self, sample_report):
        """deliver_webhook never raises exceptions, always returns DeliveryResult."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Unexpected catastrophic error"))

        # Should NOT raise
        result = await deliver_webhook(
            report=sample_report,
            webhook_url="https://example.com/webhook",
            timeout_seconds=30,
            http_client=mock_client,
        )
        assert isinstance(result, DeliveryResult), (
            "Should always return DeliveryResult, never raise"
        )

    @pytest.mark.asyncio
    async def test_invariant_url_match(self, sample_report):
        """DeliveryResult.url always matches the input webhook_url."""
        test_url = "https://my-service.example.com/hooks/audit"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await deliver_webhook(
            report=sample_report,
            webhook_url=test_url,
            timeout_seconds=30,
            http_client=mock_client,
        )
        assert result.url == test_url, (
            f"DeliveryResult.url ({result.url}) should match input ({test_url})"
        )

    @pytest.mark.asyncio
    async def test_delivery_has_timestamp(self, sample_report):
        """DeliveryResult always has attempt_timestamp set."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await deliver_webhook(
            report=sample_report,
            webhook_url="https://example.com/webhook",
            timeout_seconds=30,
            http_client=mock_client,
        )
        assert result.attempt_timestamp, "attempt_timestamp should be set"

    @pytest.mark.asyncio
    async def test_delivery_has_duration(self, sample_report):
        """DeliveryResult always has duration_ms reflecting elapsed time."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await deliver_webhook(
            report=sample_report,
            webhook_url="https://example.com/webhook",
            timeout_seconds=30,
            http_client=mock_client,
        )
        assert result.duration_ms >= 0, "duration_ms should be non-negative"


# ============================================================
# 6. Cross-cutting Invariant Tests
# ============================================================

class TestInvariants:
    """Tests for cross-cutting contract invariants."""

    def test_all_event_types_exist(self):
        """The EventType enum has exactly 9 variants."""
        assert len(EventType) == 9, f"Expected 9 EventType variants, got {len(EventType)}"
        expected = {
            "REQUEST_ROUTED", "TRAINING_EXAMPLE_STORED", "FINE_TUNE_STARTED",
            "FINE_TUNE_COMPLETED", "MODEL_VALIDATED", "MODEL_PROMOTED",
            "PHASE_TRANSITION", "BUDGET_WARNING", "CONFIDENCE_ALERT",
        }
        actual = {e.name for e in EventType}
        assert actual == expected, f"EventType variants mismatch: {actual.symmetric_difference(expected)}"

    def test_trend_direction_values(self):
        """TrendDirection has lowercase string values."""
        for member in TrendDirection:
            assert member.value == member.value.lower(), (
                f"TrendDirection.{member.name} value should be lowercase"
            )

    def test_alert_severity_values(self):
        """AlertSeverity has lowercase string values."""
        for member in AlertSeverity:
            assert member.value == member.value.lower(), (
                f"AlertSeverity.{member.name} value should be lowercase"
            )

    def test_report_format_variants(self):
        """ReportFormat has json and cli_summary variants."""
        names = {e.name for e in ReportFormat}
        assert "json" in names
        assert "cli_summary" in names

    def test_delivery_status_variants(self):
        """DeliveryStatus has success, failure, timeout variants."""
        names = {e.name for e in DeliveryStatus}
        assert names == {"success", "failure", "timeout"}

    def test_system_report_frozen(self):
        """SystemReport is frozen (immutable)."""
        report = _make_system_report()
        with pytest.raises((AttributeError, TypeError, Exception)):
            report.report_id = "new-id"

    def test_cost_breakdown_frozen(self):
        """CostBreakdown is frozen (immutable)."""
        cb = CostBreakdown(
            total_remote_spend="10.00",
            estimated_local_savings="20.00",
            cost_per_task={},
            budget_limit="100.00",
            budget_remaining="90.00",
            budget_utilization_pct=10.0,
        )
        with pytest.raises((AttributeError, TypeError, Exception)):
            cb.total_remote_spend = "99.99"

    def test_all_monetary_values_are_strings(self):
        """Monetary values in CostBreakdown are strings (Decimal serialized)."""
        cb = CostBreakdown(
            total_remote_spend="45.50",
            estimated_local_savings="120.00",
            cost_per_task={"task": "0.05"},
            budget_limit="500.00",
            budget_remaining="454.50",
            budget_utilization_pct=9.1,
        )
        assert isinstance(cb.total_remote_spend, str)
        assert isinstance(cb.estimated_local_savings, str)
        assert isinstance(cb.budget_limit, str)
        assert isinstance(cb.budget_remaining, str)
        for v in cb.cost_per_task.values():
            assert isinstance(v, str), "cost_per_task values should be strings"

    def test_unique_report_ids(
        self,
    ):
        """Two generated reports have different report_ids."""
        # Use mock sources to generate two reports
        confidence = MagicMock()
        confidence.get_task_types.return_value = ["t1"]
        confidence.get_confidence.return_value = 0.5
        confidence.get_phase.return_value = "shadow"
        confidence.get_trend.return_value = TrendDirection.stable
        confidence.get_sampling_frequency.return_value = 0.1
        confidence.get_correlation_windows.return_value = []

        budget = MagicMock()
        budget.get_total_remote_spend.return_value = "0.00"
        budget.get_estimated_local_savings.return_value = "0.00"
        budget.get_cost_per_task.return_value = {}
        budget.get_budget_limit.return_value = "100.00"
        budget.get_budget_remaining.return_value = "100.00"
        budget.get_budget_utilization_pct.return_value = 0.0

        audit = MagicMock()
        audit.entries_in_range.return_value = []

        config = ReportConfig(
            correlation_window_count=5,
            alerts_lookback_hours=24,
            webhook_url="",
            webhook_timeout_seconds=30,
        )
        r1 = generate_report(confidence, budget, audit, config)
        r2 = generate_report(confidence, budget, audit, config)
        assert r1.report_id != r2.report_id, "Each report should have a unique UUID"
