"""
Contract tests for the audit_log component.
Covers AuditEntry, AuditConfig, EventType, and JsonLinesAuditLogger.

Run with: pytest contract_test.py -v
"""

import json
import logging
import os
import stat
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.audit_log import (
    AuditEntry,
    AuditConfig,
    AuditLogger,
    EventType,
    JsonLinesAuditLogger,
)


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

ALL_EVENT_TYPES = [
    "request_routed",
    "training_example_stored",
    "fine_tune_started",
    "fine_tune_completed",
    "model_validated",
    "model_promoted",
    "phase_transition",
    "budget_warning",
    "confidence_alert",
]


def _make_entry(
    task_name="test_task",
    event_type=EventType.request_routed,
    details=None,
    timestamp=None,
):
    kwargs = {
        "task_name": task_name,
        "event_type": event_type,
        "details": details if details is not None else {"key": "value"},
    }
    if timestamp is not None:
        kwargs["timestamp"] = timestamp
    return AuditEntry(**kwargs)


@pytest.fixture
def audit_log_path(tmp_path):
    return str(tmp_path / "audit.jsonl")


@pytest.fixture
def audit_config(audit_log_path):
    return AuditConfig(log_path=audit_log_path)


@pytest.fixture
def audit_logger(audit_config):
    """Fresh JsonLinesAuditLogger for each test; cleans up logging handlers."""
    logger_instance = JsonLinesAuditLogger(config=audit_config)
    yield logger_instance
    # Cleanup: remove handlers from the named logger to avoid leaks between tests
    named_logger = logging.getLogger("apprentice.audit")
    for handler in named_logger.handlers[:]:
        handler.close()
        named_logger.removeHandler(handler)


@pytest.fixture
def sample_entries():
    """Diverse set of AuditEntry instances for testing."""
    ts_base = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    entries = []
    for i, et in enumerate(EventType):
        ts = (ts_base + timedelta(minutes=i)).isoformat()
        entries.append(
            _make_entry(
                task_name=f"task_{i % 3}",
                event_type=et,
                details={"index": i, "info": f"detail_{i}"},
                timestamp=ts,
            )
        )
    return entries


# ---------------------------------------------------------------------------
# TestEventType
# ---------------------------------------------------------------------------


class TestEventType:
    def test_all_variants_exist(self):
        """EventType enum has exactly 9 variants with correct snake_case values."""
        members = list(EventType)
        assert len(members) == 9, f"Expected 9 EventType variants, got {len(members)}"
        for name in ALL_EVENT_TYPES:
            variant = EventType(name)
            assert variant.value == name, (
                f"EventType variant {name} value mismatch: {variant.value}"
            )

    def test_rejects_unknown_string(self):
        """EventType rejects unrecognized event strings."""
        with pytest.raises(ValueError):
            EventType("unknown_event")

    def test_rejects_empty_string(self):
        """EventType rejects empty string."""
        with pytest.raises(ValueError):
            EventType("")


# ---------------------------------------------------------------------------
# TestAuditEntry
# ---------------------------------------------------------------------------


class TestAuditEntryToJsonLine:
    def test_happy_path(self):
        """to_json_line returns valid compact JSON with all required keys."""
        entry = _make_entry()
        result = entry.to_json_line()

        assert isinstance(result, str), "Result should be a string"
        assert result, "Result should be non-empty"
        assert "\n" not in result, "Result must not contain newline characters"

        parsed = json.loads(result)
        for key in ("timestamp", "task_name", "event_type", "details"):
            assert key in parsed, f"Missing key '{key}' in JSON output"

    def test_roundtrip_serialization(self):
        """to_json_line output deserializes back to matching field values."""
        entry = _make_entry(
            task_name="roundtrip_task",
            event_type=EventType.fine_tune_started,
            details={"x": 42, "nested": {"a": True}},
        )
        result = entry.to_json_line()
        parsed = json.loads(result)

        assert parsed["task_name"] == "roundtrip_task"
        assert parsed["event_type"] == "fine_tune_started"
        assert parsed["details"]["x"] == 42
        assert parsed["details"]["nested"]["a"] is True

    def test_special_characters_and_unicode(self):
        """to_json_line handles unicode and special chars properly."""
        entry = _make_entry(
            task_name="tâche_spéciale_日本語",
            details={"msg": 'quote"and\\backslash', "emoji": "🎉"},
        )
        result = entry.to_json_line()
        assert "\n" not in result
        parsed = json.loads(result)
        assert parsed["task_name"] == "tâche_spéciale_日本語"
        assert "🎉" in parsed["details"]["emoji"]

    def test_empty_details(self):
        """to_json_line works with empty details dict."""
        entry = _make_entry(details={})
        result = entry.to_json_line()
        parsed = json.loads(result)
        assert parsed["details"] == {}

    def test_details_json_serializable(self):
        """Details with standard JSON types serialize successfully."""
        entry = _make_entry(
            details={
                "str_val": "hello",
                "int_val": 123,
                "float_val": 1.5,
                "bool_val": True,
                "null_val": None,
                "list_val": [1, 2, 3],
                "dict_val": {"nested": "ok"},
            }
        )
        result = entry.to_json_line()
        parsed = json.loads(result)
        assert parsed["details"]["int_val"] == 123
        assert parsed["details"]["null_val"] is None


class TestAuditEntryFrozen:
    def test_immutable_after_construction(self):
        """AuditEntry is frozen and cannot be mutated after construction."""
        entry = _make_entry()
        with pytest.raises((AttributeError, TypeError)):
            entry.task_name = "modified"
        with pytest.raises((AttributeError, TypeError)):
            entry.event_type = EventType.budget_warning
        with pytest.raises((AttributeError, TypeError)):
            entry.details = {"new": "dict"}


# ---------------------------------------------------------------------------
# TestAuditConfig
# ---------------------------------------------------------------------------


class TestAuditConfig:
    def test_valid_config(self, tmp_path):
        """AuditConfig accepts valid log_path with existing writable parent."""
        log_path = str(tmp_path / "audit.jsonl")
        config = AuditConfig(log_path=log_path)
        assert config.log_path == log_path

    def test_empty_log_path_rejected(self):
        """AuditConfig rejects empty log_path string."""
        with pytest.raises((ValueError, Exception)):
            AuditConfig(log_path="")

    def test_parent_not_exists_rejected(self, tmp_path):
        """AuditConfig rejects log_path whose parent directory does not exist."""
        bad_path = str(tmp_path / "nonexistent_dir" / "audit.jsonl")
        with pytest.raises((ValueError, Exception)):
            AuditConfig(log_path=bad_path)


# ---------------------------------------------------------------------------
# TestJsonLinesAuditLoggerInit
# ---------------------------------------------------------------------------


class TestJsonLinesAuditLoggerInit:
    def test_happy_path(self, audit_config, audit_log_path):
        """JsonLinesAuditLogger initializes successfully with valid config."""
        logger = JsonLinesAuditLogger(config=audit_config)
        assert Path(audit_log_path).exists(), "Log file should be created"
        named = logging.getLogger("apprentice.audit")
        assert named.propagate is False, "audit logger must have propagate=False"
        # Cleanup
        for h in named.handlers[:]:
            h.close()
            named.removeHandler(h)

    def test_creates_file_if_not_exists(self, audit_config, audit_log_path):
        """Logger creates the log file if it does not exist."""
        assert not Path(audit_log_path).exists()
        logger = JsonLinesAuditLogger(config=audit_config)
        assert Path(audit_log_path).exists(), "Log file should have been created"
        for h in logging.getLogger("apprentice.audit").handlers[:]:
            h.close()
            logging.getLogger("apprentice.audit").removeHandler(h)

    def test_no_truncate_existing_content(self, audit_config, audit_log_path):
        """Logger does not truncate existing log file content."""
        existing_line = '{"existing": "data"}\n'
        Path(audit_log_path).write_text(existing_line)
        logger = JsonLinesAuditLogger(config=audit_config)
        content = Path(audit_log_path).read_text()
        assert existing_line in content, "Existing content must be preserved"
        for h in logging.getLogger("apprentice.audit").handlers[:]:
            h.close()
            logging.getLogger("apprentice.audit").removeHandler(h)

    def test_propagate_false(self, audit_config):
        """The apprentice.audit logger has propagate=False."""
        logger = JsonLinesAuditLogger(config=audit_config)
        named = logging.getLogger("apprentice.audit")
        assert named.propagate is False
        for h in named.handlers[:]:
            h.close()
            named.removeHandler(h)


# ---------------------------------------------------------------------------
# TestLog
# ---------------------------------------------------------------------------


class TestLog:
    def test_single_entry(self, audit_logger, audit_log_path):
        """Logging a single entry appends exactly one JSONL line."""
        entry = _make_entry()
        audit_logger.log(entry)
        lines = Path(audit_log_path).read_text().strip().splitlines()
        assert len(lines) == 1, f"Expected 1 line, got {len(lines)}"
        parsed = json.loads(lines[0])
        assert parsed["task_name"] == "test_task"
        assert parsed["event_type"] == "request_routed"

    def test_multiple_entries_append_only(self, audit_logger, audit_log_path):
        """Logging multiple entries appends in order without modifying previous."""
        entries = [
            _make_entry(task_name=f"task_{i}", event_type=EventType.request_routed)
            for i in range(5)
        ]

        for i, e in enumerate(entries):
            audit_logger.log(e)
            lines = Path(audit_log_path).read_text().strip().splitlines()
            assert len(lines) == i + 1, f"Expected {i+1} lines after logging entry {i}"

        lines = Path(audit_log_path).read_text().strip().splitlines()
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed["task_name"] == f"task_{i}", (
                f"Line {i} task_name mismatch"
            )

    def test_entry_flushed_immediately(self, audit_logger, audit_log_path):
        """Logged entry is readable from file immediately after log() returns."""
        entry = _make_entry(task_name="flush_test")
        audit_logger.log(entry)
        content = Path(audit_log_path).read_text()
        assert "flush_test" in content, "Entry should be flushed to disk"

    def test_append_only_invariant(self, audit_logger, audit_log_path):
        """File size only grows; previous lines remain unchanged."""
        entry1 = _make_entry(task_name="first")
        audit_logger.log(entry1)
        content_after_first = Path(audit_log_path).read_text()

        entry2 = _make_entry(task_name="second")
        audit_logger.log(entry2)
        content_after_second = Path(audit_log_path).read_text()

        assert content_after_second.startswith(content_after_first), (
            "New content must be appended, not replacing old content"
        )
        assert len(content_after_second) > len(content_after_first)


# ---------------------------------------------------------------------------
# TestEntriesByTask
# ---------------------------------------------------------------------------


class TestEntriesByTask:
    def test_matching_entries(self, audit_logger, sample_entries):
        """entries_by_task returns all entries matching the given task_name."""
        for e in sample_entries:
            audit_logger.log(e)

        results = audit_logger.entries_by_task("task_0")
        assert len(results) > 0, "Should find at least one match"
        for entry in results:
            assert entry.task_name == "task_0", (
                f"Returned entry has wrong task_name: {entry.task_name}"
            )

    def test_no_match_returns_empty(self, audit_logger, sample_entries):
        """entries_by_task returns empty list when no entries match."""
        for e in sample_entries:
            audit_logger.log(e)
        results = audit_logger.entries_by_task("nonexistent_task")
        assert results == [], f"Expected empty list, got {len(results)} entries"

    def test_case_sensitive_match(self, audit_logger):
        """entries_by_task uses case-sensitive exact match."""
        audit_logger.log(_make_entry(task_name="TaskA"))
        audit_logger.log(_make_entry(task_name="taska"))

        results_upper = audit_logger.entries_by_task("TaskA")
        results_lower = audit_logger.entries_by_task("taska")

        assert len(results_upper) == 1
        assert results_upper[0].task_name == "TaskA"
        assert len(results_lower) == 1
        assert results_lower[0].task_name == "taska"

    def test_empty_task_name_error(self, audit_logger):
        """entries_by_task raises error for empty task_name."""
        with pytest.raises((ValueError, Exception)):
            audit_logger.entries_by_task("")

    def test_does_not_modify_file(self, audit_logger, audit_log_path, sample_entries):
        """entries_by_task does not modify the audit log file."""
        for e in sample_entries:
            audit_logger.log(e)
        content_before = Path(audit_log_path).read_text()
        audit_logger.entries_by_task("task_0")
        content_after = Path(audit_log_path).read_text()
        assert content_before == content_after, "Query must not modify the file"

    def test_skips_corrupted_lines(self, audit_logger, audit_log_path):
        """entries_by_task skips corrupted lines without failing."""
        audit_logger.log(_make_entry(task_name="valid_task"))
        # Manually inject a corrupted line
        with open(audit_log_path, "a") as f:
            f.write("THIS IS NOT VALID JSON\n")
        audit_logger.log(_make_entry(task_name="valid_task"))

        results = audit_logger.entries_by_task("valid_task")
        assert len(results) == 2, (
            f"Expected 2 valid entries, got {len(results)}"
        )

    def test_chronological_order(self, audit_logger):
        """entries_by_task returns entries in chronological (file) order."""
        ts_base = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        for i in range(5):
            ts = (ts_base + timedelta(seconds=i)).isoformat()
            audit_logger.log(_make_entry(task_name="ordered", timestamp=ts))

        results = audit_logger.entries_by_task("ordered")
        timestamps = [e.timestamp for e in results]
        assert timestamps == sorted(timestamps), "Entries must be in chronological order"


# ---------------------------------------------------------------------------
# TestEntriesByType
# ---------------------------------------------------------------------------


class TestEntriesByType:
    def test_happy_path(self, audit_logger, sample_entries):
        """entries_by_type returns all entries matching the given EventType."""
        for e in sample_entries:
            audit_logger.log(e)

        target_type = EventType.fine_tune_started
        results = audit_logger.entries_by_type(target_type)
        assert len(results) > 0, "Should find at least one match"
        for entry in results:
            assert entry.event_type == target_type, (
                f"Returned entry has wrong event_type: {entry.event_type}"
            )

    def test_no_match_returns_empty(self, audit_logger):
        """entries_by_type returns empty list when no entries match."""
        audit_logger.log(_make_entry(event_type=EventType.request_routed))
        results = audit_logger.entries_by_type(EventType.budget_warning)
        assert results == []

    def test_does_not_modify_file(self, audit_logger, audit_log_path, sample_entries):
        """entries_by_type does not modify the audit log file."""
        for e in sample_entries:
            audit_logger.log(e)
        content_before = Path(audit_log_path).read_text()
        audit_logger.entries_by_type(EventType.request_routed)
        content_after = Path(audit_log_path).read_text()
        assert content_before == content_after

    def test_skips_corrupted_lines(self, audit_logger, audit_log_path):
        """entries_by_type skips corrupted lines without failing."""
        audit_logger.log(_make_entry(event_type=EventType.model_promoted))
        with open(audit_log_path, "a") as f:
            f.write("{bad json\n")
        audit_logger.log(_make_entry(event_type=EventType.model_promoted))

        results = audit_logger.entries_by_type(EventType.model_promoted)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# TestEntriesInRange
# ---------------------------------------------------------------------------


class TestEntriesInRange:
    def _log_timed_entries(self, audit_logger):
        """Log entries with known timestamps, return (entries, timestamps)."""
        ts_base = datetime(2024, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
        timestamps = []
        for i in range(5):
            ts = (ts_base + timedelta(hours=i)).isoformat()
            timestamps.append(ts)
            audit_logger.log(_make_entry(task_name=f"timed_{i}", timestamp=ts))
        return timestamps

    def test_happy_path(self, audit_logger):
        """entries_in_range returns entries within [start, end] inclusive."""
        timestamps = self._log_timed_entries(audit_logger)
        start = timestamps[1]
        end = timestamps[3]
        results = audit_logger.entries_in_range(start, end)
        assert len(results) == 3, f"Expected 3 entries in range, got {len(results)}"
        for entry in results:
            assert entry.timestamp >= start
            assert entry.timestamp <= end

    def test_boundary_inclusivity(self, audit_logger):
        """entries_in_range includes entries exactly at start and end boundaries."""
        timestamps = self._log_timed_entries(audit_logger)
        start = timestamps[0]
        end = timestamps[4]
        results = audit_logger.entries_in_range(start, end)
        result_timestamps = [e.timestamp for e in results]
        # The boundary timestamps should be included
        assert any(t == start for t in result_timestamps), "Start boundary must be inclusive"
        assert any(t == end for t in result_timestamps), "End boundary must be inclusive"

    def test_no_match_empty_list(self, audit_logger):
        """entries_in_range returns empty list when no entries in range."""
        self._log_timed_entries(audit_logger)
        far_future_start = datetime(2099, 1, 1, tzinfo=timezone.utc).isoformat()
        far_future_end = datetime(2099, 12, 31, tzinfo=timezone.utc).isoformat()
        results = audit_logger.entries_in_range(far_future_start, far_future_end)
        assert results == []

    def test_start_after_end(self, audit_logger):
        """entries_in_range raises error or returns empty when start > end."""
        self._log_timed_entries(audit_logger)
        later = datetime(2024, 6, 1, tzinfo=timezone.utc).isoformat()
        earlier = datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()
        try:
            results = audit_logger.entries_in_range(later, earlier)
            # If no exception, should return empty list
            assert results == [], "start > end should yield empty list or raise error"
        except (ValueError, Exception):
            pass  # Raising an error is also acceptable per contract

    def test_does_not_modify_file(self, audit_logger, audit_log_path):
        """entries_in_range does not modify the audit log file."""
        timestamps = self._log_timed_entries(audit_logger)
        content_before = Path(audit_log_path).read_text()
        audit_logger.entries_in_range(timestamps[0], timestamps[-1])
        content_after = Path(audit_log_path).read_text()
        assert content_before == content_after

    def test_skips_corrupted_lines(self, audit_logger, audit_log_path):
        """entries_in_range skips corrupted lines without failing."""
        timestamps = self._log_timed_entries(audit_logger)
        with open(audit_log_path, "a") as f:
            f.write("CORRUPTED LINE HERE\n")
        results = audit_logger.entries_in_range(timestamps[0], timestamps[-1])
        assert len(results) == 5, "Should return all 5 valid entries, skipping corrupted"


# ---------------------------------------------------------------------------
# TestRoundTrip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_log_and_query_all_back(self, audit_logger):
        """Log N entries then query all back, verify count and content integrity."""
        entries = []
        for i in range(10):
            e = _make_entry(
                task_name="roundtrip",
                event_type=EventType.request_routed,
                details={"index": i},
            )
            entries.append(e)
            audit_logger.log(e)

        results = audit_logger.entries_by_task("roundtrip")
        assert len(results) == 10, f"Expected 10 entries, got {len(results)}"

        for i, result in enumerate(results):
            assert result.task_name == "roundtrip"
            assert result.event_type == EventType.request_routed
            assert result.details["index"] == i, (
                f"Entry {i} details mismatch: {result.details}"
            )

    def test_mixed_types_filtered_correctly(self, audit_logger):
        """Log entries of various types, filter returns only the requested type."""
        types_logged = [
            EventType.request_routed,
            EventType.budget_warning,
            EventType.request_routed,
            EventType.fine_tune_completed,
            EventType.budget_warning,
        ]
        for et in types_logged:
            audit_logger.log(_make_entry(event_type=et))

        routed = audit_logger.entries_by_type(EventType.request_routed)
        assert len(routed) == 2

        warnings = audit_logger.entries_by_type(EventType.budget_warning)
        assert len(warnings) == 2

        completed = audit_logger.entries_by_type(EventType.fine_tune_completed)
        assert len(completed) == 1
