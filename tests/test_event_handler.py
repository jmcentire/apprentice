"""Tests for the EventDispatcher configurable event handling system."""

import json
import logging
import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.event_handler import EventDispatcher, _CallableHandler, _level_from_string


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeEntry:
    """Minimal stand-in for AuditEntry with to_json_line()."""

    def __init__(self, task_name="test_task", event_type="request_routed"):
        self._data = {
            "timestamp": "2025-01-01T00:00:00+00:00",
            "task_name": task_name,
            "event_type": event_type,
            "details": {"key": "value"},
        }

    def to_json_line(self) -> str:
        return json.dumps(self._data, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Tests: File handler
# ---------------------------------------------------------------------------

class TestFileHandler:
    """EventDispatcher with a file handler writes events to the log file."""

    def test_events_appear_in_file(self, tmp_path):
        cfg = {
            "type": "file",
            "level": "INFO",
            "path": str(tmp_path / "events.log"),
            "format": "json",
        }
        dispatcher = EventDispatcher([cfg], base_logger_name="test.file")
        entry = FakeEntry()
        dispatcher.emit(entry)
        dispatcher.close()

        content = (tmp_path / "events.log").read_text()
        assert "test_task" in content
        assert "request_routed" in content

    def test_multiple_events_are_appended(self, tmp_path):
        cfg = {
            "type": "file",
            "level": "INFO",
            "path": str(tmp_path / "multi.log"),
            "format": "json",
        }
        dispatcher = EventDispatcher([cfg], base_logger_name="test.file.multi")
        for i in range(3):
            dispatcher.emit(FakeEntry(task_name=f"task_{i}"))
        dispatcher.close()

        lines = [l for l in (tmp_path / "multi.log").read_text().splitlines() if l.strip()]
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# Tests: Stdout handler
# ---------------------------------------------------------------------------

class TestStdoutHandler:
    """EventDispatcher with a stdout handler emits to captured stdout."""

    def test_events_appear_on_stdout(self, capsys):
        cfg = {"type": "stdout", "level": "INFO", "format": "json"}
        dispatcher = EventDispatcher([cfg], base_logger_name="test.stdout")
        dispatcher.emit(FakeEntry())
        dispatcher.close()

        captured = capsys.readouterr()
        assert "test_task" in captured.out

    def test_text_format_on_stdout(self, capsys):
        cfg = {"type": "stdout", "level": "INFO", "format": "text"}
        dispatcher = EventDispatcher([cfg], base_logger_name="test.stdout.text")
        dispatcher.emit(FakeEntry())
        dispatcher.close()

        captured = capsys.readouterr()
        assert "request_routed" in captured.out


# ---------------------------------------------------------------------------
# Tests: Callable handler
# ---------------------------------------------------------------------------

class TestCallableHandler:
    """EventDispatcher with a callable handler invokes the callback."""

    def test_callable_is_invoked(self, tmp_path):
        # We'll use a built-in function as the callable.
        # Write a small module to tmp_path and import it.
        mod_path = tmp_path / "my_handler.py"
        mod_path.write_text(
            "collected = []\n"
            "def handle(msg):\n"
            "    collected.append(msg)\n"
        )

        # Add tmp_path to sys.path temporarily
        sys.path.insert(0, str(tmp_path))
        try:
            cfg = {
                "type": "callable",
                "level": "INFO",
                "handler": "my_handler.handle",
                "format": "json",
            }
            dispatcher = EventDispatcher([cfg], base_logger_name="test.callable")
            dispatcher.emit(FakeEntry())
            dispatcher.close()

            import my_handler
            assert len(my_handler.collected) == 1
            assert "test_task" in my_handler.collected[0]
        finally:
            sys.path.pop(0)
            # Clean up the imported module
            sys.modules.pop("my_handler", None)


# ---------------------------------------------------------------------------
# Tests: Level filtering
# ---------------------------------------------------------------------------

class TestLevelFiltering:
    """Events below a handler's level are filtered out."""

    def test_debug_events_filtered_at_info_handler(self, tmp_path):
        cfg = {
            "type": "file",
            "level": "WARNING",
            "path": str(tmp_path / "filtered.log"),
            "format": "json",
        }
        dispatcher = EventDispatcher([cfg], base_logger_name="test.filter")
        # emit() uses logger.info() — level INFO < WARNING, so should be filtered
        dispatcher.emit(FakeEntry())
        dispatcher.close()

        content = (tmp_path / "filtered.log").read_text()
        assert content.strip() == "", "INFO event should be filtered by WARNING-level handler"


# ---------------------------------------------------------------------------
# Tests: Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """When no handlers are configured, default file handler works."""

    def test_no_handlers_creates_empty_dispatcher(self):
        dispatcher = EventDispatcher([], base_logger_name="test.empty")
        # Should not crash, just log nowhere
        dispatcher.emit(FakeEntry())
        dispatcher.close()


# ---------------------------------------------------------------------------
# Tests: Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_level_from_string_valid(self):
        assert _level_from_string("DEBUG") == logging.DEBUG
        assert _level_from_string("INFO") == logging.INFO
        assert _level_from_string("WARNING") == logging.WARNING
        assert _level_from_string("ERROR") == logging.ERROR

    def test_level_from_string_invalid(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            _level_from_string("BOGUS")

    def test_level_from_string_case_insensitive(self):
        assert _level_from_string("info") == logging.INFO
        assert _level_from_string("Debug") == logging.DEBUG
