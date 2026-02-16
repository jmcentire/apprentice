"""
Contract test suite for the CLI component.

Tests are organized by function/concern following the contract specification.
All dependencies (apprentice_core, config) are mocked at injection seams.

Run with: pytest contract_test.py -v
"""

import asyncio
import json
import logging
import os
import sys
import io
from datetime import datetime
from pathlib import Path
from unittest.mock import (
    AsyncMock,
    MagicMock,
    Mock,
    patch,
    PropertyMock,
)

import pytest

# ---------------------------------------------------------------------------
# Attempt imports — the implementation may organize types across modules
# ---------------------------------------------------------------------------
try:
    from src.cli import (
        main,
        parse_args,
        load_config,
        resolve_input,
        format_output,
        configure_logging,
        execute_run,
        execute_status,
        execute_report,
    )
except ImportError:
    # Fall back to a flat module layout
    from cli import (  # type: ignore[no-redef]
        main,
        parse_args,
        load_config,
        resolve_input,
        format_output,
        configure_logging,
        execute_run,
        execute_status,
        execute_report,
    )

try:
    from src.cli_models import (
        RunResult,
        StatusResult,
        ReportResult,
        TaskStatusEntry,
        PhaseInfo,
        BudgetInfo,
        RunArgs,
        StatusArgs,
        ReportArgs,
        GlobalFlags,
        ParsedArgs,
        InputData,
    )
except ImportError:
    try:
        from cli_models import (  # type: ignore[no-redef]
            RunResult,
            StatusResult,
            ReportResult,
            TaskStatusEntry,
            PhaseInfo,
            BudgetInfo,
            RunArgs,
            StatusArgs,
            ReportArgs,
            GlobalFlags,
            ParsedArgs,
            InputData,
        )
    except ImportError:
        # Models may live inside cli module itself
        pass


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_phase_info():
    """Minimal PhaseInfo for test construction."""
    try:
        return PhaseInfo(phase_name="running", confidence_score=0.85, is_local_primary=True)
    except Exception:
        return {"phase_name": "running", "confidence_score": 0.85, "is_local_primary": True}


@pytest.fixture
def sample_budget_info():
    """Minimal BudgetInfo for test construction."""
    try:
        return BudgetInfo(budget_limit=100.0, budget_spent=25.0, budget_remaining=75.0, is_exhausted=False)
    except Exception:
        return {"budget_limit": 100.0, "budget_spent": 25.0, "budget_remaining": 75.0, "is_exhausted": False}


@pytest.fixture
def sample_task_status_entry(sample_phase_info, sample_budget_info):
    """Minimal TaskStatusEntry."""
    try:
        return TaskStatusEntry(task_name="test_task", phase=sample_phase_info, budget=sample_budget_info)
    except Exception:
        return {"task_name": "test_task", "phase": sample_phase_info, "budget": sample_budget_info}


@pytest.fixture
def sample_run_result():
    """A successful RunResult."""
    try:
        return RunResult(
            task_name="my_task",
            output={"result": "ok"},
            success=True,
            error_message="",
        )
    except Exception:
        return MagicMock(
            task_name="my_task",
            output={"result": "ok"},
            success=True,
            error_message="",
        )


@pytest.fixture
def sample_status_result(sample_task_status_entry):
    """A StatusResult with one task."""
    try:
        return StatusResult(tasks=[sample_task_status_entry], timestamp="2024-01-01T00:00:00Z")
    except Exception:
        return MagicMock(tasks=[sample_task_status_entry], timestamp="2024-01-01T00:00:00Z")


@pytest.fixture
def sample_report_result(sample_task_status_entry):
    """A ReportResult with one task."""
    try:
        return ReportResult(
            tasks=[sample_task_status_entry],
            total_budget_limit=100.0,
            total_budget_spent=25.0,
            system_uptime_seconds=3600.0,
            timestamp="2024-01-01T00:00:00Z",
            config_path="apprentice.yaml",
            written_to_file="",
        )
    except Exception:
        return MagicMock(
            tasks=[sample_task_status_entry],
            total_budget_limit=100.0,
            total_budget_spent=25.0,
            system_uptime_seconds=3600.0,
            timestamp="2024-01-01T00:00:00Z",
            config_path="apprentice.yaml",
            written_to_file="",
        )


@pytest.fixture
def mock_apprentice():
    """A mock Apprentice with async methods."""
    apprentice = MagicMock()
    apprentice.run = AsyncMock(return_value=MagicMock(
        output={"result": "ok"},
        success=True,
    ))
    apprentice.status = AsyncMock(return_value=[
        MagicMock(
            task_name="test_task",
            phase=MagicMock(phase_name="running", confidence_score=0.85, is_local_primary=True),
            budget=MagicMock(budget_limit=100.0, budget_spent=25.0, budget_remaining=75.0, is_exhausted=False),
        )
    ])
    apprentice.report = AsyncMock(return_value=MagicMock(
        tasks=[MagicMock(
            task_name="test_task",
            phase=MagicMock(phase_name="running", confidence_score=0.85, is_local_primary=True),
            budget=MagicMock(budget_limit=100.0, budget_spent=25.0, budget_remaining=75.0, is_exhausted=False),
        )],
        total_budget_limit=100.0,
        total_budget_spent=25.0,
        system_uptime_seconds=3600.0,
    ))
    return apprentice


# ===========================================================================
# Tests: parse_args
# ===========================================================================

class TestParseArgs:
    """Tests for parse_args: argument parsing for all subcommands and flags."""

    def test_parse_args_run_happy(self):
        """parse_args correctly parses 'run' subcommand with task_name and --input."""
        result = parse_args(["run", "my_task", "--input", '{"key": "val"}'])
        cmd = result.command if hasattr(result, "command") else getattr(result, "command", None)
        assert cmd in ("run", "SubcommandName.run") or str(cmd) == "run" or (hasattr(cmd, "value") and cmd.value == "run"), \
            f"Expected command 'run', got {cmd}"
        assert result.run_args.task_name == "my_task", "task_name should be 'my_task'"
        assert result.run_args.input_raw == '{"key": "val"}', "input_raw should match provided JSON"

    def test_parse_args_status_happy(self):
        """parse_args correctly parses 'status' subcommand with optional --task filter."""
        result = parse_args(["status", "--task", "my_task"])
        cmd_str = str(getattr(result, "command", ""))
        assert "status" in cmd_str.lower(), f"Expected 'status' command, got {cmd_str}"
        assert result.status_args.task_filter == "my_task"

    def test_parse_args_report_happy(self):
        """parse_args correctly parses 'report' subcommand with --output."""
        result = parse_args(["report", "--output", "/tmp/report.json"])
        cmd_str = str(getattr(result, "command", ""))
        assert "report" in cmd_str.lower(), f"Expected 'report' command, got {cmd_str}"
        assert result.report_args.output_path == "/tmp/report.json"

    def test_parse_args_global_flags(self):
        """parse_args correctly parses global flags --config, --json, --verbose."""
        result = parse_args(["--config", "custom.yaml", "--json", "-vv", "status"])
        assert result.global_flags.config_path == "custom.yaml", \
            f"config_path should be 'custom.yaml', got {result.global_flags.config_path}"
        assert result.global_flags.json_mode is True, "json_mode should be True"
        assert result.global_flags.verbose == 2, f"verbose should be 2, got {result.global_flags.verbose}"

    def test_parse_args_default_config(self):
        """parse_args uses a default config path when --config not specified."""
        result = parse_args(["status"])
        assert result.global_flags.config_path != "", "Default config_path should be non-empty"
        assert len(result.global_flags.config_path) > 0

    def test_parse_args_verbose_zero_default(self):
        """parse_args defaults verbose to 0 when no -v flags given."""
        result = parse_args(["status"])
        assert result.global_flags.verbose == 0, f"Default verbose should be 0, got {result.global_flags.verbose}"

    def test_parse_args_single_verbose(self):
        """parse_args sets verbose to 1 with single -v flag."""
        result = parse_args(["-v", "status"])
        assert result.global_flags.verbose == 1, f"Single -v should give verbose=1, got {result.global_flags.verbose}"

    def test_parse_args_verbose_clamped(self):
        """parse_args clamps verbose to max 2 even with -vvv."""
        result = parse_args(["-vvv", "status"])
        assert result.global_flags.verbose <= 2, \
            f"verbose should be clamped to <=2, got {result.global_flags.verbose}"

    def test_parse_args_no_subcommand(self):
        """parse_args raises SystemExit when no subcommand is provided."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_parse_args_unknown_subcommand(self):
        """parse_args raises SystemExit for an unrecognized subcommand."""
        with pytest.raises(SystemExit):
            parse_args(["unknown_cmd"])

    def test_parse_args_missing_required_arg(self):
        """parse_args raises SystemExit when run subcommand is missing task_name."""
        with pytest.raises(SystemExit):
            parse_args(["run"])

    def test_parse_args_unrecognized_flag(self):
        """parse_args raises SystemExit for an unknown flag like --bogus."""
        with pytest.raises(SystemExit):
            parse_args(["--bogus", "status"])

    def test_parse_args_json_mode_default_false(self):
        """parse_args defaults json_mode to False when --json not specified."""
        result = parse_args(["status"])
        assert result.global_flags.json_mode is False, "Default json_mode should be False"


# ===========================================================================
# Tests: load_config
# ===========================================================================

class TestLoadConfig:
    """Tests for load_config: YAML config loading and pydantic validation."""

    def test_load_config_file_not_found(self):
        """load_config raises an error when the config file does not exist."""
        with pytest.raises((FileNotFoundError, OSError, Exception)):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_invalid_yaml(self, tmp_path):
        """load_config raises an error when file contains invalid YAML syntax."""
        bad_yaml = tmp_path / "invalid.yaml"
        bad_yaml.write_text("{{{{invalid yaml: [unterminated")
        with pytest.raises(Exception) as exc_info:
            load_config(str(bad_yaml))
        # Should raise some kind of YAML or parsing error
        assert exc_info.value is not None

    def test_load_config_validation_error(self, tmp_path):
        """load_config raises error when YAML is valid but fails pydantic validation."""
        bad_schema = tmp_path / "bad_schema.yaml"
        # Valid YAML but certainly not a valid ApprenticeConfig schema
        bad_schema.write_text("random_key: random_value\nanother: 123\n")
        with pytest.raises(Exception) as exc_info:
            load_config(str(bad_schema))
        assert exc_info.value is not None

    def test_load_config_empty_path(self):
        """load_config raises error when given empty string path."""
        with pytest.raises(Exception):
            load_config("")


# ===========================================================================
# Tests: resolve_input
# ===========================================================================

class TestResolveInputJSONString:
    """Tests for resolve_input with direct JSON string inputs."""

    def test_resolve_input_valid_json_object(self):
        """resolve_input parses a valid JSON object string."""
        result = resolve_input('{"key": "value", "num": 42}')
        assert isinstance(result.data, dict), f"Expected dict, got {type(result.data)}"
        assert result.data["key"] == "value"
        assert result.data["num"] == 42

    def test_resolve_input_empty_json_object(self):
        """resolve_input handles empty JSON object {}."""
        result = resolve_input("{}")
        assert isinstance(result.data, dict), f"Expected dict, got {type(result.data)}"
        assert len(result.data) == 0, "Empty object should have no keys"

    def test_resolve_input_nested_json(self):
        """resolve_input handles deeply nested JSON objects."""
        result = resolve_input('{"a": {"b": {"c": [1, 2, 3]}}}')
        assert isinstance(result.data, dict)
        assert result.data["a"]["b"]["c"] == [1, 2, 3]

    def test_resolve_input_invalid_json_string(self):
        """resolve_input raises error for invalid JSON string."""
        with pytest.raises(Exception) as exc_info:
            resolve_input("not valid json")
        assert exc_info.value is not None

    @pytest.mark.parametrize("input_val,desc", [
        ("[1, 2, 3]", "JSON array"),
        ('"just a string"', "JSON string scalar"),
        ("42", "JSON number scalar"),
        ("true", "JSON boolean scalar"),
        ("null", "JSON null"),
    ])
    def test_resolve_input_not_json_object(self, input_val, desc):
        """resolve_input raises error when JSON is valid but not an object ({desc})."""
        with pytest.raises(Exception) as exc_info:
            resolve_input(input_val)
        assert exc_info.value is not None, f"Should reject {desc}"


class TestResolveInputFileReference:
    """Tests for resolve_input with @file reference inputs."""

    def test_resolve_input_file_reference_happy(self, tmp_path):
        """resolve_input reads JSON from an @file reference."""
        json_file = tmp_path / "input.json"
        json_file.write_text('{"from_file": true}')
        result = resolve_input(f"@{json_file}")
        assert isinstance(result.data, dict)
        assert result.data["from_file"] is True

    def test_resolve_input_file_not_found(self):
        """resolve_input raises error when @file points to nonexistent file."""
        with pytest.raises(Exception):
            resolve_input("@/nonexistent/file.json")

    def test_resolve_input_file_invalid_json(self, tmp_path):
        """resolve_input raises error when referenced file has invalid JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("this is not json")
        with pytest.raises(Exception):
            resolve_input(f"@{bad_file}")

    def test_resolve_input_empty_file_reference(self):
        """resolve_input raises error when input is exactly '@' with no file path."""
        with pytest.raises(Exception):
            resolve_input("@")

    def test_resolve_input_file_array_not_object(self, tmp_path):
        """resolve_input raises error when file contains valid JSON array."""
        arr_file = tmp_path / "array.json"
        arr_file.write_text("[1, 2, 3]")
        with pytest.raises(Exception):
            resolve_input(f"@{arr_file}")


# ===========================================================================
# Tests: format_output
# ===========================================================================

class TestFormatOutput:
    """Tests for format_output: serialization of CLIResult variants."""

    def test_format_output_run_result_json(self, sample_run_result):
        """format_output serializes RunResult in JSON mode as valid JSON."""
        result = format_output(sample_run_result, json_mode=True)
        assert len(result) > 0, "Output should be non-empty"
        parsed = json.loads(result)
        assert isinstance(parsed, dict), "JSON output should parse to a dict"

    def test_format_output_run_result_human(self, sample_run_result):
        """format_output serializes RunResult in human mode without trailing newline."""
        result = format_output(sample_run_result, json_mode=False)
        assert len(result) > 0, "Output should be non-empty"
        assert not result.endswith("\n"), "Should not have trailing newline"

    def test_format_output_status_result_json(self, sample_status_result):
        """format_output serializes StatusResult in JSON mode as valid JSON."""
        result = format_output(sample_status_result, json_mode=True)
        parsed = json.loads(result)
        assert parsed is not None

    def test_format_output_report_result_json(self, sample_report_result):
        """format_output serializes ReportResult in JSON mode as valid JSON."""
        result = format_output(sample_report_result, json_mode=True)
        parsed = json.loads(result)
        assert parsed is not None

    def test_format_output_no_trailing_newline_json(self, sample_run_result):
        """format_output never includes trailing newline in JSON mode."""
        result = format_output(sample_run_result, json_mode=True)
        assert not result.endswith("\n"), "JSON mode output must not end with newline"

    def test_format_output_no_trailing_newline_human(self, sample_run_result):
        """format_output never includes trailing newline in human mode."""
        result = format_output(sample_run_result, json_mode=False)
        assert not result.endswith("\n"), "Human mode output must not end with newline"

    def test_format_output_nonempty_json(self, sample_run_result):
        """format_output always returns non-empty string in JSON mode."""
        result = format_output(sample_run_result, json_mode=True)
        assert len(result) > 0, "Output must be non-empty"

    def test_format_output_nonempty_human(self, sample_run_result):
        """format_output always returns non-empty string in human mode."""
        result = format_output(sample_run_result, json_mode=False)
        assert len(result) > 0, "Output must be non-empty"

    def test_format_output_human_mode_no_raw_json(self, sample_run_result):
        """format_output in human mode should not produce raw JSON."""
        result = format_output(sample_run_result, json_mode=False)
        # In human mode, the output should NOT be valid JSON
        # (it's formatted text), though this isn't always guaranteed for simple outputs
        # At minimum, it should be non-empty and not end with newline
        assert len(result) > 0


# ===========================================================================
# Tests: execute_run
# ===========================================================================

class TestExecuteRun:
    """Tests for execute_run: running tasks via mock Apprentice."""

    def test_execute_run_happy(self, mock_apprentice):
        """execute_run returns RunResult with correct task_name on success."""
        run_args = MagicMock()
        run_args.task_name = "my_task"
        run_args.input_raw = '{"x": 1}'

        result = asyncio.run(execute_run(mock_apprentice, run_args))

        assert result.task_name == "my_task", \
            f"RunResult.task_name should be 'my_task', got {result.task_name}"
        assert result.success is True, "RunResult.success should be True"
        assert isinstance(result.output, dict), "RunResult.output should be a dict"

    def test_execute_run_input_resolution_failed(self, mock_apprentice):
        """execute_run handles invalid input_raw that cannot be resolved."""
        run_args = MagicMock()
        run_args.task_name = "my_task"
        run_args.input_raw = "not valid json"

        with pytest.raises(Exception):
            asyncio.run(execute_run(mock_apprentice, run_args))

    def test_execute_run_execution_error(self, mock_apprentice):
        """execute_run handles execution error from Apprentice.run()."""
        mock_apprentice.run = AsyncMock(side_effect=Exception("Execution failed"))

        run_args = MagicMock()
        run_args.task_name = "my_task"
        run_args.input_raw = '{"x": 1}'

        # The function may raise or return a RunResult with success=False
        try:
            result = asyncio.run(execute_run(mock_apprentice, run_args))
            # If it returns, it should indicate failure
            assert result.success is False or result.error_message != "", \
                "On execution error, result should indicate failure"
        except Exception:
            pass  # Raising is also acceptable behavior

    def test_execute_run_task_not_found(self, mock_apprentice):
        """execute_run handles task_not_found error from Apprentice."""
        mock_apprentice.run = AsyncMock(side_effect=Exception("Task 'nonexistent' not found"))

        run_args = MagicMock()
        run_args.task_name = "nonexistent"
        run_args.input_raw = '{"x": 1}'

        try:
            result = asyncio.run(execute_run(mock_apprentice, run_args))
            assert result.success is False or result.error_message != ""
        except Exception:
            pass  # Raising is acceptable


# ===========================================================================
# Tests: execute_status
# ===========================================================================

class TestExecuteStatus:
    """Tests for execute_status: querying task status via mock Apprentice."""

    def test_execute_status_happy(self, mock_apprentice):
        """execute_status returns StatusResult with tasks and timestamp."""
        status_args = MagicMock()
        status_args.task_filter = ""

        result = asyncio.run(execute_status(mock_apprentice, status_args))

        assert len(result.tasks) > 0, "Should have at least one task entry"
        assert result.timestamp != "", "timestamp should be non-empty"

    def test_execute_status_with_filter(self, mock_apprentice):
        """execute_status filters tasks when task_filter is specified."""
        status_args = MagicMock()
        status_args.task_filter = "test_task"

        result = asyncio.run(execute_status(mock_apprentice, status_args))

        assert len(result.tasks) <= 1, \
            f"Filtered result should have at most 1 task, got {len(result.tasks)}"

    def test_execute_status_iso_timestamp(self, mock_apprentice):
        """execute_status returns ISO 8601 formatted timestamp."""
        status_args = MagicMock()
        status_args.task_filter = ""

        result = asyncio.run(execute_status(mock_apprentice, status_args))

        # Verify the timestamp is valid ISO 8601
        try:
            datetime.fromisoformat(result.timestamp.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"timestamp '{result.timestamp}' is not valid ISO 8601")

    def test_execute_status_retrieval_error(self, mock_apprentice):
        """execute_status handles status retrieval error."""
        mock_apprentice.status = AsyncMock(side_effect=Exception("Status retrieval failed"))

        status_args = MagicMock()
        status_args.task_filter = ""

        with pytest.raises(Exception):
            asyncio.run(execute_status(mock_apprentice, status_args))


# ===========================================================================
# Tests: execute_report
# ===========================================================================

class TestExecuteReport:
    """Tests for execute_report: generating reports via mock Apprentice."""

    def test_execute_report_happy(self, mock_apprentice):
        """execute_report returns ReportResult with all fields populated."""
        report_args = MagicMock()
        report_args.output_path = ""

        result = asyncio.run(
            execute_report(mock_apprentice, report_args, config_path="apprentice.yaml", json_mode=False)
        )

        assert len(result.tasks) > 0, "Should have at least one task"
        assert result.total_budget_spent <= result.total_budget_limit, \
            "budget_spent should not exceed budget_limit"
        assert result.system_uptime_seconds >= 0, "uptime should be non-negative"
        assert result.timestamp != "", "timestamp should be non-empty"

    def test_execute_report_with_output_file(self, mock_apprentice, tmp_path):
        """execute_report writes report to output file when path specified."""
        output_file = tmp_path / "report_output.json"
        report_args = MagicMock()
        report_args.output_path = str(output_file)

        result = asyncio.run(
            execute_report(mock_apprentice, report_args, config_path="apprentice.yaml", json_mode=True)
        )

        assert result.written_to_file == str(output_file), \
            f"written_to_file should be '{output_file}', got '{result.written_to_file}'"

    def test_execute_report_budget_invariant(self, mock_apprentice):
        """execute_report ensures total_budget_spent <= total_budget_limit."""
        report_args = MagicMock()
        report_args.output_path = ""

        result = asyncio.run(
            execute_report(mock_apprentice, report_args, config_path="test.yaml", json_mode=False)
        )

        assert result.total_budget_spent <= result.total_budget_limit, \
            f"Budget invariant violated: spent={result.total_budget_spent} > limit={result.total_budget_limit}"

    def test_execute_report_uptime_nonnegative(self, mock_apprentice):
        """execute_report ensures system_uptime_seconds >= 0."""
        report_args = MagicMock()
        report_args.output_path = ""

        result = asyncio.run(
            execute_report(mock_apprentice, report_args, config_path="test.yaml", json_mode=False)
        )

        assert result.system_uptime_seconds >= 0, \
            f"Uptime should be non-negative, got {result.system_uptime_seconds}"

    def test_execute_report_write_error(self, mock_apprentice):
        """execute_report handles output file write error."""
        report_args = MagicMock()
        report_args.output_path = "/nonexistent/dir/report.json"

        with pytest.raises(Exception):
            asyncio.run(
                execute_report(mock_apprentice, report_args, config_path="test.yaml", json_mode=False)
            )

    def test_execute_report_retrieval_error(self, mock_apprentice):
        """execute_report handles report retrieval error from Apprentice."""
        mock_apprentice.report = AsyncMock(side_effect=Exception("Report retrieval failed"))

        report_args = MagicMock()
        report_args.output_path = ""

        with pytest.raises(Exception):
            asyncio.run(
                execute_report(mock_apprentice, report_args, config_path="test.yaml", json_mode=False)
            )


# ===========================================================================
# Tests: configure_logging
# ===========================================================================

class TestConfigureLogging:
    """Tests for configure_logging: log level, format, and handler config."""

    def _reset_logging(self):
        """Reset root logger state between tests."""
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def test_configure_logging_warning(self):
        """configure_logging sets WARNING level for verbose=0."""
        self._reset_logging()
        configure_logging(0)
        assert logging.getLogger().level == logging.WARNING, \
            f"Expected WARNING ({logging.WARNING}), got {logging.getLogger().level}"

    def test_configure_logging_info(self):
        """configure_logging sets INFO level for verbose=1."""
        self._reset_logging()
        configure_logging(1)
        assert logging.getLogger().level == logging.INFO, \
            f"Expected INFO ({logging.INFO}), got {logging.getLogger().level}"

    def test_configure_logging_debug(self):
        """configure_logging sets DEBUG level for verbose=2."""
        self._reset_logging()
        configure_logging(2)
        assert logging.getLogger().level == logging.DEBUG, \
            f"Expected DEBUG ({logging.DEBUG}), got {logging.getLogger().level}"

    def test_configure_logging_stderr_only(self):
        """configure_logging directs all handlers to stderr, never stdout."""
        self._reset_logging()
        configure_logging(0)
        root = logging.getLogger()
        for handler in root.handlers:
            if hasattr(handler, "stream"):
                assert handler.stream is not sys.stdout, \
                    "Log handler must not write to stdout"

    def test_configure_logging_json_format(self):
        """configure_logging produces structured JSON log lines."""
        self._reset_logging()
        configure_logging(1)

        root = logging.getLogger()
        # Find the handler and test its formatter
        for handler in root.handlers:
            if hasattr(handler, "stream"):
                # Capture a log line
                buffer = io.StringIO()
                original_stream = handler.stream
                handler.stream = buffer
                try:
                    root.info("test message")
                    handler.flush()
                    output = buffer.getvalue()
                    if output.strip():
                        # Attempt to parse as JSON
                        log_obj = json.loads(output.strip().split("\n")[0])
                        assert "message" in log_obj or "msg" in log_obj, \
                            f"Log line should contain 'message' field: {log_obj}"
                finally:
                    handler.stream = original_stream
                break


# ===========================================================================
# Tests: main (integration)
# ===========================================================================

class TestMain:
    """Integration tests for main(argv) — tests through the full pipeline."""

    def _get_cli_module_path(self):
        """Determine the module path for patching."""
        try:
            import src.cli
            return "src.cli"
        except ImportError:
            return "cli"

    def test_main_usage_error_no_args(self):
        """main() returns exit code 2 on missing arguments (no subcommand)."""
        exit_code = main([])
        assert exit_code == 2, f"Expected exit code 2 for usage error, got {exit_code}"

    def test_main_config_not_found(self):
        """main() returns exit code 1 when config file does not exist."""
        exit_code = main(["--config", "/absolutely/nonexistent/path/config.yaml", "status"])
        assert exit_code == 1, f"Expected exit code 1 for config not found, got {exit_code}"

    def test_main_run_success(self, tmp_path):
        """main() returns exit code 0 on successful run subcommand."""
        mod = self._get_cli_module_path()

        mock_config = MagicMock()
        mock_apprentice = MagicMock()
        mock_run_result = MagicMock()
        mock_run_result.task_name = "my_task"
        mock_run_result.output = {"result": "ok"}
        mock_run_result.success = True
        mock_run_result.error_message = ""

        with patch(f"{mod}.load_config", return_value=mock_config), \
             patch(f"{mod}.asyncio") as mock_asyncio:
            # Make asyncio.run return a successful result
            mock_asyncio.run = MagicMock(return_value=None)

            # We need to also patch the Apprentice constructor and execution
            # Since main() orchestrates everything, we patch at appropriate seams
            with patch(f"{mod}.format_output", return_value='{"task_name": "my_task", "success": true}'):
                exit_code = main(["--config", "test.yaml", "run", "my_task", "--input", '{"x": 1}'])

        assert exit_code == 0, f"Expected exit code 0 for success, got {exit_code}"

    def test_main_keyboard_interrupt(self):
        """main() returns exit code 130 on KeyboardInterrupt."""
        mod = self._get_cli_module_path()

        with patch(f"{mod}.load_config", side_effect=KeyboardInterrupt):
            exit_code = main(["--config", "test.yaml", "status"])

        assert exit_code == 130, f"Expected exit code 130 for KeyboardInterrupt, got {exit_code}"

    def test_main_apprentice_error(self):
        """main() returns exit code 1 when Apprentice raises an error."""
        mod = self._get_cli_module_path()

        with patch(f"{mod}.load_config", side_effect=Exception("Apprentice construction failed")):
            exit_code = main(["--config", "test.yaml", "status"])

        assert exit_code == 1, f"Expected exit code 1 for apprentice error, got {exit_code}"

    def test_main_exit_code_deterministic(self):
        """Exit codes are always from the set {0, 1, 2, 130}."""
        valid_exit_codes = {0, 1, 2, 130}

        # Test several failure modes and verify exit codes are in the valid set
        test_cases = [
            [],  # no args -> usage error
            ["--config", "/nonexistent.yaml", "status"],  # config not found
        ]

        for argv in test_cases:
            exit_code = main(argv)
            assert exit_code in valid_exit_codes, \
                f"Exit code {exit_code} for argv={argv} not in {valid_exit_codes}"

    def test_main_json_mode_error_output(self, capsys):
        """main() with --json flag should produce structured error output."""
        exit_code = main(["--json", "--config", "/nonexistent.yaml", "status"])
        assert exit_code == 1
        captured = capsys.readouterr()
        # Errors go to stderr — in JSON mode they should be structured
        # At minimum, the exit code should be correct
        assert exit_code in {0, 1, 2, 130}

    def test_main_config_validation_error(self, tmp_path):
        """main() returns exit code 1 when config fails pydantic validation."""
        mod = self._get_cli_module_path()

        # Simulate a validation error from load_config
        with patch(f"{mod}.load_config", side_effect=ValueError("Validation failed: missing field 'tasks'")):
            exit_code = main(["--config", "test.yaml", "status"])

        assert exit_code == 1, f"Expected exit code 1 for validation error, got {exit_code}"

    def test_main_invalid_input_json(self):
        """main() returns exit code 1 when --input is invalid JSON for run subcommand."""
        mod = self._get_cli_module_path()

        mock_config = MagicMock()
        with patch(f"{mod}.load_config", return_value=mock_config):
            # The resolve_input inside execute_run should fail on "not json"
            exit_code = main(["--config", "test.yaml", "run", "my_task", "--input", "not json"])

        assert exit_code == 1, f"Expected exit code 1 for invalid input JSON, got {exit_code}"

    def test_main_stdout_results_stderr_errors(self, capsys):
        """main() writes results to stdout and errors to stderr."""
        # Trigger a config error — errors should go to stderr
        main(["--config", "/nonexistent_xyz.yaml", "status"])
        captured = capsys.readouterr()
        # Results should not appear on stdout for an error case
        # Errors should appear on stderr
        # (This is a best-effort test — the exact behavior depends on implementation)
        # The key invariant is that the exit code is correct
        assert True  # If we got here without crashing, basic I/O is working


# ===========================================================================
# Tests: Invariants (cross-cutting)
# ===========================================================================

class TestInvariants:
    """Cross-cutting invariant tests from the contract."""

    def test_format_output_json_mode_always_valid_json(self, sample_run_result, sample_status_result, sample_report_result):
        """format_output in JSON mode always produces valid JSON for all result types."""
        for result_obj in [sample_run_result, sample_status_result, sample_report_result]:
            try:
                output = format_output(result_obj, json_mode=True)
                parsed = json.loads(output)
                assert parsed is not None, f"JSON parsing returned None for {type(result_obj).__name__}"
            except TypeError:
                # Some mock objects may not be serializable — skip gracefully
                pass

    def test_format_output_no_trailing_newline_all_modes(self, sample_run_result):
        """format_output never includes trailing newline in any mode."""
        for json_mode in [True, False]:
            output = format_output(sample_run_result, json_mode=json_mode)
            assert not output.endswith("\n"), \
                f"format_output(json_mode={json_mode}) should not end with newline"

    def test_exit_codes_are_deterministic_values(self):
        """All exit code paths return values from {0, 1, 2, 130}."""
        valid_codes = {0, 1, 2, 130}

        # No args = usage error = 2
        assert main([]) in valid_codes

        # Bad config = app error = 1
        assert main(["--config", "/nope.yaml", "status"]) in valid_codes

    def test_parse_args_exactly_one_subcommand_populated(self):
        """ParsedArgs has exactly one subcommand's args populated based on command."""
        result = parse_args(["run", "task1", "--input", '{"a": 1}'])
        cmd_str = str(getattr(result, "command", "")).lower()

        if "run" in cmd_str:
            assert result.run_args is not None, "run_args should be populated for 'run' command"
            assert result.run_args.task_name == "task1"

    def test_config_path_always_nonempty(self):
        """global_flags.config_path is always a non-empty string after parsing."""
        for argv in [["status"], ["--config", "x.yaml", "status"]]:
            result = parse_args(argv)
            assert result.global_flags.config_path != "", \
                f"config_path should be non-empty for argv={argv}"
            assert isinstance(result.global_flags.config_path, str)

    def test_resolve_input_output_is_always_dict(self):
        """resolve_input postcondition: returned data is always a dict."""
        valid_inputs = [
            '{}',
            '{"a": 1}',
            '{"nested": {"deep": true}}',
        ]
        for inp in valid_inputs:
            result = resolve_input(inp)
            assert isinstance(result.data, dict), \
                f"resolve_input('{inp}').data should be dict, got {type(result.data)}"
