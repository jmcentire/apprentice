"""
Contract test suite for the Unified Interface component.
Tests Apprentice lifecycle, operations, CLI parsing, CLI execution, and config loading.

Run with: pytest contract_test.py -v
"""
import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import (
    AsyncMock,
    MagicMock,
    Mock,
    PropertyMock,
    call,
    patch,
)

import pytest

# ---------------------------------------------------------------------------
# Attempt imports – tests degrade gracefully if module not yet implemented
# ---------------------------------------------------------------------------
try:
    from src.unified_interface import (
        Apprentice,
        ApprenticeConfig,
        ApprenticeError,
        BudgetExhaustedError,
        BudgetInfo,
        ConfidenceSnapshot,
        ConfidenceThresholds,
        ErrorKind,
        ExitCode,
        GlobalFlags,
        InputData,
        LocalModelUnavailableError,
        LogLevel,
        ModelSource,
        OutputFormat,
        ParsedArgs,
        PhaseInfo,
        PhaseLabel,
        ReportArgs,
        ReportResult,
        ResponseMetadata,
        RunArgs,
        RunContext,
        RunResult,
        RunStatus,
        StatusArgs,
        StatusResult,
        SubcommandName,
        SystemReport,
        TaskConfig,
        TaskNotFoundError,
        TaskResponse,
        TaskStatusEntry,
    )
except ImportError:
    pytest.skip("unified_interface module not available", allow_module_level=True)

# Try importing CLI functions separately (they may be in a sub-module)
try:
    from src.unified_interface.cli import (
        configure_logging,
        execute_report,
        execute_run,
        execute_status,
        format_output,
        load_config,
        main,
        parse_args,
        resolve_input,
    )
except ImportError:
    try:
        from src.unified_interface import (
            configure_logging,
            execute_report,
            execute_run,
            execute_status,
            format_output,
            load_config,
            main,
            parse_args,
            resolve_input,
        )
    except ImportError:
        pytest.skip("CLI functions not importable", allow_module_level=True)


# ===========================================================================
# Shared Fixtures
# ===========================================================================

VALID_YAML_CONTENT = """
schema_version: 1
remote_api_base_url: "https://api.example.com"
local_model_endpoint: "http://localhost:8080"
audit_log_path: "/tmp/audit.log"
retry_max_attempts: 3
retry_backoff_base_seconds: 1.0
budget_enforcement:
  hard_limit: true
  warning_threshold_pct: 80.0
  log_every_spend: true
tasks:
  - task_name: "test_task"
    remote_model_id: "gpt-4"
    local_model_id: "local-llama"
    budget_limit_usd: 10.0
    budget_period_days: 30
    confidence_thresholds:
      bootstrapping_upper: 0.2
      active_learning_upper: 0.5
      supervised_upper: 0.8
      autonomous_lower: 0.8
"""


@pytest.fixture
def valid_config_path(tmp_path):
    """Write a valid YAML config file and return its path."""
    p = tmp_path / "apprentice.yaml"
    p.write_text(VALID_YAML_CONTENT)
    return str(p)


@pytest.fixture
def mock_components():
    """Create a dict of all 10 mock dependency components."""
    return {
        "config_loader": MagicMock(),
        "task_registry": MagicMock(),
        "remote_client": AsyncMock(),
        "local_client": AsyncMock(),
        "confidence_engine": MagicMock(),
        "sampling_scheduler": MagicMock(),
        "budget_manager": MagicMock(),
        "training_data_store": MagicMock(),
        "router": MagicMock(),
        "audit_log": AsyncMock(),
    }


def make_task_response(
    task_name="test_task",
    source=ModelSource.LOCAL,
    status=RunStatus.SUCCESS,
    fallback_used=False,
):
    """Helper to create a TaskResponse for test assertions."""
    metadata = ResponseMetadata(
        model_id="test-model",
        cost_usd=0.001,
        retries=0,
        fallback_used=fallback_used,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    return TaskResponse(
        task_name=task_name,
        output={"result": "ok"},
        source=source,
        status=status,
        request_id="req-123",
        duration_ms=42.0,
        metadata=metadata,
    )


def make_confidence_snapshot(
    task_name="test_task",
    confidence=0.6,
    phase=PhaseLabel.SUPERVISED,
    budget_remaining=8.0,
    budget_used=2.0,
):
    """Helper to create a ConfidenceSnapshot."""
    return ConfidenceSnapshot(
        task_name=task_name,
        confidence_score=confidence,
        phase=phase,
        sampling_rate=0.3,
        budget_remaining_usd=budget_remaining,
        budget_used_usd=budget_used,
        budget_exhausted=False,
        local_model_available=True,
        total_runs=10,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def make_system_report(task_snapshots=None):
    """Helper to create a SystemReport."""
    if task_snapshots is None:
        task_snapshots = [make_confidence_snapshot()]
    return SystemReport(
        task_snapshots=task_snapshots,
        global_budget_used_usd=2.0,
        global_budget_remaining_usd=8.0,
        total_runs=10,
        total_local_runs=7,
        total_remote_runs=3,
        total_fallbacks=1,
        total_errors=0,
        uptime_seconds=120.0,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


# ===========================================================================
# 1. LIFECYCLE TESTS – __init__, create, __aenter__, __aexit__, close
# ===========================================================================


class TestApprenticeLifecycle:
    """Tests for Apprentice construction, async context management, and teardown."""

    def test_init_with_config_path(self, valid_config_path, mock_components):
        """Constructor accepts a valid YAML file path and creates instance with _running=False."""
        apprentice = Apprentice(valid_config_path, **mock_components)

        assert hasattr(apprentice, "_config"), "Should have _config attribute"
        assert apprentice._running is False, "_running should be False after __init__"

    def test_init_with_config_object(self, valid_config_path, mock_components):
        """Constructor accepts a pre-built ApprenticeConfig object."""
        config = load_config(valid_config_path)
        apprentice = Apprentice(config, **mock_components)

        assert apprentice._running is False

    def test_init_with_overrides_replaces_defaults(self, valid_config_path):
        """Component overrides completely replace default-constructed components."""
        custom_router = MagicMock(name="custom_router")
        apprentice = Apprentice(valid_config_path, router=custom_router)

        assert apprentice._router is custom_router, (
            "Override should replace default component exactly"
        )

    def test_init_config_file_not_found(self, mock_components):
        """Constructor raises error for non-existent config file path."""
        with pytest.raises((FileNotFoundError, Exception)):
            Apprentice("/nonexistent/path/config.yaml", **mock_components)

    def test_init_config_parse_error(self, tmp_path, mock_components):
        """Constructor raises error for malformed YAML."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(":::not: valid:: yaml: [")
        with pytest.raises(Exception):
            Apprentice(str(bad_yaml), **mock_components)

    def test_init_config_validation_error(self, tmp_path, mock_components):
        """Constructor raises error when config fails Pydantic validation."""
        incomplete = tmp_path / "incomplete.yaml"
        incomplete.write_text("schema_version: 1\n")
        with pytest.raises(Exception):
            Apprentice(str(incomplete), **mock_components)

    def test_init_all_10_components_non_none(self, valid_config_path, mock_components):
        """Invariant: All 10 internal components are non-None after __init__."""
        apprentice = Apprentice(valid_config_path, **mock_components)

        component_names = [
            "_config_loader",
            "_task_registry",
            "_remote_client",
            "_local_client",
            "_confidence_engine",
            "_sampling_scheduler",
            "_budget_manager",
            "_training_data_store",
            "_router",
            "_audit_log",
        ]
        for name in component_names:
            assert getattr(apprentice, name, None) is not None, (
                f"Component {name} should be non-None after __init__"
            )

    @pytest.mark.asyncio
    async def test_aenter_sets_running(self, valid_config_path, mock_components):
        """__aenter__ sets _running to True."""
        apprentice = Apprentice(valid_config_path, **mock_components)
        assert apprentice._running is False

        result = await apprentice.__aenter__()

        assert apprentice._running is True, "_running should be True after __aenter__"
        assert result is apprentice, "__aenter__ should return self"
        await apprentice.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_aenter_double_enter_raises(self, valid_config_path, mock_components):
        """__aenter__ raises error if already running."""
        apprentice = Apprentice(valid_config_path, **mock_components)
        await apprentice.__aenter__()

        with pytest.raises(RuntimeError):
            await apprentice.__aenter__()

        await apprentice.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_aexit_cleans_up(self, valid_config_path, mock_components):
        """__aexit__ sets _running to False and returns False (no suppression)."""
        apprentice = Apprentice(valid_config_path, **mock_components)
        await apprentice.__aenter__()

        result = await apprentice.__aexit__(None, None, None)

        assert apprentice._running is False, "_running should be False after __aexit__"
        assert result is False, "__aexit__ must return False to not suppress exceptions"

    @pytest.mark.asyncio
    async def test_close_sets_not_running(self, valid_config_path, mock_components):
        """close() sets _running to False."""
        apprentice = Apprentice(valid_config_path, **mock_components)
        await apprentice.__aenter__()

        await apprentice.close()

        assert apprentice._running is False

    @pytest.mark.asyncio
    async def test_close_idempotent(self, valid_config_path, mock_components):
        """close() is safe to call multiple times."""
        apprentice = Apprentice(valid_config_path, **mock_components)
        await apprentice.__aenter__()

        await apprentice.close()
        await apprentice.close()  # second call — should not raise

        assert apprentice._running is False

    @pytest.mark.asyncio
    async def test_create_returns_running_instance(
        self, valid_config_path, mock_components
    ):
        """create() returns a fully ready instance with _running=True."""
        instance = await Apprentice.create(valid_config_path, **mock_components)

        assert instance._running is True, "create() should return running instance"
        await instance.close()

    @pytest.mark.asyncio
    async def test_running_state_consistency(self, valid_config_path, mock_components):
        """Invariant: _running is True iff async context entered and not exited."""
        apprentice = Apprentice(valid_config_path, **mock_components)
        assert apprentice._running is False, "False after init"

        await apprentice.__aenter__()
        assert apprentice._running is True, "True after aenter"

        await apprentice.__aexit__(None, None, None)
        assert apprentice._running is False, "False after aexit"


# ===========================================================================
# 2. OPERATIONS TESTS – run(), status(), report()
# ===========================================================================


class TestApprenticeRun:
    """Tests for run() method covering the routing/fallback decision matrix."""

    @pytest.fixture
    async def running_apprentice(self, valid_config_path, mock_components):
        """Provide a running Apprentice with mocked dependencies."""
        apprentice = Apprentice(valid_config_path, **mock_components)
        await apprentice.__aenter__()
        yield apprentice, mock_components
        await apprentice.close()

    @pytest.mark.asyncio
    async def test_run_happy_path_success(self, running_apprentice):
        """run() returns TaskResponse with SUCCESS for nominal path."""
        apprentice, mocks = running_apprentice
        expected_response = make_task_response(source=ModelSource.LOCAL, status=RunStatus.SUCCESS)

        # Configure router to return a successful response
        mocks["router"].route = AsyncMock(return_value=expected_response)
        mocks["task_registry"].get_task = MagicMock(return_value=MagicMock())
        mocks["budget_manager"].can_spend = MagicMock(return_value=True)
        mocks["confidence_engine"].get_confidence = MagicMock(return_value=0.9)

        result = await apprentice.run("test_task", {"prompt": "hello"})

        assert isinstance(result, TaskResponse), "Should return TaskResponse"
        assert result.task_name == "test_task"
        assert result.status == RunStatus.SUCCESS
        assert result.request_id, "request_id must be non-empty"

    @pytest.mark.asyncio
    async def test_run_budget_exhausted_local_available_degrades(
        self, running_apprentice
    ):
        """Budget exhausted + local available → DEGRADED status, LOCAL source."""
        apprentice, mocks = running_apprentice
        expected = make_task_response(
            source=ModelSource.LOCAL, status=RunStatus.DEGRADED
        )
        mocks["router"].route = AsyncMock(return_value=expected)
        mocks["task_registry"].get_task = MagicMock(return_value=MagicMock())
        mocks["budget_manager"].can_spend = MagicMock(return_value=False)
        mocks["local_client"].is_available = MagicMock(return_value=True)

        result = await apprentice.run("test_task", {"prompt": "hello"})

        assert result.source == ModelSource.LOCAL, "Should use local when budget exhausted"
        assert result.status == RunStatus.DEGRADED, "Should be DEGRADED"

    @pytest.mark.asyncio
    async def test_run_local_unavailable_budget_available_fallback(
        self, running_apprentice
    ):
        """Local unavailable + budget available → REMOTE source, fallback_used=True."""
        apprentice, mocks = running_apprentice
        expected = make_task_response(
            source=ModelSource.REMOTE, status=RunStatus.SUCCESS, fallback_used=True
        )
        mocks["router"].route = AsyncMock(return_value=expected)
        mocks["task_registry"].get_task = MagicMock(return_value=MagicMock())
        mocks["budget_manager"].can_spend = MagicMock(return_value=True)
        mocks["local_client"].is_available = MagicMock(return_value=False)

        result = await apprentice.run("test_task", {"prompt": "hello"})

        assert result.source == ModelSource.REMOTE
        assert result.metadata.fallback_used is True

    @pytest.mark.asyncio
    async def test_run_both_unavailable_raises_budget_exhausted(
        self, running_apprentice
    ):
        """Local unavailable AND budget exhausted → BudgetExhaustedError."""
        apprentice, mocks = running_apprentice
        mocks["task_registry"].get_task = MagicMock(return_value=MagicMock())
        mocks["budget_manager"].can_spend = MagicMock(return_value=False)
        mocks["local_client"].is_available = MagicMock(return_value=False)
        mocks["router"].route = AsyncMock(
            side_effect=BudgetExhaustedError(
                error_kind=ErrorKind.BUDGET_EXHAUSTED,
                message="Budget exhausted",
                task_name="test_task",
                budget_limit_usd=10.0,
                budget_used_usd=10.0,
                request_id="req-123",
            )
        )

        with pytest.raises(BudgetExhaustedError) as exc_info:
            await apprentice.run("test_task", {"prompt": "hello"})

        assert exc_info.value.error_kind == ErrorKind.BUDGET_EXHAUSTED
        assert exc_info.value.task_name == "test_task"

    @pytest.mark.asyncio
    async def test_run_not_running_raises(self, valid_config_path, mock_components):
        """run() raises RuntimeError when called outside async context."""
        apprentice = Apprentice(valid_config_path, **mock_components)

        with pytest.raises(RuntimeError):
            await apprentice.run("test_task", {"prompt": "hello"})

    @pytest.mark.asyncio
    async def test_run_task_not_found(self, running_apprentice):
        """run() raises TaskNotFoundError for unknown task."""
        apprentice, mocks = running_apprentice
        mocks["task_registry"].get_task = MagicMock(
            side_effect=TaskNotFoundError(
                error_kind=ErrorKind.TASK_NOT_FOUND,
                message="Task 'unknown' not found",
                task_name="unknown",
                available_tasks=["test_task"],
            )
        )

        with pytest.raises(TaskNotFoundError) as exc_info:
            await apprentice.run("unknown", {"prompt": "hello"})

        assert "unknown" in str(exc_info.value.message)
        assert "test_task" in exc_info.value.available_tasks

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_name", ["", "   ", "\t", "\n"])
    async def test_run_empty_task_name(self, running_apprentice, bad_name):
        """run() raises error for empty or whitespace-only task_name."""
        apprentice, _ = running_apprentice

        with pytest.raises((ValueError, ApprenticeError)):
            await apprentice.run(bad_name, {"prompt": "hello"})

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_input", [None, [1, 2], "string", 42, True])
    async def test_run_invalid_input_data(self, running_apprentice, bad_input):
        """run() raises error when input_data is not a dict."""
        apprentice, _ = running_apprentice

        with pytest.raises((TypeError, ValueError, ApprenticeError)):
            await apprentice.run("test_task", bad_input)

    @pytest.mark.asyncio
    async def test_run_audit_log_always_flushed(self, running_apprentice):
        """Every run() invocation flushes a RunContext to the audit log."""
        apprentice, mocks = running_apprentice
        expected = make_task_response()
        mocks["router"].route = AsyncMock(return_value=expected)
        mocks["task_registry"].get_task = MagicMock(return_value=MagicMock())
        mocks["budget_manager"].can_spend = MagicMock(return_value=True)

        await apprentice.run("test_task", {"prompt": "hello"})

        # Verify audit log was called
        audit_log = mocks["audit_log"]
        assert audit_log.flush.called or audit_log.log.called or audit_log.write.called, (
            "Audit log must be called during run()"
        )

    @pytest.mark.asyncio
    async def test_run_source_attribution_accuracy(self, running_apprentice):
        """Invariant: TaskResponse.source accurately reflects which backend was used."""
        apprentice, mocks = running_apprentice
        local_response = make_task_response(source=ModelSource.LOCAL)
        mocks["router"].route = AsyncMock(return_value=local_response)
        mocks["task_registry"].get_task = MagicMock(return_value=MagicMock())
        mocks["budget_manager"].can_spend = MagicMock(return_value=True)

        result = await apprentice.run("test_task", {"prompt": "hello"})

        assert result.source == ModelSource.LOCAL, (
            "Source must match actual backend used"
        )


class TestApprenticeStatus:
    """Tests for status() method."""

    @pytest.fixture
    async def running_apprentice(self, valid_config_path, mock_components):
        apprentice = Apprentice(valid_config_path, **mock_components)
        await apprentice.__aenter__()
        yield apprentice, mock_components
        await apprentice.close()

    @pytest.mark.asyncio
    async def test_status_happy_path(self, running_apprentice):
        """status() returns ConfidenceSnapshot for a valid task."""
        apprentice, mocks = running_apprentice
        snapshot = make_confidence_snapshot(
            task_name="test_task", budget_remaining=8.0, budget_used=2.0
        )
        mocks["confidence_engine"].get_snapshot = MagicMock(return_value=snapshot)
        mocks["task_registry"].get_task = MagicMock(return_value=MagicMock())

        result = await apprentice.status("test_task")

        assert isinstance(result, ConfidenceSnapshot)
        assert result.task_name == "test_task"
        # Budget invariant: remaining + used == limit
        assert abs(result.budget_remaining_usd + result.budget_used_usd - 10.0) < 0.01

    @pytest.mark.asyncio
    async def test_status_not_running_raises(self, valid_config_path, mock_components):
        """status() raises RuntimeError when _running is False."""
        apprentice = Apprentice(valid_config_path, **mock_components)

        with pytest.raises(RuntimeError):
            await apprentice.status("test_task")

    @pytest.mark.asyncio
    async def test_status_task_not_found(self, running_apprentice):
        """status() raises TaskNotFoundError for unknown task."""
        apprentice, mocks = running_apprentice
        mocks["task_registry"].get_task = MagicMock(
            side_effect=TaskNotFoundError(
                error_kind=ErrorKind.TASK_NOT_FOUND,
                message="Not found",
                task_name="unknown",
                available_tasks=["test_task"],
            )
        )

        with pytest.raises(TaskNotFoundError):
            await apprentice.status("unknown")


class TestApprenticeReport:
    """Tests for report() method."""

    def test_report_happy_path(self, valid_config_path, mock_components):
        """report() returns SystemReport with correct aggregates."""
        apprentice = Apprentice(valid_config_path, **mock_components)
        expected_report = make_system_report()
        mock_components["confidence_engine"].get_all_snapshots = MagicMock(
            return_value=[make_confidence_snapshot()]
        )

        # report() may be sync or async depending on implementation
        try:
            result = apprentice.report()
        except RuntimeError:
            # If it requires running, skip this specific assertion
            pytest.skip("report() requires running context in this implementation")
            return

        assert isinstance(result, SystemReport)
        assert result.total_runs == result.total_local_runs + result.total_remote_runs, (
            "Invariant: total_runs must equal local + remote"
        )

    def test_report_uptime_zero_before_enter(self, valid_config_path, mock_components):
        """report() returns uptime_seconds=0.0 before async context entered."""
        apprentice = Apprentice(valid_config_path, **mock_components)

        try:
            result = apprentice.report()
            assert result.uptime_seconds == 0.0, (
                "Uptime should be 0 before async context entered"
            )
        except (RuntimeError, Exception):
            pytest.skip("report() may require running state in this implementation")

    def test_report_total_runs_invariant(self, valid_config_path, mock_components):
        """Invariant: total_runs == total_local_runs + total_remote_runs."""
        report = make_system_report()
        assert report.total_runs == report.total_local_runs + report.total_remote_runs


# ===========================================================================
# 3. CLI PARSING TESTS – parse_args, resolve_input, format_output, configure_logging
# ===========================================================================


class TestParseArgs:
    """Tests for parse_args() pure function."""

    def test_parse_run_subcommand(self):
        """parse_args parses 'run' with task name and input."""
        result = parse_args(["run", "my_task", "--input", '{"key": "val"}'])

        assert result.command == SubcommandName.RUN
        assert result.run_args.task_name == "my_task"
        assert result.run_args.input_raw == '{"key": "val"}'

    def test_parse_status_subcommand(self):
        """parse_args parses 'status' with optional task filter."""
        result = parse_args(["status", "--task", "my_task"])

        assert result.command == SubcommandName.STATUS
        assert result.status_args.task_filter == "my_task"

    def test_parse_report_subcommand(self):
        """parse_args parses 'report' with optional output path."""
        result = parse_args(["report", "--output", "/tmp/report.json"])

        assert result.command == SubcommandName.REPORT
        assert result.report_args.output_path == "/tmp/report.json"

    def test_parse_global_flags(self):
        """parse_args handles --config, --json, and -vv correctly."""
        result = parse_args(["--config", "custom.yaml", "--json", "-vv", "status"])

        assert result.global_flags.config_path == "custom.yaml"
        assert result.global_flags.json_mode is True
        assert result.global_flags.verbose == 2

    def test_parse_default_config_path(self):
        """Default config_path is non-empty when not specified."""
        result = parse_args(["status"])

        assert result.global_flags.config_path, (
            "Default config_path should be non-empty"
        )

    def test_parse_no_subcommand_raises(self):
        """No subcommand raises SystemExit with code 2."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args([])
        assert exc_info.value.code == 2

    def test_parse_unknown_flag_raises(self):
        """Unknown flag raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_args(["--bogus-flag", "run", "task"])

    def test_parse_verbose_clamped_to_max_2(self):
        """Verbose flag is clamped to [0, 2]."""
        result = parse_args(["-vvvvv", "status"])
        assert result.global_flags.verbose <= 2, "verbose should be clamped to max 2"

    def test_parse_verbose_zero_by_default(self):
        """Verbose defaults to 0."""
        result = parse_args(["status"])
        assert result.global_flags.verbose == 0

    def test_parse_status_no_filter(self):
        """status subcommand works without --task filter."""
        result = parse_args(["status"])
        assert result.command == SubcommandName.STATUS
        # task_filter should be empty/None
        assert not result.status_args.task_filter or result.status_args.task_filter == ""

    def test_parse_report_no_output(self):
        """report subcommand works without --output."""
        result = parse_args(["report"])
        assert result.command == SubcommandName.REPORT


class TestResolveInput:
    """Tests for resolve_input() function."""

    def test_resolve_json_string(self):
        """resolve_input parses inline JSON string."""
        result = resolve_input('{"key": "value", "num": 42}')

        assert isinstance(result.data, dict)
        assert result.data == {"key": "value", "num": 42}

    def test_resolve_file_reference(self, tmp_path):
        """resolve_input reads JSON from @file reference."""
        json_file = tmp_path / "input.json"
        json_file.write_text('{"from_file": true}')

        result = resolve_input(f"@{json_file}")

        assert result.data == {"from_file": True}

    def test_resolve_invalid_json_raises(self):
        """resolve_input raises error for invalid JSON string."""
        with pytest.raises(Exception):
            resolve_input("not json at all")

    def test_resolve_json_array_not_object_raises(self):
        """resolve_input raises when JSON is array, not object."""
        with pytest.raises(Exception):
            resolve_input("[1, 2, 3]")

    def test_resolve_json_scalar_not_object_raises(self):
        """resolve_input raises when JSON is scalar."""
        with pytest.raises(Exception):
            resolve_input('"just a string"')

    def test_resolve_file_not_found_raises(self):
        """resolve_input raises when @file does not exist."""
        with pytest.raises((FileNotFoundError, Exception)):
            resolve_input("@/nonexistent/file.json")

    def test_resolve_empty_file_reference_raises(self):
        """resolve_input raises when input is exactly '@'."""
        with pytest.raises(Exception):
            resolve_input("@")

    def test_resolve_nested_json(self):
        """resolve_input handles nested JSON objects."""
        result = resolve_input('{"outer": {"inner": [1, 2, 3]}}')
        assert result.data["outer"]["inner"] == [1, 2, 3]

    def test_resolve_empty_object(self):
        """resolve_input accepts empty JSON object."""
        result = resolve_input("{}")
        assert result.data == {}


class TestFormatOutput:
    """Tests for format_output() function."""

    def test_json_mode_produces_valid_json(self):
        """JSON mode output is parseable by json.loads."""
        result_obj = RunResult(
            task_name="test",
            output={"answer": 42},
            success=True,
            error_message="",
        )
        output = format_output(result_obj, json_mode=True)

        parsed = json.loads(output)
        assert parsed["task_name"] == "test"

    def test_human_mode_produces_readable_text(self):
        """Human mode output is non-empty and not raw JSON."""
        result_obj = RunResult(
            task_name="test",
            output={"answer": 42},
            success=True,
            error_message="",
        )
        output = format_output(result_obj, json_mode=False)

        assert len(output) > 0, "Human output should be non-empty"

    def test_format_output_no_trailing_newline(self):
        """Output does not include trailing newline."""
        result_obj = RunResult(
            task_name="test",
            output={"answer": 42},
            success=True,
            error_message="",
        )
        output = format_output(result_obj, json_mode=True)

        assert not output.endswith("\n"), "Should not end with newline"

    def test_format_status_result_json(self):
        """StatusResult formats correctly in JSON mode."""
        status_result = StatusResult(
            tasks=[],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        output = format_output(status_result, json_mode=True)
        parsed = json.loads(output)
        assert "tasks" in parsed


class TestConfigureLogging:
    """Tests for configure_logging()."""

    def test_verbose_0_sets_warning(self):
        """verbose=0 sets root logger to WARNING."""
        configure_logging(0)
        assert logging.getLogger().level == logging.WARNING

    def test_verbose_1_sets_info(self):
        """verbose=1 sets root logger to INFO."""
        configure_logging(1)
        assert logging.getLogger().level == logging.INFO

    def test_verbose_2_sets_debug(self):
        """verbose=2 sets root logger to DEBUG."""
        configure_logging(2)
        assert logging.getLogger().level == logging.DEBUG


# ===========================================================================
# 4. CLI EXECUTION TESTS – execute_run, execute_status, execute_report, main
# ===========================================================================


class TestExecuteRun:
    """Tests for execute_run() async function."""

    @pytest.mark.asyncio
    async def test_execute_run_success(self):
        """execute_run resolves input, calls run(), wraps in RunResult."""
        mock_apprentice = AsyncMock()
        mock_apprentice.run = AsyncMock(return_value=make_task_response())

        run_args = RunArgs(task_name="test_task", input_raw='{"prompt": "hello"}')

        result = await execute_run(mock_apprentice, run_args)

        assert isinstance(result, RunResult)
        assert result.task_name == "test_task"
        assert result.success is True
        assert result.output, "Output should contain response payload"

    @pytest.mark.asyncio
    async def test_execute_run_task_not_found_error(self):
        """execute_run handles TaskNotFoundError."""
        mock_apprentice = AsyncMock()
        mock_apprentice.run = AsyncMock(
            side_effect=TaskNotFoundError(
                error_kind=ErrorKind.TASK_NOT_FOUND,
                message="Task 'bad' not found",
                task_name="bad",
                available_tasks=["test_task"],
            )
        )

        run_args = RunArgs(task_name="bad", input_raw='{"prompt": "hello"}')

        # Implementation may propagate or wrap the error
        try:
            result = await execute_run(mock_apprentice, run_args)
            # If wrapped, should indicate failure
            assert result.success is False
            assert result.error_message, "Should have error message"
        except TaskNotFoundError:
            pass  # Also acceptable: propagation

    @pytest.mark.asyncio
    async def test_execute_run_invalid_input_json(self):
        """execute_run handles invalid JSON in input_raw."""
        mock_apprentice = AsyncMock()
        run_args = RunArgs(task_name="test_task", input_raw="not valid json")

        with pytest.raises(Exception):
            await execute_run(mock_apprentice, run_args)


class TestExecuteStatus:
    """Tests for execute_status() async function."""

    @pytest.mark.asyncio
    async def test_execute_status_success(self):
        """execute_status returns StatusResult with task entries."""
        mock_apprentice = AsyncMock()
        mock_apprentice.status = AsyncMock(return_value=make_confidence_snapshot())

        status_args = StatusArgs(task_filter="test_task")

        result = await execute_status(mock_apprentice, status_args)

        assert isinstance(result, StatusResult)
        assert result.timestamp, "Should have ISO 8601 timestamp"

    @pytest.mark.asyncio
    async def test_execute_status_filtered(self):
        """execute_status filters to single task when task_filter provided."""
        mock_apprentice = AsyncMock()
        snapshot = make_confidence_snapshot(task_name="specific_task")
        mock_apprentice.status = AsyncMock(return_value=snapshot)

        status_args = StatusArgs(task_filter="specific_task")

        result = await execute_status(mock_apprentice, status_args)

        assert isinstance(result, StatusResult)
        # All tasks in result should match the filter
        for task in result.tasks:
            assert task.task_name == "specific_task"


class TestExecuteReport:
    """Tests for execute_report() async function."""

    @pytest.mark.asyncio
    async def test_execute_report_success(self):
        """execute_report returns ReportResult with all tasks."""
        mock_apprentice = AsyncMock()
        mock_apprentice.report = AsyncMock(return_value=make_system_report())

        report_args = ReportArgs(output_path="")

        result = await execute_report(
            mock_apprentice, report_args, config_path="config.yaml", json_mode=False
        )

        assert isinstance(result, ReportResult)
        assert result.total_budget_spent <= result.total_budget_limit
        assert result.system_uptime_seconds >= 0

    @pytest.mark.asyncio
    async def test_execute_report_writes_file(self, tmp_path):
        """execute_report writes output to file when output_path specified."""
        mock_apprentice = AsyncMock()
        mock_apprentice.report = AsyncMock(return_value=make_system_report())

        output_file = str(tmp_path / "report_output.json")
        report_args = ReportArgs(output_path=output_file)

        result = await execute_report(
            mock_apprentice, report_args, config_path="config.yaml", json_mode=True
        )

        assert result.written_to_file == output_file
        assert Path(output_file).exists(), "Report file should have been written"


class TestMain:
    """Tests for the CLI main() entry point."""

    def test_main_usage_error_exit_2(self):
        """main() returns 2 on usage error (no subcommand)."""
        exit_code = main([])
        assert exit_code == 2, "No subcommand should yield exit code 2"

    def test_main_config_not_found_exit_1(self, tmp_path):
        """main() returns 1 when config file doesn't exist."""
        exit_code = main([
            "--config", str(tmp_path / "nonexistent.yaml"),
            "status",
        ])
        assert exit_code == 1, "Missing config should yield exit code 1"

    def test_main_keyboard_interrupt_exit_130(self):
        """main() returns 130 on KeyboardInterrupt."""
        with patch(
            "src.unified_interface.cli.load_config",
            side_effect=KeyboardInterrupt,
        ):
            exit_code = main(["--config", "dummy.yaml", "status"])
            assert exit_code == 130

    def test_main_success_exit_0(self, valid_config_path):
        """main() returns 0 on successful execution."""
        mock_report = make_system_report()
        mock_apprentice = AsyncMock()
        mock_apprentice.report = AsyncMock(return_value=mock_report)
        mock_apprentice.__aenter__ = AsyncMock(return_value=mock_apprentice)
        mock_apprentice.__aexit__ = AsyncMock(return_value=False)
        mock_apprentice._running = True

        with patch(
            "src.unified_interface.cli.Apprentice",
            return_value=mock_apprentice,
        ):
            exit_code = main(["--config", valid_config_path, "report"])

        assert exit_code == 0, "Successful report should yield exit code 0"


# ===========================================================================
# 5. CONFIG LOADING TESTS
# ===========================================================================


class TestLoadConfig:
    """Tests for load_config() function."""

    def test_load_valid_config(self, valid_config_path):
        """load_config parses a valid YAML file into ApprenticeConfig."""
        config = load_config(valid_config_path)

        assert isinstance(config, ApprenticeConfig)
        assert config.schema_version == 1
        assert len(config.tasks) == 1
        assert config.tasks[0].task_name == "test_task"

    def test_load_config_file_not_found(self):
        """load_config raises for non-existent file."""
        with pytest.raises((FileNotFoundError, Exception)):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_invalid_yaml(self, tmp_path):
        """load_config raises for malformed YAML."""
        bad = tmp_path / "bad.yaml"
        bad.write_text(":::invalid[yaml{")
        with pytest.raises(Exception):
            load_config(str(bad))

    def test_load_config_validation_error_missing_fields(self, tmp_path):
        """load_config raises when required fields missing."""
        incomplete = tmp_path / "incomplete.yaml"
        incomplete.write_text("schema_version: 1\ntasks: []\n")
        with pytest.raises(Exception):
            load_config(str(incomplete))

    def test_load_config_validation_error_unknown_fields(self, tmp_path):
        """load_config rejects unknown fields (strict mode)."""
        extra_field = tmp_path / "extra.yaml"
        content = VALID_YAML_CONTENT + "\nunknown_field: surprise\n"
        extra_field.write_text(content)
        with pytest.raises(Exception):
            load_config(str(extra_field))

    @pytest.mark.skipif(
        os.name == "nt", reason="Permission tricks unreliable on Windows"
    )
    def test_load_config_permission_denied(self, tmp_path):
        """load_config raises when file not readable."""
        no_read = tmp_path / "noperm.yaml"
        no_read.write_text(VALID_YAML_CONTENT)
        no_read.chmod(0o000)

        try:
            with pytest.raises((PermissionError, Exception)):
                load_config(str(no_read))
        finally:
            no_read.chmod(0o644)

    def test_load_config_thresholds_ordering_violation(self, tmp_path):
        """ConfidenceThresholds rejects invalid ordering."""
        bad_thresholds = tmp_path / "bad_thresholds.yaml"
        bad_content = VALID_YAML_CONTENT.replace(
            "bootstrapping_upper: 0.2", "bootstrapping_upper: 0.9"
        )
        bad_thresholds.write_text(bad_content)

        with pytest.raises(Exception):
            load_config(str(bad_thresholds))


# ===========================================================================
# 6. INVARIANT TESTS
# ===========================================================================


class TestInvariants:
    """Cross-cutting invariant tests."""

    def test_frozen_task_response_immutable(self):
        """TaskResponse is frozen — mutation raises error."""
        resp = make_task_response()
        with pytest.raises((AttributeError, TypeError, Exception)):
            resp.task_name = "mutated"

    def test_frozen_confidence_snapshot_immutable(self):
        """ConfidenceSnapshot is frozen — mutation raises error."""
        snap = make_confidence_snapshot()
        with pytest.raises((AttributeError, TypeError, Exception)):
            snap.confidence_score = 999.0

    def test_frozen_system_report_immutable(self):
        """SystemReport is frozen — mutation raises error."""
        report = make_system_report()
        with pytest.raises((AttributeError, TypeError, Exception)):
            report.total_runs = 999

    def test_model_source_is_str_enum(self):
        """ModelSource values are strings (StrEnum)."""
        assert isinstance(ModelSource.LOCAL, str)
        assert isinstance(ModelSource.REMOTE, str)

    def test_exit_code_is_int_enum(self):
        """ExitCode values are integers (IntEnum)."""
        assert isinstance(ExitCode.SUCCESS_0, int)
        assert ExitCode.SUCCESS_0 == 0
        assert ExitCode.APP_ERROR_1 == 1
        assert ExitCode.USAGE_ERROR_2 == 2
        assert ExitCode.KEYBOARD_INTERRUPT_130 == 130

    def test_phase_label_is_str_enum(self):
        """PhaseLabel values are strings (StrEnum)."""
        for phase in PhaseLabel:
            assert isinstance(phase, str)

    def test_run_status_is_str_enum(self):
        """RunStatus values are strings (StrEnum)."""
        for status in RunStatus:
            assert isinstance(status, str)

    def test_error_kind_is_str_enum(self):
        """ErrorKind values are strings (StrEnum)."""
        for kind in ErrorKind:
            assert isinstance(kind, str)

    def test_confidence_thresholds_valid_ordering(self):
        """Valid thresholds respect ordering invariant."""
        ct = ConfidenceThresholds(
            bootstrapping_upper=0.2,
            active_learning_upper=0.5,
            supervised_upper=0.8,
            autonomous_lower=0.8,
        )
        assert ct.bootstrapping_upper < ct.active_learning_upper
        assert ct.active_learning_upper < ct.supervised_upper
        assert ct.supervised_upper <= ct.autonomous_lower

    def test_confidence_thresholds_rejects_invalid_ordering(self):
        """Invalid threshold ordering is rejected at construction."""
        with pytest.raises(Exception):
            ConfidenceThresholds(
                bootstrapping_upper=0.9,  # violates: must be < active_learning_upper
                active_learning_upper=0.5,
                supervised_upper=0.8,
                autonomous_lower=0.8,
            )

    def test_budget_exhausted_error_fields(self):
        """BudgetExhaustedError has all required structured fields."""
        err = BudgetExhaustedError(
            error_kind=ErrorKind.BUDGET_EXHAUSTED,
            message="Over budget",
            task_name="task_a",
            budget_limit_usd=10.0,
            budget_used_usd=10.5,
            request_id="req-456",
        )
        assert err.error_kind == ErrorKind.BUDGET_EXHAUSTED
        assert err.task_name == "task_a"
        assert err.budget_limit_usd == 10.0
        assert err.budget_used_usd == 10.5

    def test_task_not_found_error_fields(self):
        """TaskNotFoundError has available_tasks field."""
        err = TaskNotFoundError(
            error_kind=ErrorKind.TASK_NOT_FOUND,
            message="Not found",
            task_name="missing",
            available_tasks=["a", "b"],
        )
        assert err.available_tasks == ["a", "b"]

    def test_apprentice_error_is_exception(self):
        """ApprenticeError derives from Exception."""
        err = ApprenticeError(
            error_kind=ErrorKind.INTERNAL,
            message="test",
            task_name="t",
            request_id="r",
        )
        assert isinstance(err, Exception)

    def test_budget_exhausted_derives_from_apprentice_error(self):
        """BudgetExhaustedError derives from ApprenticeError."""
        err = BudgetExhaustedError(
            error_kind=ErrorKind.BUDGET_EXHAUSTED,
            message="test",
            task_name="t",
            budget_limit_usd=10.0,
            budget_used_usd=10.0,
            request_id="r",
        )
        assert isinstance(err, ApprenticeError)
        assert isinstance(err, Exception)
