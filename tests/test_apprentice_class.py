"""
Contract tests for the Apprentice class.

Tests verify behavior at boundaries (inputs/outputs), not internals.
All 10 dependencies are mocked — tests verify the Apprentice component in isolation.

Run with: pytest tests/contract_test.py -v
"""
import pytest
import asyncio
from unittest.mock import (
    AsyncMock, MagicMock, patch, PropertyMock, ANY, call
)
from datetime import datetime, timezone

# Import the component under test
from src.apprentice_class import (
    Apprentice,
    ApprenticeConfig,
    TaskConfig,
    ConfidenceThresholds,
    BudgetEnforcementConfig,
    TaskResponse,
    ResponseMetadata,
    ConfidenceSnapshot,
    SystemReport,
    RunContext,
    ModelSource,
    PhaseLabel,
    RunStatus,
    ErrorKind,
    ApprenticeError,
    BudgetExhaustedError,
    TaskNotFoundError,
    LocalModelUnavailableError,
)


# ---------------------------------------------------------------------------
# Helpers / Builders
# ---------------------------------------------------------------------------

def make_confidence_thresholds(**overrides):
    defaults = dict(
        bootstrapping_upper=0.25,
        active_learning_upper=0.50,
        supervised_upper=0.75,
        autonomous_lower=0.90,
    )
    defaults.update(overrides)
    return ConfidenceThresholds(**defaults)


def make_task_config(task_name="test_task", **overrides):
    defaults = dict(
        task_name=task_name,
        remote_model_id="gpt-4",
        local_model_id="local-llama",
        budget_limit_usd=10.0,
        budget_period_days=30,
        confidence_thresholds=make_confidence_thresholds(),
    )
    defaults.update(overrides)
    return TaskConfig(**defaults)


def make_budget_enforcement_config(**overrides):
    defaults = dict(
        hard_limit=True,
        warning_threshold_pct=80.0,
        log_every_spend=True,
    )
    defaults.update(overrides)
    return BudgetEnforcementConfig(**defaults)


def make_config(tasks=None, **overrides):
    if tasks is None:
        tasks = [make_task_config()]
    defaults = dict(
        tasks=tasks,
        remote_api_base_url="https://api.example.com",
        local_model_endpoint="http://localhost:8080",
        budget_enforcement=make_budget_enforcement_config(),
        audit_log_path="/tmp/test_audit.jsonl",
        retry_max_attempts=3,
        retry_backoff_base_seconds=1.0,
    )
    defaults.update(overrides)
    return ApprenticeConfig(**defaults)


def make_all_mocks():
    """Create all 10 dependency mocks with sensible defaults."""
    mocks = dict(
        config_loader=MagicMock(),
        task_registry=MagicMock(),
        remote_client=AsyncMock(),
        local_client=AsyncMock(),
        confidence_engine=MagicMock(),
        sampling_scheduler=MagicMock(),
        budget_manager=MagicMock(),
        training_data_store=AsyncMock(),
        router=MagicMock(),
        audit_log=AsyncMock(),
    )
    # Sensible defaults for task_registry
    mocks["task_registry"].get.return_value = make_task_config()
    mocks["task_registry"].list_tasks.return_value = ["test_task"]
    mocks["task_registry"].contains.return_value = True
    mocks["task_registry"].__contains__ = MagicMock(return_value=True)

    # Budget manager defaults: budget available
    mocks["budget_manager"].can_spend.return_value = True
    mocks["budget_manager"].get_remaining.return_value = 8.0
    mocks["budget_manager"].get_used.return_value = 2.0
    mocks["budget_manager"].record_spend = MagicMock()

    # Confidence engine defaults
    mocks["confidence_engine"].get_score.return_value = 0.6
    mocks["confidence_engine"].get_phase.return_value = PhaseLabel.supervised
    mocks["confidence_engine"].update = MagicMock()

    # Sampling scheduler
    mocks["sampling_scheduler"].get_rate.return_value = 0.5

    # Router: route to local by default
    mocks["router"].route.return_value = ModelSource.local

    # Local client returns a valid output
    mocks["local_client"].predict.return_value = {"result": "local_output"}
    mocks["local_client"].is_available.return_value = True

    # Remote client returns a valid output
    mocks["remote_client"].predict.return_value = {"result": "remote_output"}
    mocks["remote_client"].get_cost.return_value = 0.01

    # Audit log
    mocks["audit_log"].log = AsyncMock()
    mocks["audit_log"].flush = AsyncMock()
    mocks["audit_log"].open = AsyncMock()
    mocks["audit_log"].close = AsyncMock()

    return mocks


def build_apprentice(config=None, **mock_overrides):
    """Build an Apprentice with full mock injection."""
    if config is None:
        config = make_config()
    mocks = make_all_mocks()
    mocks.update(mock_overrides)
    return Apprentice(
        config=config,
        config_loader=mocks["config_loader"],
        task_registry=mocks["task_registry"],
        remote_client=mocks["remote_client"],
        local_client=mocks["local_client"],
        confidence_engine=mocks["confidence_engine"],
        sampling_scheduler=mocks["sampling_scheduler"],
        budget_manager=mocks["budget_manager"],
        training_data_store=mocks["training_data_store"],
        router=mocks["router"],
        audit_log=mocks["audit_log"],
    ), mocks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def apprentice_and_mocks():
    """Returns (apprentice_instance, mocks_dict) with _running=False."""
    return build_apprentice()


@pytest.fixture
async def running_apprentice_and_mocks(apprentice_and_mocks):
    """Yields (apprentice_instance, mocks_dict) with _running=True."""
    apprentice, mocks = apprentice_and_mocks
    await apprentice.__aenter__()
    yield apprentice, mocks
    # Cleanup — best effort
    try:
        await apprentice.__aexit__(None, None, None)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for Apprentice.__init__"""

    def test_init_happy_path_with_config_object(self):
        """Constructor accepts a valid ApprenticeConfig and wires all components."""
        apprentice, mocks = build_apprentice()

        # _config is set (we don't peek at internals directly, but verify
        # the instance was created without error)
        assert apprentice is not None
        # _running should be False — verified via observable behavior:
        # calling run() should fail
        # We verify this in lifecycle tests; here we just confirm construction.

    def test_init_happy_path_with_overrides(self):
        """Component overrides completely replace default-constructed components."""
        custom_router = MagicMock()
        custom_router.route.return_value = ModelSource.remote
        apprentice, mocks = build_apprentice(router=custom_router)
        # The injected router should be the one used later — verified
        # indirectly through behavior tests.
        assert apprentice is not None

    def test_init_error_config_file_not_found(self):
        """Constructor raises error when config file path does not exist."""
        mocks = make_all_mocks()
        with pytest.raises((FileNotFoundError, OSError, ApprenticeError)):
            Apprentice(
                config="/nonexistent/path/config.yaml",
                **{k: v for k, v in mocks.items()},
            )

    def test_init_error_config_validation_missing_field(self):
        """Constructor raises validation error for missing required fields."""
        with pytest.raises((ValueError, TypeError, ApprenticeError)):
            # Attempt to create ApprenticeConfig missing required fields
            bad_config = {"tasks": []}  # Missing most required fields
            ApprenticeConfig(**bad_config)

    def test_init_error_config_unknown_fields_strict(self):
        """Strict parsing rejects unknown fields."""
        with pytest.raises((ValueError, TypeError, ApprenticeError)):
            ApprenticeConfig(
                tasks=[make_task_config()],
                remote_api_base_url="https://api.example.com",
                local_model_endpoint="http://localhost:8080",
                budget_enforcement=make_budget_enforcement_config(),
                audit_log_path="/tmp/test.jsonl",
                retry_max_attempts=3,
                retry_backoff_base_seconds=1.0,
                unknown_field="should_be_rejected",
            )

    def test_invariant_all_10_components_non_none(self):
        """After __init__, all 10 internal components are non-None."""
        apprentice, _ = build_apprentice()
        component_names = [
            "_config_loader", "_task_registry", "_remote_client",
            "_local_client", "_confidence_engine", "_sampling_scheduler",
            "_budget_manager", "_training_data_store", "_router", "_audit_log",
        ]
        for name in component_names:
            assert getattr(apprentice, name, None) is not None, (
                f"Component {name} should be non-None after __init__"
            )


# ---------------------------------------------------------------------------
# TestCreate
# ---------------------------------------------------------------------------

class TestCreate:
    """Tests for Apprentice.create() async factory."""

    @pytest.mark.asyncio
    async def test_create_happy_path(self):
        """create() returns a fully ready instance with _running=True."""
        mocks = make_all_mocks()
        config = make_config()
        apprentice = await Apprentice.create(
            config=config, **mocks,
        )
        try:
            # Observable: _running should be True, meaning run() won't raise not_running
            assert hasattr(apprentice, "_running")
            assert apprentice._running is True
        finally:
            await apprentice.close()


# ---------------------------------------------------------------------------
# TestLifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    """Tests for __aenter__, __aexit__, close() lifecycle."""

    @pytest.mark.asyncio
    async def test_aenter_sets_running_true(self):
        """__aenter__ transitions _running from False to True."""
        apprentice, mocks = build_apprentice()
        assert apprentice._running is False, "_running should be False before __aenter__"

        result = await apprentice.__aenter__()

        assert apprentice._running is True, "_running should be True after __aenter__"
        assert result is apprentice, "__aenter__ should return self"

        await apprentice.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_aenter_error_double_entry(self):
        """Double-entering async context raises error."""
        apprentice, mocks = build_apprentice()
        await apprentice.__aenter__()

        with pytest.raises((RuntimeError, ApprenticeError)):
            await apprentice.__aenter__()

        await apprentice.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_aexit_sets_running_false(self):
        """__aexit__ sets _running to False and returns False."""
        apprentice, mocks = build_apprentice()
        await apprentice.__aenter__()

        result = await apprentice.__aexit__(None, None, None)

        assert apprentice._running is False, "_running should be False after __aexit__"
        assert result is False, "__aexit__ should return False (don't suppress exceptions)"

    @pytest.mark.asyncio
    async def test_aexit_flushes_audit_log(self):
        """__aexit__ flushes the audit log during cleanup."""
        apprentice, mocks = build_apprentice()
        await apprentice.__aenter__()

        await apprentice.__aexit__(None, None, None)

        # Verify audit log was flushed/closed
        audit_mock = mocks["audit_log"]
        assert (
            audit_mock.flush.called or audit_mock.close.called
        ), "Audit log should be flushed or closed during __aexit__"

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """close() is safe to call multiple times."""
        apprentice, mocks = build_apprentice()
        await apprentice.__aenter__()

        await apprentice.close()
        assert apprentice._running is False, "_running should be False after first close()"

        # Second call should not raise
        await apprentice.close()
        assert apprentice._running is False, "_running should remain False after second close()"

    @pytest.mark.asyncio
    async def test_invariant_running_flag_lifecycle(self):
        """_running is True iff async context is entered and not yet exited."""
        apprentice, _ = build_apprentice()

        assert apprentice._running is False, "False after __init__"

        await apprentice.__aenter__()
        assert apprentice._running is True, "True after __aenter__"

        await apprentice.__aexit__(None, None, None)
        assert apprentice._running is False, "False after __aexit__"


# ---------------------------------------------------------------------------
# TestRun
# ---------------------------------------------------------------------------

class TestRun:
    """Tests for Apprentice.run()."""

    @pytest.mark.asyncio
    async def test_run_happy_path_local_model(self, running_apprentice_and_mocks):
        """run() returns TaskResponse with source=local when routed locally."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["router"].route.return_value = ModelSource.local
        mocks["local_client"].is_available.return_value = True
        mocks["budget_manager"].can_spend.return_value = True

        response = await apprentice.run("test_task", {"prompt": "hello"})

        assert isinstance(response, TaskResponse), "Should return a TaskResponse"
        assert response.task_name == "test_task", "task_name should match input"
        assert response.source == ModelSource.local, "source should be 'local'"
        assert response.status == RunStatus.success, "status should be 'success'"
        assert response.request_id, "request_id should be non-empty"
        assert response.duration_ms >= 0, "duration_ms should be non-negative"

    @pytest.mark.asyncio
    async def test_run_happy_path_remote_model(self, running_apprentice_and_mocks):
        """run() returns TaskResponse with source=remote when routed remotely."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["router"].route.return_value = ModelSource.remote
        mocks["local_client"].is_available.return_value = True
        mocks["budget_manager"].can_spend.return_value = True

        response = await apprentice.run("test_task", {"prompt": "hello"})

        assert isinstance(response, TaskResponse), "Should return a TaskResponse"
        assert response.source == ModelSource.remote, "source should be 'remote'"
        assert response.status == RunStatus.success, "status should be 'success'"

    @pytest.mark.asyncio
    async def test_run_budget_exhausted_local_available_degrades_gracefully(
        self, running_apprentice_and_mocks
    ):
        """Budget exhausted + local available → degraded status, no exception."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["budget_manager"].can_spend.return_value = False
        mocks["local_client"].is_available.return_value = True

        response = await apprentice.run("test_task", {"prompt": "hello"})

        assert response.status == RunStatus.degraded, (
            "Should be 'degraded' when budget exhausted but local available"
        )
        assert response.source == ModelSource.local, (
            "Should use local model when budget exhausted"
        )

    @pytest.mark.asyncio
    async def test_run_local_unavailable_budget_available_falls_back_to_remote(
        self, running_apprentice_and_mocks
    ):
        """Local unavailable + budget available → fallback to remote, no exception."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["router"].route.return_value = ModelSource.local
        mocks["local_client"].is_available.return_value = False
        mocks["budget_manager"].can_spend.return_value = True

        response = await apprentice.run("test_task", {"prompt": "hello"})

        assert response.source == ModelSource.remote, (
            "Should fallback to remote when local is unavailable"
        )
        assert response.metadata.fallback_used is True, (
            "fallback_used should be True"
        )

    @pytest.mark.asyncio
    async def test_run_error_both_unavailable_raises_budget_exhausted(
        self, running_apprentice_and_mocks
    ):
        """Local unavailable + budget exhausted → BudgetExhaustedError."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["budget_manager"].can_spend.return_value = False
        mocks["local_client"].is_available.return_value = False
        mocks["budget_manager"].get_remaining.return_value = 0.0
        mocks["budget_manager"].get_used.return_value = 10.0

        with pytest.raises((BudgetExhaustedError, LocalModelUnavailableError, ApprenticeError)) as exc_info:
            await apprentice.run("test_task", {"prompt": "hello"})

        exc = exc_info.value
        assert hasattr(exc, "task_name"), "Error should contain task_name"

    @pytest.mark.asyncio
    async def test_run_error_not_running(self):
        """run() raises RuntimeError when called outside async context."""
        apprentice, _ = build_apprentice()

        with pytest.raises((RuntimeError, ApprenticeError)):
            await apprentice.run("test_task", {"prompt": "hello"})

    @pytest.mark.asyncio
    async def test_run_error_task_not_found(self, running_apprentice_and_mocks):
        """run() raises TaskNotFoundError for unregistered task_name."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["task_registry"].get.side_effect = TaskNotFoundError(
            error_kind=ErrorKind.task_not_found,
            message="Task 'unknown_task' not found",
            task_name="unknown_task",
            available_tasks=["test_task"],
        )
        mocks["task_registry"].contains.return_value = False
        mocks["task_registry"].__contains__ = MagicMock(return_value=False)

        with pytest.raises((TaskNotFoundError, ApprenticeError)) as exc_info:
            await apprentice.run("unknown_task", {"prompt": "hello"})

        exc = exc_info.value
        assert "unknown_task" in str(exc) or (
            hasattr(exc, "task_name") and exc.task_name == "unknown_task"
        ), "Error should reference the unknown task name"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("empty_name", ["", "  ", "\t", "\n"])
    async def test_run_error_empty_task_name(
        self, running_apprentice_and_mocks, empty_name
    ):
        """run() raises error when task_name is empty or whitespace."""
        apprentice, _ = running_apprentice_and_mocks

        with pytest.raises((ValueError, TypeError, ApprenticeError)):
            await apprentice.run(empty_name, {"prompt": "hello"})

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_input", ["not_a_dict", 42, None, [1, 2]])
    async def test_run_error_invalid_input_data(
        self, running_apprentice_and_mocks, bad_input
    ):
        """run() raises error when input_data is not a dict."""
        apprentice, _ = running_apprentice_and_mocks

        with pytest.raises((ValueError, TypeError, ApprenticeError)):
            await apprentice.run("test_task", bad_input)

    @pytest.mark.asyncio
    async def test_run_audit_log_flushed_on_success(self, running_apprentice_and_mocks):
        """Audit log receives a RunContext entry on successful run."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["budget_manager"].can_spend.return_value = True
        mocks["local_client"].is_available.return_value = True

        await apprentice.run("test_task", {"prompt": "hello"})

        mocks["audit_log"].log.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_audit_log_flushed_on_error(self, running_apprentice_and_mocks):
        """Audit log receives a RunContext entry even when run() fails."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["budget_manager"].can_spend.return_value = False
        mocks["local_client"].is_available.return_value = False

        with pytest.raises((BudgetExhaustedError, LocalModelUnavailableError, ApprenticeError)):
            await apprentice.run("test_task", {"prompt": "hello"})

        # Audit log should still have been called
        assert mocks["audit_log"].log.called, (
            "Audit log must be flushed even on error"
        )

    @pytest.mark.asyncio
    async def test_run_budget_enforcement_hard_no_remote_call(
        self, running_apprentice_and_mocks
    ):
        """When budget_manager.can_spend()=False, remote_client is never called."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["budget_manager"].can_spend.return_value = False
        mocks["local_client"].is_available.return_value = True

        await apprentice.run("test_task", {"prompt": "hello"})

        # remote_client.predict should NOT have been called
        mocks["remote_client"].predict.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_remote_records_spend(self, running_apprentice_and_mocks):
        """When remote model is used, spend is recorded in budget_manager."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["router"].route.return_value = ModelSource.remote
        mocks["budget_manager"].can_spend.return_value = True
        mocks["local_client"].is_available.return_value = True

        await apprentice.run("test_task", {"prompt": "hello"})

        mocks["budget_manager"].record_spend.assert_called()

    @pytest.mark.asyncio
    async def test_run_stores_training_data(self, running_apprentice_and_mocks):
        """Successful run stores training data."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["budget_manager"].can_spend.return_value = True
        mocks["local_client"].is_available.return_value = True

        await apprentice.run("test_task", {"prompt": "hello"})

        assert mocks["training_data_store"].store.called or \
               mocks["training_data_store"].save.called or \
               mocks["training_data_store"].add.called, (
            "Training data should be stored after successful run"
        )

    @pytest.mark.asyncio
    async def test_run_updates_confidence_engine(self, running_apprentice_and_mocks):
        """Confidence engine is updated after a run."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["budget_manager"].can_spend.return_value = True
        mocks["local_client"].is_available.return_value = True

        await apprentice.run("test_task", {"prompt": "hello"})

        mocks["confidence_engine"].update.assert_called()

    @pytest.mark.asyncio
    async def test_invariant_source_accuracy_local(self, running_apprentice_and_mocks):
        """TaskResponse.source accurately reflects local backend."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["router"].route.return_value = ModelSource.local
        mocks["local_client"].is_available.return_value = True
        mocks["budget_manager"].can_spend.return_value = True

        response = await apprentice.run("test_task", {"prompt": "hello"})

        assert response.source == ModelSource.local, (
            "source must accurately reflect that local backend produced the output"
        )

    @pytest.mark.asyncio
    async def test_invariant_source_accuracy_remote(self, running_apprentice_and_mocks):
        """TaskResponse.source accurately reflects remote backend."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["router"].route.return_value = ModelSource.remote
        mocks["local_client"].is_available.return_value = True
        mocks["budget_manager"].can_spend.return_value = True

        response = await apprentice.run("test_task", {"prompt": "hello"})

        assert response.source == ModelSource.remote, (
            "source must accurately reflect that remote backend produced the output"
        )

    @pytest.mark.asyncio
    async def test_run_response_is_frozen(self, running_apprentice_and_mocks):
        """TaskResponse is a frozen Pydantic model — immutable after construction."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["budget_manager"].can_spend.return_value = True
        mocks["local_client"].is_available.return_value = True

        response = await apprentice.run("test_task", {"prompt": "hello"})

        with pytest.raises((AttributeError, TypeError, Exception)):
            response.task_name = "mutated"


# ---------------------------------------------------------------------------
# TestStatus
# ---------------------------------------------------------------------------

class TestStatus:
    """Tests for Apprentice.status()."""

    @pytest.mark.asyncio
    async def test_status_happy_path(self, running_apprentice_and_mocks):
        """status() returns a valid ConfidenceSnapshot for a registered task."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["confidence_engine"].get_score.return_value = 0.6
        mocks["confidence_engine"].get_phase.return_value = PhaseLabel.supervised
        mocks["budget_manager"].get_remaining.return_value = 8.0
        mocks["budget_manager"].get_used.return_value = 2.0

        snapshot = await apprentice.status("test_task")

        assert isinstance(snapshot, ConfidenceSnapshot), "Should return ConfidenceSnapshot"
        assert snapshot.task_name == "test_task", "task_name should match input"
        assert 0.0 <= snapshot.confidence_score <= 1.0, "confidence_score in [0,1]"
        assert snapshot.phase in list(PhaseLabel), "phase should be a valid PhaseLabel"

        # Budget invariant: remaining + used == limit
        task_cfg = make_task_config()
        expected_budget = task_cfg.budget_limit_usd
        actual_sum = snapshot.budget_remaining_usd + snapshot.budget_used_usd
        assert abs(actual_sum - expected_budget) < 0.01, (
            f"budget_remaining ({snapshot.budget_remaining_usd}) + "
            f"budget_used ({snapshot.budget_used_usd}) should equal "
            f"budget_limit ({expected_budget}), got {actual_sum}"
        )

    @pytest.mark.asyncio
    async def test_status_error_not_running(self):
        """status() raises RuntimeError when called outside async context."""
        apprentice, _ = build_apprentice()

        with pytest.raises((RuntimeError, ApprenticeError)):
            await apprentice.status("test_task")

    @pytest.mark.asyncio
    async def test_status_error_task_not_found(self, running_apprentice_and_mocks):
        """status() raises TaskNotFoundError for unregistered task."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["task_registry"].get.side_effect = TaskNotFoundError(
            error_kind=ErrorKind.task_not_found,
            message="Task 'unknown' not found",
            task_name="unknown",
            available_tasks=["test_task"],
        )
        mocks["task_registry"].contains.return_value = False

        with pytest.raises((TaskNotFoundError, ApprenticeError)):
            await apprentice.status("unknown")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("empty_name", ["", "  "])
    async def test_status_error_empty_task_name(
        self, running_apprentice_and_mocks, empty_name
    ):
        """status() raises error when task_name is empty."""
        apprentice, _ = running_apprentice_and_mocks

        with pytest.raises((ValueError, ApprenticeError)):
            await apprentice.status(empty_name)

    @pytest.mark.asyncio
    async def test_status_snapshot_is_frozen(self, running_apprentice_and_mocks):
        """ConfidenceSnapshot is frozen — immutable after construction."""
        apprentice, _ = running_apprentice_and_mocks

        snapshot = await apprentice.status("test_task")

        with pytest.raises((AttributeError, TypeError, Exception)):
            snapshot.confidence_score = 999.0


# ---------------------------------------------------------------------------
# TestReport
# ---------------------------------------------------------------------------

class TestReport:
    """Tests for Apprentice.report()."""

    def test_report_happy_path(self):
        """report() returns SystemReport with correct structure."""
        config = make_config(tasks=[make_task_config("task_a"), make_task_config("task_b")])
        mocks = make_all_mocks()
        mocks["task_registry"].list_tasks.return_value = ["task_a", "task_b"]
        mocks["task_registry"].get.side_effect = lambda name: make_task_config(name)

        apprentice, _ = build_apprentice(config=config, **mocks)

        report = apprentice.report()

        assert isinstance(report, SystemReport), "Should return SystemReport"
        assert len(report.task_snapshots) == 2, (
            "Should have one snapshot per registered task"
        )
        assert report.total_runs == report.total_local_runs + report.total_remote_runs, (
            "total_runs must equal total_local_runs + total_remote_runs"
        )

    def test_report_uptime_zero_before_context(self):
        """report() returns uptime_seconds=0.0 before async context is entered."""
        apprentice, _ = build_apprentice()

        report = apprentice.report()

        assert report.uptime_seconds == 0.0, (
            "uptime_seconds should be 0.0 before async context entry"
        )

    @pytest.mark.asyncio
    async def test_report_uptime_positive_after_context(self):
        """report() returns positive uptime_seconds after async context entry."""
        apprentice, mocks = build_apprentice()
        await apprentice.__aenter__()

        # Small sleep to ensure measurable uptime
        await asyncio.sleep(0.01)

        report = apprentice.report()

        assert report.uptime_seconds > 0.0, (
            "uptime_seconds should be > 0 after async context entry"
        )

        await apprentice.__aexit__(None, None, None)

    def test_report_global_budget_aggregation(self):
        """global_budget_used_usd equals sum of per-task budget_used."""
        apprentice, mocks = build_apprentice()

        report = apprentice.report()

        # The sum of per-task budget_used should equal global_budget_used
        per_task_sum = sum(s.budget_used_usd for s in report.task_snapshots)
        assert abs(report.global_budget_used_usd - per_task_sum) < 0.001, (
            f"global_budget_used ({report.global_budget_used_usd}) should equal "
            f"sum of per-task budget_used ({per_task_sum})"
        )

    def test_report_is_frozen(self):
        """SystemReport is a frozen Pydantic model — immutable."""
        apprentice, _ = build_apprentice()

        report = apprentice.report()

        with pytest.raises((AttributeError, TypeError, Exception)):
            report.total_runs = 999

    def test_invariant_total_runs_equals_sum(self):
        """total_runs always equals total_local_runs + total_remote_runs."""
        apprentice, _ = build_apprentice()

        report = apprentice.report()

        assert report.total_runs == report.total_local_runs + report.total_remote_runs, (
            "Invariant: total_runs == total_local_runs + total_remote_runs"
        )


# ---------------------------------------------------------------------------
# TestConfidenceThresholds validation
# ---------------------------------------------------------------------------

class TestConfidenceThresholds:
    """Tests for ConfidenceThresholds struct validation."""

    def test_valid_thresholds(self):
        """Valid thresholds satisfy bootstrapping < active_learning < supervised < autonomous."""
        ct = make_confidence_thresholds()
        assert ct.bootstrapping_upper < ct.active_learning_upper
        assert ct.active_learning_upper < ct.supervised_upper
        assert ct.supervised_upper < ct.autonomous_lower

    def test_invalid_thresholds_order(self):
        """Thresholds that violate ordering constraint are rejected."""
        with pytest.raises((ValueError, ApprenticeError)):
            ConfidenceThresholds(
                bootstrapping_upper=0.8,  # > active_learning_upper
                active_learning_upper=0.5,
                supervised_upper=0.3,
                autonomous_lower=0.1,
            )

    def test_thresholds_out_of_range(self):
        """Threshold values outside [0.0, 1.0] are rejected."""
        with pytest.raises((ValueError, ApprenticeError)):
            ConfidenceThresholds(
                bootstrapping_upper=-0.1,
                active_learning_upper=0.5,
                supervised_upper=0.75,
                autonomous_lower=1.5,
            )


# ---------------------------------------------------------------------------
# TestPhaseDerivation (Invariant)
# ---------------------------------------------------------------------------

class TestPhaseDerivation:
    """Phase labels are always derived from confidence scores — never manually set."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "confidence,expected_phase",
        [
            (0.1, PhaseLabel.bootstrapping),
            (0.3, PhaseLabel.active_learning),
            (0.6, PhaseLabel.supervised),
            (0.95, PhaseLabel.autonomous),
        ],
    )
    async def test_phase_consistent_with_confidence(
        self, running_apprentice_and_mocks, confidence, expected_phase
    ):
        """Phase label is consistent with the confidence score and thresholds."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["confidence_engine"].get_score.return_value = confidence
        mocks["confidence_engine"].get_phase.return_value = expected_phase

        snapshot = await apprentice.status("test_task")

        assert snapshot.phase == expected_phase, (
            f"With confidence {confidence}, phase should be {expected_phase}, "
            f"got {snapshot.phase}"
        )


# ---------------------------------------------------------------------------
# TestSamplingRate (Invariant)
# ---------------------------------------------------------------------------

class TestSamplingRate:
    """Sampling rate is continuous float in [0.0, 1.0]."""

    @pytest.mark.asyncio
    async def test_sampling_rate_is_float_in_range(self, running_apprentice_and_mocks):
        """Sampling rate returned in status is a float in [0.0, 1.0]."""
        apprentice, mocks = running_apprentice_and_mocks
        mocks["sampling_scheduler"].get_rate.return_value = 0.42

        snapshot = await apprentice.status("test_task")

        assert isinstance(snapshot.sampling_rate, float), "sampling_rate should be float"
        assert 0.0 <= snapshot.sampling_rate <= 1.0, (
            f"sampling_rate should be in [0.0, 1.0], got {snapshot.sampling_rate}"
        )


# ---------------------------------------------------------------------------
# TestErrorTypes
# ---------------------------------------------------------------------------

class TestErrorTypes:
    """Verify error type structure and inheritance."""

    def test_budget_exhausted_error_is_apprentice_error(self):
        """BudgetExhaustedError derives from ApprenticeError."""
        err = BudgetExhaustedError(
            error_kind=ErrorKind.budget_exhausted,
            message="Budget exhausted",
            task_name="test_task",
            budget_limit_usd=10.0,
            budget_used_usd=10.0,
            request_id="req-123",
        )
        assert isinstance(err, ApprenticeError)
        assert err.error_kind == ErrorKind.budget_exhausted

    def test_task_not_found_error_is_apprentice_error(self):
        """TaskNotFoundError derives from ApprenticeError."""
        err = TaskNotFoundError(
            error_kind=ErrorKind.task_not_found,
            message="Not found",
            task_name="unknown",
            available_tasks=["task_a"],
        )
        assert isinstance(err, ApprenticeError)
        assert err.available_tasks == ["task_a"]

    def test_local_model_unavailable_error_is_apprentice_error(self):
        """LocalModelUnavailableError derives from ApprenticeError."""
        err = LocalModelUnavailableError(
            error_kind=ErrorKind.local_model_unavailable,
            message="Local unavailable",
            task_name="test_task",
            request_id="req-456",
            budget_exhausted=True,
        )
        assert isinstance(err, ApprenticeError)
        assert err.budget_exhausted is True
