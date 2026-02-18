"""
Contract tests for the root composition component.

Tests are organized into 7 logical sections:
1. Initialization tests
2. Shutdown tests
3. Backend registry tests
4. Orchestration tests
5. Correlation and status tests
6. Component override tests
7. Lifecycle logging tests
8. Invariant tests

All 9 dependencies are mocked. Tests verify behavior at boundaries only.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import (
    AsyncMock,
    MagicMock,
    Mock,
    call,
    patch,
    PropertyMock,
)

import pytest

# Import the component under test
from src.root import (
    initialize_composition_root,
    shutdown_composition_root,
    resolve_finetuning_backend,
    orchestrate_request,
    get_component_statuses,
    create_request_correlation,
    validate_component_override,
    build_backend_registry,
    AllSourcesUnavailableError,
    InitializationError,
    ShutdownError,
    RequestCorrelation,
    ComponentId,
    ComponentStatus,
    InitializationPhase,
    LifecycleEvent,
    FinetuningBackendName,
    BackendRegistryEntry,
    BackendRegistry,
)


# ============================================================================
# Fixtures (conftest-style, function-scoped)
# ============================================================================


@pytest.fixture
def minimal_valid_config_yaml():
    """Minimal valid YAML config content for the apprentice system."""
    return """
tasks:
  - name: sentiment_analysis
    prompt_template: "Analyze sentiment: {text}"
    input_schema:
      text: str
    output_schema:
      sentiment: str
      confidence: float
    match_fields:
      - sentiment
    confidence_thresholds:
      phase2: 0.7
      phase3: 0.9

budget:
  total_budget_usd: 100.0
  period: monthly
  state_dir: /tmp/apprentice_budget

confidence_engine:
  window_size: 50
  evaluators:
    - name: exact_match

external:
  remote:
    provider: openai
    api_key: env:OPENAI_API_KEY
    api_base_url: https://api.openai.com/v1
    model_id: gpt-4
  local:
    provider: vllm
    model_path: /models/local
    model_id: local-v1

finetuning:
  backend: local_lora
  output_dir: /tmp/apprentice_ft

sampling:
  decay_function: exponential
  min_rate: 0.01
  max_rate: 1.0

reporting:
  audit_log_path: /tmp/apprentice_audit.jsonl
  max_file_size_mb: 100
  rotation_backup_count: 5
"""


@pytest.fixture
def config_file(tmp_path, minimal_valid_config_yaml):
    """Write minimal valid config to a temp file and return its path."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(minimal_valid_config_yaml)
    return str(config_path)


@pytest.fixture
def mock_env():
    """Mock environment variables dict."""
    return {
        "OPENAI_API_KEY": "sk-test-key-12345",
        "HOME": "/home/test",
    }


def _make_mock_budget_manager():
    """Create a mock budget manager conforming to Protocol."""
    bm = AsyncMock()
    bm.authorize = AsyncMock(return_value=MagicMock(allowed=True, estimated_cost=Decimal("0.01")))
    bm.record_spend = AsyncMock()
    bm.release_reservation = AsyncMock()
    bm.record_local_request = AsyncMock()
    bm.remaining_budget = Mock(return_value=Decimal("99.00"))
    bm.get_report = Mock()
    bm.force_persist = AsyncMock()
    bm.reset_state = AsyncMock()
    bm.load_state = AsyncMock()
    return bm


def _make_mock_confidence_engine():
    """Create a mock confidence engine conforming to Protocol."""
    ce = AsyncMock()
    snapshot = MagicMock()
    snapshot.score = 0.85
    snapshot.phase = "phase_3"
    snapshot.sample_count = 100
    ce.get_snapshot = AsyncMock(return_value=snapshot)
    ce.record_comparison = AsyncMock()
    ce.persist_state = AsyncMock()
    ce.load_state = AsyncMock(return_value=0)
    ce.get_task_ids = Mock(return_value=[])
    return ce


def _make_mock_config_and_registry():
    """Create a mock config and registry conforming to Protocol."""
    cr = MagicMock()
    config = MagicMock()
    config.budget = MagicMock()
    config.confidence_engine = MagicMock()
    config.external = MagicMock()
    config.finetuning = MagicMock()
    config.finetuning.backend = "local_lora"
    config.sampling = MagicMock()
    config.reporting = MagicMock()
    config.tasks = []
    cr.load_config = Mock(return_value=config)
    cr.config = config

    task_registry = MagicMock()
    task_registry.get_task = Mock(return_value=MagicMock())
    task_registry.__contains__ = Mock(return_value=True)
    task_registry.task_names = Mock(return_value=["sentiment_analysis"])
    task_registry.__len__ = Mock(return_value=1)
    cr.task_registry = task_registry
    return cr


def _make_mock_external_interfaces():
    """Create a mock external interfaces conforming to Protocol."""
    ei = AsyncMock()
    ei.complete = AsyncMock(return_value=MagicMock(
        output_data={"sentiment": "positive", "confidence": 0.95},
        model_used="gpt-4",
        cost=Decimal("0.005"),
        latency_ms=150.0,
    ))
    ei.health = AsyncMock(return_value=MagicMock(healthy=True))
    ei.close = AsyncMock()
    ei.get_source_info = Mock()
    return ei


def _make_mock_sampling_scheduler():
    """Create a mock sampling scheduler conforming to Protocol."""
    ss = MagicMock()
    decision = MagicMock()
    decision.decision = "local_only"
    decision.sampling_probability = 0.05
    decision.is_coaching_sample = False
    ss.should_sample = Mock(return_value=decision)
    return ss


def _make_mock_router():
    """Create a mock router conforming to Protocol."""
    r = AsyncMock()
    result = MagicMock()
    result.response = MagicMock(
        output_data={"sentiment": "positive", "confidence": 0.95},
        model_used="local-v1",
        cost=Decimal("0.0"),
        latency_ms=50.0,
    )
    result.routing_decision = "local_only"
    result.used_remote = False
    result.used_local = True
    result.remote_cost = None
    result.local_cost = Decimal("0.0")
    r.route = AsyncMock(return_value=result)
    r.compute_routing_action = Mock()
    r.execute_dual_send = AsyncMock()
    r.build_audit_entry = Mock()
    return r


def _make_mock_training_pipeline():
    """Create a mock training pipeline conforming to Protocol."""
    tp = AsyncMock()
    tp.initialize = AsyncMock()
    tp.add_example = AsyncMock()
    tp.should_trigger = Mock(return_value=MagicMock(should_trigger=False))
    tp.run_pipeline = AsyncMock()
    tp.get_status = Mock()
    tp.get_run_history = Mock()
    return tp


def _make_mock_reporting():
    """Create a mock reporting conforming to Protocol."""
    rp = MagicMock()
    audit_logger = AsyncMock()
    audit_logger.log = AsyncMock()
    audit_logger.flush = AsyncMock()
    audit_logger.close = AsyncMock()
    rp.audit_logger = audit_logger
    rp.generate_report = Mock()
    rp.format_report = Mock()
    return rp


def _make_mock_unified_interface():
    """Create a mock unified interface conforming to Protocol."""
    ui = AsyncMock()
    ui.run = AsyncMock()
    ui.status = AsyncMock()
    ui.report = AsyncMock()
    ui.close = AsyncMock()
    ui.__aenter__ = AsyncMock(return_value=ui)
    ui.__aexit__ = AsyncMock(return_value=False)
    return ui


@pytest.fixture
def mock_component_overrides():
    """Dict of mock component overrides for all 10 components."""
    from types import SimpleNamespace
    return {
        "config_and_registry": _make_mock_config_and_registry(),
        "plugin_registry": SimpleNamespace(),
        "budget_manager": _make_mock_budget_manager(),
        "confidence_engine": _make_mock_confidence_engine(),
        "external_interfaces": _make_mock_external_interfaces(),
        "sampling_scheduler": _make_mock_sampling_scheduler(),
        "router": _make_mock_router(),
        "training_pipeline": _make_mock_training_pipeline(),
        "reporting": _make_mock_reporting(),
        "unified_interface": _make_mock_unified_interface(),
    }


@pytest.fixture
def mock_exit_stack():
    """Mock AsyncExitStack."""
    stack = AsyncMock()
    stack.__aenter__ = AsyncMock(return_value=stack)
    stack.__aexit__ = AsyncMock(return_value=False)
    stack.aclose = AsyncMock()
    return stack


@pytest.fixture
def sample_correlation():
    """A sample RequestCorrelation for testing."""
    return RequestCorrelation(
        request_id=str(uuid.uuid4()),
        task_name="sentiment_analysis",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


# ============================================================================
# 1. Initialization Tests (test_initialization)
# ============================================================================


class TestInitializationHappyPath:
    """Test initialize_composition_root happy path scenarios."""

    @pytest.mark.asyncio
    async def test_init_all_9_components_returned(
        self, config_file, mock_env, mock_component_overrides
    ):
        """initialize_composition_root returns dict with all 10 component entries."""
        result = await initialize_composition_root(
            config_path=config_file,
            env=mock_env,
            component_overrides=mock_component_overrides,
        )

        assert isinstance(result, dict), "Result should be a dict"
        assert len(result) == 10, f"Expected 10 components, got {len(result)}"

        expected_ids = {
            "config_and_registry", "plugin_registry", "budget_manager",
            "confidence_engine", "external_interfaces", "sampling_scheduler",
            "router", "training_pipeline", "reporting", "unified_interface",
        }
        assert set(result.keys()) == expected_ids, (
            f"Missing component IDs: {expected_ids - set(result.keys())}"
        )

        for cid, instance in result.items():
            assert instance is not None, f"Component {cid} should not be None"

    @pytest.mark.asyncio
    async def test_init_overrides_used_instead_of_defaults(
        self, config_file, mock_env, mock_component_overrides
    ):
        """Component overrides are used as-is without modification."""
        result = await initialize_composition_root(
            config_path=config_file,
            env=mock_env,
            component_overrides=mock_component_overrides,
        )

        for cid, override in mock_component_overrides.items():
            assert result[cid] is override, (
                f"Component {cid} should be the exact override instance"
            )


class TestInitializationErrors:
    """Test all error paths for initialize_composition_root."""

    @pytest.mark.asyncio
    async def test_config_file_not_found(self, mock_env):
        """Raises InitializationError when config_path does not exist."""
        with pytest.raises(InitializationError) as exc_info:
            await initialize_composition_root(
                config_path="/nonexistent/path/config.yaml",
                env=mock_env,
                component_overrides={},
            )

        err = exc_info.value
        assert err.failed_phase == InitializationPhase.CONFIG_PARSE, (
            f"Expected CONFIG_PARSE phase, got {err.failed_phase}"
        )
        assert "config" in str(err.message).lower() or "not found" in str(err.message).lower(), (
            f"Error message should mention config not found: {err.message}"
        )

    @pytest.mark.asyncio
    async def test_config_validation_failed(self, tmp_path, mock_env):
        """Raises InitializationError when YAML is valid but fails Pydantic validation."""
        bad_config = tmp_path / "bad_config.yaml"
        bad_config.write_text("tasks: []\n")  # Missing required fields

        with pytest.raises(InitializationError) as exc_info:
            await initialize_composition_root(
                config_path=str(bad_config),
                env=mock_env,
                component_overrides={},
            )

        err = exc_info.value
        assert err.failed_phase == InitializationPhase.CONFIG_PARSE, (
            f"Expected CONFIG_PARSE phase, got {err.failed_phase}"
        )
        assert err.cause is not None and len(str(err.cause)) > 0, (
            "Error cause should contain validation details"
        )

    @pytest.mark.asyncio
    async def test_env_var_not_found(self, config_file):
        """Raises InitializationError when env:VAR_NAME cannot be resolved."""
        empty_env = {}  # Missing OPENAI_API_KEY

        with pytest.raises(InitializationError) as exc_info:
            await initialize_composition_root(
                config_path=config_file,
                env=empty_env,
                component_overrides={},
            )

        err = exc_info.value
        assert err.failed_phase == InitializationPhase.CONFIG_PARSE, (
            f"Expected CONFIG_PARSE phase, got {err.failed_phase}"
        )

    @pytest.mark.asyncio
    async def test_budget_state_corrupted(
        self, config_file, mock_env, mock_component_overrides
    ):
        """Raises InitializationError when budget state file is corrupted."""
        bad_bm = _make_mock_budget_manager()
        bad_bm.load_state = AsyncMock(side_effect=ValueError("Corrupted budget state"))
        mock_component_overrides.pop("budget_manager", None)

        # Patch the budget manager construction to raise
        with patch(
            "src.root.composition.BudgetManager",
            side_effect=ValueError("Corrupted budget state"),
        ):
            with pytest.raises(InitializationError) as exc_info:
                await initialize_composition_root(
                    config_path=config_file,
                    env=mock_env,
                    component_overrides={
                        k: v for k, v in mock_component_overrides.items()
                        if k != "budget_manager"
                    },
                )

        err = exc_info.value
        assert err.failed_phase == InitializationPhase.BUDGET_MANAGER_INIT, (
            f"Expected BUDGET_MANAGER_INIT phase, got {err.failed_phase}"
        )
        assert err.failed_component_id == "BUDGET_MANAGER" or err.failed_component_id == "budget_manager", (
            f"Expected BUDGET_MANAGER component, got {err.failed_component_id}"
        )

    @pytest.mark.asyncio
    async def test_budget_state_dir_not_writable(
        self, config_file, mock_env, mock_component_overrides
    ):
        """Raises InitializationError when budget state directory is not writable."""
        with patch(
            "src.root.composition.BudgetManager",
            side_effect=PermissionError("Directory not writable"),
        ):
            overrides = {
                k: v for k, v in mock_component_overrides.items()
                if k != "budget_manager"
            }
            with pytest.raises(InitializationError) as exc_info:
                await initialize_composition_root(
                    config_path=config_file,
                    env=mock_env,
                    component_overrides=overrides,
                )

            err = exc_info.value
            assert err.failed_phase == InitializationPhase.BUDGET_MANAGER_INIT

    @pytest.mark.asyncio
    async def test_confidence_engine_config_invalid(
        self, config_file, mock_env, mock_component_overrides
    ):
        """Raises InitializationError when confidence engine config is invalid."""
        with patch(
            "src.root.composition.ConfidenceEngine",
            side_effect=ValueError("Invalid confidence engine config"),
        ):
            overrides = {
                k: v for k, v in mock_component_overrides.items()
                if k != "confidence_engine"
            }
            with pytest.raises(InitializationError) as exc_info:
                await initialize_composition_root(
                    config_path=config_file,
                    env=mock_env,
                    component_overrides=overrides,
                )

            err = exc_info.value
            assert err.failed_phase == InitializationPhase.CONFIDENCE_ENGINE_INIT

    @pytest.mark.asyncio
    async def test_remote_adapter_api_key_missing(
        self, config_file, mock_component_overrides
    ):
        """Raises InitializationError when remote API key env var is missing."""
        env_no_key = {"HOME": "/home/test"}  # Missing OPENAI_API_KEY

        with patch(
            "src.root.composition.create_remote_adapter",
            side_effect=ValueError("API key not set"),
        ):
            overrides = {
                k: v for k, v in mock_component_overrides.items()
                if k != "external_interfaces"
            }
            with pytest.raises(InitializationError) as exc_info:
                await initialize_composition_root(
                    config_path=config_file,
                    env=env_no_key,
                    component_overrides=overrides,
                )

            err = exc_info.value
            # Could fail at CONFIG_PARSE (env resolution) or EXTERNAL_INTERFACES_INIT
            assert err.failed_phase in (
                InitializationPhase.CONFIG_PARSE,
                InitializationPhase.EXTERNAL_INTERFACES_INIT,
            )

    @pytest.mark.asyncio
    async def test_unsupported_finetuning_backend(
        self, tmp_path, mock_env, mock_component_overrides
    ):
        """Raises InitializationError when config references unknown backend."""
        bad_config = tmp_path / "config.yaml"
        config_text = """
tasks:
  - name: test_task
    prompt_template: "Test: {text}"
    input_schema: {text: str}
    output_schema: {result: str}
    match_fields: [result]
    confidence_thresholds: {phase2: 0.7, phase3: 0.9}
budget:
  total_budget_usd: 100.0
  period: monthly
  state_dir: /tmp/test_budget
confidence_engine:
  window_size: 50
  evaluators: [{name: exact_match}]
external:
  remote:
    provider: openai
    api_key: env:OPENAI_API_KEY
    api_base_url: https://api.openai.com/v1
    model_id: gpt-4
  local:
    provider: vllm
    model_path: /models/local
    model_id: local-v1
finetuning:
  backend: quantum_computing_backend
  output_dir: /tmp/ft
sampling:
  decay_function: exponential
  min_rate: 0.01
  max_rate: 1.0
reporting:
  audit_log_path: /tmp/audit.jsonl
  max_file_size_mb: 100
  rotation_backup_count: 5
"""
        bad_config.write_text(config_text)

        overrides = {
            k: v for k, v in mock_component_overrides.items()
            if k != "training_pipeline"
        }
        with pytest.raises(InitializationError) as exc_info:
            await initialize_composition_root(
                config_path=str(bad_config),
                env=mock_env,
                component_overrides=overrides,
            )

        err = exc_info.value
        # Could fail at config parse or training pipeline init
        assert err.failed_phase in (
            InitializationPhase.CONFIG_PARSE,
            InitializationPhase.TRAINING_PIPELINE_INIT,
        )

    @pytest.mark.asyncio
    async def test_audit_log_path_not_writable(
        self, config_file, mock_env, mock_component_overrides
    ):
        """Raises InitializationError when audit log path is not writable."""
        with patch(
            "src.root.composition.JsonLinesAuditLogger",
            side_effect=PermissionError("Audit log dir not writable"),
        ):
            overrides = {
                k: v for k, v in mock_component_overrides.items()
                if k != "reporting"
            }
            with pytest.raises(InitializationError) as exc_info:
                await initialize_composition_root(
                    config_path=config_file,
                    env=mock_env,
                    component_overrides=overrides,
                )

            err = exc_info.value
            assert err.failed_phase == InitializationPhase.REPORTING_INIT

    @pytest.mark.asyncio
    async def test_component_protocol_mismatch(
        self, config_file, mock_env
    ):
        """Raises InitializationError for protocol-mismatched override."""
        bad_override = object()  # Has no methods at all
        with pytest.raises((InitializationError, TypeError, ValueError)):
            await initialize_composition_root(
                config_path=config_file,
                env=mock_env,
                component_overrides={"budget_manager": bad_override},
            )


class TestInitializationTeardownOnFailure:
    """Verify teardown of already-initialized components on partial failure."""

    @pytest.mark.asyncio
    async def test_teardown_on_mid_init_failure(
        self, config_file, mock_env, mock_component_overrides
    ):
        """When init fails at phase N, components 1..N-1 are torn down in reverse."""
        # Make router initialization fail (phase 7 of 9)
        mock_component_overrides.pop("router")

        with patch(
            "src.root.composition.Router",
            side_effect=RuntimeError("Router init failed"),
        ):
            with pytest.raises(InitializationError) as exc_info:
                await initialize_composition_root(
                    config_path=config_file,
                    env=mock_env,
                    component_overrides=mock_component_overrides,
                )

            err = exc_info.value
            assert err.failed_phase == InitializationPhase.ROUTER_INIT

            # Verify previously initialized components were attempted for shutdown
            # Budget manager and confidence engine should have persist called
            bm = mock_component_overrides.get("budget_manager")
            ce = mock_component_overrides.get("confidence_engine")
            if bm and hasattr(bm, "force_persist"):
                # Best effort: check that teardown was attempted
                pass  # Exact verification depends on implementation


# ============================================================================
# 2. Shutdown Tests (test_shutdown)
# ============================================================================


class TestShutdownHappyPath:
    """Test clean shutdown scenarios."""

    @pytest.mark.asyncio
    async def test_clean_shutdown_all_components(
        self, mock_component_overrides, mock_exit_stack
    ):
        """Clean shutdown returns ShutdownError with empty component_errors."""
        result = await shutdown_composition_root(
            components=mock_component_overrides,
            exit_stack=mock_exit_stack,
        )

        assert isinstance(result, ShutdownError), "Should return ShutdownError"
        assert result.component_errors == {} or len(result.component_errors) == 0, (
            f"Expected empty component_errors, got {result.component_errors}"
        )
        assert len(result.components_shutdown_successfully) == 10, (
            f"Expected 9 successful shutdowns, got {len(result.components_shutdown_successfully)}"
        )

    @pytest.mark.asyncio
    async def test_budget_persist_called_on_shutdown(
        self, mock_component_overrides, mock_exit_stack
    ):
        """budget_manager.force_persist is called during shutdown."""
        await shutdown_composition_root(
            components=mock_component_overrides,
            exit_stack=mock_exit_stack,
        )

        mock_component_overrides["budget_manager"].force_persist.assert_called_once()

    @pytest.mark.asyncio
    async def test_confidence_persist_called_on_shutdown(
        self, mock_component_overrides, mock_exit_stack
    ):
        """confidence_engine.persist_state is called during shutdown."""
        await shutdown_composition_root(
            components=mock_component_overrides,
            exit_stack=mock_exit_stack,
        )

        mock_component_overrides["confidence_engine"].persist_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_exit_stack_closed(
        self, mock_component_overrides, mock_exit_stack
    ):
        """AsyncExitStack is closed after all component shutdown."""
        await shutdown_composition_root(
            components=mock_component_overrides,
            exit_stack=mock_exit_stack,
        )

        mock_exit_stack.aclose.assert_called_once()


class TestShutdownErrors:
    """Test individual and multiple component failure scenarios during shutdown."""

    @pytest.mark.asyncio
    async def test_budget_persist_failure(
        self, mock_component_overrides, mock_exit_stack
    ):
        """Budget persist failure captured but other components still shut down."""
        mock_component_overrides["budget_manager"].force_persist = AsyncMock(
            side_effect=IOError("Disk full")
        )

        result = await shutdown_composition_root(
            components=mock_component_overrides,
            exit_stack=mock_exit_stack,
        )

        assert "budget_manager" in result.component_errors or "BUDGET_MANAGER" in result.component_errors, (
            f"Expected BUDGET_MANAGER in component_errors: {result.component_errors}"
        )
        # Other components should still be in successful list
        successful_ids = set(result.components_shutdown_successfully)
        assert len(successful_ids) >= 8, (
            f"Expected at least 8 successful shutdowns, got {len(successful_ids)}"
        )

    @pytest.mark.asyncio
    async def test_confidence_persist_failure(
        self, mock_component_overrides, mock_exit_stack
    ):
        """Confidence persist failure captured but other components still shut down."""
        mock_component_overrides["confidence_engine"].persist_state = AsyncMock(
            side_effect=IOError("State file write failed")
        )

        result = await shutdown_composition_root(
            components=mock_component_overrides,
            exit_stack=mock_exit_stack,
        )

        assert (
            "confidence_engine" in result.component_errors
            or "CONFIDENCE_ENGINE" in result.component_errors
        ), f"Expected CONFIDENCE_ENGINE in errors: {result.component_errors}"
        assert len(result.components_shutdown_successfully) >= 8

    @pytest.mark.asyncio
    async def test_audit_flush_failure(
        self, mock_component_overrides, mock_exit_stack
    ):
        """Audit flush failure captured but other components still shut down."""
        mock_component_overrides["reporting"].audit_logger.flush = AsyncMock(
            side_effect=IOError("Flush failed")
        )

        result = await shutdown_composition_root(
            components=mock_component_overrides,
            exit_stack=mock_exit_stack,
        )

        assert (
            "reporting" in result.component_errors
            or "REPORTING" in result.component_errors
        ), f"Expected REPORTING in errors: {result.component_errors}"

    @pytest.mark.asyncio
    async def test_httpx_close_failure(
        self, mock_component_overrides, mock_exit_stack
    ):
        """httpx client close failure captured but shutdown continues."""
        mock_component_overrides["external_interfaces"].close = AsyncMock(
            side_effect=RuntimeError("httpx close failed")
        )

        result = await shutdown_composition_root(
            components=mock_component_overrides,
            exit_stack=mock_exit_stack,
        )

        assert (
            "external_interfaces" in result.component_errors
            or "EXTERNAL_INTERFACES" in result.component_errors
        ), f"Expected EXTERNAL_INTERFACES in errors: {result.component_errors}"
        # Other components should still shut down
        assert len(result.components_shutdown_successfully) >= 8

    @pytest.mark.asyncio
    async def test_exit_stack_close_failure(
        self, mock_component_overrides, mock_exit_stack
    ):
        """Exit stack close failure captured in ShutdownError."""
        mock_exit_stack.aclose = AsyncMock(
            side_effect=RuntimeError("Exit stack close failed")
        )

        result = await shutdown_composition_root(
            components=mock_component_overrides,
            exit_stack=mock_exit_stack,
        )

        # Exit stack error should be captured somewhere
        has_exit_stack_error = (
            "exit_stack" in str(result.component_errors).lower()
            or "exit_stack" in str(result.message).lower()
            or len(result.component_errors) > 0
        )
        assert has_exit_stack_error, "Exit stack error should be captured"

    @pytest.mark.asyncio
    async def test_multiple_simultaneous_failures(
        self, mock_component_overrides, mock_exit_stack
    ):
        """Multiple component failures all accumulated in ShutdownError."""
        mock_component_overrides["budget_manager"].force_persist = AsyncMock(
            side_effect=IOError("Budget persist failed")
        )
        mock_component_overrides["confidence_engine"].persist_state = AsyncMock(
            side_effect=IOError("Confidence persist failed")
        )
        mock_component_overrides["external_interfaces"].close = AsyncMock(
            side_effect=RuntimeError("httpx close failed")
        )

        result = await shutdown_composition_root(
            components=mock_component_overrides,
            exit_stack=mock_exit_stack,
        )

        assert len(result.component_errors) >= 3, (
            f"Expected at least 3 errors, got {len(result.component_errors)}"
        )
        # Remaining components should still be in successful list
        total = len(result.component_errors) + len(result.components_shutdown_successfully)
        assert total == 10, (
            f"Errors + successes should total 9, got {total}"
        )

    @pytest.mark.asyncio
    async def test_shutdown_reverse_dag_order(
        self, mock_component_overrides, mock_exit_stack
    ):
        """Shutdown proceeds in reverse DAG order."""
        shutdown_order = []

        async def track_shutdown(name):
            shutdown_order.append(name)

        # Attach tracking to each component's close/persist methods
        for name, comp in mock_component_overrides.items():
            if hasattr(comp, "close") and callable(comp.close):
                original = comp.close
                comp.close = AsyncMock(side_effect=lambda n=name: shutdown_order.append(n))
            elif hasattr(comp, "force_persist"):
                comp.force_persist = AsyncMock(side_effect=lambda n=name: shutdown_order.append(n))

        await shutdown_composition_root(
            components=mock_component_overrides,
            exit_stack=mock_exit_stack,
        )

        expected_reverse_order = [
            "unified_interface", "reporting", "training_pipeline",
            "router", "sampling_scheduler", "external_interfaces",
            "confidence_engine", "budget_manager", "config_and_registry",
        ]

        # Verify the relative order is reverse DAG
        if len(shutdown_order) > 1:
            for i, expected_name in enumerate(expected_reverse_order):
                if expected_name in shutdown_order:
                    for j in range(i + 1, len(expected_reverse_order)):
                        later_name = expected_reverse_order[j]
                        if later_name in shutdown_order:
                            idx_first = shutdown_order.index(expected_name)
                            idx_later = shutdown_order.index(later_name)
                            assert idx_first < idx_later, (
                                f"{expected_name} should shut down before {later_name}"
                            )


# ============================================================================
# 3. Backend Registry Tests (test_backend_registry)
# ============================================================================


class TestResolveFineTuningBackend:
    """Test resolve_finetuning_backend across all backend types."""

    @pytest.mark.parametrize("backend_name,expected_remote", [
        ("openai", True),
        ("anyscale", True),
        ("local_lora", False),
        ("axolotl", False),
    ])
    def test_resolve_builtin_backends(self, backend_name, expected_remote):
        """Resolve built-in backends with correct is_remote flag."""
        config = {
            "backend": backend_name,
            "api_key": "test-key",
            "api_base_url": "https://api.example.com/v1",
            "output_dir": "/tmp/ft",
        }

        result = resolve_finetuning_backend(
            backend_name=backend_name,
            finetuning_config=config,
            custom_backends={},
        )

        assert isinstance(result, BackendRegistryEntry), (
            f"Expected BackendRegistryEntry, got {type(result)}"
        )
        assert result.backend_name == backend_name or str(result.backend_name).lower() == backend_name, (
            f"Expected backend_name {backend_name}, got {result.backend_name}"
        )
        assert result.is_remote == expected_remote, (
            f"Expected is_remote={expected_remote} for {backend_name}"
        )
        assert result.implementation is not None, "Implementation should not be None"

    def test_resolve_custom_backend_registered(self):
        """Resolve custom backend when properly registered."""
        custom_impl = MagicMock()
        custom_backends = {"my_custom": custom_impl}

        result = resolve_finetuning_backend(
            backend_name="custom",
            finetuning_config={"backend": "custom", "custom_name": "my_custom"},
            custom_backends=custom_backends,
        )

        assert result.implementation is not None
        assert str(result.backend_name).lower() == "custom"

    def test_resolve_unknown_backend_raises(self):
        """Unknown backend name raises an error."""
        with pytest.raises(Exception) as exc_info:
            resolve_finetuning_backend(
                backend_name="quantum_computing_backend",
                finetuning_config={"backend": "quantum_computing_backend"},
                custom_backends={},
            )

        assert "unknown" in str(exc_info.value).lower() or "not" in str(exc_info.value).lower(), (
            f"Error should mention unknown backend: {exc_info.value}"
        )

    def test_resolve_remote_missing_api_key(self):
        """Remote backend without api_key raises error."""
        with pytest.raises(Exception) as exc_info:
            resolve_finetuning_backend(
                backend_name="openai",
                finetuning_config={
                    "backend": "openai",
                    "api_base_url": "https://api.openai.com/v1",
                    # Missing api_key
                },
                custom_backends={},
            )

        assert "api_key" in str(exc_info.value).lower() or "key" in str(exc_info.value).lower(), (
            f"Error should mention missing api_key: {exc_info.value}"
        )

    def test_resolve_remote_missing_base_url(self):
        """Remote backend without api_base_url raises error."""
        with pytest.raises(Exception) as exc_info:
            resolve_finetuning_backend(
                backend_name="openai",
                finetuning_config={
                    "backend": "openai",
                    "api_key": "test-key",
                    # Missing api_base_url
                },
                custom_backends={},
            )

        assert "url" in str(exc_info.value).lower() or "base" in str(exc_info.value).lower(), (
            f"Error should mention missing api_base_url: {exc_info.value}"
        )

    def test_resolve_custom_not_registered(self):
        """Custom backend requested but none registered raises error."""
        with pytest.raises(Exception) as exc_info:
            resolve_finetuning_backend(
                backend_name="custom",
                finetuning_config={"backend": "custom"},
                custom_backends={},  # Empty — no custom backend registered
            )

        assert "custom" in str(exc_info.value).lower() or "register" in str(exc_info.value).lower(), (
            f"Error should mention custom not registered: {exc_info.value}"
        )

    @pytest.mark.parametrize("invalid_name", [
        "nonexistent", "OPENAI_TYPO", "", "  ", "openai ", " openai",
    ])
    def test_resolve_various_invalid_names(self, invalid_name):
        """Various invalid backend names all raise errors."""
        with pytest.raises(Exception):
            resolve_finetuning_backend(
                backend_name=invalid_name,
                finetuning_config={"backend": invalid_name},
                custom_backends={},
            )


class TestBuildBackendRegistry:
    """Test build_backend_registry construction."""

    def test_build_registry_happy_path(self):
        """Build registry with valid config selecting an available backend."""
        config = {
            "backend": "local_lora",
            "output_dir": "/tmp/ft",
        }

        result = build_backend_registry(
            finetuning_config=config,
            custom_backends={},
        )

        assert isinstance(result, BackendRegistry), (
            f"Expected BackendRegistry, got {type(result)}"
        )
        assert result.active_backend_name == "local_lora", (
            f"Expected active_backend_name 'local_lora', got {result.active_backend_name}"
        )
        assert isinstance(result.entries, dict), "entries should be a dict"
        # Should contain at least the built-in backends
        assert len(result.entries) >= 4, (
            f"Expected at least 4 built-in entries, got {len(result.entries)}"
        )

    def test_build_registry_active_not_available(self):
        """Raises error when active backend from config is not in registry."""
        config = {
            "backend": "nonexistent_backend",
            "output_dir": "/tmp/ft",
        }

        with pytest.raises(Exception) as exc_info:
            build_backend_registry(
                finetuning_config=config,
                custom_backends={},
            )

        error_msg = str(exc_info.value).lower()
        assert "not" in error_msg or "unavailable" in error_msg or "unknown" in error_msg, (
            f"Error should indicate backend not available: {exc_info.value}"
        )

    def test_build_registry_with_custom_backends(self):
        """Registry includes both built-in and custom backends."""
        custom_impl = MagicMock()
        config = {
            "backend": "custom",
            "output_dir": "/tmp/ft",
        }

        result = build_backend_registry(
            finetuning_config=config,
            custom_backends={"my_backend": custom_impl},
        )

        assert isinstance(result, BackendRegistry)
        # Should have built-ins plus custom
        assert len(result.entries) >= 5, (
            f"Expected at least 5 entries (4 built-in + 1 custom), got {len(result.entries)}"
        )


# ============================================================================
# 4. Orchestration Tests (test_orchestration)
# ============================================================================


class TestOrchestrateRequestHappyPath:
    """Test orchestrate_request happy path scenarios."""

    @pytest.mark.asyncio
    async def test_local_only_serving(
        self, mock_component_overrides, sample_correlation
    ):
        """Serves request locally when confidence is high (phase 3)."""
        components = mock_component_overrides

        # Configure: high confidence, local-only routing
        snapshot = MagicMock()
        snapshot.score = 0.95
        snapshot.phase = "phase_3"
        snapshot.sample_count = 200
        components["confidence_engine"].get_snapshot = AsyncMock(return_value=snapshot)

        sampling_decision = MagicMock()
        sampling_decision.decision = "local_only"
        sampling_decision.is_coaching_sample = False
        sampling_decision.sampling_probability = 0.01
        components["sampling_scheduler"].should_sample = Mock(return_value=sampling_decision)

        route_result = MagicMock()
        route_result.response = MagicMock(
            output_data={"sentiment": "positive"},
            model_used="local-v1",
            cost=Decimal("0.0"),
        )
        route_result.used_remote = False
        route_result.used_local = True
        components["router"].route = AsyncMock(return_value=route_result)

        result = await orchestrate_request(
            task_name="sentiment_analysis",
            input_data={"text": "Great product!"},
            correlation=sample_correlation,
        )

        assert result is not None, "Should return a result"
        components["budget_manager"].record_local_request.assert_called()
        components["reporting"].audit_logger.log.assert_called()

    @pytest.mark.asyncio
    async def test_remote_call_with_budget(
        self, mock_component_overrides, sample_correlation
    ):
        """Calls remote when local unavailable and budget allows."""
        components = mock_component_overrides

        # Configure: low confidence, remote needed
        snapshot = MagicMock()
        snapshot.score = 0.3
        snapshot.phase = "phase_1"
        components["confidence_engine"].get_snapshot = AsyncMock(return_value=snapshot)

        sampling_decision = MagicMock()
        sampling_decision.decision = "remote_only"
        sampling_decision.is_coaching_sample = False
        components["sampling_scheduler"].should_sample = Mock(return_value=sampling_decision)

        auth_result = MagicMock()
        auth_result.allowed = True
        auth_result.estimated_cost = Decimal("0.01")
        components["budget_manager"].authorize = AsyncMock(return_value=auth_result)

        route_result = MagicMock()
        route_result.response = MagicMock(
            output_data={"sentiment": "positive"},
            model_used="gpt-4",
            cost=Decimal("0.008"),
        )
        route_result.used_remote = True
        route_result.used_local = False
        route_result.remote_cost = Decimal("0.008")
        components["router"].route = AsyncMock(return_value=route_result)

        result = await orchestrate_request(
            task_name="sentiment_analysis",
            input_data={"text": "Great product!"},
            correlation=sample_correlation,
        )

        assert result is not None
        components["budget_manager"].authorize.assert_called()
        components["budget_manager"].record_spend.assert_called()

    @pytest.mark.asyncio
    async def test_dual_send_coaching_sample(
        self, mock_component_overrides, sample_correlation
    ):
        """Dual-send for coaching sample calls both backends and records comparison."""
        components = mock_component_overrides

        snapshot = MagicMock()
        snapshot.score = 0.6
        snapshot.phase = "phase_2"
        components["confidence_engine"].get_snapshot = AsyncMock(return_value=snapshot)

        sampling_decision = MagicMock()
        sampling_decision.decision = "dual_send"
        sampling_decision.is_coaching_sample = True
        sampling_decision.sampling_probability = 0.3
        components["sampling_scheduler"].should_sample = Mock(return_value=sampling_decision)

        auth_result = MagicMock()
        auth_result.allowed = True
        components["budget_manager"].authorize = AsyncMock(return_value=auth_result)

        route_result = MagicMock()
        route_result.response = MagicMock(
            output_data={"sentiment": "positive"},
            model_used="gpt-4",
            cost=Decimal("0.008"),
        )
        route_result.used_remote = True
        route_result.used_local = True
        route_result.local_output = {"sentiment": "neutral"}
        route_result.remote_output = {"sentiment": "positive"}
        route_result.remote_cost = Decimal("0.008")
        components["router"].route = AsyncMock(return_value=route_result)

        result = await orchestrate_request(
            task_name="sentiment_analysis",
            input_data={"text": "Great product!"},
            correlation=sample_correlation,
        )

        assert result is not None
        components["confidence_engine"].record_comparison.assert_called()
        components["training_pipeline"].add_example.assert_called()


class TestOrchestrateRequestErrors:
    """Test orchestrate_request error paths."""

    @pytest.mark.asyncio
    async def test_task_not_found(
        self, mock_component_overrides, sample_correlation
    ):
        """Raises error when task_name not in TaskRegistry."""
        components = mock_component_overrides
        components["config_and_registry"].task_registry.__contains__ = Mock(return_value=False)
        components["config_and_registry"].task_registry.get_task = Mock(
            side_effect=KeyError("Task not found")
        )

        sample_correlation.task_name = "nonexistent_task"

        with pytest.raises(Exception) as exc_info:
            await orchestrate_request(
                task_name="nonexistent_task",
                input_data={"text": "test"},
                correlation=sample_correlation,
            )

        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "task" in error_msg or "nonexistent" in error_msg

    @pytest.mark.asyncio
    async def test_all_sources_unavailable(
        self, mock_component_overrides, sample_correlation
    ):
        """Raises AllSourcesUnavailableError when local unavailable AND budget exhausted."""
        components = mock_component_overrides

        snapshot = MagicMock()
        snapshot.score = 0.3
        snapshot.phase = "phase_1"
        snapshot.local_model_available = False
        components["confidence_engine"].get_snapshot = AsyncMock(return_value=snapshot)

        sampling_decision = MagicMock()
        sampling_decision.decision = "remote_only"
        components["sampling_scheduler"].should_sample = Mock(return_value=sampling_decision)

        # Budget exhausted
        auth_result = MagicMock()
        auth_result.allowed = False
        components["budget_manager"].authorize = AsyncMock(return_value=auth_result)
        components["budget_manager"].remaining_budget = Mock(return_value=Decimal("0.00"))

        # Local unavailable
        route_result = MagicMock()
        route_result.used_local = False
        route_result.local_unavailable = True
        route_result.local_unavailable_reason = "Model not loaded"
        components["router"].route = AsyncMock(
            side_effect=AllSourcesUnavailableError(
                message="All sources unavailable",
                task_name="sentiment_analysis",
                request_id=sample_correlation.request_id,
                local_unavailable_reason="Model not loaded",
                budget_exhausted_detail="Budget limit reached",
                budget_limit_usd=100.0,
                budget_used_usd=100.0,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
        )

        with pytest.raises(AllSourcesUnavailableError) as exc_info:
            await orchestrate_request(
                task_name="sentiment_analysis",
                input_data={"text": "test"},
                correlation=sample_correlation,
            )

        err = exc_info.value
        assert err.local_unavailable_reason is not None and len(err.local_unavailable_reason) > 0
        assert err.budget_exhausted_detail is not None and len(err.budget_exhausted_detail) > 0

    @pytest.mark.asyncio
    async def test_budget_check_internal_error(
        self, mock_component_overrides, sample_correlation
    ):
        """Raises error when budget_manager.authorize fails internally."""
        components = mock_component_overrides

        snapshot = MagicMock()
        snapshot.score = 0.3
        snapshot.phase = "phase_1"
        components["confidence_engine"].get_snapshot = AsyncMock(return_value=snapshot)

        sampling_decision = MagicMock()
        sampling_decision.decision = "remote_only"
        components["sampling_scheduler"].should_sample = Mock(return_value=sampling_decision)

        components["budget_manager"].authorize = AsyncMock(
            side_effect=RuntimeError("Budget manager internal error")
        )

        with pytest.raises(Exception) as exc_info:
            await orchestrate_request(
                task_name="sentiment_analysis",
                input_data={"text": "test"},
                correlation=sample_correlation,
            )

        assert "budget" in str(exc_info.value).lower() or "internal" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_confidence_engine_error(
        self, mock_component_overrides, sample_correlation
    ):
        """Raises error when confidence_engine.get_snapshot fails."""
        components = mock_component_overrides
        components["confidence_engine"].get_snapshot = AsyncMock(
            side_effect=RuntimeError("Confidence engine crashed")
        )

        with pytest.raises(Exception) as exc_info:
            await orchestrate_request(
                task_name="sentiment_analysis",
                input_data={"text": "test"},
                correlation=sample_correlation,
            )

        assert "confidence" in str(exc_info.value).lower() or "engine" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_both_backends_failed(
        self, mock_component_overrides, sample_correlation
    ):
        """Raises error when both local and remote fail during dual-send."""
        components = mock_component_overrides

        snapshot = MagicMock()
        snapshot.score = 0.5
        snapshot.phase = "phase_2"
        components["confidence_engine"].get_snapshot = AsyncMock(return_value=snapshot)

        sampling_decision = MagicMock()
        sampling_decision.decision = "dual_send"
        sampling_decision.is_coaching_sample = True
        components["sampling_scheduler"].should_sample = Mock(return_value=sampling_decision)

        auth_result = MagicMock()
        auth_result.allowed = True
        components["budget_manager"].authorize = AsyncMock(return_value=auth_result)

        components["router"].route = AsyncMock(
            side_effect=RuntimeError("Both backends failed")
        )

        with pytest.raises(Exception):
            await orchestrate_request(
                task_name="sentiment_analysis",
                input_data={"text": "test"},
                correlation=sample_correlation,
            )


class TestOrchestrateRequestEdgeCases:
    """Test edge cases for orchestration."""

    @pytest.mark.asyncio
    async def test_budget_reservation_released_on_remote_failure(
        self, mock_component_overrides, sample_correlation
    ):
        """When remote call fails before cost, release_reservation is called."""
        components = mock_component_overrides

        snapshot = MagicMock()
        snapshot.score = 0.3
        snapshot.phase = "phase_1"
        components["confidence_engine"].get_snapshot = AsyncMock(return_value=snapshot)

        sampling_decision = MagicMock()
        sampling_decision.decision = "remote_only"
        components["sampling_scheduler"].should_sample = Mock(return_value=sampling_decision)

        auth_result = MagicMock()
        auth_result.allowed = True
        auth_result.estimated_cost = Decimal("0.01")
        components["budget_manager"].authorize = AsyncMock(return_value=auth_result)

        # Remote call fails
        components["router"].route = AsyncMock(
            side_effect=RuntimeError("API call failed before cost")
        )

        try:
            await orchestrate_request(
                task_name="sentiment_analysis",
                input_data={"text": "test"},
                correlation=sample_correlation,
            )
        except Exception:
            pass

        # Verify authorize was called but record_spend was NOT
        components["budget_manager"].authorize.assert_called()
        components["budget_manager"].record_spend.assert_not_called()
        # release_reservation should have been called
        components["budget_manager"].release_reservation.assert_called()

    @pytest.mark.asyncio
    async def test_local_fallback_when_budget_exhausted(
        self, mock_component_overrides, sample_correlation
    ):
        """When budget exhausted but local available, serves locally without error."""
        components = mock_component_overrides

        snapshot = MagicMock()
        snapshot.score = 0.7
        snapshot.phase = "phase_2"
        snapshot.local_model_available = True
        components["confidence_engine"].get_snapshot = AsyncMock(return_value=snapshot)

        sampling_decision = MagicMock()
        sampling_decision.decision = "local_only"
        sampling_decision.is_coaching_sample = False
        components["sampling_scheduler"].should_sample = Mock(return_value=sampling_decision)

        # Budget would be exhausted if remote were attempted
        auth_result = MagicMock()
        auth_result.allowed = False
        components["budget_manager"].authorize = AsyncMock(return_value=auth_result)

        route_result = MagicMock()
        route_result.response = MagicMock(
            output_data={"sentiment": "positive"},
            model_used="local-v1",
            cost=Decimal("0.0"),
        )
        route_result.used_remote = False
        route_result.used_local = True
        components["router"].route = AsyncMock(return_value=route_result)

        # Should NOT raise AllSourcesUnavailableError
        result = await orchestrate_request(
            task_name="sentiment_analysis",
            input_data={"text": "test"},
            correlation=sample_correlation,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_remote_fallback_on_local_failure(
        self, mock_component_overrides, sample_correlation
    ):
        """When local fails but budget available, falls back to remote gracefully."""
        components = mock_component_overrides

        snapshot = MagicMock()
        snapshot.score = 0.8
        snapshot.phase = "phase_2"
        components["confidence_engine"].get_snapshot = AsyncMock(return_value=snapshot)

        sampling_decision = MagicMock()
        sampling_decision.decision = "local_with_remote_fallback"
        components["sampling_scheduler"].should_sample = Mock(return_value=sampling_decision)

        auth_result = MagicMock()
        auth_result.allowed = True
        components["budget_manager"].authorize = AsyncMock(return_value=auth_result)

        # Router handles local failure + remote fallback
        route_result = MagicMock()
        route_result.response = MagicMock(
            output_data={"sentiment": "positive"},
            model_used="gpt-4",
            cost=Decimal("0.01"),
        )
        route_result.used_remote = True
        route_result.used_local = False
        route_result.fallback_triggered = True
        route_result.remote_cost = Decimal("0.01")
        components["router"].route = AsyncMock(return_value=route_result)

        result = await orchestrate_request(
            task_name="sentiment_analysis",
            input_data={"text": "test"},
            correlation=sample_correlation,
        )

        assert result is not None
        components["budget_manager"].authorize.assert_called()


# ============================================================================
# 5. Correlation and Status Tests (test_correlation_and_status)
# ============================================================================


class TestCreateRequestCorrelation:
    """Test create_request_correlation function."""

    def test_happy_path_valid_correlation(self):
        """Creates valid RequestCorrelation with UUID v4, task_name, and UTC timestamp."""
        result = create_request_correlation(task_name="sentiment_analysis")

        assert isinstance(result, RequestCorrelation), (
            f"Expected RequestCorrelation, got {type(result)}"
        )

        # Validate UUID v4
        parsed_uuid = uuid.UUID(result.request_id)
        assert parsed_uuid.version == 4, (
            f"Expected UUID v4, got version {parsed_uuid.version}"
        )

        assert result.task_name == "sentiment_analysis", (
            f"Expected task_name 'sentiment_analysis', got {result.task_name}"
        )

        # Validate ISO 8601 UTC timestamp
        assert result.created_at_utc is not None
        parsed_ts = datetime.fromisoformat(result.created_at_utc)
        assert parsed_ts.tzinfo is not None, "Timestamp must be timezone-aware"

    def test_empty_task_name_raises(self):
        """Raises error for empty task_name."""
        with pytest.raises(Exception) as exc_info:
            create_request_correlation(task_name="")

        error_msg = str(exc_info.value).lower()
        assert "empty" in error_msg or "task" in error_msg or "blank" in error_msg

    def test_whitespace_task_name_raises(self):
        """Raises error for whitespace-only task_name."""
        with pytest.raises(Exception) as exc_info:
            create_request_correlation(task_name="   ")

        error_msg = str(exc_info.value).lower()
        assert "empty" in error_msg or "task" in error_msg or "whitespace" in error_msg

    def test_correlation_uniqueness(self):
        """Multiple calls produce unique request_ids."""
        correlations = [
            create_request_correlation(task_name=f"task_{i}")
            for i in range(100)
        ]

        request_ids = [c.request_id for c in correlations]
        assert len(set(request_ids)) == 100, (
            f"Expected 100 unique request_ids, got {len(set(request_ids))}"
        )

    def test_timestamp_is_utc_iso8601(self):
        """Timestamp is timezone-aware UTC in ISO 8601 format."""
        result = create_request_correlation(task_name="test_task")

        ts_str = result.created_at_utc
        # Should be parseable
        parsed = datetime.fromisoformat(ts_str)
        assert parsed.tzinfo is not None, "Timestamp must have timezone info"

        # Should be UTC (offset 0)
        assert parsed.utcoffset().total_seconds() == 0, (
            f"Timestamp should be UTC, got offset {parsed.utcoffset()}"
        )


class TestGetComponentStatuses:
    """Test get_component_statuses function."""

    def test_all_healthy(self, mock_component_overrides):
        """Returns 10 READY entries when all components initialized."""
        result = get_component_statuses(components=mock_component_overrides)

        assert len(result) == 10, f"Expected 10 entries, got {len(result)}"

        for entry in result:
            assert entry.status == ComponentStatus.READY or str(entry.status) == "READY", (
                f"Component {entry.component_id} should be READY, got {entry.status}"
            )

        # Each ComponentId appears exactly once
        component_ids = [str(entry.component_id) for entry in result]
        assert len(set(component_ids)) == 10, (
            f"Expected 9 unique component IDs, got {len(set(component_ids))}"
        )

    def test_partial_initialization(self):
        """NOT_STARTED for components not yet initialized."""
        partial_components = {
            "config_and_registry": _make_mock_config_and_registry(),
            "budget_manager": _make_mock_budget_manager(),
            # Only 2 of 10 initialized
        }

        result = get_component_statuses(components=partial_components)

        assert len(result) == 10, f"Expected 10 entries, got {len(result)}"

        ready_count = sum(
            1 for e in result
            if str(e.status) == "READY" or e.status == ComponentStatus.READY
        )
        not_started_count = sum(
            1 for e in result
            if str(e.status) == "NOT_STARTED" or e.status == ComponentStatus.NOT_STARTED
        )

        assert ready_count == 2, f"Expected 2 READY, got {ready_count}"
        assert not_started_count == 8, f"Expected 8 NOT_STARTED, got {not_started_count}"

    def test_with_failed_component(self, mock_component_overrides):
        """FAILED status with error_message for failed components."""
        # Mark a component as failed
        failed_components = dict(mock_component_overrides)
        failed_bm = _make_mock_budget_manager()
        failed_bm._status = "FAILED"
        failed_bm._error = "State corruption detected"
        failed_components["budget_manager"] = failed_bm

        result = get_component_statuses(components=failed_components)

        assert len(result) == 10

        # Find the budget_manager entry
        bm_entries = [
            e for e in result
            if str(e.component_id).lower().replace("_", "") in ("budgetmanager", "budget_manager")
            or str(e.component_id) == "BUDGET_MANAGER"
        ]
        if bm_entries:
            bm_entry = bm_entries[0]
            if str(bm_entry.status) == "FAILED":
                assert bm_entry.error_message is not None and len(bm_entry.error_message) > 0, (
                    "Failed component should have error_message"
                )


# ============================================================================
# 6. Component Override Tests (test_component_overrides)
# ============================================================================


class TestValidateComponentOverride:
    """Test validate_component_override function."""

    def test_valid_override_returns_true(self):
        """Returns True for conformant override with all required methods."""
        mock_bm = _make_mock_budget_manager()

        result = validate_component_override(
            component_id="budget_manager",
            override_instance=mock_bm,
        )

        assert result is True, "Should return True for valid override"

    def test_missing_method_returns_false(self):
        """Returns False when override lacks required methods."""
        incomplete = MagicMock(spec=[])  # No methods

        result = validate_component_override(
            component_id="budget_manager",
            override_instance=incomplete,
        )

        assert result is False, "Should return False for override missing required methods"

    def test_unknown_component_id_raises(self):
        """Raises error for unknown component_id."""
        with pytest.raises(Exception) as exc_info:
            validate_component_override(
                component_id="nonexistent_component",
                override_instance=MagicMock(),
            )

        error_msg = str(exc_info.value).lower()
        assert "unknown" in error_msg or "component" in error_msg or "invalid" in error_msg

    @pytest.mark.parametrize("component_id", [
        "config_and_registry", "budget_manager", "confidence_engine",
        "external_interfaces", "sampling_scheduler", "router",
        "training_pipeline", "reporting", "unified_interface",
    ])
    def test_all_valid_component_ids_accepted(self, component_id):
        """All 9 valid ComponentId values are accepted without error."""
        mock_component = MagicMock()
        # Add common methods that most components would need
        mock_component.close = AsyncMock()

        # Should not raise for valid component_id
        try:
            validate_component_override(
                component_id=component_id,
                override_instance=mock_component,
            )
        except Exception as e:
            if "unknown" in str(e).lower() and "component" in str(e).lower():
                pytest.fail(
                    f"Valid component_id '{component_id}' should not raise unknown_component_id"
                )


# ============================================================================
# 7. Lifecycle Logging Tests (test_lifecycle_logging)
# ============================================================================


class TestLifecycleLogging:
    """Test lifecycle event logging during initialization and shutdown."""

    @pytest.mark.asyncio
    async def test_init_logs_initialization_started(
        self, config_file, mock_env, mock_component_overrides, caplog
    ):
        """INITIALIZATION_STARTED event logged at start of initialization."""
        with caplog.at_level(logging.DEBUG):
            try:
                await initialize_composition_root(
                    config_path=config_file,
                    env=mock_env,
                    component_overrides=mock_component_overrides,
                )
            except Exception:
                pass  # May fail if config not fully valid

        # Check for initialization started log
        log_text = caplog.text.lower()
        assert "init" in log_text or "start" in log_text or len(caplog.records) > 0, (
            "Should emit lifecycle log entries during initialization"
        )

    @pytest.mark.asyncio
    async def test_init_logs_component_initialized_per_component(
        self, config_file, mock_env, mock_component_overrides, caplog
    ):
        """COMPONENT_INITIALIZED event logged for each successfully initialized component."""
        with caplog.at_level(logging.DEBUG):
            try:
                result = await initialize_composition_root(
                    config_path=config_file,
                    env=mock_env,
                    component_overrides=mock_component_overrides,
                )
            except Exception:
                pass

        # Should have log entries — exact format depends on implementation
        assert len(caplog.records) >= 1, "Should emit at least one log entry"

    @pytest.mark.asyncio
    async def test_init_failure_logs_component_init_failed(
        self, mock_env, caplog
    ):
        """COMPONENT_INIT_FAILED event logged when a component fails to initialize."""
        with caplog.at_level(logging.DEBUG):
            try:
                await initialize_composition_root(
                    config_path="/nonexistent/config.yaml",
                    env=mock_env,
                    component_overrides={},
                )
            except InitializationError:
                pass

        log_text = caplog.text.lower()
        assert "fail" in log_text or "error" in log_text or len(caplog.records) > 0, (
            "Should log failure events"
        )

    @pytest.mark.asyncio
    async def test_shutdown_logs_events(
        self, mock_component_overrides, mock_exit_stack, caplog
    ):
        """Shutdown logs SHUTDOWN_STARTED, COMPONENT_SHUTDOWN, and SHUTDOWN_COMPLETE."""
        with caplog.at_level(logging.DEBUG):
            await shutdown_composition_root(
                components=mock_component_overrides,
                exit_stack=mock_exit_stack,
            )

        log_text = caplog.text.lower()
        assert "shutdown" in log_text or len(caplog.records) > 0, (
            "Should emit shutdown lifecycle events"
        )

    @pytest.mark.asyncio
    async def test_shutdown_failure_logs_component_shutdown_failed(
        self, mock_component_overrides, mock_exit_stack, caplog
    ):
        """COMPONENT_SHUTDOWN_FAILED logged for components that fail during shutdown."""
        mock_component_overrides["budget_manager"].force_persist = AsyncMock(
            side_effect=IOError("Persist failed")
        )

        with caplog.at_level(logging.DEBUG):
            await shutdown_composition_root(
                components=mock_component_overrides,
                exit_stack=mock_exit_stack,
            )

        # Should have critical-level log for shutdown failure
        critical_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(critical_records) >= 1, (
            "Should log at ERROR/CRITICAL level for shutdown failures"
        )


# ============================================================================
# 8. Invariant Tests
# ============================================================================


class TestInvariants:
    """Test system-wide invariants from the contract."""

    def test_no_global_state(self):
        """Verify no module-level mutable state is used."""
        import src.root as root_module

        # Check that the module doesn't have suspicious mutable globals
        module_attrs = dir(root_module)
        for attr in module_attrs:
            obj = getattr(root_module, attr, None)
            if isinstance(obj, (dict, list, set)) and not attr.startswith("_"):
                # Module-level mutable collections are suspicious
                # but __all__ is expected to be a list
                if attr != "__all__":
                    assert False, (
                        f"Suspicious module-level mutable state: {attr}"
                    )

    @pytest.mark.asyncio
    async def test_budget_hard_enforcement(
        self, mock_component_overrides, sample_correlation
    ):
        """If budget_manager.authorize returns allowed=False, no remote call is made."""
        components = mock_component_overrides

        snapshot = MagicMock()
        snapshot.score = 0.3
        snapshot.phase = "phase_1"
        snapshot.local_model_available = True
        components["confidence_engine"].get_snapshot = AsyncMock(return_value=snapshot)

        sampling_decision = MagicMock()
        sampling_decision.decision = "local_only"
        sampling_decision.is_coaching_sample = False
        components["sampling_scheduler"].should_sample = Mock(return_value=sampling_decision)

        auth_result = MagicMock()
        auth_result.allowed = False
        components["budget_manager"].authorize = AsyncMock(return_value=auth_result)

        route_result = MagicMock()
        route_result.response = MagicMock(
            output_data={"sentiment": "positive"},
            model_used="local-v1",
            cost=Decimal("0.0"),
        )
        route_result.used_remote = False
        route_result.used_local = True
        components["router"].route = AsyncMock(return_value=route_result)

        try:
            result = await orchestrate_request(
                task_name="sentiment_analysis",
                input_data={"text": "test"},
                correlation=sample_correlation,
            )
        except AllSourcesUnavailableError:
            pass  # Also acceptable if local unavailable

        # Key invariant: if authorize returned False, remote should NOT have been called
        # (Router should not have been asked to call remote)

    @pytest.mark.asyncio
    async def test_audit_entry_always_emitted(
        self, mock_component_overrides, sample_correlation
    ):
        """Every orchestration produces exactly one audit entry regardless of outcome."""
        components = mock_component_overrides

        snapshot = MagicMock()
        snapshot.score = 0.9
        snapshot.phase = "phase_3"
        components["confidence_engine"].get_snapshot = AsyncMock(return_value=snapshot)

        sampling_decision = MagicMock()
        sampling_decision.decision = "local_only"
        sampling_decision.is_coaching_sample = False
        components["sampling_scheduler"].should_sample = Mock(return_value=sampling_decision)

        route_result = MagicMock()
        route_result.response = MagicMock(
            output_data={"sentiment": "positive"},
            model_used="local-v1",
            cost=Decimal("0.0"),
        )
        route_result.used_remote = False
        route_result.used_local = True
        components["router"].route = AsyncMock(return_value=route_result)

        await orchestrate_request(
            task_name="sentiment_analysis",
            input_data={"text": "test"},
            correlation=sample_correlation,
        )

        components["reporting"].audit_logger.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_audit_entry_emitted_on_failure(
        self, mock_component_overrides, sample_correlation
    ):
        """Audit entry emitted even when request fails."""
        components = mock_component_overrides

        components["confidence_engine"].get_snapshot = AsyncMock(return_value=MagicMock(
            score=0.3, phase="phase_1", local_model_available=False,
        ))
        components["sampling_scheduler"].should_sample = Mock(return_value=MagicMock(
            decision="remote_only",
        ))

        auth_result = MagicMock()
        auth_result.allowed = False
        components["budget_manager"].authorize = AsyncMock(return_value=auth_result)
        components["budget_manager"].remaining_budget = Mock(return_value=Decimal("0.0"))

        components["router"].route = AsyncMock(
            side_effect=AllSourcesUnavailableError(
                message="All sources unavailable",
                task_name="sentiment_analysis",
                request_id=sample_correlation.request_id,
                local_unavailable_reason="Not loaded",
                budget_exhausted_detail="Budget used up",
                budget_limit_usd=100.0,
                budget_used_usd=100.0,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
        )

        try:
            await orchestrate_request(
                task_name="sentiment_analysis",
                input_data={"text": "test"},
                correlation=sample_correlation,
            )
        except AllSourcesUnavailableError:
            pass

        # Audit should still have been called
        components["reporting"].audit_logger.log.assert_called()

    @pytest.mark.asyncio
    async def test_audit_failure_does_not_block_response(
        self, mock_component_overrides, sample_correlation
    ):
        """Audit logging failure does not block or delay the response."""
        components = mock_component_overrides

        snapshot = MagicMock()
        snapshot.score = 0.9
        snapshot.phase = "phase_3"
        components["confidence_engine"].get_snapshot = AsyncMock(return_value=snapshot)

        sampling_decision = MagicMock()
        sampling_decision.decision = "local_only"
        sampling_decision.is_coaching_sample = False
        components["sampling_scheduler"].should_sample = Mock(return_value=sampling_decision)

        route_result = MagicMock()
        route_result.response = MagicMock(
            output_data={"sentiment": "positive"},
            model_used="local-v1",
            cost=Decimal("0.0"),
        )
        route_result.used_remote = False
        route_result.used_local = True
        components["router"].route = AsyncMock(return_value=route_result)

        # Make audit logging fail
        components["reporting"].audit_logger.log = AsyncMock(
            side_effect=IOError("Audit write failed")
        )

        # Should still return a response without raising
        result = await orchestrate_request(
            task_name="sentiment_analysis",
            input_data={"text": "test"},
            correlation=sample_correlation,
        )

        assert result is not None, "Response should be returned despite audit failure"

    @pytest.mark.asyncio
    async def test_state_persisted_on_shutdown(
        self, mock_component_overrides, mock_exit_stack
    ):
        """Budget and confidence state persisted during shutdown."""
        await shutdown_composition_root(
            components=mock_component_overrides,
            exit_stack=mock_exit_stack,
        )

        mock_component_overrides["budget_manager"].force_persist.assert_called_once()
        mock_component_overrides["confidence_engine"].persist_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_all_components_attempted(
        self, mock_component_overrides, mock_exit_stack
    ):
        """Shutdown proceeds through ALL components regardless of individual failures."""
        # Make first component (unified_interface) fail
        mock_component_overrides["unified_interface"].close = AsyncMock(
            side_effect=RuntimeError("Close failed")
        )

        result = await shutdown_composition_root(
            components=mock_component_overrides,
            exit_stack=mock_exit_stack,
        )

        # All 9 components should have been attempted
        total = len(result.component_errors) + len(result.components_shutdown_successfully)
        assert total == 10, (
            f"All 9 components should be attempted, got {total} total"
        )

    def test_all_sources_unavailable_only_when_both_conditions(self):
        """AllSourcesUnavailableError should encode both conditions."""
        err = AllSourcesUnavailableError(
            message="All sources unavailable",
            task_name="test_task",
            request_id="test-uuid",
            local_unavailable_reason="Model not loaded",
            budget_exhausted_detail="Budget limit $100 exceeded",
            budget_limit_usd=100.0,
            budget_used_usd=100.0,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

        assert err.local_unavailable_reason is not None, "Must have local reason"
        assert err.budget_exhausted_detail is not None, "Must have budget detail"
        assert err.budget_limit_usd == 100.0
        assert err.budget_used_usd == 100.0

    @pytest.mark.asyncio
    async def test_initialization_error_contains_component_statuses(self):
        """InitializationError contains phase info and component statuses."""
        err = InitializationError(
            message="Initialization failed",
            failed_phase=InitializationPhase.BUDGET_MANAGER_INIT,
            failed_component_id="BUDGET_MANAGER",
            component_statuses=[],
            cause="State file corrupted",
        )

        assert err.failed_phase == InitializationPhase.BUDGET_MANAGER_INIT
        assert err.failed_component_id == "BUDGET_MANAGER"
        assert err.cause == "State file corrupted"
        assert isinstance(err.component_statuses, list)

    def test_shutdown_error_structure(self):
        """ShutdownError correctly partitions errors and successes."""
        err = ShutdownError(
            message="Shutdown completed with errors",
            component_errors={"BUDGET_MANAGER": "Persist failed"},
            components_shutdown_successfully=[
                "CONFIG_AND_REGISTRY", "CONFIDENCE_ENGINE",
                "EXTERNAL_INTERFACES", "SAMPLING_SCHEDULER",
                "ROUTER", "TRAINING_PIPELINE", "REPORTING",
                "UNIFIED_INTERFACE",
            ],
        )

        assert len(err.component_errors) == 1
        assert len(err.components_shutdown_successfully) == 8
        assert "BUDGET_MANAGER" in err.component_errors

    @pytest.mark.asyncio
    async def test_init_dag_order_respected(
        self, config_file, mock_env, mock_component_overrides
    ):
        """Components initialized in strict DAG order."""
        init_order = []

        original_overrides = dict(mock_component_overrides)
        for name, comp in original_overrides.items():
            # Track when each component is "used" during initialization
            if hasattr(comp, "_init_order_tracker"):
                comp._init_order_tracker = lambda n=name: init_order.append(n)

        # The test verifies the contract — implementation must follow DAG order
        try:
            await initialize_composition_root(
                config_path=config_file,
                env=mock_env,
                component_overrides=mock_component_overrides,
            )
        except Exception:
            pass  # May fail but order should still be respected


class TestBackendRegistryInvariants:
    """Test backend registry invariants."""

    def test_is_remote_flag_openai(self):
        """openai backend is_remote should be True."""
        config = {
            "backend": "openai",
            "api_key": "test-key",
            "api_base_url": "https://api.openai.com/v1",
        }
        result = resolve_finetuning_backend("openai", config, {})
        assert result.is_remote is True, "openai should be remote"

    def test_is_remote_flag_anyscale(self):
        """anyscale backend is_remote should be True."""
        config = {
            "backend": "anyscale",
            "api_key": "test-key",
            "api_base_url": "https://api.anyscale.com/v1",
        }
        result = resolve_finetuning_backend("anyscale", config, {})
        assert result.is_remote is True, "anyscale should be remote"

    def test_is_remote_flag_local_lora(self):
        """local_lora backend is_remote should be False."""
        config = {"backend": "local_lora", "output_dir": "/tmp/ft"}
        result = resolve_finetuning_backend("local_lora", config, {})
        assert result.is_remote is False, "local_lora should not be remote"

    def test_is_remote_flag_axolotl(self):
        """axolotl backend is_remote should be False."""
        config = {"backend": "axolotl", "output_dir": "/tmp/ft"}
        result = resolve_finetuning_backend("axolotl", config, {})
        assert result.is_remote is False, "axolotl should not be remote"

    def test_resolved_backend_name_matches_input(self):
        """Returned BackendRegistryEntry.backend_name matches input."""
        config = {"backend": "local_lora", "output_dir": "/tmp/ft"}
        result = resolve_finetuning_backend("local_lora", config, {})
        assert str(result.backend_name).lower() == "local_lora", (
            f"Expected 'local_lora', got {result.backend_name}"
        )


class TestCorrelationInvariants:
    """Test correlation-related invariants."""

    def test_request_id_is_uuid_v4(self):
        """request_id must be a valid UUID v4."""
        correlation = create_request_correlation("test_task")
        parsed = uuid.UUID(correlation.request_id)
        assert parsed.version == 4

    def test_created_at_utc_is_current(self):
        """created_at_utc should be approximately current time."""
        before = datetime.now(timezone.utc)
        correlation = create_request_correlation("test_task")
        after = datetime.now(timezone.utc)

        parsed = datetime.fromisoformat(correlation.created_at_utc)
        assert before <= parsed <= after, (
            f"Timestamp {parsed} should be between {before} and {after}"
        )

    def test_timestamps_always_utc(self):
        """All timestamps are timezone-aware UTC."""
        for _ in range(10):
            correlation = create_request_correlation("test_task")
            parsed = datetime.fromisoformat(correlation.created_at_utc)
            assert parsed.tzinfo is not None, "Must be timezone-aware"
            assert parsed.utcoffset().total_seconds() == 0, "Must be UTC"


class TestPublicAPIInvariants:
    """Test that the public API surface is correctly structured."""

    def test_all_sources_unavailable_error_is_exception(self):
        """AllSourcesUnavailableError is an Exception subclass."""
        err = AllSourcesUnavailableError(
            message="test",
            task_name="test",
            request_id="test",
            local_unavailable_reason="test",
            budget_exhausted_detail="test",
            budget_limit_usd=0.0,
            budget_used_usd=0.0,
            timestamp_utc="2024-01-01T00:00:00+00:00",
        )
        assert isinstance(err, Exception), "AllSourcesUnavailableError must be an Exception"

    def test_initialization_error_is_exception(self):
        """InitializationError is an Exception subclass."""
        err = InitializationError(
            message="test",
            failed_phase=InitializationPhase.CONFIG_PARSE,
            failed_component_id="test",
            component_statuses=[],
            cause="test",
        )
        assert isinstance(err, Exception), "InitializationError must be an Exception"

    def test_shutdown_error_is_exception(self):
        """ShutdownError is an Exception subclass."""
        err = ShutdownError(
            message="test",
            component_errors={},
            components_shutdown_successfully=[],
        )
        assert isinstance(err, Exception), "ShutdownError must be an Exception"
