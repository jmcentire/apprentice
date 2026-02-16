"""
Contract test suite for RequestRouter component.
Tests are organized into sections covering:
  - compute_routing_action (exhaustive decision table)
  - route() happy paths
  - route() error paths
  - route() fallback/degradation scenarios
  - execute_dual_send
  - build_audit_entry
  - RouterConfig validation
  - Invariants

Run with: pytest contract_test.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime, timezone

from src.router import (
    RequestRouter,
    TaskRequest,
    RoutingAction,
    Phase,
    ModelResponse,
    BudgetSnapshot,
    PhaseContext,
    RoutingDecision,
    RoutingResult,
    RoutingAuditEntry,
    DualSendOutcome,
    TrainingExampleType,
    TrainingExample,
    RouterConfig,
    compute_routing_action,
    build_audit_entry,
)


# ============================================================================
# Factory Helpers
# ============================================================================

def make_task_request(**overrides):
    """Create a valid TaskRequest with sensible defaults."""
    defaults = {
        "request_id": "req-test-001",
        "task_type": "summarize",
        "prompt": "Summarize this document about AI routing.",
        "metadata": {"source": "test"},
        "timestamp_utc": "2024-01-15T10:30:00Z",
    }
    defaults.update(overrides)
    return TaskRequest(**defaults)


def make_budget_snapshot(exhausted=False, **overrides):
    """Create a BudgetSnapshot."""
    if exhausted:
        defaults = {
            "total_budget_usd": 100.0,
            "spent_usd": 100.0,
            "remaining_usd": 0.0,
            "is_exhausted": True,
        }
    else:
        defaults = {
            "total_budget_usd": 100.0,
            "spent_usd": 25.0,
            "remaining_usd": 75.0,
            "is_exhausted": False,
        }
    defaults.update(overrides)
    return BudgetSnapshot(**defaults)


def make_phase_context(phase=Phase.PHASE_1, confidence=0.2):
    return PhaseContext(phase=phase, confidence=confidence)


def make_model_response(content="Test response content", model_id="test-model",
                        latency_ms=150.0, cost=0.01, token_count=50):
    return ModelResponse(
        content=content,
        model_id=model_id,
        latency_ms=latency_ms,
        cost=cost,
        token_count=token_count,
    )


def make_local_response(**overrides):
    defaults = {"content": "Local model response", "model_id": "local-llama-7b",
                "latency_ms": 50.0, "cost": 0.0, "token_count": 40}
    defaults.update(overrides)
    return make_model_response(**defaults)


def make_remote_response(**overrides):
    defaults = {"content": "Remote model response", "model_id": "gpt-4",
                "latency_ms": 200.0, "cost": 0.03, "token_count": 60}
    defaults.update(overrides)
    return make_model_response(**defaults)


def make_router_config(**overrides):
    defaults = {
        "max_concurrent_dual_sends": 10,
        "remote_call_timeout_ms": 5000,
        "local_call_timeout_ms": 2000,
        "enable_training_collection": True,
        "enable_audit_logging": True,
    }
    defaults.update(overrides)
    return RouterConfig(**defaults)


# ============================================================================
# Mock Dependency Fixtures
# ============================================================================

@pytest.fixture
def mock_budget_manager():
    mgr = MagicMock()
    mgr.check_budget = AsyncMock(return_value=make_budget_snapshot(exhausted=False))
    mgr.record_spend = AsyncMock()
    return mgr


@pytest.fixture
def mock_budget_manager_exhausted():
    mgr = MagicMock()
    mgr.check_budget = AsyncMock(return_value=make_budget_snapshot(exhausted=True))
    mgr.record_spend = AsyncMock()
    return mgr


@pytest.fixture
def mock_confidence_engine():
    engine = MagicMock()
    engine.get_phase_context = AsyncMock(
        return_value=make_phase_context(Phase.PHASE_1, 0.2)
    )
    return engine


@pytest.fixture
def mock_sampling_scheduler():
    scheduler = MagicMock()
    scheduler.should_coach = AsyncMock(return_value=False)
    return scheduler


@pytest.fixture
def mock_local_backend():
    backend = MagicMock()
    backend.call = AsyncMock(return_value=make_local_response())
    return backend


@pytest.fixture
def mock_remote_backend():
    backend = MagicMock()
    backend.call = AsyncMock(return_value=make_remote_response())
    return backend


@pytest.fixture
def mock_training_collector():
    collector = MagicMock()
    collector.collect = AsyncMock()
    return collector


@pytest.fixture
def mock_audit_logger():
    logger = MagicMock()
    logger.log = AsyncMock()
    return logger


@pytest.fixture
def make_router(mock_budget_manager, mock_confidence_engine, mock_sampling_scheduler,
                mock_local_backend, mock_remote_backend, mock_training_collector,
                mock_audit_logger):
    """Factory fixture that creates a RequestRouter with all mocks injected."""
    def _make(**overrides):
        config = make_router_config()
        deps = {
            "config": config,
            "budget_manager": mock_budget_manager,
            "confidence_engine": mock_confidence_engine,
            "sampling_scheduler": mock_sampling_scheduler,
            "local_model_backend": mock_local_backend,
            "remote_model_backend": mock_remote_backend,
            "training_data_collector": mock_training_collector,
            "audit_logger": mock_audit_logger,
        }
        deps.update(overrides)
        return RequestRouter(**deps)
    return _make


# ============================================================================
# 1. test_compute_routing_action — Exhaustive Decision Table
# ============================================================================

class TestComputeRoutingAction:
    """Exhaustive tests for the pure decision matrix function."""

    @pytest.mark.parametrize(
        "phase, is_coaching, budget_avail, expected",
        [
            # Phase 1: budget → REMOTE_ONLY; no budget → LOCAL_ONLY
            (Phase.PHASE_1, False, True, RoutingAction.REMOTE_ONLY),
            (Phase.PHASE_1, True, True, RoutingAction.REMOTE_ONLY),
            (Phase.PHASE_1, False, False, RoutingAction.LOCAL_ONLY),
            (Phase.PHASE_1, True, False, RoutingAction.LOCAL_ONLY),
            # Phase 2: budget → BOTH_RETURN_REMOTE; no budget → LOCAL_ONLY
            (Phase.PHASE_2, False, True, RoutingAction.BOTH_RETURN_REMOTE),
            (Phase.PHASE_2, True, True, RoutingAction.BOTH_RETURN_REMOTE),
            (Phase.PHASE_2, False, False, RoutingAction.LOCAL_ONLY),
            (Phase.PHASE_2, True, False, RoutingAction.LOCAL_ONLY),
            # Phase 3: coaching+budget → BOTH_RETURN_LOCAL; not coaching → LOCAL_ONLY
            (Phase.PHASE_3, True, True, RoutingAction.BOTH_RETURN_LOCAL),
            (Phase.PHASE_3, False, True, RoutingAction.LOCAL_ONLY),
            (Phase.PHASE_3, True, False, RoutingAction.LOCAL_ONLY),
            (Phase.PHASE_3, False, False, RoutingAction.LOCAL_ONLY),
        ],
        ids=[
            "P1-nocoach-budget",
            "P1-coach-budget",
            "P1-nocoach-nobudget",
            "P1-coach-nobudget",
            "P2-nocoach-budget",
            "P2-coach-budget",
            "P2-nocoach-nobudget",
            "P2-coach-nobudget",
            "P3-coach-budget",
            "P3-nocoach-budget",
            "P3-coach-nobudget",
            "P3-nocoach-nobudget",
        ],
    )
    def test_decision_matrix(self, phase, is_coaching, budget_avail, expected):
        """Verify each cell of the phase × sampling × budget matrix."""
        result = compute_routing_action(phase, is_coaching, budget_avail)
        assert result == expected, (
            f"compute_routing_action({phase}, {is_coaching}, {budget_avail}) "
            f"returned {result}, expected {expected}"
        )

    @pytest.mark.parametrize("phase", [Phase.PHASE_1, Phase.PHASE_2, Phase.PHASE_3])
    @pytest.mark.parametrize("is_coaching", [True, False])
    def test_no_budget_never_returns_remote_action(self, phase, is_coaching):
        """Invariant: When budget is exhausted, result never includes remote-primary actions."""
        result = compute_routing_action(phase, is_coaching, budget_available=False)
        remote_actions = {
            RoutingAction.REMOTE_ONLY,
            RoutingAction.BOTH_RETURN_REMOTE,
            RoutingAction.BOTH_RETURN_LOCAL,
        }
        assert result not in remote_actions, (
            f"Budget exhausted but got remote action {result} for "
            f"phase={phase}, is_coaching={is_coaching}"
        )

    def test_phase3_no_coaching_ignores_budget(self):
        """Phase 3 without coaching always returns LOCAL_ONLY regardless of budget."""
        result_with_budget = compute_routing_action(Phase.PHASE_3, False, True)
        result_no_budget = compute_routing_action(Phase.PHASE_3, False, False)
        assert result_with_budget == RoutingAction.LOCAL_ONLY
        assert result_no_budget == RoutingAction.LOCAL_ONLY

    def test_deterministic_same_inputs_same_output(self):
        """Same inputs always produce the same output."""
        for _ in range(10):
            r1 = compute_routing_action(Phase.PHASE_2, True, True)
            r2 = compute_routing_action(Phase.PHASE_2, True, True)
            assert r1 == r2, "compute_routing_action is not deterministic"

    def test_return_is_valid_routing_action(self):
        """Return value is always a valid RoutingAction enum member."""
        for phase in Phase:
            for coaching in [True, False]:
                for budget in [True, False]:
                    result = compute_routing_action(phase, coaching, budget)
                    assert isinstance(result, RoutingAction), (
                        f"Expected RoutingAction, got {type(result)}"
                    )


# ============================================================================
# 2. test_route — Happy Path Integration
# ============================================================================

class TestRouteHappyPath:
    """Happy-path tests for route() across all major phases."""

    @pytest.mark.asyncio
    async def test_phase1_remote_only(self, make_router, mock_confidence_engine,
                                       mock_budget_manager, mock_remote_backend,
                                       mock_audit_logger, mock_training_collector,
                                       mock_sampling_scheduler, mock_local_backend):
        """Phase 1 with budget: calls remote only, returns remote response."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_1, 0.15)
        )
        mock_sampling_scheduler.should_coach = AsyncMock(return_value=False)
        remote_resp = make_remote_response(content="Remote phase1 answer")
        mock_remote_backend.call = AsyncMock(return_value=remote_resp)

        router = make_router()
        request = make_task_request()
        result = await router.route(request)

        assert result.request_id == request.request_id, "request_id must match"
        assert result.response.content == "Remote phase1 answer", "Should return remote content"
        assert result.intended_action == RoutingAction.REMOTE_ONLY
        assert result.actual_action == RoutingAction.REMOTE_ONLY
        assert result.is_fallback is False
        assert result.is_degraded is False
        assert result.phase == Phase.PHASE_1
        assert result.confidence == 0.15
        mock_budget_manager.record_spend.assert_called_once()
        mock_audit_logger.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_phase2_coaching_dual_send(self, make_router, mock_confidence_engine,
                                              mock_budget_manager, mock_sampling_scheduler,
                                              mock_local_backend, mock_remote_backend,
                                              mock_training_collector, mock_audit_logger):
        """Phase 2 coaching sample: dual-send, returns remote response, collects pairs."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_2, 0.6)
        )
        mock_sampling_scheduler.should_coach = AsyncMock(return_value=True)
        local_resp = make_local_response(content="Local p2 answer")
        remote_resp = make_remote_response(content="Remote p2 answer")
        mock_local_backend.call = AsyncMock(return_value=local_resp)
        mock_remote_backend.call = AsyncMock(return_value=remote_resp)

        router = make_router()
        request = make_task_request()
        result = await router.route(request)

        assert result.request_id == request.request_id
        assert result.response.content == "Remote p2 answer", (
            "Phase 2 BOTH_RETURN_REMOTE should return remote response"
        )
        assert result.intended_action == RoutingAction.BOTH_RETURN_REMOTE
        assert result.is_coaching_sample is True
        assert result.phase == Phase.PHASE_2

        # Verify training data collected: at least COACHING_EVALUATION_PAIR
        assert mock_training_collector.collect.called, (
            "Training collector should be called for coaching dual-send"
        )
        # Check that at least one call contains the coaching pair type
        collect_calls = mock_training_collector.collect.call_args_list
        collected_types = []
        for c in collect_calls:
            # The first positional arg or keyword arg should be a TrainingExample
            arg = c[0][0] if c[0] else c[1].get("example")
            if hasattr(arg, "example_type"):
                collected_types.append(arg.example_type)
        assert TrainingExampleType.COACHING_EVALUATION_PAIR in collected_types, (
            f"Expected COACHING_EVALUATION_PAIR in {collected_types}"
        )

    @pytest.mark.asyncio
    async def test_phase3_local_only(self, make_router, mock_confidence_engine,
                                      mock_budget_manager, mock_sampling_scheduler,
                                      mock_local_backend, mock_remote_backend,
                                      mock_audit_logger):
        """Phase 3 without coaching: local only, remote never called."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_3, 0.95)
        )
        mock_sampling_scheduler.should_coach = AsyncMock(return_value=False)
        local_resp = make_local_response(content="Local autonomous answer")
        mock_local_backend.call = AsyncMock(return_value=local_resp)

        router = make_router()
        request = make_task_request()
        result = await router.route(request)

        assert result.request_id == request.request_id
        assert result.response.content == "Local autonomous answer"
        assert result.intended_action == RoutingAction.LOCAL_ONLY
        assert result.actual_action == RoutingAction.LOCAL_ONLY
        assert result.is_fallback is False
        mock_remote_backend.call.assert_not_called(), (
            "Remote backend should NOT be called in Phase 3 local-only"
        )

    @pytest.mark.asyncio
    async def test_phase3_coaching_dual_returns_local(self, make_router,
                                                       mock_confidence_engine,
                                                       mock_budget_manager,
                                                       mock_sampling_scheduler,
                                                       mock_local_backend,
                                                       mock_remote_backend,
                                                       mock_training_collector,
                                                       mock_audit_logger):
        """Phase 3 coaching: dual-send, returns LOCAL response."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_3, 0.92)
        )
        mock_sampling_scheduler.should_coach = AsyncMock(return_value=True)
        local_resp = make_local_response(content="Local p3 coaching")
        remote_resp = make_remote_response(content="Remote p3 coaching")
        mock_local_backend.call = AsyncMock(return_value=local_resp)
        mock_remote_backend.call = AsyncMock(return_value=remote_resp)

        router = make_router()
        request = make_task_request()
        result = await router.route(request)

        assert result.intended_action == RoutingAction.BOTH_RETURN_LOCAL
        assert result.response.content == "Local p3 coaching", (
            "Phase 3 BOTH_RETURN_LOCAL should return LOCAL response"
        )
        assert result.is_coaching_sample is True
        assert mock_training_collector.collect.called

    @pytest.mark.asyncio
    async def test_route_response_has_nonempty_content(self, make_router,
                                                        mock_confidence_engine,
                                                        mock_remote_backend):
        """Postcondition: response always contains non-empty content."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_1, 0.1)
        )
        mock_remote_backend.call = AsyncMock(
            return_value=make_remote_response(content="valid content")
        )

        router = make_router()
        result = await router.route(make_task_request())

        assert result.response.content, "Response content must be non-empty"
        assert len(result.response.content) > 0

    @pytest.mark.asyncio
    async def test_route_total_cost_equals_model_costs(self, make_router,
                                                        mock_confidence_engine,
                                                        mock_budget_manager,
                                                        mock_sampling_scheduler,
                                                        mock_local_backend,
                                                        mock_remote_backend):
        """Postcondition: total_cost_usd == sum of all model call costs."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_2, 0.55)
        )
        mock_sampling_scheduler.should_coach = AsyncMock(return_value=True)
        local_resp = make_local_response(cost=0.0)
        remote_resp = make_remote_response(cost=0.05)
        mock_local_backend.call = AsyncMock(return_value=local_resp)
        mock_remote_backend.call = AsyncMock(return_value=remote_resp)

        router = make_router()
        result = await router.route(make_task_request())

        expected_cost = local_resp.cost + remote_resp.cost
        assert abs(result.total_cost_usd - expected_cost) < 1e-9, (
            f"total_cost_usd {result.total_cost_usd} != expected {expected_cost}"
        )


# ============================================================================
# 3. test_route_errors — All 5 Error Paths
# ============================================================================

class TestRouteErrors:
    """Error-path tests for route()."""

    @pytest.mark.asyncio
    async def test_invalid_request_empty_id(self, make_router, mock_audit_logger):
        """Empty request_id triggers invalid_request error."""
        router = make_router()
        request = make_task_request(request_id="")

        with pytest.raises(Exception) as exc_info:
            await router.route(request)

        assert "invalid_request" in str(exc_info.value).lower() or \
               "invalid" in str(type(exc_info.value).__name__).lower(), (
            f"Expected invalid_request error, got {exc_info.value}"
        )

    @pytest.mark.asyncio
    async def test_invalid_request_empty_prompt(self, make_router):
        """Empty prompt triggers invalid_request error."""
        router = make_router()
        request = make_task_request(prompt="")

        with pytest.raises(Exception) as exc_info:
            await router.route(request)

        assert "invalid_request" in str(exc_info.value).lower() or \
               "invalid" in str(type(exc_info.value).__name__).lower() or \
               "prompt" in str(exc_info.value).lower(), (
            f"Expected invalid_request error for empty prompt, got {exc_info.value}"
        )

    @pytest.mark.asyncio
    async def test_invalid_request_empty_task_type(self, make_router):
        """Empty task_type triggers invalid_request error."""
        router = make_router()
        request = make_task_request(task_type="")

        with pytest.raises(Exception) as exc_info:
            await router.route(request)

        assert "invalid_request" in str(exc_info.value).lower() or \
               "invalid" in str(type(exc_info.value).__name__).lower() or \
               "task_type" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_budget_check_failed(self, make_router, mock_budget_manager,
                                        mock_audit_logger):
        """BudgetManager failure raises budget_check_failed."""
        mock_budget_manager.check_budget = AsyncMock(
            side_effect=RuntimeError("DB connection lost")
        )
        router = make_router()
        request = make_task_request()

        with pytest.raises(Exception) as exc_info:
            await router.route(request)

        error_str = str(exc_info.value).lower() + str(type(exc_info.value).__name__).lower()
        assert "budget" in error_str, (
            f"Expected budget_check_failed error, got {exc_info.value}"
        )
        # Audit should still be logged
        mock_audit_logger.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_confidence_engine_unavailable(self, make_router,
                                                  mock_confidence_engine,
                                                  mock_audit_logger):
        """ConfidenceEngine failure raises confidence_engine_unavailable."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            side_effect=RuntimeError("Engine down")
        )
        router = make_router()
        request = make_task_request()

        with pytest.raises(Exception) as exc_info:
            await router.route(request)

        error_str = str(exc_info.value).lower() + str(type(exc_info.value).__name__).lower()
        assert "confidence" in error_str, (
            f"Expected confidence_engine_unavailable, got {exc_info.value}"
        )
        mock_audit_logger.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_sampling_scheduler_unavailable(self, make_router,
                                                   mock_confidence_engine,
                                                   mock_sampling_scheduler,
                                                   mock_audit_logger):
        """SamplingScheduler failure raises sampling_scheduler_unavailable."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_2, 0.5)
        )
        mock_sampling_scheduler.should_coach = AsyncMock(
            side_effect=RuntimeError("Scheduler crashed")
        )
        router = make_router()
        request = make_task_request()

        with pytest.raises(Exception) as exc_info:
            await router.route(request)

        error_str = str(exc_info.value).lower() + str(type(exc_info.value).__name__).lower()
        assert "sampling" in error_str or "scheduler" in error_str, (
            f"Expected sampling_scheduler_unavailable, got {exc_info.value}"
        )
        mock_audit_logger.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_both_backends_unavailable(self, make_router, mock_confidence_engine,
                                              mock_budget_manager, mock_sampling_scheduler,
                                              mock_local_backend, mock_remote_backend,
                                              mock_audit_logger):
        """Both backends failing raises both_backends_unavailable."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_2, 0.5)
        )
        mock_sampling_scheduler.should_coach = AsyncMock(return_value=True)
        mock_local_backend.call = AsyncMock(
            side_effect=ConnectionError("Local down")
        )
        mock_remote_backend.call = AsyncMock(
            side_effect=ConnectionError("Remote down")
        )

        router = make_router()
        request = make_task_request()

        with pytest.raises(Exception) as exc_info:
            await router.route(request)

        error_str = str(exc_info.value).lower() + str(type(exc_info.value).__name__).lower()
        assert "backend" in error_str or "unavailable" in error_str or "both" in error_str, (
            f"Expected both_backends_unavailable, got {exc_info.value}"
        )
        mock_audit_logger.log.assert_called_once()


# ============================================================================
# 4. test_route_fallbacks — Degradation Scenarios
# ============================================================================

class TestRouteFallbacksAndDegradation:
    """Tests for fallback and degradation paths."""

    @pytest.mark.asyncio
    async def test_local_fails_fallback_to_remote(self, make_router,
                                                    mock_confidence_engine,
                                                    mock_budget_manager,
                                                    mock_sampling_scheduler,
                                                    mock_local_backend,
                                                    mock_remote_backend,
                                                    mock_audit_logger):
        """Local fails in LOCAL_ONLY → falls back to remote if budget allows."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_3, 0.95)
        )
        mock_sampling_scheduler.should_coach = AsyncMock(return_value=False)
        mock_local_backend.call = AsyncMock(
            side_effect=RuntimeError("Local inference failed")
        )
        remote_resp = make_remote_response(content="Remote fallback answer")
        mock_remote_backend.call = AsyncMock(return_value=remote_resp)

        router = make_router()
        result = await router.route(make_task_request())

        assert result.is_fallback is True, "Should be marked as fallback"
        assert result.actual_action == RoutingAction.LOCAL_FALLBACK_TO_REMOTE, (
            f"Expected LOCAL_FALLBACK_TO_REMOTE, got {result.actual_action}"
        )
        assert result.intended_action == RoutingAction.LOCAL_ONLY, (
            f"Intended should be LOCAL_ONLY, got {result.intended_action}"
        )
        assert result.response.content == "Remote fallback answer"

    @pytest.mark.asyncio
    async def test_remote_fails_in_dual_send_degraded(self, make_router,
                                                       mock_confidence_engine,
                                                       mock_budget_manager,
                                                       mock_sampling_scheduler,
                                                       mock_local_backend,
                                                       mock_remote_backend,
                                                       mock_audit_logger):
        """Remote fails during Phase 2 dual-send → falls back to local (degraded)."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_2, 0.6)
        )
        mock_sampling_scheduler.should_coach = AsyncMock(return_value=True)
        local_resp = make_local_response(content="Degraded local p2")
        mock_local_backend.call = AsyncMock(return_value=local_resp)
        mock_remote_backend.call = AsyncMock(
            side_effect=TimeoutError("Remote timed out")
        )

        router = make_router()
        result = await router.route(make_task_request())

        assert result.is_degraded is True, "Should be marked as degraded"
        assert result.response.content, "Should still have response content"

    @pytest.mark.asyncio
    async def test_budget_exhausted_forces_local_only(self, make_router,
                                                       mock_confidence_engine,
                                                       mock_budget_manager,
                                                       mock_sampling_scheduler,
                                                       mock_local_backend,
                                                       mock_remote_backend,
                                                       mock_audit_logger):
        """Budget exhausted in Phase 1 → forced LOCAL_ONLY, remote never called."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_1, 0.1)
        )
        mock_budget_manager.check_budget = AsyncMock(
            return_value=make_budget_snapshot(exhausted=True)
        )
        mock_sampling_scheduler.should_coach = AsyncMock(return_value=False)
        local_resp = make_local_response(content="Budget-constrained local")
        mock_local_backend.call = AsyncMock(return_value=local_resp)

        router = make_router()
        result = await router.route(make_task_request())

        assert result.intended_action == RoutingAction.LOCAL_ONLY, (
            "Budget exhausted should compute LOCAL_ONLY"
        )
        assert result.is_degraded is True, (
            "Phase 1 forced to local should be degraded"
        )
        mock_remote_backend.call.assert_not_called(), (
            "Remote backend should NEVER be called when budget is exhausted"
        )

    @pytest.mark.asyncio
    async def test_budget_exhausted_phase2_no_remote(self, make_router,
                                                      mock_confidence_engine,
                                                      mock_budget_manager,
                                                      mock_sampling_scheduler,
                                                      mock_local_backend,
                                                      mock_remote_backend):
        """Budget exhausted in Phase 2 → LOCAL_ONLY, no remote call."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_2, 0.55)
        )
        mock_budget_manager.check_budget = AsyncMock(
            return_value=make_budget_snapshot(exhausted=True)
        )
        mock_sampling_scheduler.should_coach = AsyncMock(return_value=True)
        mock_local_backend.call = AsyncMock(return_value=make_local_response())

        router = make_router()
        result = await router.route(make_task_request())

        assert result.is_degraded is True
        mock_remote_backend.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_flag_consistency(self, make_router,
                                              mock_confidence_engine,
                                              mock_budget_manager,
                                              mock_sampling_scheduler,
                                              mock_local_backend,
                                              mock_remote_backend):
        """is_fallback is True iff actual_action != intended_action."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_3, 0.95)
        )
        mock_sampling_scheduler.should_coach = AsyncMock(return_value=False)
        mock_local_backend.call = AsyncMock(
            side_effect=RuntimeError("Local failed")
        )
        mock_remote_backend.call = AsyncMock(return_value=make_remote_response())

        router = make_router()
        result = await router.route(make_task_request())

        if result.actual_action != result.intended_action:
            assert result.is_fallback is True, (
                "is_fallback must be True when actual != intended"
            )
        if result.is_fallback:
            assert result.actual_action != result.intended_action, (
                "When is_fallback is True, actual must differ from intended"
            )
            assert result.actual_action == RoutingAction.LOCAL_FALLBACK_TO_REMOTE, (
                "Fallback actual_action must be LOCAL_FALLBACK_TO_REMOTE"
            )


# ============================================================================
# 5. test_execute_dual_send — Async Concurrency
# ============================================================================

class TestExecuteDualSend:
    """Tests for execute_dual_send concurrent execution."""

    @pytest.mark.asyncio
    async def test_both_succeed(self, make_router, mock_local_backend,
                                 mock_remote_backend):
        """Both backends succeed → BOTH_SUCCEEDED with both responses."""
        local_resp = make_local_response()
        remote_resp = make_remote_response()
        mock_local_backend.call = AsyncMock(return_value=local_resp)
        mock_remote_backend.call = AsyncMock(return_value=remote_resp)

        router = make_router()
        request = make_task_request()
        dual_result = await router.execute_dual_send(request)

        assert dual_result.outcome == DualSendOutcome.BOTH_SUCCEEDED
        assert dual_result.local_response is not None, "Local response should be present"
        assert dual_result.remote_response is not None, "Remote response should be present"

    @pytest.mark.asyncio
    async def test_local_fails_remote_succeeds(self, make_router,
                                                mock_local_backend,
                                                mock_remote_backend):
        """Local fails, remote succeeds → LOCAL_FAILED_REMOTE_SUCCEEDED."""
        mock_local_backend.call = AsyncMock(
            side_effect=RuntimeError("Local OOM")
        )
        mock_remote_backend.call = AsyncMock(return_value=make_remote_response())

        router = make_router()
        dual_result = await router.execute_dual_send(make_task_request())

        assert dual_result.outcome == DualSendOutcome.LOCAL_FAILED_REMOTE_SUCCEEDED
        assert dual_result.local_response is None, "Local response should be None on failure"
        assert dual_result.remote_response is not None
        assert dual_result.local_error, "local_error should be non-empty"

    @pytest.mark.asyncio
    async def test_local_succeeds_remote_fails(self, make_router,
                                                mock_local_backend,
                                                mock_remote_backend):
        """Local succeeds, remote fails → LOCAL_SUCCEEDED_REMOTE_FAILED."""
        mock_local_backend.call = AsyncMock(return_value=make_local_response())
        mock_remote_backend.call = AsyncMock(
            side_effect=TimeoutError("Remote timeout")
        )

        router = make_router()
        dual_result = await router.execute_dual_send(make_task_request())

        assert dual_result.outcome == DualSendOutcome.LOCAL_SUCCEEDED_REMOTE_FAILED
        assert dual_result.local_response is not None
        assert dual_result.remote_response is None, "Remote response should be None on failure"
        assert dual_result.remote_error, "remote_error should be non-empty"

    @pytest.mark.asyncio
    async def test_both_fail(self, make_router, mock_local_backend,
                              mock_remote_backend):
        """Both backends fail → BOTH_FAILED with both None."""
        mock_local_backend.call = AsyncMock(
            side_effect=ConnectionError("Local unreachable")
        )
        mock_remote_backend.call = AsyncMock(
            side_effect=ConnectionError("Remote unreachable")
        )

        router = make_router()
        dual_result = await router.execute_dual_send(make_task_request())

        assert dual_result.outcome == DualSendOutcome.BOTH_FAILED
        assert dual_result.local_response is None
        assert dual_result.remote_response is None
        assert dual_result.local_error, "local_error should be non-empty"
        assert dual_result.remote_error, "remote_error should be non-empty"

    @pytest.mark.asyncio
    async def test_error_populated_only_on_failure(self, make_router,
                                                    mock_local_backend,
                                                    mock_remote_backend):
        """Error fields are non-empty iff the corresponding backend failed."""
        mock_local_backend.call = AsyncMock(return_value=make_local_response())
        mock_remote_backend.call = AsyncMock(
            side_effect=RuntimeError("Remote error")
        )

        router = make_router()
        dual_result = await router.execute_dual_send(make_task_request())

        # Local succeeded → no error
        assert not dual_result.local_error, (
            "local_error should be empty when local succeeded"
        )
        # Remote failed → has error
        assert dual_result.remote_error, (
            "remote_error should be non-empty when remote failed"
        )


# ============================================================================
# 6. test_build_audit_entry — Data Transformation
# ============================================================================

class TestBuildAuditEntry:
    """Tests for the pure build_audit_entry function."""

    def _make_decision(self, phase=Phase.PHASE_1, confidence=0.2,
                       action=RoutingAction.REMOTE_ONLY, coaching=False):
        return RoutingDecision(
            intended_action=action,
            phase_context=PhaseContext(phase=phase, confidence=confidence),
            is_coaching_sample=coaching,
            budget_snapshot=make_budget_snapshot(),
        )

    def _make_result(self, request_id="req-test-001",
                     intended=RoutingAction.REMOTE_ONLY,
                     actual=RoutingAction.REMOTE_ONLY,
                     phase=Phase.PHASE_1, confidence=0.2,
                     cost=0.03, latency=200.0):
        return RoutingResult(
            request_id=request_id,
            response=make_remote_response(),
            intended_action=intended,
            actual_action=actual,
            phase=phase,
            confidence=confidence,
            is_coaching_sample=False,
            total_cost_usd=cost,
            total_latency_ms=latency,
            is_degraded=False,
            is_fallback=False,
            local_response=None,
            remote_response=make_remote_response(),
        )

    def test_all_fields_populated(self):
        """All audit entry fields are populated from corresponding inputs."""
        request = make_task_request()
        decision = self._make_decision()
        result = self._make_result()

        entry = build_audit_entry(request, decision, result, error_detail="")

        assert entry.request_id == request.request_id
        assert entry.task_type == request.task_type
        assert entry.phase == decision.phase_context.phase
        assert entry.confidence == decision.phase_context.confidence
        assert entry.intended_action == decision.intended_action
        assert entry.actual_action == result.actual_action
        assert entry.total_cost_usd == result.total_cost_usd
        assert entry.total_latency_ms == result.total_latency_ms
        assert entry.is_degraded == result.is_degraded
        assert entry.is_fallback == result.is_fallback
        assert entry.error_detail == ""

    def test_error_detail_populated(self):
        """Error detail is preserved in audit entry."""
        request = make_task_request()
        decision = self._make_decision()
        result = self._make_result()

        entry = build_audit_entry(request, decision, result,
                                   error_detail="remote timeout after 5000ms")

        assert entry.error_detail == "remote timeout after 5000ms"
        assert entry.request_id == request.request_id

    def test_request_id_matches(self):
        """Output request_id always equals input request_id."""
        for rid in ["req-1", "req-abc-123", "x" * 100]:
            request = make_task_request(request_id=rid)
            decision = self._make_decision()
            result = self._make_result(request_id=rid)
            entry = build_audit_entry(request, decision, result, error_detail="")
            assert entry.request_id == rid

    def test_timestamp_utc_is_set(self):
        """Audit entry has a non-empty timestamp_utc."""
        request = make_task_request()
        decision = self._make_decision()
        result = self._make_result()

        entry = build_audit_entry(request, decision, result, error_detail="")

        assert entry.timestamp_utc, "timestamp_utc must be set"
        assert len(entry.timestamp_utc) > 0

    def test_cost_is_nonnegative(self):
        """Total cost in audit entry is non-negative."""
        request = make_task_request()
        decision = self._make_decision()
        result = self._make_result(cost=0.05)

        entry = build_audit_entry(request, decision, result, error_detail="")

        assert entry.total_cost_usd >= 0, (
            f"Cost should be non-negative, got {entry.total_cost_usd}"
        )

    def test_coaching_sample_preserved(self):
        """Coaching sample flag is correctly captured."""
        request = make_task_request()
        decision = self._make_decision(coaching=True)
        result = self._make_result()

        entry = build_audit_entry(request, decision, result, error_detail="")

        assert entry.is_coaching_sample == True


# ============================================================================
# 7. test_router_config — Config Validation
# ============================================================================

class TestRouterConfig:
    """Tests for RouterConfig validation at construction time."""

    def test_valid_config_accepted(self):
        """Valid config with sensible values is accepted."""
        config = make_router_config()
        assert config.max_concurrent_dual_sends == 10
        assert config.remote_call_timeout_ms == 5000
        assert config.local_call_timeout_ms == 2000
        assert config.enable_training_collection is True
        assert config.enable_audit_logging is True

    def test_negative_remote_timeout_rejected(self):
        """Negative remote_call_timeout_ms raises ValidationError."""
        with pytest.raises(Exception) as exc_info:
            make_router_config(remote_call_timeout_ms=-1)
        assert exc_info.value  # Some validation error raised

    def test_negative_local_timeout_rejected(self):
        """Negative local_call_timeout_ms raises ValidationError."""
        with pytest.raises(Exception) as exc_info:
            make_router_config(local_call_timeout_ms=-100)
        assert exc_info.value

    def test_zero_max_concurrent_rejected(self):
        """Zero max_concurrent_dual_sends raises ValidationError."""
        with pytest.raises(Exception) as exc_info:
            make_router_config(max_concurrent_dual_sends=0)
        assert exc_info.value

    def test_negative_max_concurrent_rejected(self):
        """Negative max_concurrent_dual_sends raises ValidationError."""
        with pytest.raises(Exception) as exc_info:
            make_router_config(max_concurrent_dual_sends=-5)
        assert exc_info.value


# ============================================================================
# 8. Invariant Tests — Cross-Cutting Concerns
# ============================================================================

class TestInvariants:
    """Tests for contract invariants that span multiple scenarios."""

    @pytest.mark.asyncio
    async def test_audit_always_logged_on_success(self, make_router,
                                                    mock_confidence_engine,
                                                    mock_remote_backend,
                                                    mock_audit_logger):
        """Every successful route() call produces exactly one audit entry."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_1, 0.1)
        )
        mock_remote_backend.call = AsyncMock(return_value=make_remote_response())

        router = make_router()
        await router.route(make_task_request())

        assert mock_audit_logger.log.call_count == 1, (
            f"Expected 1 audit log call, got {mock_audit_logger.log.call_count}"
        )

    @pytest.mark.asyncio
    async def test_audit_logged_on_error(self, make_router,
                                          mock_confidence_engine,
                                          mock_audit_logger):
        """Audit entry is still logged even when route() raises an error."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            side_effect=RuntimeError("Engine down")
        )

        router = make_router()
        with pytest.raises(Exception):
            await router.route(make_task_request())

        mock_audit_logger.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_audit_logger_failure_does_not_block(self, make_router,
                                                        mock_confidence_engine,
                                                        mock_remote_backend,
                                                        mock_audit_logger):
        """Audit logger failure does not block or affect the route() response."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_1, 0.1)
        )
        mock_remote_backend.call = AsyncMock(
            return_value=make_remote_response(content="Still works")
        )
        mock_audit_logger.log = AsyncMock(
            side_effect=RuntimeError("Audit storage full")
        )

        router = make_router()
        result = await router.route(make_task_request())

        assert result.response.content == "Still works", (
            "Route should still return response even when audit logger fails"
        )

    @pytest.mark.asyncio
    async def test_budget_never_exceeded_all_phases(self, make_router,
                                                     mock_confidence_engine,
                                                     mock_budget_manager,
                                                     mock_sampling_scheduler,
                                                     mock_local_backend,
                                                     mock_remote_backend):
        """When budget is exhausted, remote is never called regardless of phase."""
        mock_budget_manager.check_budget = AsyncMock(
            return_value=make_budget_snapshot(exhausted=True)
        )
        mock_local_backend.call = AsyncMock(return_value=make_local_response())
        mock_sampling_scheduler.should_coach = AsyncMock(return_value=True)

        for phase in [Phase.PHASE_1, Phase.PHASE_2, Phase.PHASE_3]:
            mock_confidence_engine.get_phase_context = AsyncMock(
                return_value=make_phase_context(phase, 0.5)
            )
            mock_remote_backend.call.reset_mock()

            router = make_router()
            result = await router.route(make_task_request())

            mock_remote_backend.call.assert_not_called(), (
                f"Remote should not be called in {phase} with exhausted budget"
            )
            assert result.is_degraded is True or result.intended_action == RoutingAction.LOCAL_ONLY

    @pytest.mark.asyncio
    async def test_training_collected_phase1_remote_response(self, make_router,
                                                              mock_confidence_engine,
                                                              mock_remote_backend,
                                                              mock_training_collector,
                                                              mock_sampling_scheduler):
        """Training data collected for remote response in Phase 1."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_1, 0.1)
        )
        mock_sampling_scheduler.should_coach = AsyncMock(return_value=False)
        mock_remote_backend.call = AsyncMock(return_value=make_remote_response())

        router = make_router()
        await router.route(make_task_request())

        assert mock_training_collector.collect.called, (
            "Training collector should be called for Phase 1 remote response"
        )
        # Verify REMOTE_RESPONSE_EXAMPLE was submitted
        collect_calls = mock_training_collector.collect.call_args_list
        found_remote_example = False
        for c in collect_calls:
            arg = c[0][0] if c[0] else c[1].get("example")
            if hasattr(arg, "example_type") and \
               arg.example_type == TrainingExampleType.REMOTE_RESPONSE_EXAMPLE:
                found_remote_example = True
                break
        assert found_remote_example, (
            "Expected REMOTE_RESPONSE_EXAMPLE training example in Phase 1"
        )

    @pytest.mark.asyncio
    async def test_training_collected_phase2_remote_response(self, make_router,
                                                              mock_confidence_engine,
                                                              mock_sampling_scheduler,
                                                              mock_local_backend,
                                                              mock_remote_backend,
                                                              mock_training_collector):
        """Training data collected for remote response in Phase 2."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_2, 0.6)
        )
        mock_sampling_scheduler.should_coach = AsyncMock(return_value=False)
        mock_local_backend.call = AsyncMock(return_value=make_local_response())
        mock_remote_backend.call = AsyncMock(return_value=make_remote_response())

        router = make_router()
        await router.route(make_task_request())

        assert mock_training_collector.collect.called, (
            "Training collector should be called for Phase 2 remote response"
        )

    @pytest.mark.asyncio
    async def test_record_spend_called_with_correct_cost(self, make_router,
                                                          mock_confidence_engine,
                                                          mock_budget_manager,
                                                          mock_remote_backend):
        """BudgetManager.record_spend() is called with total_cost_usd."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_1, 0.1)
        )
        remote_resp = make_remote_response(cost=0.05)
        mock_remote_backend.call = AsyncMock(return_value=remote_resp)

        router = make_router()
        result = await router.route(make_task_request())

        mock_budget_manager.record_spend.assert_called_once()
        # The cost passed should match total_cost_usd
        spend_call_args = mock_budget_manager.record_spend.call_args
        if spend_call_args[0]:
            recorded_cost = spend_call_args[0][0]
        else:
            # Could be keyword arg
            recorded_cost = list(spend_call_args[1].values())[0]
        assert abs(recorded_cost - result.total_cost_usd) < 1e-9, (
            f"record_spend called with {recorded_cost}, "
            f"but total_cost_usd is {result.total_cost_usd}"
        )

    @pytest.mark.asyncio
    async def test_no_global_state_between_requests(self, make_router,
                                                     mock_confidence_engine,
                                                     mock_remote_backend,
                                                     mock_budget_manager):
        """Consecutive calls produce independent results (no global state)."""
        mock_confidence_engine.get_phase_context = AsyncMock(
            return_value=make_phase_context(Phase.PHASE_1, 0.1)
        )
        mock_remote_backend.call = AsyncMock(return_value=make_remote_response())

        router = make_router()

        req1 = make_task_request(request_id="req-001")
        req2 = make_task_request(request_id="req-002")

        result1 = await router.route(req1)
        result2 = await router.route(req2)

        assert result1.request_id == "req-001"
        assert result2.request_id == "req-002"
        assert result1.request_id != result2.request_id
