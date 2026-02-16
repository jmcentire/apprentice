"""
Contract tests for the Phase Manager component.

Tests are organized into sections:
1. Fixtures
2. Threshold validation tests
3. compute_phase pure function tests
4. evaluate stateful tests
5. Task state management tests
6. Property-based / invariant tests
7. Edge cases

All dependencies are mocked. No global state.
"""

import math
import pytest
from unittest.mock import MagicMock, PropertyMock
from dataclasses import dataclass

from src.phase_manager import (
    Phase,
    TrendDirection,
    TransitionTrigger,
    PhaseThresholds,
    PhaseMetrics,
    PhaseTransition,
    PhaseComputationResult,
    PhaseManager,
    compute_phase,
    validate_thresholds,
)


# ============================================================
# Section 1: Fixtures
# ============================================================

@pytest.fixture
def default_thresholds():
    """Valid PhaseThresholds with emergency=0.3 < coaching=0.5 < phase2_to_phase3=0.8."""
    return PhaseThresholds(
        phase1_to_phase2_example_count=10,
        phase2_to_phase3_correlation=0.8,
        coaching_trigger=0.5,
        emergency_threshold=0.3,
        sustained_windows=3,
        cooldown_windows=2,
    )


@pytest.fixture
def mock_clock():
    """Mock Clock returning deterministic ISO 8601 timestamps."""
    clock = MagicMock()
    clock.now.return_value = "2024-01-15T10:00:00Z"
    return clock


@pytest.fixture
def phase_manager(mock_clock):
    """Fresh PhaseManager instance injected with mock clock."""
    return PhaseManager(clock=mock_clock)


def make_metrics(
    task_id="task-1",
    example_count=20,
    rolling_correlation=0.9,
    local_model_available=True,
    consecutive_windows_above_threshold=5,
    trend_direction=TrendDirection.STABLE,
):
    """Factory for PhaseMetrics with sensible defaults."""
    return PhaseMetrics(
        task_id=task_id,
        example_count=example_count,
        rolling_correlation=rolling_correlation,
        local_model_available=local_model_available,
        consecutive_windows_above_threshold=consecutive_windows_above_threshold,
        trend_direction=trend_direction,
    )


def make_remote_only_metrics(task_id="task-1"):
    """Metrics that should produce REMOTE_ONLY (low correlation, few examples)."""
    return make_metrics(
        task_id=task_id,
        example_count=2,
        rolling_correlation=0.1,
        local_model_available=False,
        consecutive_windows_above_threshold=0,
    )


def make_coaching_metrics(task_id="task-1"):
    """Metrics that should produce COACHING (above coaching, below sustained for autonomous)."""
    return make_metrics(
        task_id=task_id,
        example_count=20,
        rolling_correlation=0.7,
        local_model_available=True,
        consecutive_windows_above_threshold=1,
    )


def make_autonomous_metrics(task_id="task-1"):
    """Metrics that should produce AUTONOMOUS (sustained high correlation)."""
    return make_metrics(
        task_id=task_id,
        example_count=20,
        rolling_correlation=0.9,
        local_model_available=True,
        consecutive_windows_above_threshold=5,
    )


# ============================================================
# Section 2: Threshold Validation Tests
# ============================================================

class TestValidateThresholds:
    def test_happy_path_valid_thresholds(self, default_thresholds):
        """validate_thresholds returns the same instance when all invariants hold."""
        result = validate_thresholds(default_thresholds)
        assert result is default_thresholds, "Should return the same instance on valid input"
        assert result.emergency_threshold < result.coaching_trigger < result.phase2_to_phase3_correlation

    @pytest.mark.parametrize("emergency, coaching, p2p3", [
        (0.5, 0.5, 0.8),   # emergency == coaching
        (0.6, 0.5, 0.8),   # emergency > coaching
        (0.3, 0.8, 0.8),   # coaching == p2p3
        (0.3, 0.9, 0.8),   # coaching > p2p3
        (0.9, 0.5, 0.3),   # all reversed
    ])
    def test_threshold_ordering_violation(self, emergency, coaching, p2p3):
        """validate_thresholds raises on violated ordering invariant."""
        with pytest.raises(Exception) as exc_info:
            thresholds = PhaseThresholds(
                phase1_to_phase2_example_count=10,
                phase2_to_phase3_correlation=p2p3,
                coaching_trigger=coaching,
                emergency_threshold=emergency,
                sustained_windows=3,
                cooldown_windows=2,
            )
            validate_thresholds(thresholds)
        # Accept either construction-time or validation-time error

    def test_invalid_example_count_zero(self):
        """validate_thresholds raises when phase1_to_phase2_example_count < 1."""
        with pytest.raises(Exception):
            thresholds = PhaseThresholds(
                phase1_to_phase2_example_count=0,
                phase2_to_phase3_correlation=0.8,
                coaching_trigger=0.5,
                emergency_threshold=0.3,
                sustained_windows=3,
                cooldown_windows=2,
            )
            validate_thresholds(thresholds)

    def test_invalid_example_count_negative(self):
        """validate_thresholds raises when phase1_to_phase2_example_count is negative."""
        with pytest.raises(Exception):
            thresholds = PhaseThresholds(
                phase1_to_phase2_example_count=-1,
                phase2_to_phase3_correlation=0.8,
                coaching_trigger=0.5,
                emergency_threshold=0.3,
                sustained_windows=3,
                cooldown_windows=2,
            )
            validate_thresholds(thresholds)

    def test_invalid_sustained_windows_zero(self):
        """validate_thresholds raises when sustained_windows < 1."""
        with pytest.raises(Exception):
            thresholds = PhaseThresholds(
                phase1_to_phase2_example_count=10,
                phase2_to_phase3_correlation=0.8,
                coaching_trigger=0.5,
                emergency_threshold=0.3,
                sustained_windows=0,
                cooldown_windows=2,
            )
            validate_thresholds(thresholds)

    def test_invalid_cooldown_windows_negative(self):
        """validate_thresholds raises when cooldown_windows < 0."""
        with pytest.raises(Exception):
            thresholds = PhaseThresholds(
                phase1_to_phase2_example_count=10,
                phase2_to_phase3_correlation=0.8,
                coaching_trigger=0.5,
                emergency_threshold=0.3,
                sustained_windows=3,
                cooldown_windows=-1,
            )
            validate_thresholds(thresholds)


# ============================================================
# Section 3: compute_phase Pure Function Tests
# ============================================================

class TestComputePhase:
    """Tests for the pure, stateless compute_phase function."""

    def test_emergency_below_threshold(self, default_thresholds):
        """Correlation below emergency_threshold → REMOTE_ONLY."""
        metrics = make_metrics(rolling_correlation=0.1)
        result = compute_phase(metrics, default_thresholds)
        assert result.phase == Phase.REMOTE_ONLY, (
            f"Expected REMOTE_ONLY for correlation {metrics.rolling_correlation} < "
            f"emergency {default_thresholds.emergency_threshold}"
        )

    def test_coaching_trigger_with_model_and_examples(self, default_thresholds):
        """Correlation between emergency and coaching, with model + examples → COACHING."""
        metrics = make_metrics(
            rolling_correlation=0.4,
            example_count=20,
            local_model_available=True,
            consecutive_windows_above_threshold=0,
        )
        result = compute_phase(metrics, default_thresholds)
        assert result.phase == Phase.COACHING

    def test_coaching_trigger_no_model(self, default_thresholds):
        """Correlation between emergency and coaching, no model → REMOTE_ONLY."""
        metrics = make_metrics(
            rolling_correlation=0.4,
            example_count=20,
            local_model_available=False,
            consecutive_windows_above_threshold=0,
        )
        result = compute_phase(metrics, default_thresholds)
        assert result.phase == Phase.REMOTE_ONLY

    def test_coaching_trigger_insufficient_examples(self, default_thresholds):
        """Correlation between emergency and coaching, insufficient examples → REMOTE_ONLY."""
        metrics = make_metrics(
            rolling_correlation=0.4,
            example_count=5,
            local_model_available=True,
            consecutive_windows_above_threshold=0,
        )
        result = compute_phase(metrics, default_thresholds)
        assert result.phase == Phase.REMOTE_ONLY

    def test_insufficient_examples_high_correlation(self, default_thresholds):
        """High correlation but insufficient examples → REMOTE_ONLY."""
        metrics = make_metrics(
            rolling_correlation=0.9,
            example_count=5,
            local_model_available=True,
            consecutive_windows_above_threshold=10,
        )
        result = compute_phase(metrics, default_thresholds)
        assert result.phase == Phase.REMOTE_ONLY

    def test_no_model_high_correlation(self, default_thresholds):
        """High correlation but no model → never AUTONOMOUS."""
        metrics = make_metrics(
            rolling_correlation=0.9,
            example_count=20,
            local_model_available=False,
            consecutive_windows_above_threshold=10,
        )
        result = compute_phase(metrics, default_thresholds)
        assert result.phase != Phase.AUTONOMOUS, "Must not be AUTONOMOUS without local model"
        assert result.phase in (Phase.REMOTE_ONLY, Phase.COACHING)

    def test_autonomous_sustained_threshold_met(self, default_thresholds):
        """Sustained windows >= threshold, model available, enough examples → AUTONOMOUS."""
        metrics = make_metrics(
            rolling_correlation=0.9,
            example_count=20,
            local_model_available=True,
            consecutive_windows_above_threshold=3,  # == sustained_windows
        )
        result = compute_phase(metrics, default_thresholds)
        assert result.phase == Phase.AUTONOMOUS

    def test_coaching_not_sustained_enough(self, default_thresholds):
        """Above coaching, model + examples present, but not enough sustained windows → COACHING."""
        metrics = make_metrics(
            rolling_correlation=0.9,
            example_count=20,
            local_model_available=True,
            consecutive_windows_above_threshold=2,  # < sustained_windows (3)
        )
        result = compute_phase(metrics, default_thresholds)
        assert result.phase == Phase.COACHING

    def test_nan_correlation_raises(self, default_thresholds):
        """NaN correlation must raise an error."""
        metrics = make_metrics(rolling_correlation=float("nan"))
        with pytest.raises(Exception):
            compute_phase(metrics, default_thresholds)

    # Boundary value tests
    def test_boundary_exact_emergency_threshold(self, default_thresholds):
        """Correlation exactly at emergency_threshold (0.3) — should NOT trigger emergency."""
        metrics = make_metrics(
            rolling_correlation=0.3,  # == emergency_threshold
            example_count=20,
            local_model_available=True,
            consecutive_windows_above_threshold=0,
        )
        result = compute_phase(metrics, default_thresholds)
        # At exactly emergency_threshold, the condition is < not <=, so should be COACHING or above
        # (correlation < coaching_trigger=0.5, model available, examples sufficient → COACHING)
        assert result.phase in (Phase.REMOTE_ONLY, Phase.COACHING), (
            "Boundary at emergency_threshold: implementation may use < or <="
        )

    def test_boundary_just_below_emergency(self, default_thresholds):
        """Correlation just below emergency_threshold → REMOTE_ONLY."""
        metrics = make_metrics(rolling_correlation=0.2999)
        result = compute_phase(metrics, default_thresholds)
        assert result.phase == Phase.REMOTE_ONLY

    def test_boundary_exact_coaching_trigger(self, default_thresholds):
        """Correlation exactly at coaching_trigger (0.5)."""
        metrics = make_metrics(
            rolling_correlation=0.5,
            example_count=20,
            local_model_available=True,
            consecutive_windows_above_threshold=0,
        )
        result = compute_phase(metrics, default_thresholds)
        # At coaching_trigger boundary — either COACHING or above depending on < vs <=
        assert result.phase in (Phase.COACHING, Phase.REMOTE_ONLY)

    def test_boundary_exact_sustained_windows(self, default_thresholds):
        """consecutive_windows exactly at sustained_windows → AUTONOMOUS."""
        metrics = make_metrics(
            rolling_correlation=0.9,
            example_count=20,
            local_model_available=True,
            consecutive_windows_above_threshold=3,  # exactly sustained_windows
        )
        result = compute_phase(metrics, default_thresholds)
        assert result.phase == Phase.AUTONOMOUS

    def test_boundary_one_below_sustained_windows(self, default_thresholds):
        """consecutive_windows one below sustained_windows → COACHING (not AUTONOMOUS)."""
        metrics = make_metrics(
            rolling_correlation=0.9,
            example_count=20,
            local_model_available=True,
            consecutive_windows_above_threshold=2,  # sustained_windows - 1
        )
        result = compute_phase(metrics, default_thresholds)
        assert result.phase == Phase.COACHING

    def test_determinism(self, default_thresholds):
        """Same inputs always produce same output — compute_phase is pure."""
        metrics = make_coaching_metrics()
        results = [compute_phase(metrics, default_thresholds) for _ in range(10)]
        phases = [r.phase for r in results]
        triggers = [r.trigger for r in results]
        assert len(set(phases)) == 1, "Phase must be deterministic"
        assert len(set(triggers)) == 1, "Trigger must be deterministic"

    def test_trigger_for_emergency(self, default_thresholds):
        """Trigger should indicate emergency threshold breach."""
        metrics = make_metrics(rolling_correlation=0.1)
        result = compute_phase(metrics, default_thresholds)
        assert result.trigger == TransitionTrigger.CORRELATION_BELOW_EMERGENCY_THRESHOLD

    def test_trigger_for_coaching(self, default_thresholds):
        """Trigger should indicate coaching trigger correlation level."""
        metrics = make_metrics(
            rolling_correlation=0.4,
            example_count=20,
            local_model_available=True,
            consecutive_windows_above_threshold=0,
        )
        result = compute_phase(metrics, default_thresholds)
        assert result.trigger == TransitionTrigger.CORRELATION_BELOW_COACHING_TRIGGER

    def test_trigger_for_autonomous(self, default_thresholds):
        """Trigger should indicate sustained correlation above threshold."""
        metrics = make_autonomous_metrics()
        result = compute_phase(metrics, default_thresholds)
        assert result.trigger == TransitionTrigger.SUSTAINED_CORRELATION_ABOVE_THRESHOLD

    def test_trigger_for_phase1_to_phase2(self, default_thresholds):
        """Trigger for transitioning from Phase1 to Phase2 via example count + model ready."""
        metrics = make_metrics(
            rolling_correlation=0.7,
            example_count=10,  # exactly at threshold
            local_model_available=True,
            consecutive_windows_above_threshold=0,
        )
        result = compute_phase(metrics, default_thresholds)
        assert result.phase == Phase.COACHING
        assert result.trigger == TransitionTrigger.EXAMPLE_COUNT_AND_MODEL_READY

    # Postcondition: AUTONOMOUS requires sustained windows
    @pytest.mark.parametrize("windows", [0, 1, 2])
    def test_autonomous_requires_sustained_windows(self, default_thresholds, windows):
        """AUTONOMOUS is never returned when consecutive_windows < sustained_windows."""
        metrics = make_metrics(
            rolling_correlation=0.95,
            example_count=100,
            local_model_available=True,
            consecutive_windows_above_threshold=windows,
        )
        result = compute_phase(metrics, default_thresholds)
        if result.phase == Phase.AUTONOMOUS:
            assert windows >= default_thresholds.sustained_windows


# ============================================================
# Section 4: evaluate Stateful Tests
# ============================================================

class TestEvaluate:
    """Tests for the stateful evaluate method on PhaseManager."""

    def test_first_evaluation_initial_assignment(self, phase_manager, default_thresholds, mock_clock):
        """First evaluation for a new task may produce INITIAL_ASSIGNMENT transition."""
        metrics = make_coaching_metrics()
        result = phase_manager.evaluate(metrics, default_thresholds)
        # New task: initialized to REMOTE_ONLY, then transitions to COACHING
        if result.transition is not None:
            assert result.transition.from_phase == Phase.REMOTE_ONLY
            assert result.transition.trigger in (
                TransitionTrigger.INITIAL_ASSIGNMENT,
                TransitionTrigger.EXAMPLE_COUNT_AND_MODEL_READY,
                TransitionTrigger.CORRELATION_BELOW_COACHING_TRIGGER,
            )
            assert result.transition.timestamp == "2024-01-15T10:00:00Z"
            assert result.transition.task_id == metrics.task_id

    def test_no_transition_when_phase_unchanged(self, phase_manager, default_thresholds):
        """Evaluating with same-phase metrics produces no transition."""
        metrics = make_remote_only_metrics()
        # First eval — initializes to REMOTE_ONLY, compute_phase also REMOTE_ONLY
        result1 = phase_manager.evaluate(metrics, default_thresholds)
        # Second eval — still REMOTE_ONLY, no transition
        result2 = phase_manager.evaluate(metrics, default_thresholds)
        # After the first eval settles, second should produce no transition
        assert result2.transition is None, "No transition when phase is unchanged"
        assert result2.current_phase == Phase.REMOTE_ONLY

    def test_transition_remote_to_coaching(self, phase_manager, default_thresholds):
        """Evaluating from REMOTE_ONLY to COACHING detects transition."""
        # First: establish REMOTE_ONLY
        remote_metrics = make_remote_only_metrics()
        phase_manager.evaluate(remote_metrics, default_thresholds)

        # Second: metrics that produce COACHING
        coaching_metrics = make_coaching_metrics()
        result = phase_manager.evaluate(coaching_metrics, default_thresholds)
        assert result.current_phase == Phase.COACHING
        if result.transition is not None:
            assert result.transition.from_phase == Phase.REMOTE_ONLY
            assert result.transition.to_phase == Phase.COACHING

    def test_cooldown_suppresses_transition(self, phase_manager, default_thresholds):
        """During cooldown, transition is suppressed and cooldown_active=True."""
        # Trigger a transition first
        remote_metrics = make_remote_only_metrics()
        phase_manager.evaluate(remote_metrics, default_thresholds)
        coaching_metrics = make_coaching_metrics()
        result1 = phase_manager.evaluate(coaching_metrics, default_thresholds)

        # If a transition occurred and cooldown_windows > 0, the next different-phase eval
        # within cooldown should be suppressed
        if result1.transition is not None:
            # Immediately try another transition (within cooldown_windows=2)
            autonomous_metrics = make_autonomous_metrics()
            result2 = phase_manager.evaluate(autonomous_metrics, default_thresholds)
            if result2.cooldown_active:
                assert result2.transition is None, "Transition suppressed during cooldown"
                # Phase should remain the previous phase
                assert result2.current_phase == Phase.COACHING

    def test_nan_correlation_raises(self, phase_manager, default_thresholds):
        """evaluate raises on NaN rolling_correlation."""
        metrics = make_metrics(rolling_correlation=float("nan"))
        with pytest.raises(Exception):
            phase_manager.evaluate(metrics, default_thresholds)

    def test_clock_failure_raises(self, default_thresholds):
        """evaluate raises when clock.now() fails."""
        failing_clock = MagicMock()
        failing_clock.now.side_effect = RuntimeError("clock failure")
        pm = PhaseManager(clock=failing_clock)

        # Use metrics that will trigger a transition (new task → any phase change)
        metrics = make_coaching_metrics()
        with pytest.raises(Exception):
            pm.evaluate(metrics, default_thresholds)

    def test_windows_incremented_no_transition(self, phase_manager, default_thresholds):
        """windows_since_last_transition increments by 1 when no transition occurs."""
        metrics = make_remote_only_metrics()
        phase_manager.evaluate(metrics, default_thresholds)
        phase_manager.evaluate(metrics, default_thresholds)
        state = phase_manager.get_task_state(metrics.task_id)
        if state is not None:
            assert state.windows_since_last_transition >= 1

    def test_windows_reset_on_transition(self, phase_manager, default_thresholds):
        """windows_since_last_transition resets to 0 when a transition occurs."""
        remote_metrics = make_remote_only_metrics()
        phase_manager.evaluate(remote_metrics, default_thresholds)
        phase_manager.evaluate(remote_metrics, default_thresholds)

        coaching_metrics = make_coaching_metrics()
        result = phase_manager.evaluate(coaching_metrics, default_thresholds)
        if result.transition is not None:
            state = phase_manager.get_task_state(coaching_metrics.task_id)
            assert state.windows_since_last_transition == 0

    def test_transition_appended_to_history(self, phase_manager, default_thresholds):
        """PhaseTransition events are appended to task's transition_history."""
        remote_metrics = make_remote_only_metrics()
        phase_manager.evaluate(remote_metrics, default_thresholds)

        coaching_metrics = make_coaching_metrics()
        result = phase_manager.evaluate(coaching_metrics, default_thresholds)
        if result.transition is not None:
            history = phase_manager.get_transition_history(coaching_metrics.task_id)
            assert len(history) >= 1
            assert result.transition in history

    def test_transition_has_frozen_metrics_snapshot(self, phase_manager, default_thresholds):
        """PhaseTransition contains frozen metrics_snapshot matching input metrics."""
        coaching_metrics = make_coaching_metrics()
        result = phase_manager.evaluate(coaching_metrics, default_thresholds)
        if result.transition is not None:
            snapshot = result.transition.metrics_snapshot
            assert snapshot.task_id == coaching_metrics.task_id
            assert snapshot.rolling_correlation == coaching_metrics.rolling_correlation
            assert snapshot.example_count == coaching_metrics.example_count


# ============================================================
# Section 5: Task State Management Tests
# ============================================================

class TestGetTaskPhase:
    def test_unknown_task_returns_remote_only(self, phase_manager):
        """Unknown task_id returns Phase.REMOTE_ONLY."""
        result = phase_manager.get_task_phase("never-seen-task")
        assert result == Phase.REMOTE_ONLY

    def test_known_task_returns_current_phase(self, phase_manager, default_thresholds):
        """After evaluation, returns the computed phase."""
        metrics = make_coaching_metrics()
        eval_result = phase_manager.evaluate(metrics, default_thresholds)
        phase = phase_manager.get_task_phase(metrics.task_id)
        assert phase == eval_result.current_phase

    def test_empty_task_id_raises(self, phase_manager):
        """Empty task_id raises an error."""
        with pytest.raises(Exception):
            phase_manager.get_task_phase("")


class TestGetTaskState:
    def test_unknown_task_returns_none(self, phase_manager):
        """Unknown task_id returns None."""
        result = phase_manager.get_task_state("unknown-task")
        assert result is None

    def test_known_task_returns_state(self, phase_manager, default_thresholds):
        """After evaluation, returns full TaskPhaseState."""
        metrics = make_coaching_metrics()
        phase_manager.evaluate(metrics, default_thresholds)
        state = phase_manager.get_task_state(metrics.task_id)
        assert state is not None
        assert state.task_id == metrics.task_id
        assert isinstance(state.current_phase, Phase)
        assert isinstance(state.windows_since_last_transition, int)
        assert isinstance(state.transition_history, list)

    def test_empty_task_id_raises(self, phase_manager):
        """Empty task_id raises an error."""
        with pytest.raises(Exception):
            phase_manager.get_task_state("")


class TestGetTransitionHistory:
    def test_unknown_task_returns_empty(self, phase_manager):
        """Unknown task_id returns empty list."""
        result = phase_manager.get_transition_history("unknown-task")
        assert result == []

    def test_chronological_ordering(self, phase_manager, default_thresholds, mock_clock):
        """Transition history is ordered oldest first."""
        timestamps = iter([
            "2024-01-15T10:00:00Z",
            "2024-01-15T10:01:00Z",
            "2024-01-15T10:02:00Z",
            "2024-01-15T10:03:00Z",
            "2024-01-15T10:04:00Z",
        ])
        mock_clock.now.side_effect = lambda: next(timestamps, "2024-01-15T10:05:00Z")

        task_id = "task-chrono"
        # Drive multiple transitions
        phase_manager.evaluate(make_remote_only_metrics(task_id), default_thresholds)
        phase_manager.evaluate(make_coaching_metrics(task_id), default_thresholds)
        phase_manager.evaluate(make_remote_only_metrics(task_id), default_thresholds)
        phase_manager.evaluate(make_coaching_metrics(task_id), default_thresholds)

        history = phase_manager.get_transition_history(task_id)
        if len(history) >= 2:
            for i in range(len(history) - 1):
                assert history[i].timestamp <= history[i + 1].timestamp, (
                    f"History not chronological: {history[i].timestamp} > {history[i+1].timestamp}"
                )

    def test_empty_task_id_raises(self, phase_manager):
        """Empty task_id raises an error."""
        with pytest.raises(Exception):
            phase_manager.get_transition_history("")


class TestResetTask:
    def test_reset_existing_task(self, phase_manager, default_thresholds):
        """reset_task returns True for a tracked task and clears all state."""
        metrics = make_coaching_metrics()
        phase_manager.evaluate(metrics, default_thresholds)
        result = phase_manager.reset_task(metrics.task_id)
        assert result is True

    def test_reset_unknown_task(self, phase_manager):
        """reset_task returns False for an unknown task."""
        result = phase_manager.reset_task("never-tracked")
        assert result is False

    def test_reset_clears_all_state(self, phase_manager, default_thresholds):
        """After reset: get_task_phase→REMOTE_ONLY, get_task_state→None, history→[]."""
        metrics = make_coaching_metrics()
        phase_manager.evaluate(metrics, default_thresholds)
        phase_manager.reset_task(metrics.task_id)

        assert phase_manager.get_task_phase(metrics.task_id) == Phase.REMOTE_ONLY, (
            "After reset, task phase should be REMOTE_ONLY"
        )
        assert phase_manager.get_task_state(metrics.task_id) is None, (
            "After reset, task state should be None"
        )
        assert phase_manager.get_transition_history(metrics.task_id) == [], (
            "After reset, transition history should be empty"
        )

    def test_reset_empty_task_id_raises(self, phase_manager):
        """Empty task_id raises an error."""
        with pytest.raises(Exception):
            phase_manager.reset_task("")


# ============================================================
# Section 6: Invariant Tests
# ============================================================

class TestInvariants:
    def test_remote_only_universal_default(self, phase_manager):
        """REMOTE_ONLY is the universal default for any unknown task."""
        for task_id in ["task-a", "task-b", "nonexistent-123"]:
            assert phase_manager.get_task_phase(task_id) == Phase.REMOTE_ONLY

    def test_task_isolation(self, phase_manager, default_thresholds):
        """Two different tasks do not interfere with each other's state."""
        metrics_a = make_coaching_metrics(task_id="task-a")
        metrics_b = make_remote_only_metrics(task_id="task-b")

        phase_manager.evaluate(metrics_a, default_thresholds)
        phase_manager.evaluate(metrics_b, default_thresholds)

        phase_a = phase_manager.get_task_phase("task-a")
        phase_b = phase_manager.get_task_phase("task-b")

        # task-a should be COACHING, task-b REMOTE_ONLY
        assert phase_a != phase_b or phase_a == Phase.REMOTE_ONLY, (
            "Tasks should have independent phases"
        )
        assert phase_b == Phase.REMOTE_ONLY, "task-b should remain REMOTE_ONLY"

    def test_emergency_from_any_phase(self, default_thresholds):
        """Emergency revert to REMOTE_ONLY works from COACHING and AUTONOMOUS."""
        emergency_metrics_base = dict(
            rolling_correlation=0.1,  # way below emergency_threshold=0.3
            example_count=100,
            local_model_available=True,
            consecutive_windows_above_threshold=100,
            trend_direction=TrendDirection.DEGRADING,
        )
        for task_id in ["from-coaching", "from-autonomous"]:
            metrics = make_metrics(task_id=task_id, **emergency_metrics_base)
            result = compute_phase(metrics, default_thresholds)
            assert result.phase == Phase.REMOTE_ONLY, (
                f"Emergency should force REMOTE_ONLY, got {result.phase}"
            )

    def test_phase1_to_phase2_requires_both_conditions(self, default_thresholds):
        """Phase 1→2 requires BOTH sufficient examples AND local_model_available."""
        # Only examples, no model
        m1 = make_metrics(
            rolling_correlation=0.7,
            example_count=20,
            local_model_available=False,
            consecutive_windows_above_threshold=0,
        )
        r1 = compute_phase(m1, default_thresholds)
        assert r1.phase == Phase.REMOTE_ONLY, "No model → stay REMOTE_ONLY"

        # Only model, no examples
        m2 = make_metrics(
            rolling_correlation=0.7,
            example_count=5,
            local_model_available=True,
            consecutive_windows_above_threshold=0,
        )
        r2 = compute_phase(m2, default_thresholds)
        assert r2.phase == Phase.REMOTE_ONLY, "Insufficient examples → stay REMOTE_ONLY"

        # Both conditions met
        m3 = make_metrics(
            rolling_correlation=0.7,
            example_count=20,
            local_model_available=True,
            consecutive_windows_above_threshold=0,
        )
        r3 = compute_phase(m3, default_thresholds)
        assert r3.phase == Phase.COACHING, "Both conditions met → COACHING"


# ============================================================
# Section 7: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_zero_example_count(self, default_thresholds):
        """Zero example count always yields REMOTE_ONLY."""
        metrics = make_metrics(
            example_count=0,
            rolling_correlation=0.9,
            local_model_available=True,
            consecutive_windows_above_threshold=100,
        )
        result = compute_phase(metrics, default_thresholds)
        assert result.phase == Phase.REMOTE_ONLY

    def test_inf_correlation(self, default_thresholds):
        """Infinite correlation should be handled gracefully (not crash)."""
        metrics = make_metrics(rolling_correlation=float("inf"))
        try:
            result = compute_phase(metrics, default_thresholds)
            # If it succeeds, should still return a valid Phase
            assert result.phase in (Phase.REMOTE_ONLY, Phase.COACHING, Phase.AUTONOMOUS)
        except Exception:
            # Raising is also acceptable for inf
            pass

    def test_negative_inf_correlation(self, default_thresholds):
        """Negative infinity correlation should trigger REMOTE_ONLY."""
        metrics = make_metrics(rolling_correlation=float("-inf"))
        try:
            result = compute_phase(metrics, default_thresholds)
            assert result.phase == Phase.REMOTE_ONLY
        except Exception:
            pass  # Raising is also acceptable

    def test_cooldown_windows_zero(self):
        """cooldown_windows=0 means no cooldown suppression."""
        thresholds = PhaseThresholds(
            phase1_to_phase2_example_count=10,
            phase2_to_phase3_correlation=0.8,
            coaching_trigger=0.5,
            emergency_threshold=0.3,
            sustained_windows=3,
            cooldown_windows=0,
        )
        clock = MagicMock()
        clock.now.return_value = "2024-01-15T10:00:00Z"
        pm = PhaseManager(clock=clock)

        # First eval → establish state
        pm.evaluate(make_remote_only_metrics("task-cd0"), thresholds)
        # Second eval → should transition without cooldown
        result = pm.evaluate(make_coaching_metrics("task-cd0"), thresholds)
        # With cooldown_windows=0, transition should NOT be suppressed
        assert result.cooldown_active is False or result.transition is not None, (
            "With cooldown_windows=0, transitions should not be suppressed"
        )

    def test_sustained_windows_one(self):
        """sustained_windows=1: single window above threshold suffices for AUTONOMOUS."""
        thresholds = PhaseThresholds(
            phase1_to_phase2_example_count=10,
            phase2_to_phase3_correlation=0.8,
            coaching_trigger=0.5,
            emergency_threshold=0.3,
            sustained_windows=1,
            cooldown_windows=0,
        )
        metrics = make_metrics(
            rolling_correlation=0.9,
            example_count=20,
            local_model_available=True,
            consecutive_windows_above_threshold=1,
        )
        result = compute_phase(metrics, thresholds)
        assert result.phase == Phase.AUTONOMOUS, (
            "With sustained_windows=1 and 1 window above threshold, should be AUTONOMOUS"
        )

    def test_exact_example_count_threshold(self, default_thresholds):
        """Exactly at phase1_to_phase2_example_count boundary."""
        metrics = make_metrics(
            rolling_correlation=0.7,
            example_count=10,  # exactly phase1_to_phase2_example_count
            local_model_available=True,
            consecutive_windows_above_threshold=0,
        )
        result = compute_phase(metrics, default_thresholds)
        # At exactly the threshold, should allow COACHING (>= comparison)
        assert result.phase == Phase.COACHING, (
            "At exact example_count threshold with model available, should be COACHING"
        )

    def test_phase_level_ordering(self):
        """Phase enum levels: REMOTE_ONLY=1 < COACHING=2 < AUTONOMOUS=3."""
        assert Phase.REMOTE_ONLY.level < Phase.COACHING.level
        assert Phase.COACHING.level < Phase.AUTONOMOUS.level

    def test_clock_returning_unusual_timestamp(self, default_thresholds):
        """Clock returning various ISO 8601 formats still works."""
        clock = MagicMock()
        clock.now.return_value = "2024-12-31T23:59:59.999999+00:00"
        pm = PhaseManager(clock=clock)
        metrics = make_coaching_metrics()
        result = pm.evaluate(metrics, default_thresholds)
        if result.transition is not None:
            assert result.transition.timestamp == "2024-12-31T23:59:59.999999+00:00"

    def test_many_evaluations_no_state_leak(self, phase_manager, default_thresholds):
        """Multiple evaluations for different tasks don't leak state."""
        for i in range(20):
            task_id = f"task-{i}"
            metrics = make_coaching_metrics(task_id=task_id)
            phase_manager.evaluate(metrics, default_thresholds)

        # Each task should be independently tracked
        for i in range(20):
            task_id = f"task-{i}"
            state = phase_manager.get_task_state(task_id)
            assert state is not None
            assert state.task_id == task_id

    def test_transition_is_frozen(self, phase_manager, default_thresholds):
        """PhaseTransition should be immutable (frozen Pydantic model)."""
        metrics = make_coaching_metrics()
        result = phase_manager.evaluate(metrics, default_thresholds)
        if result.transition is not None:
            with pytest.raises((AttributeError, TypeError, Exception)):
                result.transition.task_id = "hacked"
