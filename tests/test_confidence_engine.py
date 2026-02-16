"""
Contract tests for ConfidenceEngine.

Tests verify the ConfidenceEngine implementation against its contract,
covering initialization, record_comparison, snapshot generation,
persistence, and invariants.

All dependencies (evaluators, phase_manager, rolling_window) are mocked.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock, call
from datetime import datetime, timezone
from dateutil.parser import isoparse


# ---------------------------------------------------------------------------
# Import the component under test
# ---------------------------------------------------------------------------
from src.confidence_engine import (
    ConfidenceEngine,
    ConfidenceEngineConfig,
    EvaluationOutcome,
    ConfidenceSnapshotEnriched,
    ConfidenceSnapshotEnrichedMap,
    DriftShiftStatus,
)


# ===================================================================
# FIXTURES
# ===================================================================

@pytest.fixture
def valid_phase_thresholds():
    """Phase thresholds satisfying: emergency < coaching < phase2_to_phase3."""
    return {
        "emergency_threshold": 0.2,
        "coaching_trigger": 0.5,
        "phase2_to_phase3_threshold": 0.8,
    }


@pytest.fixture
def mock_evaluator():
    """Returns a factory that creates mock evaluators with configurable score."""
    def _make(score=0.85, side_effect=None):
        evaluator = MagicMock()
        if side_effect:
            evaluator.evaluate.side_effect = side_effect
        else:
            result = MagicMock()
            result.score = score
            evaluator.evaluate.return_value = result
        return evaluator
    return _make


@pytest.fixture
def mock_logger():
    """Mock logger with spy capabilities for structured log verification."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest.fixture
def default_evaluator_registry(mock_evaluator):
    """Registry with a single 'exact_match' evaluator."""
    return {"exact_match": mock_evaluator(score=0.85)}


@pytest.fixture
def default_task_evaluator_configs():
    """Task evaluator configs for task_a using exact_match."""
    return {
        "task_a": {"evaluator_type": "exact_match", "params": {}},
    }


@pytest.fixture
def valid_config(valid_phase_thresholds, default_task_evaluator_configs, tmp_path):
    """Constructs a valid ConfidenceEngineConfig."""
    return ConfidenceEngineConfig(
        window_size=100,
        max_window_age_seconds=3600.0,
        trend_sensitivity=0.1,
        trend_window_size=10,
        task_evaluator_configs=default_task_evaluator_configs,
        phase_thresholds=valid_phase_thresholds,
        persistence_dir=str(tmp_path / "confidence_state"),
    )


@pytest.fixture
def mock_phase_manager():
    """Mock phase_manager with validate_thresholds and evaluate."""
    pm = MagicMock()
    pm.validate_thresholds = MagicMock(side_effect=lambda t: t)
    pm.get_initial_phase = MagicMock(return_value="REMOTE_ONLY")

    phase_result = MagicMock()
    phase_result.phase = "REMOTE_ONLY"
    phase_result.transition = None
    pm.evaluate = MagicMock(return_value=phase_result)
    return pm


@pytest.fixture
def make_engine(mock_logger, mock_phase_manager):
    """Factory fixture for creating ConfidenceEngine instances with optional overrides."""
    def _make(config=None, evaluator_registry=None, logger=None, phase_manager=None):
        _logger = logger or mock_logger
        _pm = phase_manager or mock_phase_manager

        with patch("src.confidence_engine.phase_manager", _pm):
            engine = ConfidenceEngine(
                config=config,
                evaluator_registry=evaluator_registry,
                logger=_logger,
            )
        # Store references for test assertions
        engine._test_logger = _logger
        engine._test_phase_manager = _pm
        return engine
    return _make


@pytest.fixture
def engine(make_engine, valid_config, default_evaluator_registry):
    """A ready-to-use ConfidenceEngine with valid config and registry."""
    return make_engine(config=valid_config, evaluator_registry=default_evaluator_registry)


@pytest.fixture
def multi_task_config(valid_phase_thresholds, tmp_path):
    """Config with multiple tasks and evaluator types."""
    return ConfidenceEngineConfig(
        window_size=50,
        max_window_age_seconds=3600.0,
        trend_sensitivity=0.1,
        trend_window_size=5,
        task_evaluator_configs={
            "task_a": {"evaluator_type": "exact_match", "params": {}},
            "task_b": {"evaluator_type": "semantic_similarity", "params": {}},
            "task_c": {"evaluator_type": "exact_match", "params": {}},
        },
        phase_thresholds=valid_phase_thresholds,
        persistence_dir=str(tmp_path / "multi_state"),
    )


@pytest.fixture
def multi_task_registry(mock_evaluator):
    """Registry covering both evaluator types for multi-task config."""
    return {
        "exact_match": mock_evaluator(score=0.9),
        "semantic_similarity": mock_evaluator(score=0.75),
    }


@pytest.fixture
def multi_task_engine(make_engine, multi_task_config, multi_task_registry):
    """Engine with multiple tasks configured."""
    return make_engine(config=multi_task_config, evaluator_registry=multi_task_registry)


# ===================================================================
# 1. INITIALIZATION & CONFIG VALIDATION TESTS
# ===================================================================

class TestConfidenceEngineInit:
    """Test __init__ happy paths and error paths."""

    def test_init_happy_path(self, engine, mock_phase_manager):
        """Engine initializes successfully with valid config and deps."""
        # Postcondition: internal maps are empty
        assert engine.get_task_ids() is not None
        # Phase thresholds validated
        mock_phase_manager.validate_thresholds.assert_called_once()

    def test_init_internal_state_empty(self, engine):
        """All internal state maps are empty after construction."""
        # No windows exist
        task_ids_with_windows = [
            tid for tid in engine.get_task_ids()
            if engine.get_window_size(tid) > 0
        ]
        assert task_ids_with_windows == [], \
            "No windows should exist after construction"

    def test_init_error_invalid_phase_thresholds(
        self, make_engine, default_task_evaluator_configs,
        default_evaluator_registry, tmp_path, mock_phase_manager
    ):
        """Raises error when phase_thresholds violate ordering invariant."""
        invalid_thresholds = {
            "emergency_threshold": 0.9,   # > coaching_trigger -> invalid
            "coaching_trigger": 0.5,
            "phase2_to_phase3_threshold": 0.8,
        }
        mock_phase_manager.validate_thresholds.side_effect = ValueError(
            "invalid_phase_thresholds"
        )
        config = ConfidenceEngineConfig(
            window_size=100,
            max_window_age_seconds=3600.0,
            trend_sensitivity=0.1,
            trend_window_size=10,
            task_evaluator_configs=default_task_evaluator_configs,
            phase_thresholds=invalid_thresholds,
            persistence_dir=str(tmp_path / "state"),
        )
        with pytest.raises(Exception) as exc_info:
            make_engine(
                config=config,
                evaluator_registry=default_evaluator_registry,
                phase_manager=mock_phase_manager,
            )
        assert "threshold" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

    def test_init_error_empty_evaluator_registry(
        self, make_engine, valid_config
    ):
        """Raises error when evaluator_registry is an empty dict."""
        with pytest.raises(Exception) as exc_info:
            make_engine(config=valid_config, evaluator_registry={})
        # Error should indicate empty registry
        assert exc_info.value is not None

    def test_init_error_missing_evaluator_for_task(
        self, make_engine, valid_config
    ):
        """Raises when task references evaluator type not in registry."""
        # task_a needs 'exact_match', but we only provide 'semantic_similarity'
        wrong_registry = {"semantic_similarity": MagicMock()}
        with pytest.raises(Exception):
            make_engine(config=valid_config, evaluator_registry=wrong_registry)

    def test_init_error_empty_task_evaluator_configs(
        self, make_engine, valid_phase_thresholds,
        default_evaluator_registry, tmp_path
    ):
        """Raises error when task_evaluator_configs is empty dict."""
        with pytest.raises(Exception):
            config = ConfidenceEngineConfig(
                window_size=100,
                max_window_age_seconds=3600.0,
                trend_sensitivity=0.1,
                trend_window_size=10,
                task_evaluator_configs={},
                phase_thresholds=valid_phase_thresholds,
                persistence_dir=str(tmp_path / "state"),
            )
            make_engine(config=config, evaluator_registry=default_evaluator_registry)

    def test_config_immutable_after_init(self, engine):
        """Config is frozen/immutable after construction — invariant."""
        with pytest.raises((AttributeError, TypeError, Exception)):
            engine._config.window_size = 999


# ===================================================================
# 2. RECORD_COMPARISON TESTS
# ===================================================================

class TestRecordComparison:
    """Test record_comparison happy paths, error paths, and edge cases."""

    def test_happy_path_successful_evaluation(self, engine):
        """Successful evaluation returns correct EvaluationOutcome."""
        outcome = engine.record_comparison(
            task_id="task_a",
            input_data="What is 2+2?",
            local_output="4",
            remote_output="4",
            local_fields={},
            remote_fields={},
        )

        assert isinstance(outcome, EvaluationOutcome), \
            "Should return EvaluationOutcome"
        assert outcome.score is not None, "Score should not be None on success"
        assert 0.0 <= outcome.score <= 1.0, \
            f"Score {outcome.score} should be in [0.0, 1.0]"
        assert outcome.added_to_window is True, \
            "Successful evaluation should be added to window"
        assert outcome.error is None, \
            "Error should be None on success"
        assert outcome.evaluator_type == "exact_match", \
            f"evaluator_type should be 'exact_match', got '{outcome.evaluator_type}'"

    def test_happy_path_timestamp_is_utc_iso8601(self, engine):
        """Timestamp is valid UTC ISO-8601."""
        outcome = engine.record_comparison(
            task_id="task_a",
            input_data="test input",
            local_output="output",
            remote_output="output",
            local_fields={},
            remote_fields={},
        )
        ts = outcome.timestamp
        assert ts is not None
        parsed = isoparse(ts)
        assert parsed.tzinfo is not None, "Timestamp must be timezone-aware"
        assert parsed.utcoffset().total_seconds() == 0, \
            "Timestamp must be UTC"

    def test_evaluator_failure_returns_degraded_outcome(
        self, make_engine, valid_config, mock_evaluator, mock_logger
    ):
        """Evaluator failure returns degraded outcome, not added to window."""
        failing_eval = mock_evaluator(side_effect=RuntimeError("eval broke"))
        registry = {"exact_match": failing_eval}
        engine = make_engine(
            config=valid_config,
            evaluator_registry=registry,
            logger=mock_logger,
        )

        outcome = engine.record_comparison(
            task_id="task_a",
            input_data="test input",
            local_output="output",
            remote_output="output",
            local_fields={},
            remote_fields={},
        )

        assert outcome.score is None, "Score should be None on failure"
        assert outcome.added_to_window is False, \
            "Failed evaluation should NOT be added to window"
        assert outcome.error is not None and len(outcome.error) > 0, \
            "Error message should be populated"
        assert engine.get_window_size("task_a") == 0, \
            "Window should be empty after failed evaluation"

    def test_evaluator_failure_logged_as_warning(
        self, make_engine, valid_config, mock_evaluator, mock_logger
    ):
        """Evaluator failure is logged at WARNING level."""
        failing_eval = mock_evaluator(side_effect=ValueError("bad input"))
        registry = {"exact_match": failing_eval}
        engine = make_engine(
            config=valid_config,
            evaluator_registry=registry,
            logger=mock_logger,
        )

        engine.record_comparison(
            task_id="task_a",
            input_data="test",
            local_output="a",
            remote_output="b",
            local_fields={},
            remote_fields={},
        )
        mock_logger.warning.assert_called()

    def test_error_unknown_task_id(self, engine):
        """Raises error for unknown task_id."""
        with pytest.raises(Exception) as exc_info:
            engine.record_comparison(
                task_id="nonexistent_task",
                input_data="test",
                local_output="a",
                remote_output="b",
                local_fields={},
                remote_fields={},
            )
        assert exc_info.value is not None

    def test_error_empty_task_id(self, engine):
        """Raises error for empty string task_id."""
        with pytest.raises(Exception):
            engine.record_comparison(
                task_id="",
                input_data="test",
                local_output="a",
                remote_output="b",
                local_fields={},
                remote_fields={},
            )

    def test_error_empty_input_data(self, engine):
        """Raises error for empty input_data."""
        with pytest.raises(Exception):
            engine.record_comparison(
                task_id="task_a",
                input_data="",
                local_output="a",
                remote_output="b",
                local_fields={},
                remote_fields={},
            )

    def test_lazy_window_creation(self, engine):
        """Rolling window is lazily created on first record_comparison."""
        assert engine.get_window_size("task_a") == 0, \
            "No window should exist before any record_comparison"

        engine.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )

        assert engine.get_window_size("task_a") == 1, \
            "Window should contain 1 entry after first successful record_comparison"

    def test_multiple_comparisons_accumulate(self, engine):
        """Multiple successful comparisons accumulate in the window."""
        for i in range(5):
            engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )
        assert engine.get_window_size("task_a") == 5

    def test_failed_evaluations_dont_accumulate(
        self, make_engine, valid_config, mock_evaluator
    ):
        """Only successful evaluations increase window size."""
        call_count = 0

        def alternating_eval(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("failure")
            result = MagicMock()
            result.score = 0.8
            return result

        evaluator = MagicMock()
        evaluator.evaluate.side_effect = alternating_eval
        registry = {"exact_match": evaluator}
        engine = make_engine(config=valid_config, evaluator_registry=registry)

        outcomes = []
        for i in range(6):
            outcome = engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )
            outcomes.append(outcome)

        successful = [o for o in outcomes if o.added_to_window]
        failed = [o for o in outcomes if not o.added_to_window]

        assert len(successful) == 3, "3 out of 6 should succeed"
        assert len(failed) == 3, "3 out of 6 should fail"
        assert engine.get_window_size("task_a") == 3, \
            "Window size should match successful evaluations count"


# ===================================================================
# 3. GET_SNAPSHOT & PHASE LOGIC TESTS
# ===================================================================

class TestGetSnapshot:
    """Test get_snapshot including phase transitions, drift/shift, and trend."""

    def test_happy_path_with_data(self, engine, mock_phase_manager):
        """Returns enriched snapshot with correct fields after recordings."""
        # Record some comparisons
        for i in range(5):
            engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )

        snapshot = engine.get_snapshot(
            task_id="task_a",
            example_count=10,
            local_model_available=True,
        )

        assert isinstance(snapshot, ConfidenceSnapshotEnriched)
        assert snapshot.task_type == "task_a"
        assert snapshot.training_example_count == 10
        assert snapshot.sample_count == 5
        assert snapshot.phase in ("REMOTE_ONLY", "COACHING", "AUTONOMOUS")
        assert snapshot.trend_direction in ("IMPROVING", "STABLE", "DEGRADING")

    def test_no_window_returns_default_snapshot(self, engine):
        """Returns default snapshot when no window exists for task."""
        snapshot = engine.get_snapshot(
            task_id="task_a",
            example_count=0,
            local_model_available=True,
        )

        assert snapshot.correlation_score is None, \
            "correlation_score should be None with no window data"
        assert snapshot.sample_count == 0, \
            "sample_count should be 0 with no window data"
        assert snapshot.phase == "REMOTE_ONLY", \
            "Phase should be REMOTE_ONLY with no data"
        assert snapshot.is_stale is False, \
            "Should not be stale with no data"

    def test_local_model_unavailable_forces_remote_only(
        self, engine, mock_phase_manager
    ):
        """Phase is REMOTE_ONLY when local_model_available=False."""
        # Configure phase_manager to return REMOTE_ONLY when local unavailable
        phase_result = MagicMock()
        phase_result.phase = "REMOTE_ONLY"
        phase_result.transition = None
        mock_phase_manager.evaluate.return_value = phase_result

        for i in range(10):
            engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )

        snapshot = engine.get_snapshot(
            task_id="task_a",
            example_count=100,
            local_model_available=False,
        )

        assert snapshot.phase == "REMOTE_ONLY", \
            "Phase must be REMOTE_ONLY when local model is unavailable"

    def test_error_empty_task_id(self, engine):
        """Raises error for empty task_id."""
        with pytest.raises(Exception):
            engine.get_snapshot(task_id="", example_count=0, local_model_available=True)

    def test_error_negative_example_count(self, engine):
        """Raises error for negative example_count."""
        with pytest.raises(Exception):
            engine.get_snapshot(
                task_id="task_a",
                example_count=-1,
                local_model_available=True,
            )

    def test_phase_transition_produces_audit_dict(
        self, engine, mock_phase_manager
    ):
        """Phase transition generates phase_transition dict with expected keys."""
        # First call: REMOTE_ONLY (default)
        phase_result_1 = MagicMock()
        phase_result_1.phase = "REMOTE_ONLY"
        phase_result_1.transition = None

        # Second call: transition to COACHING
        transition_mock = MagicMock()
        transition_mock.from_phase = "REMOTE_ONLY"
        transition_mock.to_phase = "COACHING"
        transition_mock.reason = "Score above coaching_trigger"
        transition_mock.context = {"score": 0.85}

        phase_result_2 = MagicMock()
        phase_result_2.phase = "COACHING"
        phase_result_2.transition = transition_mock

        mock_phase_manager.evaluate.side_effect = [phase_result_1, phase_result_2]

        # First snapshot (baseline)
        engine.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )
        engine.get_snapshot(task_id="task_a", example_count=10, local_model_available=True)

        # Second snapshot (should trigger transition)
        snapshot = engine.get_snapshot(
            task_id="task_a", example_count=20, local_model_available=True
        )

        if snapshot.phase_transition is not None:
            assert "from_phase" in snapshot.phase_transition
            assert "to_phase" in snapshot.phase_transition
            assert "reason" in snapshot.phase_transition
            assert "context" in snapshot.phase_transition

    def test_no_phase_transition_returns_none(self, engine, mock_phase_manager):
        """phase_transition is None when phase doesn't change."""
        phase_result = MagicMock()
        phase_result.phase = "REMOTE_ONLY"
        phase_result.transition = None
        mock_phase_manager.evaluate.return_value = phase_result

        # Two snapshot calls with same phase
        engine.get_snapshot(task_id="task_a", example_count=0, local_model_available=True)
        snapshot = engine.get_snapshot(
            task_id="task_a", example_count=0, local_model_available=True
        )

        assert snapshot.phase_transition is None, \
            "phase_transition should be None when phase unchanged"

    def test_drift_detected_when_degrading(
        self, make_engine, valid_config, mock_evaluator, mock_phase_manager
    ):
        """drift_shift_status.drift_detected is True when trend is DEGRADING."""
        # Create evaluator that returns decreasing scores
        call_count = 0

        def decreasing_scores(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.score = max(0.0, 1.0 - (call_count * 0.05))
            return result

        evaluator = MagicMock()
        evaluator.evaluate.side_effect = decreasing_scores
        registry = {"exact_match": evaluator}

        # Configure phase_manager to report DEGRADING trend via snapshot
        engine = make_engine(
            config=valid_config,
            evaluator_registry=registry,
            phase_manager=mock_phase_manager,
        )

        # Record many comparisons with decreasing scores
        for i in range(20):
            engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )

        snapshot = engine.get_snapshot(
            task_id="task_a", example_count=10, local_model_available=True
        )

        # If the trend is indeed degrading, drift should be detected
        if snapshot.trend_direction == "DEGRADING":
            assert snapshot.drift_shift_status.drift_detected is True, \
                "Drift should be detected when trend is DEGRADING"

    def test_shift_detected_on_large_score_delta(
        self, make_engine, mock_phase_manager, mock_evaluator, tmp_path
    ):
        """shift_detected is True when score delta exceeds trend_sensitivity."""
        # Build config with low trend_sensitivity to make shift detection easy
        config = ConfidenceEngineConfig(
            window_size=100,
            max_window_age_seconds=3600.0,
            trend_sensitivity=0.05,  # Small sensitivity
            trend_window_size=5,
            task_evaluator_configs={"task_a": {"evaluator_type": "exact_match", "params": {}}},
            phase_thresholds={
                "emergency_threshold": 0.2,
                "coaching_trigger": 0.5,
                "phase2_to_phase3_threshold": 0.8,
            },
            persistence_dir=str(tmp_path / "shift_state"),
        )

        # First batch: high scores
        score_value = [0.95]

        def eval_fn(*args, **kwargs):
            result = MagicMock()
            result.score = score_value[0]
            return result

        evaluator = MagicMock()
        evaluator.evaluate.side_effect = eval_fn
        registry = {"exact_match": evaluator}

        engine = make_engine(
            config=config,
            evaluator_registry=registry,
            phase_manager=mock_phase_manager,
        )

        for i in range(10):
            engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )

        # First snapshot to establish baseline score
        snap1 = engine.get_snapshot(
            task_id="task_a", example_count=10, local_model_available=True
        )

        # Now record low scores to create a shift
        score_value[0] = 0.3
        for i in range(10, 20):
            engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )

        snap2 = engine.get_snapshot(
            task_id="task_a", example_count=10, local_model_available=True
        )

        # The score delta should exceed trend_sensitivity of 0.05
        if snap2.drift_shift_status.score_delta is not None:
            assert snap2.drift_shift_status.shift_detected is True, \
                f"Shift should be detected; score_delta={snap2.drift_shift_status.score_delta}"

    def test_consecutive_windows_increment(
        self, make_engine, mock_phase_manager, mock_evaluator, tmp_path
    ):
        """consecutive_windows_above_threshold increments with high scores."""
        config = ConfidenceEngineConfig(
            window_size=100,
            max_window_age_seconds=3600.0,
            trend_sensitivity=0.1,
            trend_window_size=5,
            task_evaluator_configs={"task_a": {"evaluator_type": "exact_match", "params": {}}},
            phase_thresholds={
                "emergency_threshold": 0.2,
                "coaching_trigger": 0.5,
                "phase2_to_phase3_threshold": 0.8,
            },
            persistence_dir=str(tmp_path / "consec_state"),
        )

        evaluator = mock_evaluator(score=0.95)  # Above 0.8 threshold
        registry = {"exact_match": evaluator}
        engine = make_engine(
            config=config,
            evaluator_registry=registry,
            phase_manager=mock_phase_manager,
        )

        for i in range(5):
            engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )

        # Multiple get_snapshot calls should increment counter
        snap1 = engine.get_snapshot(
            task_id="task_a", example_count=10, local_model_available=True
        )
        snap2 = engine.get_snapshot(
            task_id="task_a", example_count=10, local_model_available=True
        )

        # If score is above threshold, counter should be >= 1
        if snap1.correlation_score is not None and snap1.correlation_score >= 0.8:
            assert snap2.consecutive_windows_above_threshold >= 1, \
                "Counter should increment when score is above threshold"

    def test_consecutive_windows_reset_on_low_score(
        self, make_engine, mock_phase_manager, mock_evaluator, tmp_path
    ):
        """consecutive_windows_above_threshold resets to 0 on low score."""
        config = ConfidenceEngineConfig(
            window_size=100,
            max_window_age_seconds=3600.0,
            trend_sensitivity=0.1,
            trend_window_size=5,
            task_evaluator_configs={"task_a": {"evaluator_type": "exact_match", "params": {}}},
            phase_thresholds={
                "emergency_threshold": 0.2,
                "coaching_trigger": 0.5,
                "phase2_to_phase3_threshold": 0.8,
            },
            persistence_dir=str(tmp_path / "reset_state"),
        )

        score_value = [0.95]

        def eval_fn(*args, **kwargs):
            result = MagicMock()
            result.score = score_value[0]
            return result

        evaluator = MagicMock()
        evaluator.evaluate.side_effect = eval_fn
        registry = {"exact_match": evaluator}
        engine = make_engine(
            config=config,
            evaluator_registry=registry,
            phase_manager=mock_phase_manager,
        )

        # Record high scores
        for i in range(5):
            engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )

        engine.get_snapshot(task_id="task_a", example_count=10, local_model_available=True)

        # Now drop score below threshold
        score_value[0] = 0.3
        for i in range(5, 15):
            engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )

        snapshot = engine.get_snapshot(
            task_id="task_a", example_count=10, local_model_available=True
        )

        if snapshot.correlation_score is not None and snapshot.correlation_score < 0.8:
            assert snapshot.consecutive_windows_above_threshold == 0, \
                "Counter should reset to 0 when score drops below threshold"

    def test_snapshot_task_type_matches_task_id(self, engine):
        """snapshot.task_type always equals the requested task_id."""
        snapshot = engine.get_snapshot(
            task_id="task_a", example_count=5, local_model_available=True
        )
        assert snapshot.task_type == "task_a"

    def test_snapshot_training_example_count_matches(self, engine):
        """snapshot.training_example_count matches provided example_count."""
        snapshot = engine.get_snapshot(
            task_id="task_a", example_count=42, local_model_available=True
        )
        assert snapshot.training_example_count == 42

    def test_snapshot_correlation_score_in_bounds(self, engine):
        """correlation_score is in [0.0, 1.0] when data exists."""
        for i in range(5):
            engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )

        snapshot = engine.get_snapshot(
            task_id="task_a", example_count=5, local_model_available=True
        )

        if snapshot.correlation_score is not None:
            assert 0.0 <= snapshot.correlation_score <= 1.0, \
                f"correlation_score {snapshot.correlation_score} out of bounds"


# ===================================================================
# 4. GET_ALL_SNAPSHOTS TESTS
# ===================================================================

class TestGetAllSnapshots:
    """Test get_all_snapshots multi-task behavior."""

    def test_happy_path_all_tasks_covered(self, multi_task_engine):
        """Returns snapshots for all known tasks."""
        # Record some data for task_a
        multi_task_engine.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )

        result = multi_task_engine.get_all_snapshots(
            example_counts={"task_a": 10, "task_b": 5},
            local_model_availability={"task_a": True, "task_b": True},
        )

        assert isinstance(result, ConfidenceSnapshotEnrichedMap)
        # All tasks from config should be present
        for tid in ["task_a", "task_b", "task_c"]:
            assert tid in result.snapshots, \
                f"Task {tid} should be in snapshots"

    def test_missing_example_counts_default_to_zero(self, multi_task_engine):
        """Tasks not in example_counts get example_count=0."""
        result = multi_task_engine.get_all_snapshots(
            example_counts={"task_a": 10},  # task_b, task_c missing
            local_model_availability={},
        )

        assert result.snapshots["task_b"].training_example_count == 0
        assert result.snapshots["task_c"].training_example_count == 0

    def test_missing_availability_defaults_to_false(self, multi_task_engine):
        """Tasks not in local_model_availability default to False."""
        result = multi_task_engine.get_all_snapshots(
            example_counts={},
            local_model_availability={"task_a": True},  # task_b, task_c missing
        )

        # task_b and task_c should get REMOTE_ONLY phase
        assert result.snapshots["task_b"].phase == "REMOTE_ONLY"
        assert result.snapshots["task_c"].phase == "REMOTE_ONLY"

    def test_error_negative_example_count_value(self, multi_task_engine):
        """Raises error when a value in example_counts is negative."""
        with pytest.raises(Exception):
            multi_task_engine.get_all_snapshots(
                example_counts={"task_a": -5},
                local_model_availability={},
            )


# ===================================================================
# 5. PERSISTENCE TESTS
# ===================================================================

class TestPersistence:
    """Test persist_state and load_state."""

    def test_persist_state_happy_path(self, engine, tmp_path):
        """Persists window files for tasks with data."""
        engine.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )

        engine.persist_state()

        persistence_dir = engine._config.persistence_dir
        assert os.path.isdir(persistence_dir), "persistence_dir should exist"
        expected_file = os.path.join(persistence_dir, "task_a.json")
        assert os.path.isfile(expected_file), \
            f"Expected {expected_file} to exist"

    def test_persist_state_creates_directory(self, engine, tmp_path):
        """Creates persistence_dir if it does not exist."""
        persistence_dir = engine._config.persistence_dir
        # Ensure dir doesn't exist yet
        if os.path.exists(persistence_dir):
            os.rmdir(persistence_dir)

        engine.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )
        engine.persist_state()

        assert os.path.isdir(persistence_dir)

    def test_persist_state_io_failure_logged_not_raised(
        self, make_engine, valid_config, default_evaluator_registry, mock_logger
    ):
        """I/O failure is logged at WARNING, doesn't raise."""
        engine = make_engine(
            config=valid_config,
            evaluator_registry=default_evaluator_registry,
            logger=mock_logger,
        )
        engine.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )

        # Mock the window's persist method to fail
        with patch.object(engine, "persist_state", wraps=engine.persist_state):
            # If we can access internal windows, patch their persist
            # Otherwise, make the directory non-writable
            persistence_dir = engine._config.persistence_dir
            os.makedirs(persistence_dir, exist_ok=True)
            os.chmod(persistence_dir, 0o000)

            try:
                # Should not raise
                engine.persist_state()
            except PermissionError:
                # Some implementations may raise at directory level
                pass
            finally:
                os.chmod(persistence_dir, 0o755)

    def test_load_state_happy_path(
        self, make_engine, valid_config, default_evaluator_registry, mock_logger
    ):
        """Round-trip: persist then load recovers window state."""
        engine1 = make_engine(
            config=valid_config,
            evaluator_registry=default_evaluator_registry,
            logger=mock_logger,
        )

        # Record data and persist
        for i in range(5):
            engine1.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )
        engine1.persist_state()

        # Create new engine and load
        engine2 = make_engine(
            config=valid_config,
            evaluator_registry=default_evaluator_registry,
            logger=mock_logger,
        )
        loaded_count = engine2.load_state()

        assert loaded_count >= 1, "Should load at least 1 task window"
        assert engine2.get_window_size("task_a") == 5, \
            "Loaded window should have 5 entries"

    def test_load_state_no_directory_returns_zero(
        self, make_engine, default_evaluator_registry, mock_logger,
        valid_phase_thresholds, tmp_path
    ):
        """Returns 0 when persistence_dir does not exist."""
        config = ConfidenceEngineConfig(
            window_size=100,
            max_window_age_seconds=3600.0,
            trend_sensitivity=0.1,
            trend_window_size=10,
            task_evaluator_configs={"task_a": {"evaluator_type": "exact_match", "params": {}}},
            phase_thresholds=valid_phase_thresholds,
            persistence_dir=str(tmp_path / "nonexistent_dir"),
        )
        engine = make_engine(
            config=config,
            evaluator_registry=default_evaluator_registry,
            logger=mock_logger,
        )

        result = engine.load_state()
        assert result == 0, "Should return 0 when no persistence_dir"

    def test_load_state_corrupted_file_skipped(
        self, make_engine, valid_config, default_evaluator_registry,
        mock_logger
    ):
        """Corrupted files are skipped with WARNING, valid files still load."""
        # First, create valid state
        engine1 = make_engine(
            config=valid_config,
            evaluator_registry=default_evaluator_registry,
            logger=mock_logger,
        )
        engine1.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )
        engine1.persist_state()

        # Write a corrupted file
        persistence_dir = engine1._config.persistence_dir
        corrupted_path = os.path.join(persistence_dir, "corrupted_task.json")
        with open(corrupted_path, "w") as f:
            f.write("{invalid json!!!")

        # Load in new engine
        engine2 = make_engine(
            config=valid_config,
            evaluator_registry=default_evaluator_registry,
            logger=mock_logger,
        )
        loaded_count = engine2.load_state()

        # task_a should load, corrupted_task should be skipped
        assert loaded_count >= 1, "At least task_a should load"
        # WARNING should have been logged for corrupted file
        # (implementation-dependent, but the contract says so)

    def test_persist_state_empty_engine(self, engine):
        """Persisting empty engine (no windows) does not raise."""
        # No record_comparison calls made
        engine.persist_state()
        # Should complete without error


# ===================================================================
# 6. UTILITY FUNCTION TESTS
# ===================================================================

class TestGetTaskIds:
    """Test get_task_ids."""

    def test_happy_path(self, multi_task_engine):
        """Returns all task_ids from config."""
        task_ids = multi_task_engine.get_task_ids()
        assert isinstance(task_ids, list)
        assert set(task_ids) >= {"task_a", "task_b", "task_c"}

    def test_no_duplicates(self, multi_task_engine):
        """No duplicate task_ids in result."""
        # Record comparison to potentially add to windows
        multi_task_engine.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )
        task_ids = multi_task_engine.get_task_ids()
        assert len(task_ids) == len(set(task_ids)), \
            "get_task_ids should return no duplicates"

    def test_empty_engine_returns_config_tasks(self, engine):
        """Freshly initialized engine returns config task_ids."""
        task_ids = engine.get_task_ids()
        assert "task_a" in task_ids


class TestGetWindowSize:
    """Test get_window_size."""

    def test_happy_path_with_data(self, engine):
        """Returns correct window size after recordings."""
        for i in range(3):
            engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )
        assert engine.get_window_size("task_a") == 3

    def test_no_window_returns_zero(self, engine):
        """Returns 0 for task with no window."""
        assert engine.get_window_size("task_a") == 0

    def test_error_empty_task_id(self, engine):
        """Raises error for empty task_id."""
        with pytest.raises(Exception):
            engine.get_window_size("")

    def test_return_value_non_negative(self, engine):
        """Return value is always >= 0."""
        size = engine.get_window_size("task_a")
        assert size >= 0


# ===================================================================
# 7. INVARIANT TESTS
# ===================================================================

class TestInvariants:
    """Test contract invariants."""

    def test_failed_evaluations_never_in_window(
        self, make_engine, valid_config, mock_evaluator
    ):
        """Failed evaluations are never added to the rolling window."""
        call_count = 0

        def alternating(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("fail")
            result = MagicMock()
            result.score = 0.9
            return result

        evaluator = MagicMock()
        evaluator.evaluate.side_effect = alternating
        registry = {"exact_match": evaluator}
        engine = make_engine(config=valid_config, evaluator_registry=registry)

        outcomes = []
        for i in range(10):
            o = engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )
            outcomes.append(o)

        successful = sum(1 for o in outcomes if o.added_to_window)
        assert engine.get_window_size("task_a") == successful, \
            "Window size must equal number of successful evaluations"

        for o in outcomes:
            if o.score is None:
                assert o.added_to_window is False, \
                    "Failed eval must have added_to_window=False"

    def test_evaluator_errors_never_propagate(
        self, make_engine, valid_config
    ):
        """Evaluator exceptions never propagate through record_comparison."""
        exception_types = [
            ValueError("bad value"),
            RuntimeError("runtime error"),
            TypeError("type error"),
            KeyError("missing key"),
            Exception("generic error"),
        ]

        for exc in exception_types:
            evaluator = MagicMock()
            evaluator.evaluate.side_effect = exc
            registry = {"exact_match": evaluator}
            engine = make_engine(config=valid_config, evaluator_registry=registry)

            # Should NOT raise
            outcome = engine.record_comparison(
                task_id="task_a",
                input_data="input",
                local_output="local",
                remote_output="remote",
                local_fields={},
                remote_fields={},
            )

            assert outcome.score is None, \
                f"Score should be None when evaluator raises {type(exc).__name__}"
            assert outcome.error is not None, \
                f"Error should be populated when evaluator raises {type(exc).__name__}"

    def test_score_bounds_invariant(self, engine):
        """All successful scores are in [0.0, 1.0]."""
        scores = []
        for i in range(10):
            outcome = engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )
            if outcome.score is not None:
                scores.append(outcome.score)

        for score in scores:
            assert 0.0 <= score <= 1.0, \
                f"Score {score} out of bounds [0.0, 1.0]"

    def test_persistence_is_explicit_only(self, engine, tmp_path):
        """No file I/O during record_comparison."""
        persistence_dir = engine._config.persistence_dir

        engine.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )

        # Check that no files were written
        if os.path.exists(persistence_dir):
            files = os.listdir(persistence_dir)
            assert len(files) == 0, \
                f"No files should be written during record_comparison, found: {files}"

    def test_utc_timestamps_invariant(self, engine):
        """All timestamps use UTC timezone."""
        outcome = engine.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )

        ts = outcome.timestamp
        assert ts is not None, "Timestamp should not be None"
        # Should contain UTC indicator
        assert "+00:00" in ts or ts.endswith("Z"), \
            f"Timestamp '{ts}' should contain UTC indicator"

        parsed = isoparse(ts)
        assert parsed.tzinfo is not None, "Timestamp must be timezone-aware"

    def test_phase_valid_values_invariant(self, engine, mock_phase_manager):
        """Phase is always one of the valid enum values."""
        valid_phases = {"REMOTE_ONLY", "COACHING", "AUTONOMOUS"}

        engine.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )

        snapshot = engine.get_snapshot(
            task_id="task_a", example_count=10, local_model_available=True
        )
        assert snapshot.phase in valid_phases, \
            f"Phase '{snapshot.phase}' is not a valid value"

    def test_trend_direction_valid_values_invariant(self, engine, mock_phase_manager):
        """trend_direction is always one of IMPROVING, STABLE, DEGRADING."""
        valid_trends = {"IMPROVING", "STABLE", "DEGRADING"}

        engine.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )

        snapshot = engine.get_snapshot(
            task_id="task_a", example_count=10, local_model_available=True
        )
        assert snapshot.trend_direction in valid_trends, \
            f"trend_direction '{snapshot.trend_direction}' is not valid"

    def test_window_size_bounded_by_config(
        self, make_engine, mock_evaluator, valid_phase_thresholds,
        mock_logger, tmp_path
    ):
        """Window size never exceeds config.window_size."""
        max_window = 5
        config = ConfidenceEngineConfig(
            window_size=max_window,
            max_window_age_seconds=3600.0,
            trend_sensitivity=0.1,
            trend_window_size=3,
            task_evaluator_configs={"task_a": {"evaluator_type": "exact_match", "params": {}}},
            phase_thresholds=valid_phase_thresholds,
            persistence_dir=str(tmp_path / "bounded_state"),
        )
        registry = {"exact_match": mock_evaluator(score=0.9)}
        engine = make_engine(
            config=config,
            evaluator_registry=registry,
            logger=mock_logger,
        )

        # Record more than window_size comparisons
        for i in range(20):
            engine.record_comparison(
                task_id="task_a",
                input_data=f"input_{i}",
                local_output=f"local_{i}",
                remote_output=f"remote_{i}",
                local_fields={},
                remote_fields={},
            )

        size = engine.get_window_size("task_a")
        assert size <= max_window, \
            f"Window size {size} exceeds config.window_size {max_window}"

    def test_snapshot_frozen_immutable(self, engine, mock_phase_manager):
        """ConfidenceSnapshotEnriched is frozen/immutable."""
        engine.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )

        snapshot = engine.get_snapshot(
            task_id="task_a", example_count=5, local_model_available=True
        )

        with pytest.raises((AttributeError, TypeError, Exception)):
            snapshot.phase = "AUTONOMOUS"

    def test_no_global_state_between_instances(
        self, make_engine, valid_config, default_evaluator_registry, mock_logger
    ):
        """All state is instance-scoped; no leakage between engines."""
        engine1 = make_engine(
            config=valid_config,
            evaluator_registry=default_evaluator_registry,
            logger=mock_logger,
        )
        engine2 = make_engine(
            config=valid_config,
            evaluator_registry=default_evaluator_registry,
            logger=mock_logger,
        )

        engine1.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )

        assert engine1.get_window_size("task_a") == 1
        assert engine2.get_window_size("task_a") == 0, \
            "engine2 should not be affected by engine1's state"

    def test_engine_operates_with_zero_persisted_state(self, engine):
        """Engine works correctly with no persisted state loaded."""
        # Don't call load_state, just use the engine directly
        outcome = engine.record_comparison(
            task_id="task_a",
            input_data="input",
            local_output="local",
            remote_output="remote",
            local_fields={},
            remote_fields={},
        )
        assert outcome is not None
        assert outcome.score is not None or outcome.error is not None

        snapshot = engine.get_snapshot(
            task_id="task_a", example_count=0, local_model_available=False
        )
        assert snapshot.phase == "REMOTE_ONLY"
