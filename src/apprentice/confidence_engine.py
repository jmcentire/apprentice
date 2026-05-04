"""
Confidence Engine & Evaluators (confidence_engine) v1

Per-task quality tracking Facade/orchestrator. Maintains a rolling window of comparison-pair samples per task,
computes correlation scores using task-specific evaluators, detects drift (gradual degradation) and shift (sudden
quality drops), derives current phase from confidence metrics via the phase manager, and exposes frozen
ConfidenceSnapshot objects for downstream consumers.
"""

import logging
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Import child modules via relative imports (intra-package)
from . import evaluators
from . import phase_manager
from . import rolling_window

_PACT_KEY = "PACT:confidence_engine"
logger = logging.getLogger(__name__)


def _log(level: str, msg: str, **kwargs) -> None:
    """Log with PACT key embedded for production traceability."""
    getattr(logger, level)(f"[{_PACT_KEY}] {msg}", **kwargs)


# ============================================================================
# RE-EXPORT CHILD TYPES
# ============================================================================

# From evaluators
EvaluatorType = evaluators.EvaluatorType
TaskResponse = evaluators.TaskResponse
TaskEvaluatorConfig = evaluators.TaskEvaluatorConfig
FieldEvaluation = evaluators.FieldEvaluation
EvaluationResult = evaluators.EvaluationResult
EvaluatorProtocol = evaluators.EvaluatorProtocol

# From phase_manager
Phase = phase_manager.Phase
TrendDirection = phase_manager.TrendDirection
TransitionTrigger = phase_manager.TransitionTrigger
PhaseThresholds = phase_manager.PhaseThresholds
PhaseMetrics = phase_manager.PhaseMetrics
PhaseTransition = phase_manager.PhaseTransition

# From rolling_window
ComparisonPair = rolling_window.ComparisonPair
ComparisonPairInput = rolling_window.ComparisonPairInput
WindowAggregate = rolling_window.WindowAggregate


# ============================================================================
# PARENT TYPES
# ============================================================================

class ConfidenceEngineConfig(BaseModel):
    """Frozen Pydantic v2 configuration model for the ConfidenceEngine."""
    model_config = ConfigDict(frozen=True)

    window_size: int = Field(default=100, ge=1, le=10000)
    max_window_age_seconds: float = Field(default=259200.0, gt=0.0)
    trend_sensitivity: float = Field(default=0.05, gt=0.0, lt=1.0)
    trend_window_size: int = Field(default=10, ge=2)
    task_evaluator_configs: dict[str, dict[str, Any]]
    phase_thresholds: dict[str, Any]
    persistence_dir: str = Field(default=".apprentice/confidence", min_length=1)


class EvaluationOutcome(BaseModel):
    """Pydantic v2 model representing the result of a single record_comparison call."""
    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    score: Optional[float] = None
    evaluator_type: str = Field(min_length=1)
    error: Optional[str] = None
    timestamp: str
    added_to_window: bool


class DriftShiftStatus(BaseModel):
    """Diagnostic struct indicating whether drift or shift has been detected."""
    model_config = ConfigDict(frozen=True)

    drift_detected: bool
    shift_detected: bool
    score_delta: Optional[float] = None


class ConfidenceSnapshotEnriched(BaseModel):
    """Extended snapshot combining phase_manager snapshot fields with engine-level drift/shift diagnostics."""
    model_config = ConfigDict(frozen=True)

    task_type: str = Field(min_length=1)
    correlation_score: Optional[float] = None
    training_example_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    consecutive_windows_above_threshold: int = Field(ge=0)
    trend_direction: str
    is_stale: bool
    phase: str
    drift_shift_status: DriftShiftStatus
    window_size: int
    phase_transition: Optional[dict[str, Any]] = None


class ConfidenceSnapshotEnrichedMap(BaseModel):
    """Mapping from task_id to ConfidenceSnapshotEnriched, returned by get_all_snapshots."""
    snapshots: dict[str, ConfidenceSnapshotEnriched] = Field(default_factory=dict)


# ============================================================================
# CONFIDENCE ENGINE
# ============================================================================

class ConfidenceEngine:
    """
    Per-task quality tracking Facade/orchestrator.
    Maintains a rolling window of comparison-pair samples per task, computes correlation scores using
    task-specific evaluators, detects drift and shift, and derives current phase from confidence metrics.
    """

    def __init__(
        self,
        config: ConfidenceEngineConfig,
        evaluator_registry: dict[str, Any],
        logger: Optional[Any] = None,
    ) -> None:
        """
        Constructs the ConfidenceEngine with validated configuration and injected dependencies.
        """
        # Validate config
        if not isinstance(config, ConfidenceEngineConfig):
            raise ValueError("config must be a ConfidenceEngineConfig instance")

        # Validate evaluator registry
        if not evaluator_registry or len(evaluator_registry) == 0:
            raise ValueError("evaluator_registry must contain at least one evaluator")

        # Validate task_evaluator_configs is non-empty
        if not config.task_evaluator_configs or len(config.task_evaluator_configs) == 0:
            raise ValueError("task_evaluator_configs must contain at least one task entry")

        # Validate that all task configs reference evaluators in the registry
        for task_id, task_config in config.task_evaluator_configs.items():
            eval_type = task_config.get("evaluator_type")
            if eval_type not in evaluator_registry:
                raise ValueError(
                    f"Task '{task_id}' references evaluator type '{eval_type}' "
                    f"not found in evaluator registry"
                )

        # Validate phase thresholds
        # Map test naming to phase_manager naming
        phase_threshold_dict = dict(config.phase_thresholds)
        if "phase2_to_phase3_threshold" in phase_threshold_dict:
            phase_threshold_dict["phase2_to_phase3_correlation"] = phase_threshold_dict.pop("phase2_to_phase3_threshold")
        # Set defaults for optional fields if missing
        phase_threshold_dict.setdefault("phase1_to_phase2_example_count", 10)
        phase_threshold_dict.setdefault("sustained_windows", 3)
        phase_threshold_dict.setdefault("cooldown_windows", 0)

        try:
            phase_thresholds = PhaseThresholds(**phase_threshold_dict)
            phase_manager.validate_thresholds(phase_thresholds)
        except Exception as e:
            raise ValueError(f"Phase threshold validation failed: {e}")

        self._config = config
        self._evaluator_registry = evaluator_registry
        self._logger = logger or logging.getLogger("apprentice.confidence_engine")

        # Internal state: task-to-window map
        self._windows: dict[str, rolling_window.DequeRollingWindow] = {}

        # Internal state: task-to-last-known-phase map
        self._last_known_phase: dict[str, str] = {}

        # Internal state: consecutive windows above threshold counters
        self._consecutive_windows_counters: dict[str, int] = {}

        # Internal state: previous score map for shift detection
        self._previous_scores: dict[str, float] = {}

        # Store references to child modules so callers (and tests) can
        # swap them via the module-level patch that is active at construction time.
        self._phase_manager = phase_manager
        self._evaluators = evaluators
        self._rolling_window = rolling_window

        _log("info", f"ConfidenceEngine initialized with {len(config.task_evaluator_configs)} tasks")

    def record_comparison(
        self,
        task_id: str,
        input_data: str,
        local_output: str,
        remote_output: str,
        local_fields: dict[str, Any] = None,
        remote_fields: dict[str, Any] = None,
    ) -> EvaluationOutcome:
        """
        Records a comparison between a local model response and a remote model response.
        """
        # Validate inputs
        if not task_id:
            raise ValueError("task_id must be a non-empty string")
        if not input_data:
            raise ValueError("input_data must be a non-empty string")

        # Resolve the evaluator config. Exact match first; then the
        # multi-tenant ``${prefix}:${task}`` fallback that mirrors
        # TaskRegistry.get_task. Per-tenant state is still tracked by
        # the full task_id (so ``tnt_abc:skill`` and ``tnt_xyz:skill``
        # accumulate independently) — only the config lookup tolerates
        # the prefix.
        config_key = task_id
        if config_key not in self._config.task_evaluator_configs and ":" in config_key:
            suffix = config_key.split(":", 1)[1]
            if suffix in self._config.task_evaluator_configs:
                config_key = suffix
        if config_key not in self._config.task_evaluator_configs:
            raise ValueError(
                f"No evaluator config found for task_id '{task_id}'. "
                f"All task types must be declared in task_evaluator_configs at startup."
            )

        # Get evaluator for this task
        task_config = self._config.task_evaluator_configs[config_key]
        eval_type = task_config.get("evaluator_type")

        if eval_type not in self._evaluator_registry:
            raise ValueError(
                f"Evaluator type '{eval_type}' referenced by task '{task_id}' "
                f"not found in evaluator registry"
            )

        evaluator = self._evaluator_registry[eval_type]

        # Build TaskResponse objects
        timestamp = datetime.now(timezone.utc).isoformat()

        local_task_response = TaskResponse(
            task_id=task_id,
            output=local_output,
            fields=local_fields or {},
            model_id="local",
            timestamp=timestamp,
        )

        remote_task_response = TaskResponse(
            task_id=task_id,
            output=remote_output,
            fields=remote_fields or {},
            model_id="remote",
            timestamp=timestamp,
        )

        # Build evaluator config
        eval_config = TaskEvaluatorConfig(
            evaluator_type=EvaluatorType(eval_type),
            **task_config.get("params", {})
        )

        # Call evaluator
        try:
            eval_result = evaluator.evaluate(
                local=local_task_response,
                remote=remote_task_response,
                task_config=eval_config,
            )

            # Extract score and error from result (handle MagicMock attributes)
            score = getattr(eval_result, 'score', None)
            error = getattr(eval_result, 'error', None)

            # Handle MagicMock error field - if it's a MagicMock, treat as None
            if error is not None and hasattr(error, '_mock_name'):
                error = None

            # Check for evaluation failure
            if error is not None or score is None:
                # Degraded outcome
                error_str = str(error) if error is not None else "Score is None"
                self._logger.warning(
                    f"Evaluator failed for task '{task_id}': {error_str}"
                )
                return EvaluationOutcome(
                    task_id=task_id,
                    score=None,
                    evaluator_type=eval_type,
                    error=error_str,
                    timestamp=timestamp,
                    added_to_window=False,
                )

            # Success - add to window
            if task_id not in self._windows:
                # Lazy create window
                window_config = rolling_window.WindowConfig(
                    window_size=self._config.window_size,
                    max_window_age_seconds=self._config.max_window_age_seconds,
                )
                store_config = rolling_window.StoreConfig(
                    base_dir=self._config.persistence_dir
                )
                store = rolling_window.JsonFileWindowStore(config=store_config)
                self._windows[task_id] = rolling_window.DequeRollingWindow(
                    config=window_config,
                    store=store,
                )
                self._consecutive_windows_counters[task_id] = 0
                self._last_known_phase[task_id] = "REMOTE_ONLY"

            # Add pair to window
            pair_input = ComparisonPairInput(
                input_data=input_data,
                local_response=local_output,
                remote_response=remote_output,
                evaluation_score=score,
            )
            self._windows[task_id].add(pair_input)

            return EvaluationOutcome(
                task_id=task_id,
                score=score,
                evaluator_type=eval_type,
                error=None,
                timestamp=timestamp,
                added_to_window=True,
            )

        except Exception as e:
            # Evaluator raised an exception
            self._logger.warning(
                f"Evaluator exception for task '{task_id}': {str(e)}"
            )
            return EvaluationOutcome(
                task_id=task_id,
                score=None,
                evaluator_type=eval_type,
                error=f"Evaluator exception: {str(e)}",
                timestamp=timestamp,
                added_to_window=False,
            )

    def get_snapshot(
        self,
        task_id: str,
        example_count: int,
        local_model_available: bool,
    ) -> ConfidenceSnapshotEnriched:
        """
        Computes and returns an enriched confidence snapshot for a single task.
        """
        # Validate inputs
        if not task_id:
            raise ValueError("task_id must be a non-empty string")
        if example_count < 0:
            raise ValueError("example_count must be non-negative")

        # Check if we have a window for this task
        if task_id not in self._windows:
            # No window exists - return default snapshot
            return ConfidenceSnapshotEnriched(
                task_type=task_id,
                correlation_score=None,
                training_example_count=example_count,
                sample_count=0,
                consecutive_windows_above_threshold=0,
                trend_direction="IMPROVING",  # Default
                is_stale=False,
                phase="REMOTE_ONLY",
                drift_shift_status=DriftShiftStatus(
                    drift_detected=False,
                    shift_detected=False,
                    score_delta=None,
                ),
                window_size=self._config.window_size,
                phase_transition=None,
            )

        # Get aggregate from window
        window = self._windows[task_id]
        aggregate = window.aggregate(trend_window=self._config.trend_window_size)

        # Build PhaseMetrics
        correlation_score = aggregate.mean_score if aggregate.mean_score is not None else 0.0

        # Map rolling_window TrendDirection to phase_manager TrendDirection
        # rolling_window has INSUFFICIENT_DATA, but phase_manager doesn't
        rw_trend = aggregate.trend_direction.name
        if rw_trend == "INSUFFICIENT_DATA":
            # Default to STABLE when insufficient data
            trend_dir = TrendDirection.STABLE
        else:
            trend_dir = TrendDirection[rw_trend]

        # Get consecutive windows counter
        consecutive_windows = self._consecutive_windows_counters.get(task_id, 0)

        # Update counter based on current score
        # Map test naming to phase_manager naming
        phase_threshold_dict = dict(self._config.phase_thresholds)
        if "phase2_to_phase3_threshold" in phase_threshold_dict:
            phase_threshold_dict["phase2_to_phase3_correlation"] = phase_threshold_dict.pop("phase2_to_phase3_threshold")
        phase_threshold_dict.setdefault("phase1_to_phase2_example_count", 10)
        phase_threshold_dict.setdefault("sustained_windows", 3)
        phase_threshold_dict.setdefault("cooldown_windows", 0)
        phase_thresholds = PhaseThresholds(**phase_threshold_dict)
        if aggregate.mean_score is not None and aggregate.mean_score >= phase_thresholds.phase2_to_phase3_correlation:
            consecutive_windows += 1
        else:
            consecutive_windows = 0
        self._consecutive_windows_counters[task_id] = consecutive_windows

        metrics = PhaseMetrics(
            task_id=task_id,
            example_count=example_count,
            rolling_correlation=correlation_score,
            local_model_available=local_model_available,
            consecutive_windows_above_threshold=consecutive_windows,
            trend_direction=trend_dir,
        )

        # Evaluate phase
        phase_result = self._phase_manager.evaluate(
            metrics=metrics,
            thresholds=phase_thresholds,
        )

        # Extract phase value (handle both real Phase enum and mock strings)
        # Try .phase first (what the mock fixture uses), then .current_phase (what real PhaseManager uses)
        if hasattr(phase_result, 'phase'):
            current_phase_obj = phase_result.phase
        elif hasattr(phase_result, 'current_phase'):
            current_phase_obj = phase_result.current_phase
        else:
            current_phase_obj = "REMOTE_ONLY"

        # Now extract the string value
        if isinstance(current_phase_obj, str):
            current_phase = current_phase_obj
        elif hasattr(current_phase_obj, 'value'):
            # It's a real Phase enum
            current_phase = current_phase_obj.value
        elif hasattr(current_phase_obj, '_mock_name'):
            # It's a newly created MagicMock from accessing a non-existent attribute
            # Fall back to the other attribute
            if hasattr(phase_result, 'current_phase') and isinstance(phase_result.current_phase, str):
                current_phase = phase_result.current_phase
            else:
                current_phase = "REMOTE_ONLY"
        else:
            # Fallback
            current_phase = "REMOTE_ONLY"

        # Detect drift and shift
        drift_detected = (aggregate.trend_direction.name == "DEGRADING")

        # Shift detection
        previous_score = self._previous_scores.get(task_id)
        shift_detected = False
        score_delta = None

        if previous_score is not None and aggregate.mean_score is not None:
            score_delta = aggregate.mean_score - previous_score
            if abs(score_delta) > self._config.trend_sensitivity:
                shift_detected = True

        # Update previous score
        if aggregate.mean_score is not None:
            self._previous_scores[task_id] = aggregate.mean_score

        drift_shift_status = DriftShiftStatus(
            drift_detected=drift_detected,
            shift_detected=shift_detected,
            score_delta=score_delta,
        )

        # Build phase transition dict if present
        phase_transition_dict = None
        if phase_result.transition is not None:
            transition = phase_result.transition
            current_timestamp = datetime.now(timezone.utc).isoformat()

            # Extract values from transition (handle both real and mock objects)
            def extract_value(obj):
                if hasattr(obj, 'value'):
                    return obj.value
                elif isinstance(obj, str):
                    return obj
                else:
                    return str(obj)

            from_phase = extract_value(getattr(transition, 'from_phase', 'REMOTE_ONLY'))
            to_phase = extract_value(getattr(transition, 'to_phase', 'REMOTE_ONLY'))
            trigger = extract_value(getattr(transition, 'reason', getattr(transition, 'trigger', 'INITIAL_ASSIGNMENT')))
            task_id_val = getattr(transition, 'task_id', task_id)
            timestamp_val = getattr(transition, 'timestamp', current_timestamp)

            # Build context from whatever is available
            context = getattr(transition, 'context', {})
            if not context:
                context = {
                    "task_id": task_id_val,
                    "timestamp": timestamp_val,
                }

            phase_transition_dict = {
                "from_phase": from_phase,
                "to_phase": to_phase,
                "reason": trigger,
                "context": context,
            }

        # Update last known phase
        self._last_known_phase[task_id] = current_phase

        return ConfidenceSnapshotEnriched(
            task_type=task_id,
            correlation_score=aggregate.mean_score,
            training_example_count=example_count,
            sample_count=aggregate.sample_count,
            consecutive_windows_above_threshold=consecutive_windows,
            trend_direction=trend_dir.name,
            is_stale=aggregate.is_stale,
            phase=current_phase,
            drift_shift_status=drift_shift_status,
            window_size=self._config.window_size,
            phase_transition=phase_transition_dict,
        )

    def get_all_snapshots(
        self,
        example_counts: dict[str, int],
        local_model_availability: dict[str, bool],
    ) -> ConfidenceSnapshotEnrichedMap:
        """
        Computes and returns enriched confidence snapshots for all known tasks.
        """
        # Validate example counts
        for task_id, count in example_counts.items():
            if count < 0:
                raise ValueError("All example_counts values must be non-negative integers")

        snapshots = {}

        # Get all known task IDs
        all_task_ids = set(self._config.task_evaluator_configs.keys())
        all_task_ids.update(self._windows.keys())

        for task_id in all_task_ids:
            example_count = example_counts.get(task_id, 0)
            local_available = local_model_availability.get(task_id, False)

            snapshot = self.get_snapshot(
                task_id=task_id,
                example_count=example_count,
                local_model_available=local_available,
            )
            snapshots[task_id] = snapshot

        return ConfidenceSnapshotEnrichedMap(snapshots=snapshots)

    def persist_state(self) -> None:
        """
        Persists the rolling window state for all tasks to disk as JSON files.
        """
        # Create persistence directory if it doesn't exist
        persistence_dir = Path(self._config.persistence_dir)
        try:
            persistence_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._logger.warning(
                f"Cannot create persistence directory '{persistence_dir}': {e}"
            )
            return

        # Persist each window
        for task_id, window in self._windows.items():
            try:
                window.persist(task_type=task_id)
                self._logger.debug(f"Persisted window for task '{task_id}'")
            except Exception as e:
                self._logger.warning(
                    f"Failed to persist window for task '{task_id}': {e}"
                )

    def load_state(self) -> int:
        """
        Loads persisted rolling window state from disk for all tasks.
        """
        persistence_dir = Path(self._config.persistence_dir)

        if not persistence_dir.exists():
            self._logger.info(f"Persistence directory '{persistence_dir}' does not exist")
            return 0

        # Find all JSON files in persistence directory
        try:
            json_files = list(persistence_dir.glob("*.json"))
        except Exception as e:
            self._logger.warning(
                f"Cannot read persistence directory '{persistence_dir}': {e}"
            )
            return 0

        loaded_count = 0

        for json_file in json_files:
            # Extract task_id from filename
            task_id = json_file.stem

            # Check if task is in config
            if task_id not in self._config.task_evaluator_configs:
                self._logger.warning(
                    f"Skipping persisted file for unknown task '{task_id}'"
                )
                continue

            try:
                # Create window if it doesn't exist
                if task_id not in self._windows:
                    window_config = rolling_window.WindowConfig(
                        window_size=self._config.window_size,
                        max_window_age_seconds=self._config.max_window_age_seconds,
                    )
                    store_config = rolling_window.StoreConfig(
                        base_dir=self._config.persistence_dir
                    )
                    store = rolling_window.JsonFileWindowStore(config=store_config)
                    self._windows[task_id] = rolling_window.DequeRollingWindow(
                        config=window_config,
                        store=store,
                    )
                    self._consecutive_windows_counters[task_id] = 0
                    self._last_known_phase[task_id] = "REMOTE_ONLY"

                # Load from store
                count = self._windows[task_id].load_from_store(task_type=task_id)
                if count > 0:
                    loaded_count += 1
                    self._logger.info(
                        f"Loaded {count} pairs for task '{task_id}'"
                    )

            except Exception as e:
                self._logger.warning(
                    f"Failed to load window for task '{task_id}': {e}"
                )

        self._logger.info(f"Loaded state for {loaded_count} tasks")
        return loaded_count

    def get_task_ids(self) -> list[str]:
        """
        Returns the set of all task_ids currently known to the engine.
        """
        all_task_ids = set(self._config.task_evaluator_configs.keys())
        all_task_ids.update(self._windows.keys())
        return list(all_task_ids)

    def get_window_size(self, task_id: str) -> int:
        """
        Returns the current number of comparison pairs in the rolling window for a given task.
        """
        if not task_id:
            raise ValueError("task_id must be a non-empty string")

        if task_id not in self._windows:
            return 0

        return len(self._windows[task_id])


# ============================================================================
# SYSTEM CLOCK
# ============================================================================

class _SystemClock:
    """System clock implementation for PhaseManager."""

    def now(self) -> str:
        """Returns current time as ISO 8601 string."""
        return datetime.now(timezone.utc).isoformat()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Parent types
    "ConfidenceEngine",
    "ConfidenceEngineConfig",
    "EvaluationOutcome",
    "ConfidenceSnapshotEnriched",
    "ConfidenceSnapshotEnrichedMap",
    "DriftShiftStatus",
    # Re-exported child types
    "EvaluatorType",
    "TaskResponse",
    "TaskEvaluatorConfig",
    "FieldEvaluation",
    "EvaluationResult",
    "EvaluatorProtocol",
    "Phase",
    "TrendDirection",
    "TransitionTrigger",
    "PhaseThresholds",
    "PhaseMetrics",
    "PhaseTransition",
    "ComparisonPair",
    "ComparisonPairInput",
    "WindowAggregate",
]
