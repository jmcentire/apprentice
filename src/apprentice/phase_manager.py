# === Phase Manager (phase_manager) v1 ===
# Determines the current phase per task type based on training example count and confidence scores.
# Phase 1→2 transition: when training example count >= phase1_to_phase2 threshold AND a local model is available.
# Phase 2→3 transition: when rolling correlation >= phase2_to_phase3 threshold over a sustained number of consecutive windows.
# Phase 3→2 regression: when correlation drops below coaching_trigger.
# Phase 3→1 emergency: when correlation drops below emergency_threshold, revert to full remote serving.
# Transitions are logged as audit events. Phase is derived (computed from state), not stored as a discrete flag — emergent from confidence scores.

import logging
import math
from enum import Enum
from typing import Protocol
from pydantic import BaseModel, Field, model_validator, ConfigDict, field_validator

_PACT_KEY = "PACT:fb2bb1:phase_manager"
logger = logging.getLogger(__name__)


class PactFormatter(logging.Formatter):
    """Formatter that injects the PACT log key into every record."""

    def format(self, record):
        record.pact_key = _PACT_KEY
        return super().format(record)


def _log(level: str, msg: str, **kwargs) -> None:
    """Log with PACT key embedded for production traceability."""
    getattr(logger, level)(f"[{_PACT_KEY}] {msg}", **kwargs)


# ============================================================
# Enums
# ============================================================

class Phase(str, Enum):
    """Quality-based operational phase derived from confidence metrics."""
    REMOTE_ONLY = "REMOTE_ONLY"
    COACHING = "COACHING"
    AUTONOMOUS = "AUTONOMOUS"

    @property
    def level(self) -> int:
        """Explicit ordering: REMOTE_ONLY=1, COACHING=2, AUTONOMOUS=3."""
        return {Phase.REMOTE_ONLY: 1, Phase.COACHING: 2, Phase.AUTONOMOUS: 3}[self]


class TrendDirection(str, Enum):
    """Direction of the rolling correlation trend over consecutive evaluation windows."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"


class TransitionTrigger(str, Enum):
    """Structured reason for a phase transition, used in audit events."""
    EXAMPLE_COUNT_AND_MODEL_READY = "EXAMPLE_COUNT_AND_MODEL_READY"
    SUSTAINED_CORRELATION_ABOVE_THRESHOLD = "SUSTAINED_CORRELATION_ABOVE_THRESHOLD"
    CORRELATION_BELOW_COACHING_TRIGGER = "CORRELATION_BELOW_COACHING_TRIGGER"
    CORRELATION_BELOW_EMERGENCY_THRESHOLD = "CORRELATION_BELOW_EMERGENCY_THRESHOLD"
    INITIAL_ASSIGNMENT = "INITIAL_ASSIGNMENT"


# ============================================================
# Clock Protocol
# ============================================================

class Clock(Protocol):
    """Protocol for time dependency injection."""
    def now(self) -> str:
        """Returns current time as ISO 8601 string."""
        ...


# ============================================================
# Pydantic Models
# ============================================================

class PhaseThresholds(BaseModel):
    """Configuration model for phase transition thresholds."""
    model_config = ConfigDict(frozen=True)

    phase1_to_phase2_example_count: int = Field(ge=1)
    phase2_to_phase3_correlation: float = Field(gt=0.0, le=1.0)
    coaching_trigger: float = Field(gt=0.0, lt=1.0)
    emergency_threshold: float = Field(ge=0.0, lt=1.0)
    sustained_windows: int = Field(default=3, ge=1)
    cooldown_windows: int = Field(default=0, ge=0)

    @model_validator(mode='after')
    def validate_threshold_ordering(self):
        """Enforce invariant: emergency_threshold < coaching_trigger < phase2_to_phase3_correlation."""
        if not (self.emergency_threshold < self.coaching_trigger < self.phase2_to_phase3_correlation):
            raise ValueError(
                f"PhaseThresholds invariant violated: must have emergency_threshold < coaching_trigger < phase2_to_phase3_correlation. "
                f"Got emergency_threshold={self.emergency_threshold}, coaching_trigger={self.coaching_trigger}, "
                f"phase2_to_phase3_correlation={self.phase2_to_phase3_correlation}"
            )
        return self


class PhaseMetrics(BaseModel):
    """Frozen Pydantic model containing all inputs needed to compute the current quality-based phase."""
    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    example_count: int = Field(ge=0)
    rolling_correlation: float
    local_model_available: bool
    consecutive_windows_above_threshold: int = Field(ge=0)
    trend_direction: TrendDirection


class PhaseTransition(BaseModel):
    """Frozen Pydantic model representing an audit event for a phase transition."""
    model_config = ConfigDict(frozen=True)

    task_id: str
    from_phase: Phase
    to_phase: Phase
    trigger: TransitionTrigger
    timestamp: str = Field(pattern=r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')
    metrics_snapshot: PhaseMetrics


class PhaseComputationResult(BaseModel):
    """Result of the pure phase computation."""
    phase: Phase
    trigger: TransitionTrigger


class TaskPhaseState(BaseModel):
    """Per-task mutable state tracked by the PhaseManager."""
    task_id: str
    current_phase: Phase
    windows_since_last_transition: int = Field(ge=0)
    transition_history: list['PhaseTransition'] = Field(default_factory=list)


class EvaluateResult(BaseModel):
    """Result of evaluating a task's metrics through the PhaseManager."""
    current_phase: Phase
    transition: 'PhaseTransition | None'
    cooldown_active: bool


class TaskPhaseStateMap(BaseModel):
    """Dictionary-like container mapping task_id strings to their TaskPhaseState."""
    states: dict[str, TaskPhaseState] = Field(default_factory=dict)


# Type aliases
PhaseTransitionList = list[PhaseTransition]
OptionalPhaseTransition = PhaseTransition | None
OptionalTaskPhaseState = TaskPhaseState | None


# ============================================================
# Pure Computation Core
# ============================================================

def compute_phase(
    metrics: PhaseMetrics,
    thresholds: PhaseThresholds,
) -> PhaseComputationResult:
    """
    Pure computation core. Derives the quality-based Phase from a frozen PhaseMetrics snapshot and PhaseThresholds configuration.
    
    Logic:
    (1) If rolling_correlation < emergency_threshold → REMOTE_ONLY
    (2) If rolling_correlation < coaching_trigger → COACHING (if example_count >= phase1_to_phase2_example_count and local_model_available, else REMOTE_ONLY)
    (3) If example_count < phase1_to_phase2_example_count or not local_model_available → REMOTE_ONLY
    (4) If consecutive_windows_above_threshold >= sustained_windows → AUTONOMOUS
    (5) Otherwise → COACHING
    """
    # Check for NaN correlation
    if math.isnan(metrics.rolling_correlation):
        raise ValueError(
            "rolling_correlation must be a valid float, not NaN. Caller must handle insufficient data before invoking compute_phase."
        )

    # (1) Emergency threshold check
    if metrics.rolling_correlation < thresholds.emergency_threshold:
        return PhaseComputationResult(
            phase=Phase.REMOTE_ONLY,
            trigger=TransitionTrigger.CORRELATION_BELOW_EMERGENCY_THRESHOLD,
        )

    # (2) Check if we have basic requirements for Phase 2 (COACHING)
    has_model_and_examples = (
        metrics.example_count >= thresholds.phase1_to_phase2_example_count
        and metrics.local_model_available
    )

    # (3) If insufficient examples or no model → REMOTE_ONLY
    if not has_model_and_examples:
        return PhaseComputationResult(
            phase=Phase.REMOTE_ONLY,
            trigger=TransitionTrigger.EXAMPLE_COUNT_AND_MODEL_READY,
        )

    # Now we know we have enough examples and a model available
    # Check correlation thresholds

    # If correlation is below coaching_trigger → COACHING
    if metrics.rolling_correlation < thresholds.coaching_trigger:
        return PhaseComputationResult(
            phase=Phase.COACHING,
            trigger=TransitionTrigger.CORRELATION_BELOW_COACHING_TRIGGER,
        )

    # (4) Check if we meet sustained windows requirement for AUTONOMOUS
    if metrics.consecutive_windows_above_threshold >= thresholds.sustained_windows:
        return PhaseComputationResult(
            phase=Phase.AUTONOMOUS,
            trigger=TransitionTrigger.SUSTAINED_CORRELATION_ABOVE_THRESHOLD,
        )

    # (5) Otherwise → COACHING (correlation is good but not sustained enough)
    return PhaseComputationResult(
        phase=Phase.COACHING,
        trigger=TransitionTrigger.EXAMPLE_COUNT_AND_MODEL_READY,
    )


# ============================================================
# Threshold Validation
# ============================================================

def validate_thresholds(thresholds: PhaseThresholds) -> PhaseThresholds:
    """
    Validates a PhaseThresholds instance, checking both individual field constraints and the cross-field invariant.
    Returns the validated PhaseThresholds if valid; raises on any violation.
    """
    # Field validators already run by Pydantic
    # The model_validator already enforces the ordering invariant
    # This is essentially a passthrough, but we can add explicit checks if needed
    
    # Re-check ordering (redundant with model_validator, but explicit)
    if not (thresholds.emergency_threshold < thresholds.coaching_trigger < thresholds.phase2_to_phase3_correlation):
        raise ValueError(
            f"PhaseThresholds invariant violated: must have emergency_threshold < coaching_trigger < phase2_to_phase3_correlation. "
            f"Got emergency_threshold={thresholds.emergency_threshold}, coaching_trigger={thresholds.coaching_trigger}, "
            f"phase2_to_phase3_correlation={thresholds.phase2_to_phase3_correlation}"
        )
    
    if thresholds.phase1_to_phase2_example_count < 1:
        raise ValueError("phase1_to_phase2_example_count must be at least 1")
    
    if thresholds.sustained_windows < 1:
        raise ValueError("sustained_windows must be at least 1")
    
    if thresholds.cooldown_windows < 0:
        raise ValueError("cooldown_windows must be non-negative")
    
    return thresholds


# ============================================================
# PhaseManager (Stateful)
# ============================================================

class PhaseManager:
    """Stateful transition detection and phase management."""

    def __init__(self, clock: Clock):
        self._clock = clock
        self._task_states: dict[str, TaskPhaseState] = {}

    def evaluate(
        self,
        metrics: PhaseMetrics,
        thresholds: PhaseThresholds,
    ) -> EvaluateResult:
        """
        Stateful transition detection layer. Takes current PhaseMetrics for a task, computes the quality-based phase,
        diffs against the previously recorded phase, enforces cooldown, and emits a PhaseTransition audit event if changed.
        """
        # Check for NaN correlation
        if math.isnan(metrics.rolling_correlation):
            raise ValueError("rolling_correlation must not be NaN. Caller must handle insufficient data.")

        # Get timestamp BEFORE computing phase - this ensures we fail fast on clock failure
        try:
            timestamp = self._clock.now()
        except Exception as e:
            raise RuntimeError("Clock dependency failed to provide valid ISO 8601 timestamp.") from e

        # Compute the new phase
        computation_result = compute_phase(metrics, thresholds)
        new_phase = computation_result.phase

        # Get or initialize task state
        if metrics.task_id not in self._task_states:
            # First time seeing this task — initialize to REMOTE_ONLY
            self._task_states[metrics.task_id] = TaskPhaseState(
                task_id=metrics.task_id,
                current_phase=Phase.REMOTE_ONLY,
                windows_since_last_transition=0,
                transition_history=[],
            )

        task_state = self._task_states[metrics.task_id]
        old_phase = task_state.current_phase

        # Check if phase changed
        phase_changed = new_phase != old_phase

        # Check cooldown - but only if there was a previous transition
        cooldown_active = False
        if phase_changed and thresholds.cooldown_windows > 0:
            # Only enforce cooldown if we have had a previous transition
            # (i.e., transition_history is not empty)
            if len(task_state.transition_history) > 0 and task_state.windows_since_last_transition < thresholds.cooldown_windows:
                # Cooldown suppresses the transition
                cooldown_active = True
                phase_changed = False

        transition_event: OptionalPhaseTransition = None

        if phase_changed:
            # Determine trigger
            # If this is the first evaluation (no history) and we're transitioning from REMOTE_ONLY
            if old_phase == Phase.REMOTE_ONLY and len(task_state.transition_history) == 0:
                trigger = TransitionTrigger.INITIAL_ASSIGNMENT
            else:
                trigger = computation_result.trigger

            # Create transition event using the timestamp we obtained earlier
            transition_event = PhaseTransition(
                task_id=metrics.task_id,
                from_phase=old_phase,
                to_phase=new_phase,
                trigger=trigger,
                timestamp=timestamp,
                metrics_snapshot=metrics,
            )

            # Update task state
            task_state.current_phase = new_phase
            task_state.windows_since_last_transition = 0
            task_state.transition_history.append(transition_event)

            # Log the transition
            _log(
                "info",
                f"Phase transition for task {metrics.task_id}: {old_phase.value} → {new_phase.value} (trigger: {trigger.value})",
            )
        else:
            # No transition — increment window counter
            task_state.windows_since_last_transition += 1

        # Return result
        return EvaluateResult(
            current_phase=task_state.current_phase,
            transition=transition_event,
            cooldown_active=cooldown_active,
        )

    def get_task_phase(self, task_id: str) -> Phase:
        """
        Returns the current quality-based phase for a given task.
        If the task_id has never been evaluated, returns Phase.REMOTE_ONLY.
        """
        if not task_id:
            raise ValueError("task_id must be a non-empty string.")

        if task_id not in self._task_states:
            return Phase.REMOTE_ONLY

        return self._task_states[task_id].current_phase

    def get_task_state(self, task_id: str) -> OptionalTaskPhaseState:
        """
        Returns the full TaskPhaseState for a given task.
        Returns None if the task_id has never been evaluated.
        """
        if not task_id:
            raise ValueError("task_id must be a non-empty string.")

        return self._task_states.get(task_id)

    def get_transition_history(self, task_id: str) -> PhaseTransitionList:
        """
        Returns the complete ordered list of PhaseTransition audit events for a given task.
        Returns an empty list if the task has never transitioned or has never been evaluated.
        """
        if not task_id:
            raise ValueError("task_id must be a non-empty string.")

        if task_id not in self._task_states:
            return []

        return self._task_states[task_id].transition_history.copy()

    def reset_task(self, task_id: str) -> bool:
        """
        Removes all tracked state for a given task_id.
        Returns True if the task existed and was reset, False if the task_id was not tracked.
        """
        if not task_id:
            raise ValueError("task_id must be a non-empty string.")

        if task_id in self._task_states:
            del self._task_states[task_id]
            _log("info", f"Reset task state for task {task_id}")
            return True

        return False


# ============================================================
# Module-level convenience functions for stateful operations
# ============================================================

# Global PhaseManager instance (will be initialized when first used)
_global_manager: PhaseManager | None = None


def _get_global_manager() -> PhaseManager:
    """Get or create the global PhaseManager instance."""
    global _global_manager
    if _global_manager is None:
        from datetime import datetime, timezone
        
        class SystemClock:
            def now(self) -> str:
                return datetime.now(timezone.utc).isoformat()
        
        _global_manager = PhaseManager(clock=SystemClock())
    return _global_manager


def evaluate(
    metrics: PhaseMetrics,
    thresholds: PhaseThresholds,
) -> EvaluateResult:
    """
    Module-level evaluate function using global PhaseManager.
    """
    manager = _get_global_manager()
    return manager.evaluate(metrics, thresholds)


def get_task_phase(task_id: str) -> Phase:
    """
    Module-level function to get task phase using global PhaseManager.
    """
    manager = _get_global_manager()
    return manager.get_task_phase(task_id)


def get_task_state(task_id: str) -> OptionalTaskPhaseState:
    """
    Module-level function to get task state using global PhaseManager.
    """
    manager = _get_global_manager()
    return manager.get_task_state(task_id)


def get_transition_history(task_id: str) -> PhaseTransitionList:
    """
    Module-level function to get transition history using global PhaseManager.
    """
    manager = _get_global_manager()
    return manager.get_transition_history(task_id)


def reset_task(task_id: str) -> bool:
    """
    Module-level function to reset task using global PhaseManager.
    """
    manager = _get_global_manager()
    return manager.reset_task(task_id)


# ============================================================
# Required Exports
# ============================================================

__all__ = [
    'Phase',
    'TrendDirection',
    'TransitionTrigger',
    'PhaseThresholds',
    'PhaseMetrics',
    'PhaseTransition',
    'PhaseComputationResult',
    'TaskPhaseState',
    'PhaseTransitionList',
    'EvaluateResult',
    'OptionalPhaseTransition',
    'OptionalTaskPhaseState',
    'TaskPhaseStateMap',
    'Clock',
    'PhaseManager',
    'compute_phase',
    'evaluate',
    'get_task_phase',
    'get_task_state',
    'get_transition_history',
    'reset_task',
    'validate_thresholds',
]
