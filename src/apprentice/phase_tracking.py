"""Per-Journey Phase Tracking (phase_tracking) v1

Extends phase management with per-journey-type tracking. When story_type is
provided, phase computation uses metrics filtered to that journey type.
When story_type is None, falls back to existing atomic-only behavior.
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

_PACT_KEY = "PACT:phase_tracking"
logger = logging.getLogger(__name__)


def _log(level: str, msg: str, **kwargs) -> None:
    getattr(logger, level)(f"[{_PACT_KEY}] {msg}", **kwargs)


# ============================================================================
# Enums & Types
# ============================================================================

class Phase(str, Enum):
    """Learning phases, ordered from least to most autonomous."""
    observation = "observation"
    coaching = "coaching"
    autonomous = "autonomous"

    @property
    def level(self) -> int:
        return {Phase.observation: 1, Phase.coaching: 2, Phase.autonomous: 3}[self]


StoryType = str  # non-empty string identifier


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class PhaseMetrics:
    """Metrics used to determine the current learning phase.
    Extended with optional story_type (last field for backward compat)."""
    success_count: int = 0
    failure_count: int = 0
    total_attempts: int = 0
    success_rate: float = 0.0
    consecutive_successes: int = 0
    current_phase: Phase = Phase.observation
    story_type: Optional[str] = None

    def __post_init__(self):
        if self.total_attempts != self.success_count + self.failure_count:
            raise ValueError(
                f"total_attempts ({self.total_attempts}) must equal "
                f"success_count ({self.success_count}) + failure_count ({self.failure_count})"
            )

    def copy(self) -> 'PhaseMetrics':
        return PhaseMetrics(
            success_count=self.success_count,
            failure_count=self.failure_count,
            total_attempts=self.total_attempts,
            success_rate=self.success_rate,
            consecutive_successes=self.consecutive_successes,
            current_phase=self.current_phase,
            story_type=self.story_type,
        )

    def to_dict(self) -> dict:
        return {
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_attempts": self.total_attempts,
            "success_rate": self.success_rate,
            "consecutive_successes": self.consecutive_successes,
            "current_phase": self.current_phase.value,
            "story_type": self.story_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PhaseMetrics':
        return cls(
            success_count=data["success_count"],
            failure_count=data["failure_count"],
            total_attempts=data["total_attempts"],
            success_rate=data["success_rate"],
            consecutive_successes=data["consecutive_successes"],
            current_phase=Phase(data["current_phase"]),
            story_type=data.get("story_type"),
        )


class StoryMetricsStore:
    """Maps story_type -> PhaseMetrics. Lazy-initializes new types."""

    def __init__(self):
        self.metrics_by_type: dict[str, PhaseMetrics] = {}

    def get(self, story_type: str) -> PhaseMetrics:
        if story_type not in self.metrics_by_type:
            self.metrics_by_type[story_type] = PhaseMetrics(story_type=story_type)
        return self.metrics_by_type[story_type]

    def known_types(self) -> list[str]:
        return list(self.metrics_by_type.keys())


@dataclass(frozen=True)
class PhaseComputationResult:
    """Result of a phase computation."""
    phase: Phase
    metrics: PhaseMetrics
    story_type: Optional[str] = None


@dataclass
class MetricEvent:
    """A single metric event representing a task attempt outcome."""
    success: bool
    story_type: Optional[str] = None


# ============================================================================
# Utility
# ============================================================================

def normalize_story_type(story_type: Optional[str]) -> Optional[str]:
    """Normalize story_type: empty/whitespace -> None, strip whitespace."""
    if story_type is None:
        return None
    if not isinstance(story_type, str):
        raise TypeError(f"story_type must be a str or None, got {type(story_type).__name__}")
    stripped = story_type.strip()
    if not stripped:
        return None
    return stripped


# ============================================================================
# Phase Computation
# ============================================================================

# Phase transition thresholds
_COACHING_THRESHOLD = 5     # attempts before coaching
_COACHING_RATE = 0.6        # success rate for coaching
_AUTONOMOUS_THRESHOLD = 10  # attempts before autonomous
_AUTONOMOUS_RATE = 0.8      # success rate for autonomous
_AUTONOMOUS_CONSECUTIVE = 5  # consecutive successes needed


def _compute_phase_from_metrics(metrics: PhaseMetrics) -> Phase:
    """Pure function: derive phase from metrics."""
    if metrics.total_attempts < _COACHING_THRESHOLD:
        return Phase.observation

    if (metrics.total_attempts >= _AUTONOMOUS_THRESHOLD and
            metrics.success_rate >= _AUTONOMOUS_RATE and
            metrics.consecutive_successes >= _AUTONOMOUS_CONSECUTIVE):
        return Phase.autonomous

    if metrics.success_rate >= _COACHING_RATE:
        return Phase.coaching

    return Phase.observation


def _update_metrics(metrics: PhaseMetrics, success: bool) -> None:
    """Mutate a PhaseMetrics in-place with a new event."""
    metrics.total_attempts += 1
    if success:
        metrics.success_count += 1
        metrics.consecutive_successes += 1
    else:
        metrics.failure_count += 1
        metrics.consecutive_successes = 0

    if metrics.total_attempts > 0:
        metrics.success_rate = metrics.success_count / metrics.total_attempts
    else:
        metrics.success_rate = 0.0

    metrics.current_phase = _compute_phase_from_metrics(metrics)


# ============================================================================
# PhaseTracker (Stateful)
# ============================================================================

class PhaseTracker:
    """Stateful phase tracking with per-story-type support."""

    def __init__(self):
        self._global_metrics = PhaseMetrics(story_type=None)
        self._story_store = StoryMetricsStore()

    def compute_phase(self, story_type: Optional[str] = None) -> PhaseComputationResult:
        """Compute current phase, optionally scoped to a story_type."""
        story_type = normalize_story_type(story_type)

        if story_type is None:
            metrics = self._global_metrics
        else:
            metrics = self._story_store.get(story_type)

        phase = _compute_phase_from_metrics(metrics)

        return PhaseComputationResult(
            phase=phase,
            metrics=metrics.copy(),
            story_type=story_type,
        )

    def record_metric(self, event: MetricEvent) -> None:
        """Record a metric event. Dual-writes to global and per-type metrics."""
        if not isinstance(event, MetricEvent):
            raise TypeError(f"event must be a MetricEvent instance, got {type(event).__name__}")

        st = normalize_story_type(event.story_type)

        # Always update global
        _update_metrics(self._global_metrics, event.success)

        # Update per-type if applicable
        if st is not None:
            type_metrics = self._story_store.get(st)
            _update_metrics(type_metrics, event.success)

    def get_phase_metrics(self, story_type: Optional[str] = None) -> PhaseMetrics:
        """Get current metrics snapshot for a story_type or globally."""
        story_type = normalize_story_type(story_type)

        if story_type is None:
            return self._global_metrics.copy()
        else:
            return self._story_store.get(story_type).copy()

    def get_known_story_types(self) -> list[str]:
        """Return all story types that have had at least one event."""
        return self._story_store.known_types()

    def normalize_story_type(self, story_type: Optional[str]) -> Optional[str]:
        """Delegate to module-level normalize_story_type."""
        return normalize_story_type(story_type)

    def serialize_all_metrics(self) -> dict:
        """Serialize complete metrics state for persistence."""
        result = {
            "global": self._global_metrics.to_dict(),
            "by_story_type": {},
        }
        for st, metrics in self._story_store.metrics_by_type.items():
            result["by_story_type"][st] = metrics.to_dict()
        return result

    def deserialize_all_metrics(self, data: dict) -> None:
        """Restore metrics state from serialized data."""
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dict, got {type(data).__name__}")
        if "global" not in data:
            raise KeyError("Serialized data must contain 'global' key")
        if "by_story_type" not in data:
            raise KeyError("Serialized data must contain 'by_story_type' key")

        try:
            self._global_metrics = PhaseMetrics.from_dict(data["global"])
        except Exception as e:
            raise ValueError(f"Invalid PhaseMetrics data: {e}")

        self._story_store = StoryMetricsStore()
        for st, metrics_data in data["by_story_type"].items():
            try:
                self._story_store.metrics_by_type[st] = PhaseMetrics.from_dict(metrics_data)
            except Exception as e:
                raise ValueError(f"Invalid PhaseMetrics data for story_type '{st}': {e}")


# ============================================================================
# Module-level API (global instance)
# ============================================================================

_global_tracker: PhaseTracker | None = None


def _get_tracker() -> PhaseTracker:
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PhaseTracker()
    return _global_tracker


def compute_phase(story_type: Optional[str] = None) -> PhaseComputationResult:
    return _get_tracker().compute_phase(story_type)


def record_metric(event: MetricEvent) -> None:
    _get_tracker().record_metric(event)


def get_phase_metrics(story_type: Optional[str] = None) -> PhaseMetrics:
    return _get_tracker().get_phase_metrics(story_type)


def get_known_story_types() -> list[str]:
    return _get_tracker().get_known_story_types()


def serialize_all_metrics() -> dict:
    return _get_tracker().serialize_all_metrics()


def deserialize_all_metrics(data: dict) -> None:
    _get_tracker().deserialize_all_metrics(data)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'Phase',
    'PhaseMetrics',
    'StoryMetricsStore',
    'PhaseComputationResult',
    'MetricEvent',
    'PhaseTracker',
    'compute_phase',
    'record_metric',
    'get_phase_metrics',
    'get_known_story_types',
    'normalize_story_type',
    'serialize_all_metrics',
    'deserialize_all_metrics',
]
