"""
Observer Component (observer) v1

Observes user and agent actions per task, maintains a rolling context window,
and probabilistically generates shadow recommendations for later comparison.

Shadow recommendations are placeholder-only in this phase; actual model
integration is deferred.
"""

import random
import uuid
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ===========================================================================
# Enums
# ===========================================================================


class EventType(str, Enum):
    """Type of event observed by the Observer."""
    user_action = "user_action"
    agent_action = "agent_action"
    system_event = "system_event"


# ===========================================================================
# Data Models
# ===========================================================================


class ObservationEvent(BaseModel):
    """Frozen Pydantic model representing a single observed event."""
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_name: str
    event_type: EventType
    action_data: dict = Field(default_factory=dict)
    context: dict = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ShadowRecommendation(BaseModel):
    """Frozen Pydantic model representing a shadow recommendation paired with an observed event."""
    model_config = ConfigDict(frozen=True)

    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str
    recommended_action: dict = Field(default_factory=dict)
    actual_action: Optional[dict] = None
    match_score: Optional[float] = None  # 0.0-1.0 when actual_action filled
    model_used: str = ""


class ObserverConfig(BaseModel):
    """Frozen Pydantic configuration model for the Observer."""
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    context_window_size: int = Field(default=50, ge=1, le=1000)
    shadow_recommendation_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    min_context_before_recommending: int = Field(default=10, ge=1)


# ===========================================================================
# Match Score Computation
# ===========================================================================


def _compute_match_score(recommended: dict, actual: dict) -> float:
    """Compute simple dict key overlap ratio between recommended_action and actual_action.

    Returns 0.0 if both dicts are empty (no keys to compare).
    Otherwise returns |intersection of keys| / |union of keys|.
    """
    rec_keys = set(recommended.keys())
    act_keys = set(actual.keys())
    union = rec_keys | act_keys
    if not union:
        return 0.0
    intersection = rec_keys & act_keys
    return len(intersection) / len(union)


# ===========================================================================
# Observer
# ===========================================================================


class Observer:
    """Observes task events, maintains rolling context windows, and generates
    probabilistic shadow recommendations for offline comparison.

    When ``config.enabled`` is False, ``observe`` and ``record_action`` are
    no-ops (observe does nothing; record_action returns None).
    """

    def __init__(self, config: ObserverConfig) -> None:
        self._config = config
        # task_name -> deque of ObservationEvent (bounded by context_window_size)
        self._context: dict[str, deque[ObservationEvent]] = {}
        # event_id -> ShadowRecommendation
        self._recommendations: dict[str, ShadowRecommendation] = {}
        # task_name -> list of event_ids that have recommendations (preserves order)
        self._task_recommendation_ids: dict[str, list[str]] = {}
        # track total events observed per task (for min_context_before_recommending)
        self._event_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, event: ObservationEvent) -> None:
        """Add an event to the rolling context window for its task.

        If the context window exceeds ``config.context_window_size``, the
        oldest event is dropped.  When the observer is disabled this is a
        no-op.
        """
        if not self._config.enabled:
            return

        task = event.task_name

        # Ensure deque exists for this task
        if task not in self._context:
            self._context[task] = deque(maxlen=self._config.context_window_size)

        self._context[task].append(event)

        # Track total events observed (not just window contents)
        self._event_counts[task] = self._event_counts.get(task, 0) + 1

        # Possibly generate a shadow recommendation
        self._maybe_generate_recommendation(event)

    def record_action(self, event_id: str, action: dict) -> Optional[ShadowRecommendation]:
        """Record the actual action taken for a previously observed event.

        If a shadow recommendation was generated for this event, computes the
        match score and returns the updated ``ShadowRecommendation``.
        Returns ``None`` if the observer is disabled or no recommendation
        exists for the given event_id.
        """
        if not self._config.enabled:
            return None

        if event_id not in self._recommendations:
            return None

        old_rec = self._recommendations[event_id]
        score = _compute_match_score(old_rec.recommended_action, action)

        updated = old_rec.model_copy(
            update={
                "actual_action": action,
                "match_score": score,
            }
        )
        self._recommendations[event_id] = updated
        return updated

    def get_context(self, task_name: str) -> list[ObservationEvent]:
        """Return the current context window for *task_name*.

        Returns an empty list if no events have been observed for the task.
        """
        if task_name not in self._context:
            return []
        return list(self._context[task_name])

    def get_recommendations(self, task_name: str) -> list[ShadowRecommendation]:
        """Return all shadow recommendations for *task_name*.

        Returns an empty list if no recommendations exist for the task.
        """
        if task_name not in self._task_recommendation_ids:
            return []
        return [
            self._recommendations[eid]
            for eid in self._task_recommendation_ids[task_name]
            if eid in self._recommendations
        ]

    def clear(self, task_name: str) -> None:
        """Clear context window and recommendations for *task_name*."""
        self._context.pop(task_name, None)
        removed_ids = self._task_recommendation_ids.pop(task_name, [])
        for eid in removed_ids:
            self._recommendations.pop(eid, None)
        self._event_counts.pop(task_name, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_generate_recommendation(self, event: ObservationEvent) -> None:
        """Probabilistically generate a shadow recommendation for *event*.

        A recommendation is only considered when the total number of events
        observed for the task meets ``min_context_before_recommending``.
        The recommendation is then generated with probability
        ``shadow_recommendation_rate``.
        """
        task = event.task_name
        total_events = self._event_counts.get(task, 0)

        if total_events < self._config.min_context_before_recommending:
            return

        if random.random() >= self._config.shadow_recommendation_rate:
            return

        # Placeholder model call — returns empty recommended_action
        rec = ShadowRecommendation(
            event_id=event.event_id,
            recommended_action={},
            model_used="placeholder",
        )

        self._recommendations[event.event_id] = rec
        if task not in self._task_recommendation_ids:
            self._task_recommendation_ids[task] = []
        self._task_recommendation_ids[task].append(event.event_id)


# ===========================================================================
# Exports
# ===========================================================================

__all__ = [
    "EventType",
    "ObservationEvent",
    "ShadowRecommendation",
    "ObserverConfig",
    "Observer",
]
