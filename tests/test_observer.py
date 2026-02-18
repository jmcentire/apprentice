"""
Contract tests for the Observer component.

Sections:
1. Fixtures and helpers
2. ObservationEvent model tests
3. ShadowRecommendation model tests
4. ObserverConfig model tests
5. EventType enum tests
6. Observer unit tests (disabled mode, observe, context window, record_action,
   get_context, get_recommendations, clear, probabilistic generation)
"""

import uuid
from unittest.mock import patch

import pytest

from src.observer import (
    EventType,
    ObservationEvent,
    ShadowRecommendation,
    ObserverConfig,
    Observer,
    _compute_match_score,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES & HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def make_event(
    task_name: str = "test-task",
    event_type: EventType = EventType.user_action,
    action_data: dict | None = None,
    context: dict | None = None,
    event_id: str | None = None,
) -> ObservationEvent:
    kwargs: dict = {
        "task_name": task_name,
        "event_type": event_type,
    }
    if action_data is not None:
        kwargs["action_data"] = action_data
    if context is not None:
        kwargs["context"] = context
    if event_id is not None:
        kwargs["event_id"] = event_id
    return ObservationEvent(**kwargs)


def enabled_config(**overrides) -> ObserverConfig:
    defaults = {
        "enabled": True,
        "context_window_size": 50,
        "shadow_recommendation_rate": 0.1,
        "min_context_before_recommending": 10,
    }
    defaults.update(overrides)
    return ObserverConfig(**defaults)


def disabled_config() -> ObserverConfig:
    return ObserverConfig(enabled=False)


@pytest.fixture
def observer_disabled():
    return Observer(disabled_config())


@pytest.fixture
def observer_enabled():
    return Observer(enabled_config())


@pytest.fixture
def observer_always_recommend():
    """Observer that always generates recommendations after 1 event."""
    return Observer(
        enabled_config(
            shadow_recommendation_rate=1.0,
            min_context_before_recommending=1,
        )
    )


@pytest.fixture
def observer_never_recommend():
    """Observer that never generates recommendations."""
    return Observer(
        enabled_config(
            shadow_recommendation_rate=0.0,
            min_context_before_recommending=1,
        )
    )


# ═══════════════════════════════════════════════════════════════════════════
# EVENT TYPE ENUM
# ═══════════════════════════════════════════════════════════════════════════


class TestEventType:
    def test_values(self):
        assert EventType.user_action == "user_action"
        assert EventType.agent_action == "agent_action"
        assert EventType.system_event == "system_event"

    def test_member_count(self):
        assert len(EventType) == 3

    def test_is_str_enum(self):
        assert isinstance(EventType.user_action, str)


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVATION EVENT MODEL
# ═══════════════════════════════════════════════════════════════════════════


class TestObservationEvent:
    def test_creation_with_defaults(self):
        event = ObservationEvent(task_name="my-task", event_type=EventType.user_action)
        assert event.task_name == "my-task"
        assert event.event_type == EventType.user_action
        assert event.action_data == {}
        assert event.context == {}
        # event_id should be a valid UUID
        uuid.UUID(event.event_id)
        # timestamp should be a non-empty ISO string
        assert len(event.timestamp) > 0

    def test_creation_with_custom_values(self):
        event = ObservationEvent(
            event_id="custom-id",
            task_name="task-2",
            event_type=EventType.agent_action,
            action_data={"key": "value"},
            context={"session": "abc"},
            timestamp="2025-01-01T00:00:00+00:00",
        )
        assert event.event_id == "custom-id"
        assert event.action_data == {"key": "value"}
        assert event.context == {"session": "abc"}
        assert event.timestamp == "2025-01-01T00:00:00+00:00"

    def test_frozen(self):
        event = make_event()
        with pytest.raises(Exception):
            event.task_name = "changed"


# ═══════════════════════════════════════════════════════════════════════════
# SHADOW RECOMMENDATION MODEL
# ═══════════════════════════════════════════════════════════════════════════


class TestShadowRecommendation:
    def test_creation_defaults(self):
        rec = ShadowRecommendation(event_id="ev-1")
        uuid.UUID(rec.recommendation_id)
        assert rec.event_id == "ev-1"
        assert rec.recommended_action == {}
        assert rec.actual_action is None
        assert rec.match_score is None
        assert rec.model_used == ""

    def test_creation_full(self):
        rec = ShadowRecommendation(
            recommendation_id="rec-1",
            event_id="ev-1",
            recommended_action={"a": 1},
            actual_action={"a": 2},
            match_score=0.75,
            model_used="gpt-4",
        )
        assert rec.recommendation_id == "rec-1"
        assert rec.match_score == 0.75
        assert rec.model_used == "gpt-4"

    def test_frozen(self):
        rec = ShadowRecommendation(event_id="ev-1")
        with pytest.raises(Exception):
            rec.event_id = "changed"


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVER CONFIG MODEL
# ═══════════════════════════════════════════════════════════════════════════


class TestObserverConfig:
    def test_defaults(self):
        cfg = ObserverConfig()
        assert cfg.enabled is False
        assert cfg.context_window_size == 50
        assert cfg.shadow_recommendation_rate == 0.1
        assert cfg.min_context_before_recommending == 10

    def test_custom(self):
        cfg = ObserverConfig(
            enabled=True,
            context_window_size=200,
            shadow_recommendation_rate=0.5,
            min_context_before_recommending=5,
        )
        assert cfg.enabled is True
        assert cfg.context_window_size == 200
        assert cfg.shadow_recommendation_rate == 0.5
        assert cfg.min_context_before_recommending == 5

    def test_frozen(self):
        cfg = ObserverConfig()
        with pytest.raises(Exception):
            cfg.enabled = True

    def test_context_window_size_bounds(self):
        with pytest.raises(Exception):
            ObserverConfig(context_window_size=0)
        with pytest.raises(Exception):
            ObserverConfig(context_window_size=1001)
        # Edge values should succeed
        ObserverConfig(context_window_size=1)
        ObserverConfig(context_window_size=1000)

    def test_shadow_recommendation_rate_bounds(self):
        with pytest.raises(Exception):
            ObserverConfig(shadow_recommendation_rate=-0.01)
        with pytest.raises(Exception):
            ObserverConfig(shadow_recommendation_rate=1.01)
        ObserverConfig(shadow_recommendation_rate=0.0)
        ObserverConfig(shadow_recommendation_rate=1.0)

    def test_min_context_before_recommending_bounds(self):
        with pytest.raises(Exception):
            ObserverConfig(min_context_before_recommending=0)
        ObserverConfig(min_context_before_recommending=1)


# ═══════════════════════════════════════════════════════════════════════════
# MATCH SCORE HELPER
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeMatchScore:
    def test_both_empty(self):
        assert _compute_match_score({}, {}) == 0.0

    def test_identical_keys(self):
        assert _compute_match_score({"a": 1, "b": 2}, {"a": 3, "b": 4}) == 1.0

    def test_no_overlap(self):
        assert _compute_match_score({"a": 1}, {"b": 2}) == 0.0

    def test_partial_overlap(self):
        score = _compute_match_score({"a": 1, "b": 2}, {"b": 3, "c": 4})
        # intersection = {b}, union = {a, b, c} => 1/3
        assert abs(score - 1.0 / 3.0) < 1e-9

    def test_one_empty(self):
        assert _compute_match_score({"a": 1}, {}) == 0.0
        assert _compute_match_score({}, {"a": 1}) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVER — DISABLED MODE
# ═══════════════════════════════════════════════════════════════════════════


class TestObserverDisabled:
    def test_observe_is_noop(self, observer_disabled):
        event = make_event()
        observer_disabled.observe(event)
        assert observer_disabled.get_context("test-task") == []

    def test_record_action_returns_none(self, observer_disabled):
        result = observer_disabled.record_action("any-id", {"action": "do"})
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVER — OBSERVE & CONTEXT WINDOW
# ═══════════════════════════════════════════════════════════════════════════


class TestObserverObserve:
    def test_observe_adds_events(self, observer_enabled):
        e1 = make_event(task_name="task-a")
        e2 = make_event(task_name="task-a")
        observer_enabled.observe(e1)
        observer_enabled.observe(e2)
        ctx = observer_enabled.get_context("task-a")
        assert len(ctx) == 2
        assert ctx[0].event_id == e1.event_id
        assert ctx[1].event_id == e2.event_id

    def test_context_window_drops_oldest(self):
        cfg = enabled_config(context_window_size=3)
        obs = Observer(cfg)
        events = [make_event(task_name="t") for _ in range(5)]
        for e in events:
            obs.observe(e)
        ctx = obs.get_context("t")
        assert len(ctx) == 3
        # Should contain the 3 most recent events
        assert ctx[0].event_id == events[2].event_id
        assert ctx[1].event_id == events[3].event_id
        assert ctx[2].event_id == events[4].event_id

    def test_separate_task_contexts(self, observer_enabled):
        e1 = make_event(task_name="alpha")
        e2 = make_event(task_name="beta")
        observer_enabled.observe(e1)
        observer_enabled.observe(e2)
        assert len(observer_enabled.get_context("alpha")) == 1
        assert len(observer_enabled.get_context("beta")) == 1


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVER — GET CONTEXT & GET RECOMMENDATIONS (NON-EXISTING TASK)
# ═══════════════════════════════════════════════════════════════════════════


class TestObserverGetters:
    def test_get_context_nonexistent_task(self, observer_enabled):
        assert observer_enabled.get_context("no-such-task") == []

    def test_get_recommendations_nonexistent_task(self, observer_enabled):
        assert observer_enabled.get_recommendations("no-such-task") == []


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVER — RECORD ACTION
# ═══════════════════════════════════════════════════════════════════════════


class TestObserverRecordAction:
    def test_record_action_no_recommendation(self, observer_enabled):
        """record_action returns None when no recommendation exists for the event."""
        event = make_event()
        observer_enabled.observe(event)
        result = observer_enabled.record_action(event.event_id, {"do": "something"})
        # With default config (rate=0.1, min_context=10), after 1 event no rec is generated
        assert result is None

    def test_record_action_with_recommendation(self, observer_always_recommend):
        """record_action returns updated ShadowRecommendation with match score."""
        obs = observer_always_recommend
        event = make_event(task_name="t1")
        obs.observe(event)

        # Verify recommendation was generated
        recs = obs.get_recommendations("t1")
        assert len(recs) == 1
        assert recs[0].actual_action is None

        # Record actual action
        actual = {"key_a": 1, "key_b": 2}
        result = obs.record_action(event.event_id, actual)
        assert result is not None
        assert result.actual_action == actual
        assert result.match_score is not None
        assert 0.0 <= result.match_score <= 1.0
        # The recommended_action is {} (placeholder), actual has keys -> score = 0.0
        assert result.match_score == 0.0

    def test_record_action_match_score_with_overlapping_keys(self, observer_always_recommend):
        """Verify match_score computation when recommended and actual share keys."""
        obs = observer_always_recommend
        event = make_event(task_name="t1")
        obs.observe(event)

        # Manually replace the recommendation with one that has keys
        rec = obs._recommendations[event.event_id]
        patched_rec = rec.model_copy(update={"recommended_action": {"a": 1, "b": 2}})
        obs._recommendations[event.event_id] = patched_rec

        actual = {"b": 99, "c": 3}
        result = obs.record_action(event.event_id, actual)
        # intersection={b}, union={a,b,c} => 1/3
        assert result is not None
        assert abs(result.match_score - 1.0 / 3.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVER — CLEAR
# ═══════════════════════════════════════════════════════════════════════════


class TestObserverClear:
    def test_clear_removes_context_and_recommendations(self, observer_always_recommend):
        obs = observer_always_recommend
        for _ in range(5):
            obs.observe(make_event(task_name="clearing"))
        assert len(obs.get_context("clearing")) == 5
        assert len(obs.get_recommendations("clearing")) > 0

        obs.clear("clearing")
        assert obs.get_context("clearing") == []
        assert obs.get_recommendations("clearing") == []

    def test_clear_nonexistent_task_is_safe(self, observer_enabled):
        observer_enabled.clear("nonexistent")  # should not raise


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVER — MIN CONTEXT BEFORE RECOMMENDING
# ═══════════════════════════════════════════════════════════════════════════


class TestObserverMinContext:
    def test_no_recommendations_before_min_context(self):
        cfg = enabled_config(
            shadow_recommendation_rate=1.0,
            min_context_before_recommending=5,
        )
        obs = Observer(cfg)
        for i in range(4):
            obs.observe(make_event(task_name="t"))
        # 4 events < min 5 => no recommendations
        assert obs.get_recommendations("t") == []

    def test_recommendations_after_min_context(self):
        cfg = enabled_config(
            shadow_recommendation_rate=1.0,
            min_context_before_recommending=5,
        )
        obs = Observer(cfg)
        for i in range(5):
            obs.observe(make_event(task_name="t"))
        # 5th event meets threshold, rate=1.0 => recommendation generated
        recs = obs.get_recommendations("t")
        assert len(recs) == 1


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVER — SHADOW RECOMMENDATION RATE
# ═══════════════════════════════════════════════════════════════════════════


class TestObserverRecommendationRate:
    def test_rate_1_always_generates(self, observer_always_recommend):
        obs = observer_always_recommend
        events = [make_event(task_name="t") for _ in range(20)]
        for e in events:
            obs.observe(e)
        recs = obs.get_recommendations("t")
        # All 20 events should produce recommendations (rate=1.0, min_context=1)
        assert len(recs) == 20

    def test_rate_0_never_generates(self, observer_never_recommend):
        obs = observer_never_recommend
        for _ in range(50):
            obs.observe(make_event(task_name="t"))
        recs = obs.get_recommendations("t")
        assert len(recs) == 0

    def test_probabilistic_rate(self):
        """With rate=0.5, roughly half of eligible events should get recommendations."""
        cfg = enabled_config(
            shadow_recommendation_rate=0.5,
            min_context_before_recommending=1,
        )
        obs = Observer(cfg)
        n = 1000
        for _ in range(n):
            obs.observe(make_event(task_name="t"))
        recs = obs.get_recommendations("t")
        # Allow wide margin for randomness but not 0 or n
        assert 100 < len(recs) < 900


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVER — RECOMMENDATIONS SURVIVE CONTEXT EVICTION
# ═══════════════════════════════════════════════════════════════════════════


class TestObserverEvictionBehavior:
    def test_recommendations_persist_after_window_eviction(self):
        """Recommendations remain accessible even after their events are evicted from context."""
        cfg = enabled_config(
            context_window_size=3,
            shadow_recommendation_rate=1.0,
            min_context_before_recommending=1,
        )
        obs = Observer(cfg)
        early_events = [make_event(task_name="t") for _ in range(3)]
        for e in early_events:
            obs.observe(e)

        # Context has 3 events, 3 recommendations
        assert len(obs.get_context("t")) == 3
        assert len(obs.get_recommendations("t")) == 3

        # Add 3 more events to evict the earlier ones from context
        for _ in range(3):
            obs.observe(make_event(task_name="t"))

        assert len(obs.get_context("t")) == 3
        # All 6 recommendations should still be accessible
        assert len(obs.get_recommendations("t")) == 6

        # Can still record_action for evicted event
        result = obs.record_action(early_events[0].event_id, {"x": 1})
        assert result is not None
