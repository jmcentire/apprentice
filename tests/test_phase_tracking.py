"""Tests for per-journey phase tracking."""

import pytest

from apprentice.phase_tracking import (
    Phase,
    PhaseMetrics,
    PhaseTracker,
    MetricEvent,
    PhaseComputationResult,
    normalize_story_type,
)


class TestNormalizeStoryType:
    def test_none_passthrough(self):
        assert normalize_story_type(None) is None

    def test_empty_to_none(self):
        assert normalize_story_type("") is None

    def test_whitespace_to_none(self):
        assert normalize_story_type("   ") is None

    def test_strips_whitespace(self):
        assert normalize_story_type("  checkout  ") == "checkout"

    def test_valid_passthrough(self):
        assert normalize_story_type("checkout_flow") == "checkout_flow"

    def test_non_string_raises(self):
        with pytest.raises(TypeError, match="must be a str or None"):
            normalize_story_type(42)

    def test_non_string_list_raises(self):
        with pytest.raises(TypeError, match="must be a str or None"):
            normalize_story_type(["x"])


class TestPhaseMetrics:
    def test_create_defaults(self):
        m = PhaseMetrics()
        assert m.success_count == 0
        assert m.failure_count == 0
        assert m.total_attempts == 0
        assert m.success_rate == 0.0
        assert m.consecutive_successes == 0
        assert m.current_phase == Phase.observation
        assert m.story_type is None

    def test_invariant_violation(self):
        with pytest.raises(ValueError, match="total_attempts"):
            PhaseMetrics(success_count=5, failure_count=3, total_attempts=10)

    def test_with_story_type(self):
        m = PhaseMetrics(story_type="checkout")
        assert m.story_type == "checkout"

    def test_copy(self):
        m = PhaseMetrics(success_count=5, failure_count=3, total_attempts=8,
                         success_rate=0.625, story_type="checkout")
        c = m.copy()
        assert c.success_count == m.success_count
        assert c.story_type == m.story_type
        assert c is not m

    def test_to_dict_from_dict_roundtrip(self):
        m = PhaseMetrics(
            success_count=10, failure_count=2, total_attempts=12,
            success_rate=10/12, consecutive_successes=5,
            current_phase=Phase.coaching, story_type="checkout",
        )
        d = m.to_dict()
        restored = PhaseMetrics.from_dict(d)
        assert restored.success_count == m.success_count
        assert restored.failure_count == m.failure_count
        assert restored.story_type == m.story_type
        assert restored.current_phase == m.current_phase


class TestPhaseTracker:
    @pytest.fixture
    def tracker(self):
        return PhaseTracker()

    def test_initial_phase_observation(self, tracker):
        result = tracker.compute_phase()
        assert result.phase == Phase.observation

    def test_initial_phase_for_unknown_story_type(self, tracker):
        result = tracker.compute_phase("checkout")
        assert result.phase == Phase.observation

    def test_record_success_updates_global(self, tracker):
        tracker.record_metric(MetricEvent(success=True))
        m = tracker.get_phase_metrics()
        assert m.success_count == 1
        assert m.failure_count == 0
        assert m.total_attempts == 1

    def test_record_failure_updates_global(self, tracker):
        tracker.record_metric(MetricEvent(success=False))
        m = tracker.get_phase_metrics()
        assert m.success_count == 0
        assert m.failure_count == 1
        assert m.consecutive_successes == 0

    def test_record_with_story_type_dual_writes(self, tracker):
        tracker.record_metric(MetricEvent(success=True, story_type="checkout"))
        # Global updated
        global_m = tracker.get_phase_metrics()
        assert global_m.success_count == 1
        assert global_m.story_type is None
        # Per-type updated
        type_m = tracker.get_phase_metrics("checkout")
        assert type_m.success_count == 1
        assert type_m.story_type == "checkout"

    def test_per_type_independent(self, tracker):
        tracker.record_metric(MetricEvent(success=True, story_type="checkout"))
        tracker.record_metric(MetricEvent(success=False, story_type="support"))

        checkout_m = tracker.get_phase_metrics("checkout")
        assert checkout_m.success_count == 1
        assert checkout_m.failure_count == 0

        support_m = tracker.get_phase_metrics("support")
        assert support_m.success_count == 0
        assert support_m.failure_count == 1

        global_m = tracker.get_phase_metrics()
        assert global_m.total_attempts == 2

    def test_consecutive_successes_reset_on_failure(self, tracker):
        for _ in range(3):
            tracker.record_metric(MetricEvent(success=True))
        m = tracker.get_phase_metrics()
        assert m.consecutive_successes == 3

        tracker.record_metric(MetricEvent(success=False))
        m = tracker.get_phase_metrics()
        assert m.consecutive_successes == 0

    def test_phase_progression_to_coaching(self, tracker):
        """After enough successful attempts, should reach coaching."""
        for _ in range(6):
            tracker.record_metric(MetricEvent(success=True))
        result = tracker.compute_phase()
        assert result.phase in (Phase.coaching, Phase.observation)
        # At 6 successes with 100% rate >= 0.6 and >= 5 attempts
        assert result.phase == Phase.coaching

    def test_phase_progression_to_autonomous(self, tracker):
        """After many consecutive successes, should reach autonomous."""
        for _ in range(15):
            tracker.record_metric(MetricEvent(success=True))
        result = tracker.compute_phase()
        assert result.phase == Phase.autonomous

    def test_empty_string_story_type_normalized(self, tracker):
        tracker.record_metric(MetricEvent(success=True, story_type=""))
        # Should go to global only, not create a "" key
        assert "" not in tracker.get_known_story_types()

    def test_get_known_story_types(self, tracker):
        tracker.record_metric(MetricEvent(success=True, story_type="checkout"))
        tracker.record_metric(MetricEvent(success=True, story_type="support"))
        types = tracker.get_known_story_types()
        assert set(types) == {"checkout", "support"}

    def test_get_known_story_types_empty(self, tracker):
        assert tracker.get_known_story_types() == []

    def test_invalid_event_type(self, tracker):
        with pytest.raises(TypeError, match="MetricEvent"):
            tracker.record_metric({"success": True})

    def test_per_type_phase_independent(self, tracker):
        """One story type can be autonomous while another is observation."""
        for _ in range(15):
            tracker.record_metric(MetricEvent(success=True, story_type="checkout"))

        checkout = tracker.compute_phase("checkout")
        support = tracker.compute_phase("support")
        assert checkout.phase == Phase.autonomous
        assert support.phase == Phase.observation


class TestSerializationRoundtrip:
    def test_roundtrip(self):
        tracker = PhaseTracker()
        for _ in range(5):
            tracker.record_metric(MetricEvent(success=True, story_type="checkout"))
        tracker.record_metric(MetricEvent(success=False, story_type="support"))

        # Export
        serialized = tracker.serialize_all_metrics()
        assert "global" in serialized
        assert "by_story_type" in serialized
        assert "checkout" in serialized["by_story_type"]

        # Import into fresh tracker
        tracker2 = PhaseTracker()
        tracker2.deserialize_all_metrics(serialized)

        # Verify
        m1 = tracker.get_phase_metrics()
        m2 = tracker2.get_phase_metrics()
        assert m1.total_attempts == m2.total_attempts

        m1c = tracker.get_phase_metrics("checkout")
        m2c = tracker2.get_phase_metrics("checkout")
        assert m1c.success_count == m2c.success_count

    def test_deserialize_missing_global_raises(self):
        tracker = PhaseTracker()
        with pytest.raises(KeyError, match="global"):
            tracker.deserialize_all_metrics({"by_story_type": {}})

    def test_deserialize_missing_by_story_type_raises(self):
        tracker = PhaseTracker()
        with pytest.raises(KeyError, match="by_story_type"):
            tracker.deserialize_all_metrics({"global": {}})

    def test_deserialize_invalid_data_type(self):
        tracker = PhaseTracker()
        with pytest.raises(TypeError, match="dict"):
            tracker.deserialize_all_metrics("not a dict")
