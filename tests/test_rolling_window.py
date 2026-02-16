"""
Contract tests for the Rolling Comparison Window component.

Organized into sections:
1. conftest-style fixtures and helpers (inline)
2. DequeRollingWindow unit tests (mocked store)
3. JsonFileWindowStore integration tests (tmp_path)
4. Invariant and property tests
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List, Optional

import pytest

# ── Component imports ──────────────────────────────────────────────────────
from src.rolling_window import (
    DequeRollingWindow,
    JsonFileWindowStore,
    ComparisonPair,
    ComparisonPairInput,
    WindowAggregate,
    WindowConfig,
    StoreConfig,
    TrendDirection,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES & HELPERS
# ═══════════════════════════════════════════════════════════════════════════

class FakeWindowStore:
    """In-memory fake implementing the WindowStore interface for unit tests."""

    def __init__(self):
        self._data = {}
        self.save_calls = []
        self.load_calls = []

    def save(self, task_type: str, pairs):
        self.save_calls.append((task_type, list(pairs)))
        self._data[task_type] = list(pairs)

    def load(self, task_type: str):
        self.load_calls.append(task_type)
        return list(self._data.get(task_type, []))


def make_config(window_size=10, max_window_age_seconds=3600.0):
    return WindowConfig(window_size=window_size, max_window_age_seconds=max_window_age_seconds)


def make_pair_input(input_data="test prompt", local_response="local", remote_response="remote", evaluation_score=0.5):
    return ComparisonPairInput(
        input_data=input_data,
        local_response=local_response,
        remote_response=remote_response,
        evaluation_score=evaluation_score,
    )


@pytest.fixture
def fake_store():
    return FakeWindowStore()


@pytest.fixture
def window(fake_store):
    """A default DequeRollingWindow with size 10, 1hr max age, and a fake store."""
    return DequeRollingWindow(config=make_config(10, 3600.0), store=fake_store)


@pytest.fixture
def small_window(fake_store):
    """A small window of size 3 for eviction tests."""
    return DequeRollingWindow(config=make_config(3, 3600.0), store=fake_store)


# ═══════════════════════════════════════════════════════════════════════════
# DEQUE ROLLING WINDOW — INIT
# ═══════════════════════════════════════════════════════════════════════════

class TestDequeRollingWindowInit:

    def test_init_happy_path(self, fake_store):
        w = DequeRollingWindow(config=make_config(10, 3600.0), store=fake_store)
        assert len(w) == 0, "Newly constructed window should be empty"

    def test_init_window_size_one(self, fake_store):
        w = DequeRollingWindow(config=make_config(1, 1.0), store=fake_store)
        assert len(w) == 0

    def test_init_window_size_max(self, fake_store):
        w = DequeRollingWindow(config=make_config(10000, 1.0), store=fake_store)
        assert len(w) == 0

    def test_init_invalid_window_size_zero(self, fake_store):
        with pytest.raises(Exception):
            DequeRollingWindow(config=make_config(0, 3600.0), store=fake_store)

    def test_init_invalid_window_size_negative(self, fake_store):
        with pytest.raises(Exception):
            DequeRollingWindow(config=make_config(-1, 3600.0), store=fake_store)

    def test_init_invalid_window_size_exceeds_max(self, fake_store):
        with pytest.raises(Exception):
            DequeRollingWindow(config=make_config(10001, 3600.0), store=fake_store)

    def test_init_invalid_max_age_zero(self, fake_store):
        with pytest.raises(Exception):
            DequeRollingWindow(config=make_config(5, 0.0), store=fake_store)

    def test_init_invalid_max_age_negative(self, fake_store):
        with pytest.raises(Exception):
            DequeRollingWindow(config=make_config(5, -1.0), store=fake_store)

    def test_init_no_store_accepted(self):
        """Passing None as store should be accepted (no-op persistence)."""
        w = DequeRollingWindow(config=make_config(5, 3600.0), store=None)
        assert len(w) == 0


# ═══════════════════════════════════════════════════════════════════════════
# DEQUE ROLLING WINDOW — ADD
# ═══════════════════════════════════════════════════════════════════════════

class TestDequeRollingWindowAdd:

    def test_add_happy_path(self, window):
        pair_input = make_pair_input(input_data="hello", evaluation_score=0.85)
        result = window.add(pair_input)

        assert result.input_data == "hello", "input_data should match"
        assert result.local_response == "local", "local_response should match"
        assert result.remote_response == "remote", "remote_response should match"
        assert result.evaluation_score == pytest.approx(0.85), "score should match"
        assert result.timestamp is not None, "timestamp must be set"
        assert len(window) == 1, "Window should have 1 pair"

    def test_add_timestamp_is_utc_and_recent(self, window):
        before = datetime.now(timezone.utc)
        result = window.add(make_pair_input())
        after = datetime.now(timezone.utc)

        ts = datetime.fromisoformat(result.timestamp)
        assert ts >= before - timedelta(seconds=2), "timestamp should be near current time"
        assert ts <= after + timedelta(seconds=2), "timestamp should be near current time"

    def test_add_boundary_score_zero(self, window):
        result = window.add(make_pair_input(evaluation_score=0.0))
        assert result.evaluation_score == pytest.approx(0.0)

    def test_add_boundary_score_one(self, window):
        result = window.add(make_pair_input(evaluation_score=1.0))
        assert result.evaluation_score == pytest.approx(1.0)

    def test_add_invalid_score_negative(self, window):
        with pytest.raises(Exception):
            window.add(make_pair_input(evaluation_score=-0.1))

    def test_add_invalid_score_above_one(self, window):
        with pytest.raises(Exception):
            window.add(make_pair_input(evaluation_score=1.1))

    def test_add_empty_input_data(self, window):
        with pytest.raises(Exception):
            window.add(make_pair_input(input_data=""))

    def test_add_at_capacity_evicts_oldest(self, small_window):
        pairs = []
        for i in range(3):
            p = small_window.add(make_pair_input(input_data=f"prompt_{i}", evaluation_score=0.1 * (i + 1)))
            pairs.append(p)

        assert len(small_window) == 3

        # Add a 4th pair — should evict the first
        p4 = small_window.add(make_pair_input(input_data="prompt_3", evaluation_score=0.9))
        assert len(small_window) == 3, "Window should still be at capacity"

        current_pairs = small_window.pairs()
        assert current_pairs[0].input_data == "prompt_1", "Oldest should now be prompt_1 (prompt_0 evicted)"
        assert current_pairs[-1].input_data == "prompt_3", "Newest should be prompt_3"

    def test_add_newest_is_last(self, window):
        window.add(make_pair_input(input_data="first"))
        p2 = window.add(make_pair_input(input_data="second"))
        pairs = window.pairs()
        assert pairs[-1].input_data == "second", "Newest pair should be last"

    def test_rapid_adds_no_data_loss(self, fake_store):
        ws = 50
        w = DequeRollingWindow(config=make_config(ws, 3600.0), store=fake_store)
        added = []
        for i in range(ws):
            p = w.add(make_pair_input(input_data=f"rapid_{i}", evaluation_score=i / ws))
            added.append(p)
        assert len(w) == ws, f"Window should contain exactly {ws} pairs"
        current = w.pairs()
        for i, pair in enumerate(current):
            assert pair.input_data == f"rapid_{i}", f"Pair at index {i} should match"


# ═══════════════════════════════════════════════════════════════════════════
# DEQUE ROLLING WINDOW — AGGREGATE
# ═══════════════════════════════════════════════════════════════════════════

class TestDequeRollingWindowAggregate:

    def test_aggregate_empty_window(self, window):
        result = window.aggregate(trend_window=None)
        assert result.sample_count == 0
        assert result.mean_score is None
        assert result.trend_slope is None
        assert result.trend_direction == TrendDirection.INSUFFICIENT_DATA
        assert result.is_stale is True
        assert result.oldest_timestamp is None

    def test_aggregate_single_pair(self, window):
        window.add(make_pair_input(evaluation_score=0.7))
        result = window.aggregate(trend_window=None)
        assert result.sample_count == 1
        assert result.mean_score == pytest.approx(0.7)
        assert result.trend_slope is None, "Need >= 2 pairs for trend"
        assert result.trend_direction == TrendDirection.INSUFFICIENT_DATA

    def test_aggregate_improving_trend(self, window):
        scores = [0.2, 0.4, 0.6, 0.8]
        for s in scores:
            window.add(make_pair_input(evaluation_score=s))

        result = window.aggregate(trend_window=None)
        assert result.sample_count == 4
        assert result.mean_score == pytest.approx(0.5)
        assert result.trend_slope is not None
        assert result.trend_slope > 0, "Slope should be positive for improving scores"
        assert result.trend_direction == TrendDirection.IMPROVING

    def test_aggregate_degrading_trend(self, window):
        for s in [0.8, 0.6, 0.4, 0.2]:
            window.add(make_pair_input(evaluation_score=s))

        result = window.aggregate(trend_window=None)
        assert result.trend_slope is not None
        assert result.trend_slope < 0, "Slope should be negative for degrading scores"
        assert result.trend_direction == TrendDirection.DEGRADING

    def test_aggregate_stable_trend(self, window):
        for _ in range(5):
            window.add(make_pair_input(evaluation_score=0.5))

        result = window.aggregate(trend_window=None)
        assert result.trend_slope is not None
        assert result.trend_slope == pytest.approx(0.0), "Slope should be 0 for identical scores"
        assert result.trend_direction == TrendDirection.STABLE

    def test_aggregate_with_trend_window(self, window):
        # First 2 pairs: high scores, last 3 pairs: increasing
        scores = [0.9, 0.1, 0.3, 0.5, 0.7]
        for s in scores:
            window.add(make_pair_input(evaluation_score=s))

        result = window.aggregate(trend_window=3)
        assert result.sample_count == 5
        assert result.mean_score == pytest.approx(sum(scores) / len(scores))
        # Trend only considers last 3: 0.3, 0.5, 0.7 → improving
        assert result.trend_slope is not None
        assert result.trend_slope > 0, "Trend over last 3 should be improving"
        assert result.trend_direction == TrendDirection.IMPROVING

    def test_aggregate_trend_window_exceeds_sample_count(self, window):
        window.add(make_pair_input(evaluation_score=0.3))
        window.add(make_pair_input(evaluation_score=0.7))
        result = window.aggregate(trend_window=5)
        assert result.trend_slope is None, "trend_window > sample_count → None slope"
        assert result.trend_direction == TrendDirection.INSUFFICIENT_DATA

    def test_aggregate_invalid_trend_window_one(self, window):
        with pytest.raises(Exception):
            window.aggregate(trend_window=1)

    def test_aggregate_invalid_trend_window_zero(self, window):
        with pytest.raises(Exception):
            window.aggregate(trend_window=0)

    def test_aggregate_mean_in_range(self, window):
        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            window.add(make_pair_input(evaluation_score=s))
        result = window.aggregate(trend_window=None)
        assert result.mean_score is not None
        assert 0.0 <= result.mean_score <= 1.0, "mean_score must be in [0, 1]"

    def test_aggregate_is_stale_false_when_fresh(self, window):
        window.add(make_pair_input())
        result = window.aggregate(trend_window=None)
        # max_window_age_seconds is 3600, pair was just added → not stale
        assert result.is_stale is False, "Freshly added data should not be stale"

    def test_aggregate_oldest_timestamp_set_when_nonempty(self, window):
        window.add(make_pair_input())
        result = window.aggregate(trend_window=None)
        assert result.oldest_timestamp is not None


# ═══════════════════════════════════════════════════════════════════════════
# DEQUE ROLLING WINDOW — CHECK_STALENESS
# ═══════════════════════════════════════════════════════════════════════════

class TestDequeRollingWindowCheckStaleness:

    def test_check_staleness_empty_window(self, window):
        assert window.check_staleness(max_age_seconds=3600.0) is True

    def test_check_staleness_fresh(self, window):
        window.add(make_pair_input())
        assert window.check_staleness(max_age_seconds=3600.0) is False

    def test_check_staleness_stale_pair(self, fake_store):
        """Simulate staleness by using a very small max_age and time.sleep."""
        w = DequeRollingWindow(config=make_config(10, 3600.0), store=fake_store)
        w.add(make_pair_input())
        # Check with a tiny max_age to ensure staleness
        # The pair was just added, so 0.0001 sec is likely too small
        # Use time.sleep to ensure the pair becomes stale
        time.sleep(0.05)
        result = w.check_staleness(max_age_seconds=0.01)
        assert result is True, "Pair older than 0.01s should be stale after 0.05s sleep"

    def test_check_staleness_not_stale_with_large_age(self, window):
        window.add(make_pair_input())
        assert window.check_staleness(max_age_seconds=999999.0) is False

    def test_check_staleness_invalid_max_age_negative(self, window):
        with pytest.raises(Exception):
            window.check_staleness(max_age_seconds=-1.0)

    def test_check_staleness_invalid_max_age_zero(self, window):
        with pytest.raises(Exception):
            window.check_staleness(max_age_seconds=0.0)


# ═══════════════════════════════════════════════════════════════════════════
# DEQUE ROLLING WINDOW — PAIRS
# ═══════════════════════════════════════════════════════════════════════════

class TestDequeRollingWindowPairs:

    def test_pairs_empty(self, window):
        result = window.pairs()
        assert result == [], "Empty window should return empty list"

    def test_pairs_returns_ordered_copy(self, window):
        for i in range(3):
            window.add(make_pair_input(input_data=f"p_{i}"))

        result = window.pairs()
        assert len(result) == 3
        assert result[0].input_data == "p_0"
        assert result[1].input_data == "p_1"
        assert result[2].input_data == "p_2"

    def test_pairs_snapshot_isolation(self, window):
        window.add(make_pair_input(input_data="before"))
        snapshot = window.pairs()
        assert len(snapshot) == 1

        window.add(make_pair_input(input_data="after"))
        assert len(snapshot) == 1, "Snapshot should not be affected by subsequent adds"
        assert len(window.pairs()) == 2, "Window should have 2 pairs now"

    def test_pairs_mutation_does_not_affect_window(self, window):
        window.add(make_pair_input())
        snapshot = window.pairs()
        snapshot.clear()
        assert len(window) == 1, "Clearing snapshot should not affect window"


# ═══════════════════════════════════════════════════════════════════════════
# DEQUE ROLLING WINDOW — __LEN__
# ═══════════════════════════════════════════════════════════════════════════

class TestDequeRollingWindowLen:

    def test_len_empty(self, window):
        assert len(window) == 0

    def test_len_after_adds(self, window):
        for i in range(5):
            window.add(make_pair_input(input_data=f"item_{i}"))
        assert len(window) == 5

    def test_len_never_exceeds_window_size(self, small_window):
        for i in range(10):
            small_window.add(make_pair_input(input_data=f"item_{i}"))
        assert len(small_window) == 3, "len should never exceed window_size=3"


# ═══════════════════════════════════════════════════════════════════════════
# DEQUE ROLLING WINDOW — PERSIST
# ═══════════════════════════════════════════════════════════════════════════

class TestDequeRollingWindowPersist:

    def test_persist_happy_path(self, window, fake_store):
        window.add(make_pair_input(input_data="p1"))
        window.add(make_pair_input(input_data="p2"))
        window.persist(task_type="summarization")

        assert len(fake_store.save_calls) == 1
        task, pairs = fake_store.save_calls[0]
        assert task == "summarization"
        assert len(pairs) == 2
        assert pairs[0].input_data == "p1"
        assert pairs[1].input_data == "p2"

    def test_persist_empty_task_type(self, window):
        with pytest.raises(Exception):
            window.persist(task_type="")

    def test_persist_no_store_noop(self):
        w = DequeRollingWindow(config=make_config(5, 3600.0), store=None)
        w.add(make_pair_input())
        # Should not raise
        w.persist(task_type="test")

    def test_persist_store_write_error(self, fake_store):
        w = DequeRollingWindow(config=make_config(5, 3600.0), store=fake_store)
        w.add(make_pair_input())
        fake_store.save = MagicMock(side_effect=IOError("disk full"))
        with pytest.raises(Exception):
            w.persist(task_type="test")


# ═══════════════════════════════════════════════════════════════════════════
# DEQUE ROLLING WINDOW — LOAD_FROM_STORE
# ═══════════════════════════════════════════════════════════════════════════

class TestDequeRollingWindowLoadFromStore:

    def test_load_from_store_happy_path(self, window, fake_store):
        # Pre-populate store via persist
        for i in range(3):
            window.add(make_pair_input(input_data=f"item_{i}", evaluation_score=0.1 * (i + 1)))
        window.persist(task_type="test_task")

        # Create fresh window and load
        w2 = DequeRollingWindow(config=make_config(10, 3600.0), store=fake_store)
        count = w2.load_from_store(task_type="test_task")
        assert count == 3
        assert len(w2) == 3
        loaded_pairs = w2.pairs()
        assert loaded_pairs[0].input_data == "item_0"
        assert loaded_pairs[2].input_data == "item_2"

    def test_load_from_store_empty_task_type(self, window):
        with pytest.raises(Exception):
            window.load_from_store(task_type="")

    def test_load_from_store_no_store_noop(self):
        w = DequeRollingWindow(config=make_config(5, 3600.0), store=None)
        result = w.load_from_store(task_type="test")
        assert result == 0

    def test_load_from_store_truncates_excess(self, fake_store):
        # Create a big window, add 5 pairs, persist
        big_w = DequeRollingWindow(config=make_config(10, 3600.0), store=fake_store)
        for i in range(5):
            big_w.add(make_pair_input(input_data=f"pair_{i}"))
        big_w.persist(task_type="trunc_test")

        # Load into a window of size 3 → should only keep last 3
        small_w = DequeRollingWindow(config=make_config(3, 3600.0), store=fake_store)
        count = small_w.load_from_store(task_type="trunc_test")
        assert count == 3, "Should load only window_size pairs"
        assert len(small_w) == 3
        loaded = small_w.pairs()
        # Most recent 3 should be pair_2, pair_3, pair_4
        assert loaded[0].input_data == "pair_2"
        assert loaded[1].input_data == "pair_3"
        assert loaded[2].input_data == "pair_4"

    def test_load_from_store_store_read_error(self, fake_store):
        w = DequeRollingWindow(config=make_config(5, 3600.0), store=fake_store)
        fake_store.load = MagicMock(side_effect=IOError("read error"))
        with pytest.raises(Exception):
            w.load_from_store(task_type="test")

    def test_load_from_store_corrupt_data_empty_fallback(self, fake_store):
        """Store returns empty list (corrupt data graceful fallback)."""
        fake_store._data["corrupt_task"] = []
        w = DequeRollingWindow(config=make_config(5, 3600.0), store=fake_store)
        count = w.load_from_store(task_type="corrupt_task")
        assert count == 0
        assert len(w) == 0

    def test_load_replaces_existing_data(self, fake_store):
        """load_from_store replaces current window contents, not appends."""
        w = DequeRollingWindow(config=make_config(10, 3600.0), store=fake_store)
        w.add(make_pair_input(input_data="original"))
        assert len(w) == 1

        # Store has different data
        fake_store._data["replace_test"] = []
        count = w.load_from_store(task_type="replace_test")
        assert count == 0, "Loading empty data should clear the window"
        assert len(w) == 0


# ═══════════════════════════════════════════════════════════════════════════
# PERSIST/LOAD ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════════════

class TestPersistLoadRoundTrip:

    def test_roundtrip_preserves_all_fields(self, fake_store):
        w1 = DequeRollingWindow(config=make_config(10, 3600.0), store=fake_store)
        p1 = w1.add(make_pair_input(
            input_data="prompt A",
            local_response="local A",
            remote_response="remote A",
            evaluation_score=0.42,
        ))
        p2 = w1.add(make_pair_input(
            input_data="prompt B",
            local_response="local B",
            remote_response="remote B",
            evaluation_score=0.99,
        ))
        w1.persist(task_type="roundtrip")

        w2 = DequeRollingWindow(config=make_config(10, 3600.0), store=fake_store)
        count = w2.load_from_store(task_type="roundtrip")
        assert count == 2

        loaded = w2.pairs()
        assert loaded[0].input_data == "prompt A"
        assert loaded[0].local_response == "local A"
        assert loaded[0].remote_response == "remote A"
        assert loaded[0].evaluation_score == pytest.approx(0.42)
        assert loaded[0].timestamp == p1.timestamp

        assert loaded[1].input_data == "prompt B"
        assert loaded[1].local_response == "local B"
        assert loaded[1].remote_response == "remote B"
        assert loaded[1].evaluation_score == pytest.approx(0.99)
        assert loaded[1].timestamp == p2.timestamp


# ═══════════════════════════════════════════════════════════════════════════
# JSON FILE WINDOW STORE — INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestJsonFileWindowStoreInit:

    def test_init_creates_directory(self, tmp_path):
        base = str(tmp_path / "new_subdir" / "store")
        store = JsonFileWindowStore(config=StoreConfig(base_dir=base))
        assert os.path.isdir(base), "base_dir should be created"

    def test_init_existing_directory(self, tmp_path):
        base = str(tmp_path)
        store = JsonFileWindowStore(config=StoreConfig(base_dir=base))
        assert os.path.isdir(base)

    def test_init_empty_base_dir(self):
        with pytest.raises(Exception):
            JsonFileWindowStore(config=StoreConfig(base_dir=""))


class TestJsonFileWindowStoreSaveLoad:

    def _make_pairs(self, window, n=3):
        """Helper: create n pairs via a window and return them."""
        pairs = []
        for i in range(n):
            p = window.add(make_pair_input(
                input_data=f"data_{i}",
                local_response=f"local_{i}",
                remote_response=f"remote_{i}",
                evaluation_score=round(i * 0.25, 2),
            ))
            pairs.append(p)
        return pairs

    def test_save_load_roundtrip(self, tmp_path):
        store = JsonFileWindowStore(config=StoreConfig(base_dir=str(tmp_path)))
        # Use a window to create valid ComparisonPair objects
        w = DequeRollingWindow(config=make_config(10, 3600.0), store=store)
        pairs = self._make_pairs(w, 3)

        store.save("my_task", pairs)
        loaded = store.load("my_task")

        assert len(loaded) == 3, "Should load all 3 pairs"
        for orig, loaded_p in zip(pairs, loaded):
            assert loaded_p.input_data == orig.input_data
            assert loaded_p.local_response == orig.local_response
            assert loaded_p.remote_response == orig.remote_response
            assert loaded_p.evaluation_score == pytest.approx(orig.evaluation_score)
            assert loaded_p.timestamp == orig.timestamp

    def test_save_empty_task_type(self, tmp_path):
        store = JsonFileWindowStore(config=StoreConfig(base_dir=str(tmp_path)))
        with pytest.raises(Exception):
            store.save("", [])

    def test_save_empty_pairs_list(self, tmp_path):
        store = JsonFileWindowStore(config=StoreConfig(base_dir=str(tmp_path)))
        store.save("empty_task", [])
        loaded = store.load("empty_task")
        assert loaded == []

    def test_load_missing_file(self, tmp_path):
        store = JsonFileWindowStore(config=StoreConfig(base_dir=str(tmp_path)))
        result = store.load("nonexistent_task")
        assert result == [], "Missing file should return empty list"

    def test_load_corrupt_json(self, tmp_path):
        store = JsonFileWindowStore(config=StoreConfig(base_dir=str(tmp_path)))
        # Write garbage to the expected file
        # We need to figure out the filename pattern; try common patterns
        # The contract says filename is derived from task_type with sanitization
        corrupt_file = tmp_path / "corrupt_task.json"
        corrupt_file.write_text("{{{not valid json!!!", encoding="utf-8")

        result = store.load("corrupt_task")
        assert result == [], "Corrupt JSON should return empty list"

    def test_load_empty_task_type(self, tmp_path):
        store = JsonFileWindowStore(config=StoreConfig(base_dir=str(tmp_path)))
        with pytest.raises(Exception):
            store.load("")

    def test_save_overwrite(self, tmp_path):
        store = JsonFileWindowStore(config=StoreConfig(base_dir=str(tmp_path)))
        w = DequeRollingWindow(config=make_config(10, 3600.0), store=store)
        p1 = w.add(make_pair_input(input_data="first_version"))
        store.save("overwrite_task", [p1])

        p2 = w.add(make_pair_input(input_data="second_version"))
        store.save("overwrite_task", [p2])

        loaded = store.load("overwrite_task")
        assert len(loaded) == 1
        assert loaded[0].input_data == "second_version"


# ═══════════════════════════════════════════════════════════════════════════
# END-TO-END: WINDOW + REAL STORE
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndWithRealStore:

    def test_full_workflow(self, tmp_path):
        store = JsonFileWindowStore(config=StoreConfig(base_dir=str(tmp_path)))
        w1 = DequeRollingWindow(config=make_config(5, 3600.0), store=store)

        for i in range(4):
            w1.add(make_pair_input(input_data=f"q_{i}", evaluation_score=0.2 * (i + 1)))

        w1.persist(task_type="e2e_test")

        # New window, load from store
        w2 = DequeRollingWindow(config=make_config(5, 3600.0), store=store)
        count = w2.load_from_store(task_type="e2e_test")
        assert count == 4
        assert len(w2) == 4

        agg = w2.aggregate(trend_window=None)
        assert agg.sample_count == 4
        assert agg.mean_score is not None
        assert 0.0 <= agg.mean_score <= 1.0
        assert agg.trend_direction == TrendDirection.IMPROVING


# ═══════════════════════════════════════════════════════════════════════════
# INVARIANT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestInvariants:

    def test_deque_maxlen_never_exceeded(self, fake_store):
        ws = 5
        w = DequeRollingWindow(config=make_config(ws, 3600.0), store=fake_store)
        for i in range(100):
            w.add(make_pair_input(input_data=f"inv_{i}", evaluation_score=(i % 11) / 10.0))
            assert len(w) <= ws, f"len={len(w)} exceeded window_size={ws} at add #{i}"

    def test_all_scores_in_range(self, window):
        scores = [0.0, 0.1, 0.5, 0.9, 1.0]
        for s in scores:
            window.add(make_pair_input(evaluation_score=s))
        for p in window.pairs():
            assert 0.0 <= p.evaluation_score <= 1.0, f"Score {p.evaluation_score} out of range"

    def test_all_input_data_nonempty(self, window):
        for i in range(5):
            window.add(make_pair_input(input_data=f"nonempty_{i}"))
        for p in window.pairs():
            assert len(p.input_data) > 0, "input_data must be non-empty"

    def test_chronological_order(self, window):
        for i in range(5):
            window.add(make_pair_input(input_data=f"chrono_{i}"))
            # small sleep to ensure distinct timestamps in edge-case environments
            time.sleep(0.001)

        pairs = window.pairs()
        for i in range(len(pairs) - 1):
            ts_a = datetime.fromisoformat(pairs[i].timestamp)
            ts_b = datetime.fromisoformat(pairs[i + 1].timestamp)
            assert ts_a <= ts_b, f"Pair {i} timestamp {ts_a} should be <= pair {i+1} timestamp {ts_b}"

    def test_frozen_pairs_immutable(self, window):
        pair = window.add(make_pair_input())
        with pytest.raises(Exception):
            pair.input_data = "mutated"
        with pytest.raises(Exception):
            pair.evaluation_score = 0.99

    def test_empty_window_is_stale(self, window):
        result = window.aggregate(trend_window=None)
        assert result.is_stale is True, "Empty window must always be stale"

    def test_trend_requires_two_points(self, window):
        window.add(make_pair_input(evaluation_score=0.5))
        result = window.aggregate(trend_window=None)
        assert result.trend_slope is None
        assert result.trend_direction == TrendDirection.INSUFFICIENT_DATA

    def test_aggregate_snapshot_immutability(self, window):
        """WindowAggregate should not change when the window changes after computation."""
        window.add(make_pair_input(evaluation_score=0.3))
        window.add(make_pair_input(evaluation_score=0.7))
        agg = window.aggregate(trend_window=None)

        # Add more pairs
        window.add(make_pair_input(evaluation_score=0.1))
        window.add(make_pair_input(evaluation_score=0.9))

        # Original aggregate should be unchanged
        assert agg.sample_count == 2, "Aggregate snapshot should not be affected by later adds"
        assert agg.mean_score == pytest.approx(0.5)

    def test_window_aggregate_is_frozen(self, window):
        window.add(make_pair_input())
        agg = window.aggregate(trend_window=None)
        with pytest.raises(Exception):
            agg.sample_count = 999

    def test_timestamps_are_valid_iso8601(self, window):
        window.add(make_pair_input())
        pair = window.pairs()[0]
        # Should not raise
        dt = datetime.fromisoformat(pair.timestamp)
        assert dt.tzinfo is not None or "Z" in pair.timestamp or "+" in pair.timestamp, \
            "Timestamp should include timezone info"


# ═══════════════════════════════════════════════════════════════════════════
# PROPERTY-BASED TESTS (using parameterize as lightweight hypothesis)
# ═══════════════════════════════════════════════════════════════════════════

class TestPropertyBased:
    """Parameterized tests covering property-like invariants across various inputs."""

    @pytest.mark.parametrize("ws", [1, 2, 5, 10, 100])
    def test_len_bounded_by_window_size(self, ws, fake_store):
        w = DequeRollingWindow(config=make_config(ws, 3600.0), store=fake_store)
        for i in range(ws * 3):
            w.add(make_pair_input(input_data=f"prop_{i}", evaluation_score=(i % 101) / 100.0))
        assert len(w) == ws

    @pytest.mark.parametrize("scores", [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.1, 0.2, 0.3, 0.4],
        [0.9, 0.7, 0.5, 0.3, 0.1],
    ])
    def test_mean_score_correctness(self, scores, fake_store):
        w = DequeRollingWindow(config=make_config(len(scores) + 5, 3600.0), store=fake_store)
        for s in scores:
            w.add(make_pair_input(evaluation_score=s))
        agg = w.aggregate(trend_window=None)
        expected_mean = sum(scores) / len(scores)
        assert agg.mean_score == pytest.approx(expected_mean), \
            f"Expected mean {expected_mean}, got {agg.mean_score}"

    @pytest.mark.parametrize("n_pairs", [0, 1, 2, 5, 10])
    def test_sample_count_matches_len(self, n_pairs, fake_store):
        w = DequeRollingWindow(config=make_config(20, 3600.0), store=fake_store)
        for i in range(n_pairs):
            w.add(make_pair_input(input_data=f"sc_{i}", evaluation_score=0.5))
        agg = w.aggregate(trend_window=None)
        assert agg.sample_count == n_pairs == len(w)

    @pytest.mark.parametrize("ws", [1, 3, 5])
    def test_persist_load_identity(self, ws, fake_store):
        w1 = DequeRollingWindow(config=make_config(ws, 3600.0), store=fake_store)
        for i in range(ws):
            w1.add(make_pair_input(input_data=f"id_{i}", evaluation_score=i / max(ws, 1)))
        w1.persist(task_type="identity_test")

        w2 = DequeRollingWindow(config=make_config(ws, 3600.0), store=fake_store)
        count = w2.load_from_store(task_type="identity_test")
        assert count == ws

        original = w1.pairs()
        restored = w2.pairs()
        assert len(original) == len(restored)
        for o, r in zip(original, restored):
            assert o.input_data == r.input_data
            assert o.evaluation_score == pytest.approx(r.evaluation_score)
            assert o.timestamp == r.timestamp
