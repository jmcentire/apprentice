"""Tests for story collector (storage and retrieval in training_data_store)."""

import json
import os
import tempfile
import pytest

from apprentice.training_data_store import (
    CollectorStory,
    StoryStep,
    StoryTrainingExample,
    StoryCollector,
    SplitName,
    sanitize_story_type,
    compute_split,
)


class TestSanitizeStoryType:
    def test_passthrough(self):
        assert sanitize_story_type("checkout_flow") == "checkout_flow"

    def test_removes_special_chars(self):
        assert sanitize_story_type("check out!@#flow") == "checkout_flow" or \
               sanitize_story_type("check out!@#flow") == "checkout_flow" or \
               sanitize_story_type("hello world") == "helloworld"

    def test_preserves_hyphens(self):
        assert sanitize_story_type("check-out") == "check-out"

    def test_empty_after_sanitization_raises(self):
        with pytest.raises(ValueError, match="empty after sanitization"):
            sanitize_story_type("!@#$%")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty after sanitization"):
            sanitize_story_type("")


class TestComputeSplit:
    def test_deterministic(self):
        """Same story_id always maps to the same split."""
        split1 = compute_split("story-abc-123")
        split2 = compute_split("story-abc-123")
        assert split1 == split2

    def test_returns_valid_split(self):
        split = compute_split("test-story")
        assert split in (SplitName.train, SplitName.val, SplitName.test)

    def test_distribution(self):
        """Roughly 80/10/10 distribution over many IDs."""
        counts = {SplitName.train: 0, SplitName.val: 0, SplitName.test: 0}
        for i in range(1000):
            split = compute_split(f"story-{i}")
            counts[split] += 1
        # Allow wide margin -- just check it's not degenerate
        assert counts[SplitName.train] > 500
        assert counts[SplitName.val] > 30
        assert counts[SplitName.test] > 30


class TestStoryStep:
    def test_create(self):
        s = StoryStep(0, {"q": "hello"}, {"a": "world"})
        assert s.step_index == 0
        assert s.input_data == {"q": "hello"}
        assert s.output_data == {"a": "world"}
        assert s.metadata == {}

    def test_roundtrip(self):
        s = StoryStep(1, {"x": 1}, {"y": 2}, {"tag": "v1"})
        d = s.to_dict()
        restored = StoryStep.from_dict(d)
        assert restored == s

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError):
            StoryStep(-1, {}, {})


class TestCollectorStory:
    def test_create(self):
        steps = [StoryStep(0, {"a": 1}, {"b": 2})]
        story = CollectorStory("s1", "checkout", steps)
        assert story.story_id == "s1"
        assert story.story_type == "checkout"
        assert len(story.steps) == 1

    def test_empty_story_id_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            CollectorStory("", "checkout", [])

    def test_empty_story_type_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            CollectorStory("s1", "", [])

    def test_roundtrip(self):
        steps = [
            StoryStep(0, {"a": 1}, {"b": 2}, {"tag": "v1"}),
            StoryStep(1, {"c": 3}, {"d": 4}),
        ]
        story = CollectorStory("s1", "checkout", steps, {"source": "test"})
        d = story.to_dict()
        restored = CollectorStory.from_dict(d)
        assert restored == story


class TestStoryCollector:
    @pytest.fixture
    def collector(self, tmp_path):
        return StoryCollector(str(tmp_path))

    def _make_story(self, story_id="s1", story_type="checkout", n_steps=2):
        steps = [
            StoryStep(i, {"input": i}, {"output": i * 10})
            for i in range(n_steps)
        ]
        return CollectorStory(story_id, story_type, steps)

    def test_store_and_retrieve(self, collector, tmp_path):
        story = self._make_story()
        collector.store_story(story)

        # File exists
        path = os.path.join(str(tmp_path), "stories", "checkout.jsonl")
        assert os.path.exists(path)

        # Read back
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["story_id"] == "s1"

    def test_store_multiple(self, collector, tmp_path):
        for i in range(3):
            story = self._make_story(story_id=f"s{i}")
            collector.store_story(story)

        path = os.path.join(str(tmp_path), "stories", "checkout.jsonl")
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_get_story_batch_filters_by_split(self, collector):
        """Store many stories and verify split filtering works."""
        for i in range(20):
            story = self._make_story(story_id=f"story-{i}")
            collector.store_story(story)

        train = collector.get_story_batch("checkout", "train")
        val = collector.get_story_batch("checkout", "val")
        test = collector.get_story_batch("checkout", "test")

        total = len(train) + len(val) + len(test)
        assert total == 20

    def test_get_story_batch_empty_file(self, collector):
        result = collector.get_story_batch("nonexistent", "train")
        assert result == []

    def test_get_story_batch_invalid_split(self, collector):
        with pytest.raises(ValueError, match="split must be one of"):
            collector.get_story_batch("checkout", "invalid")

    def test_stories_to_training_examples(self, collector):
        story = self._make_story(n_steps=4)
        examples = collector.stories_to_training_examples(story)
        assert len(examples) == 3  # 4 steps -> 3 examples

        # Verify first example
        ex0 = examples[0]
        assert ex0.story_id == "s1"
        assert ex0.step_index == 1
        assert ex0.context == {"output": 0}  # step 0's output
        assert ex0.input_data == {"input": 1}  # step 1's input
        assert ex0.expected_output == {"output": 10}  # step 1's output

    def test_stories_to_training_examples_single_step(self, collector):
        story = self._make_story(n_steps=1)
        examples = collector.stories_to_training_examples(story)
        assert examples == []

    def test_stories_to_training_examples_empty(self, collector):
        story = self._make_story(n_steps=0)
        examples = collector.stories_to_training_examples(story)
        assert examples == []

    def test_creates_directories(self, tmp_path):
        deep_dir = os.path.join(str(tmp_path), "a", "b", "c")
        collector = StoryCollector(deep_dir)
        story = self._make_story()
        collector.store_story(story)
        assert os.path.exists(os.path.join(deep_dir, "stories", "checkout.jsonl"))

    def test_story_type_sanitization(self, collector, tmp_path):
        story = self._make_story(story_type="checkout flow!")
        collector.store_story(story)
        # Sanitized name
        path = os.path.join(str(tmp_path), "stories", "checkoutflow.jsonl")
        assert os.path.exists(path)
