"""Tests for Story and StoryStep data models."""

import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from apprentice.data_models import Story, StoryStep


class TestStoryStep:
    """Tests for StoryStep model."""

    def test_create_minimal(self):
        step = StoryStep(
            step_id="s1",
            task_type="code_gen",
            input_data={"prompt": "hello"},
            model_used="gpt-4",
        )
        assert step.step_id == "s1"
        assert step.task_type == "code_gen"
        assert step.input_data == {"prompt": "hello"}
        assert step.output_data is None
        assert step.model_used == "gpt-4"
        assert step.confidence_at_step is None
        assert step.metadata == {}

    def test_create_full(self):
        step = StoryStep(
            step_id="s1",
            task_type="code_gen",
            input_data={"prompt": "hello"},
            output_data={"code": "print('hi')"},
            model_used="local",
            confidence_at_step=0.95,
            metadata={"latency_ms": 100},
        )
        assert step.output_data == {"code": "print('hi')"}
        assert step.confidence_at_step == 0.95
        assert step.metadata == {"latency_ms": 100}

    def test_frozen(self):
        step = StoryStep(
            step_id="s1", task_type="t", input_data={}, model_used="m"
        )
        with pytest.raises(ValidationError):
            step.step_id = "s2"

    def test_empty_step_id_rejected(self):
        with pytest.raises(ValidationError):
            StoryStep(step_id="", task_type="t", input_data={}, model_used="m")

    def test_empty_task_type_rejected(self):
        with pytest.raises(ValidationError):
            StoryStep(step_id="s1", task_type="", input_data={}, model_used="m")

    def test_empty_model_used_rejected(self):
        with pytest.raises(ValidationError):
            StoryStep(step_id="s1", task_type="t", input_data={}, model_used="")

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            StoryStep(
                step_id="s1", task_type="t", input_data={},
                model_used="m", confidence_at_step=1.5
            )

    def test_confidence_negative_rejected(self):
        with pytest.raises(ValidationError):
            StoryStep(
                step_id="s1", task_type="t", input_data={},
                model_used="m", confidence_at_step=-0.1
            )

    def test_confidence_boundary_zero(self):
        step = StoryStep(
            step_id="s1", task_type="t", input_data={},
            model_used="m", confidence_at_step=0.0
        )
        assert step.confidence_at_step == 0.0

    def test_confidence_boundary_one(self):
        step = StoryStep(
            step_id="s1", task_type="t", input_data={},
            model_used="m", confidence_at_step=1.0
        )
        assert step.confidence_at_step == 1.0

    def test_json_roundtrip(self):
        step = StoryStep(
            step_id="s1", task_type="code_gen",
            input_data={"prompt": "hello"},
            output_data={"result": "world"},
            model_used="gpt-4", confidence_at_step=0.8,
            metadata={"key": "value"},
        )
        json_str = step.model_dump_json()
        restored = StoryStep.model_validate_json(json_str)
        assert restored == step

    def test_input_data_any_type(self):
        """input_data can be any JSON-compatible type."""
        step = StoryStep(
            step_id="s1", task_type="t",
            input_data="just a string",
            model_used="m",
        )
        assert step.input_data == "just a string"

    def test_model_copy(self):
        step = StoryStep(
            step_id="s1", task_type="t", input_data={}, model_used="m"
        )
        updated = step.model_copy(update={"step_id": "s2"})
        assert updated.step_id == "s2"
        assert step.step_id == "s1"  # original unchanged


class TestStory:
    """Tests for Story model."""

    def _utc_now(self):
        return datetime.now(timezone.utc)

    def _make_step(self, step_id="s1"):
        return StoryStep(
            step_id=step_id, task_type="t", input_data={}, model_used="m"
        )

    def test_create_minimal(self):
        now = self._utc_now()
        story = Story(
            story_id="st1", story_type="request",
            steps=[], created_at=now,
        )
        assert story.story_id == "st1"
        assert story.story_type == "request"
        assert story.steps == []
        assert story.created_at == now
        assert story.closed_at is None
        assert story.context == {}

    def test_create_with_steps(self):
        now = self._utc_now()
        s1 = self._make_step("s1")
        s2 = self._make_step("s2")
        story = Story(
            story_id="st1", story_type="journey",
            steps=[s1, s2], created_at=now,
            context={"goal": "deploy"},
        )
        assert len(story.steps) == 2
        assert story.steps[0].step_id == "s1"
        assert story.context == {"goal": "deploy"}

    def test_frozen(self):
        now = self._utc_now()
        story = Story(
            story_id="st1", story_type="request",
            steps=[], created_at=now,
        )
        with pytest.raises(ValidationError):
            story.story_id = "st2"

    def test_empty_story_id_rejected(self):
        with pytest.raises(ValidationError):
            Story(
                story_id="", story_type="request",
                steps=[], created_at=self._utc_now(),
            )

    def test_empty_story_type_rejected(self):
        with pytest.raises(ValidationError):
            Story(
                story_id="st1", story_type="",
                steps=[], created_at=self._utc_now(),
            )

    def test_naive_datetime_rejected(self):
        with pytest.raises(ValidationError):
            Story(
                story_id="st1", story_type="request",
                steps=[], created_at=datetime.now(),
            )

    def test_non_utc_datetime_rejected(self):
        est = timezone(timedelta(hours=-5))
        with pytest.raises(ValidationError):
            Story(
                story_id="st1", story_type="request",
                steps=[], created_at=datetime.now(est),
            )

    def test_closed_at_utc(self):
        now = self._utc_now()
        story = Story(
            story_id="st1", story_type="request",
            steps=[], created_at=now, closed_at=now,
        )
        assert story.closed_at == now

    def test_closed_at_naive_rejected(self):
        now = self._utc_now()
        with pytest.raises(ValidationError):
            Story(
                story_id="st1", story_type="request",
                steps=[], created_at=now,
                closed_at=datetime.now(),
            )

    def test_closed_at_non_utc_rejected(self):
        now = self._utc_now()
        est = timezone(timedelta(hours=-5))
        with pytest.raises(ValidationError):
            Story(
                story_id="st1", story_type="request",
                steps=[], created_at=now,
                closed_at=datetime.now(est),
            )

    def test_json_roundtrip(self):
        now = self._utc_now()
        s1 = StoryStep(
            step_id="s1", task_type="code_gen",
            input_data={"x": 1}, output_data={"y": 2},
            model_used="m", confidence_at_step=0.5,
        )
        story = Story(
            story_id="st1", story_type="journey",
            steps=[s1], created_at=now, closed_at=now,
            context={"goal": "test"},
        )
        json_str = story.model_dump_json()
        restored = Story.model_validate_json(json_str)
        assert restored == story

    def test_model_copy(self):
        now = self._utc_now()
        story = Story(
            story_id="st1", story_type="request",
            steps=[], created_at=now,
        )
        later = self._utc_now()
        closed = story.model_copy(update={"closed_at": later})
        assert closed.closed_at == later
        assert story.closed_at is None  # original unchanged

    def test_steps_order_preserved(self):
        now = self._utc_now()
        steps = [self._make_step(f"s{i}") for i in range(5)]
        story = Story(
            story_id="st1", story_type="journey",
            steps=steps, created_at=now,
        )
        for i, step in enumerate(story.steps):
            assert step.step_id == f"s{i}"
