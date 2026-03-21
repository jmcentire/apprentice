"""Tests for ApprenticeStoryLearner (root coordinator)."""

import os
import pytest
from datetime import datetime, timezone

from apprentice.data_models import Story as DMStory, StoryStep as DMStoryStep
from apprentice.journey_evaluator import JourneyEvaluator, EvaluatorConfig
from apprentice.phase_tracking import PhaseTracker, Phase, MetricEvent
from apprentice.training_data_store import StoryCollector
from apprentice.story_learner import (
    ApprenticeStoryLearner,
    RootConfig,
    ProcessingResult,
    LearningSummary,
    ExportedState,
    StoryProcessingError,
    LearningStateError,
)


class TestRootConfig:
    def test_defaults(self):
        cfg = RootConfig()
        assert cfg.min_steps_for_evaluation == 1
        assert cfg.auto_record_phase_metric is True
        assert cfg.goal_threshold == 0.8

    def test_custom(self):
        cfg = RootConfig(min_steps_for_evaluation=3, goal_threshold=0.5)
        assert cfg.min_steps_for_evaluation == 3
        assert cfg.goal_threshold == 0.5

    def test_invalid_min_steps(self):
        with pytest.raises(ValueError):
            RootConfig(min_steps_for_evaluation=0)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            RootConfig(goal_threshold=1.5)


class TestApprenticeStoryLearnerInit:
    @pytest.fixture
    def deps(self, tmp_path):
        return {
            "evaluator": JourneyEvaluator(),
            "phase_tracker": PhaseTracker(),
            "collector": StoryCollector(str(tmp_path)),
        }

    def test_valid_init(self, deps):
        learner = ApprenticeStoryLearner(**deps)
        assert learner is not None

    def test_missing_evaluator_method(self, deps):
        deps["evaluator"] = object()
        with pytest.raises(TypeError, match="evaluate_detailed"):
            ApprenticeStoryLearner(**deps)

    def test_missing_phase_tracker_method(self, deps):
        class BadTracker:
            pass
        deps["phase_tracker"] = BadTracker()
        with pytest.raises(TypeError, match="PhaseTrackerProtocol"):
            ApprenticeStoryLearner(**deps)

    def test_missing_collector_method(self, deps):
        class BadCollector:
            pass
        deps["collector"] = BadCollector()
        with pytest.raises(TypeError, match="StoryCollectorProtocol"):
            ApprenticeStoryLearner(**deps)

    def test_invalid_config_type(self, deps):
        with pytest.raises(TypeError, match="RootConfig"):
            ApprenticeStoryLearner(**deps, config="bad")


def _make_dm_story(story_id="st1", story_type="checkout", n_steps=3):
    """Helper to create a data_models.Story for testing."""
    now = datetime.now(timezone.utc)
    steps = []
    for i in range(n_steps):
        steps.append(DMStoryStep(
            step_id=f"step-{i}",
            task_type=f"task_{i}",
            input_data={"step": i},
            output_data={"result": i * 10},
            model_used="local",
            confidence_at_step=0.5 + (i * 0.1),
        ))
    return DMStory(
        story_id=story_id,
        story_type=story_type,
        steps=steps,
        created_at=now,
        context={"goal": "test"},
    )


class TestProcessStory:
    @pytest.fixture
    def learner(self, tmp_path):
        return ApprenticeStoryLearner(
            evaluator=JourneyEvaluator(),
            phase_tracker=PhaseTracker(),
            collector=StoryCollector(str(tmp_path)),
        )

    def test_basic_processing(self, learner):
        story = _make_dm_story()
        result = learner.process_story(story)
        assert isinstance(result, ProcessingResult)
        assert result.story_id == "st1"
        assert result.story_type == "checkout"
        assert result.stored is True
        assert result.step_count == 3
        assert result.phase == Phase.observation

    def test_journey_score_computed(self, learner):
        story = _make_dm_story(n_steps=3)
        result = learner.process_story(story)
        assert result.journey_score is not None
        assert 0.0 <= result.journey_score.composite <= 1.0

    def test_phase_metric_recorded(self, learner):
        story = _make_dm_story(n_steps=3)
        result = learner.process_story(story)
        assert result.phase_metric_recorded is True

    def test_skip_evaluation_below_min_steps(self, tmp_path):
        learner = ApprenticeStoryLearner(
            evaluator=JourneyEvaluator(),
            phase_tracker=PhaseTracker(),
            collector=StoryCollector(str(tmp_path)),
            config=RootConfig(min_steps_for_evaluation=5),
        )
        story = _make_dm_story(n_steps=3)
        result = learner.process_story(story)
        assert result.journey_score is None
        assert result.phase_metric_recorded is False

    def test_auto_record_disabled(self, tmp_path):
        learner = ApprenticeStoryLearner(
            evaluator=JourneyEvaluator(),
            phase_tracker=PhaseTracker(),
            collector=StoryCollector(str(tmp_path)),
            config=RootConfig(auto_record_phase_metric=False),
        )
        story = _make_dm_story(n_steps=3)
        result = learner.process_story(story)
        assert result.phase_metric_recorded is False

    def test_empty_story_type_rejected(self, learner):
        now = datetime.now(timezone.utc)
        # Create a story-like object with empty story_type
        class FakeStory:
            story_id = "st1"
            story_type = "!@#$"  # sanitizes to empty
            steps = []
        with pytest.raises(StoryProcessingError, match="non-empty"):
            learner.process_story(FakeStory())

    def test_invalid_story_object(self, learner):
        with pytest.raises(StoryProcessingError, match="valid Story"):
            learner.process_story("not a story")

    def test_story_persisted_to_disk(self, tmp_path):
        learner = ApprenticeStoryLearner(
            evaluator=JourneyEvaluator(),
            phase_tracker=PhaseTracker(),
            collector=StoryCollector(str(tmp_path)),
        )
        story = _make_dm_story()
        learner.process_story(story)
        jsonl = os.path.join(str(tmp_path), "stories", "checkout.jsonl")
        assert os.path.exists(jsonl)


class TestGetTrainingBatch:
    @pytest.fixture
    def learner(self, tmp_path):
        return ApprenticeStoryLearner(
            evaluator=JourneyEvaluator(),
            phase_tracker=PhaseTracker(),
            collector=StoryCollector(str(tmp_path)),
        )

    def test_empty_batch(self, learner):
        batch = learner.get_training_batch("checkout", "train")
        assert batch == []

    def test_retrieve_after_store(self, learner):
        for i in range(10):
            story = _make_dm_story(story_id=f"st-{i}", n_steps=3)
            learner.process_story(story)

        # Get all splits
        train = learner.get_training_batch("checkout", "train")
        val = learner.get_training_batch("checkout", "val")
        test = learner.get_training_batch("checkout", "test")

        # Each 3-step story -> 2 training examples
        total = len(train) + len(val) + len(test)
        assert total == 20  # 10 stories * 2 examples each

    def test_batch_size_limit(self, learner):
        for i in range(10):
            story = _make_dm_story(story_id=f"st-{i}", n_steps=3)
            learner.process_story(story)
        batch = learner.get_training_batch("checkout", "train", batch_size=3)
        assert len(batch) <= 3

    def test_invalid_split(self, learner):
        with pytest.raises(ValueError, match="split must be one of"):
            learner.get_training_batch("checkout", "invalid")

    def test_invalid_batch_size(self, learner):
        with pytest.raises(ValueError, match="positive integer"):
            learner.get_training_batch("checkout", "train", batch_size=0)


class TestGetLearningSummary:
    @pytest.fixture
    def learner(self, tmp_path):
        return ApprenticeStoryLearner(
            evaluator=JourneyEvaluator(),
            phase_tracker=PhaseTracker(),
            collector=StoryCollector(str(tmp_path)),
        )

    def test_global_summary(self, learner):
        summary = learner.get_learning_summary()
        assert isinstance(summary, LearningSummary)
        assert summary.story_type is None
        assert summary.phase == Phase.observation

    def test_per_type_summary(self, learner):
        story = _make_dm_story()
        learner.process_story(story)
        summary = learner.get_learning_summary("checkout")
        assert summary.story_type == "checkout"

    def test_known_story_types_in_global(self, learner):
        for st in ["checkout", "support"]:
            story = _make_dm_story(story_type=st)
            learner.process_story(story)
        summary = learner.get_learning_summary()
        assert set(summary.known_story_types) == {"checkout", "support"}

    def test_known_story_types_empty_for_per_type(self, learner):
        story = _make_dm_story()
        learner.process_story(story)
        summary = learner.get_learning_summary("checkout")
        assert summary.known_story_types == []

    def test_invalid_story_type_type(self, learner):
        with pytest.raises(TypeError, match="str or None"):
            learner.get_learning_summary(42)


class TestExportImportState:
    @pytest.fixture
    def learner(self, tmp_path):
        return ApprenticeStoryLearner(
            evaluator=JourneyEvaluator(),
            phase_tracker=PhaseTracker(),
            collector=StoryCollector(str(tmp_path)),
        )

    def test_export_state(self, learner):
        story = _make_dm_story()
        learner.process_story(story)
        state = learner.export_state()
        assert isinstance(state, ExportedState)
        assert state.version == 1
        assert "global" in state.phase_metrics

    def test_roundtrip(self, tmp_path):
        learner1 = ApprenticeStoryLearner(
            evaluator=JourneyEvaluator(),
            phase_tracker=PhaseTracker(),
            collector=StoryCollector(str(tmp_path / "l1")),
        )
        for i in range(5):
            story = _make_dm_story(story_id=f"st-{i}")
            learner1.process_story(story)

        state = learner1.export_state()

        learner2 = ApprenticeStoryLearner(
            evaluator=JourneyEvaluator(),
            phase_tracker=PhaseTracker(),
            collector=StoryCollector(str(tmp_path / "l2")),
        )
        learner2.import_state(state)

        s1 = learner1.get_learning_summary()
        s2 = learner2.get_learning_summary()
        assert s1.metrics.total_attempts == s2.metrics.total_attempts

    def test_import_dict(self, learner):
        data = {
            "version": 1,
            "phase_metrics": {
                "global": {
                    "success_count": 3, "failure_count": 1,
                    "total_attempts": 4, "success_rate": 0.75,
                    "consecutive_successes": 2,
                    "current_phase": "observation",
                    "story_type": None,
                },
                "by_story_type": {},
            },
        }
        learner.import_state(data)
        m = learner.get_learning_summary().metrics
        assert m.total_attempts == 4

    def test_import_invalid_type(self, learner):
        with pytest.raises(LearningStateError, match="ExportedState"):
            learner.import_state("not valid")

    def test_import_missing_global(self, learner):
        with pytest.raises(LearningStateError, match="global"):
            learner.import_state({"version": 1, "phase_metrics": {"by_story_type": {}}})

    def test_import_unsupported_version(self, learner):
        with pytest.raises(LearningStateError, match="version"):
            learner.import_state({"version": 0, "phase_metrics": {}})


class TestBackwardCompatibility:
    """Verify story features don't break existing atomic routing."""

    def test_existing_training_data_store_unaffected(self, tmp_path):
        """The original TrainingDataStore still works."""
        from apprentice.training_data_store import (
            TrainingDataStore,
            TrainingDataStoreConfig,
            TrainingExample,
        )
        store = TrainingDataStore()
        config = TrainingDataStoreConfig(base_dir=str(tmp_path))
        store.initialize(config)

        ex = TrainingExample(
            input="hello",
            output="world",
            task_type="test_task",
            timestamp="2024-01-01T00:00:00Z",
        )
        result = store.add_example(ex)
        assert result.example_count == 1

    def test_existing_evaluators_unaffected(self):
        """ExactMatchEvaluator still works."""
        from apprentice.evaluators import (
            ExactMatchEvaluator,
            TaskResponse,
            TaskEvaluatorConfig,
            EvaluatorType,
        )
        evaluator = ExactMatchEvaluator()
        local = TaskResponse(
            task_id="t1", output="hello",
            timestamp="2024-01-01T00:00:00Z",
        )
        remote = TaskResponse(
            task_id="t1", output="hello",
            timestamp="2024-01-01T00:00:00Z",
        )
        config = TaskEvaluatorConfig(evaluator_type=EvaluatorType.exact_match)
        result = evaluator.evaluate(local, remote, config)
        assert result.score == 1.0

    def test_existing_phase_manager_unaffected(self):
        """The original PhaseManager still works."""
        from apprentice.phase_manager import (
            PhaseManager,
            PhaseMetrics,
            PhaseThresholds,
            Phase,
            TrendDirection,
        )

        class FakeClock:
            def now(self):
                return datetime.now(timezone.utc).isoformat()

        pm = PhaseManager(clock=FakeClock())
        metrics = PhaseMetrics(
            task_id="test_task",
            example_count=100,
            rolling_correlation=0.95,
            local_model_available=True,
            consecutive_windows_above_threshold=5,
            trend_direction=TrendDirection.IMPROVING,
        )
        thresholds = PhaseThresholds(
            phase1_to_phase2_example_count=10,
            phase2_to_phase3_correlation=0.9,
            coaching_trigger=0.7,
            emergency_threshold=0.3,
        )
        result = pm.evaluate(metrics, thresholds)
        assert result.current_phase == Phase.AUTONOMOUS
