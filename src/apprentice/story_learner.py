"""Apprentice Story Learner (root coordinator) v1

Facade/Coordinator that orchestrates story-level learning by composing
journey evaluation, phase tracking, and story collection behind a unified API.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from apprentice.journey_evaluator import JourneyEvaluator, JourneyScore, Story as JEStory, Step as JEStep
from apprentice.phase_tracking import (
    PhaseTracker,
    Phase,
    MetricEvent,
    PhaseMetrics,
    normalize_story_type,
)
from apprentice.training_data_store import (
    StoryCollector,
    CollectorStory,
    StoryStep as CollectorStep,
    StoryTrainingExample,
    sanitize_story_type,
    SplitName,
)

_PACT_KEY = "PACT:story_learner"
logger = logging.getLogger(__name__)


def _log(level: str, msg: str, **kwargs) -> None:
    getattr(logger, level)(f"[{_PACT_KEY}] {msg}", **kwargs)


# ============================================================================
# Error Classes
# ============================================================================

class StoryProcessingError(Exception):
    """Raised when story processing fails."""
    pass


class LearningStateError(Exception):
    """Raised when learning state import/export fails."""
    pass


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class RootConfig:
    """Configuration for the root coordinator."""
    min_steps_for_evaluation: int = 1
    auto_record_phase_metric: bool = True
    goal_threshold: float = 0.8

    def __post_init__(self):
        if self.min_steps_for_evaluation < 1:
            raise ValueError("min_steps_for_evaluation must be >= 1")
        if not (0.0 <= self.goal_threshold <= 1.0):
            raise ValueError("goal_threshold must be in [0.0, 1.0]")


# ============================================================================
# Result Types
# ============================================================================

@dataclass(frozen=True)
class ProcessingResult:
    """Result of processing a single story through the full pipeline."""
    story_id: str
    story_type: str
    stored: bool
    journey_score: Optional[JourneyScore] = None
    phase: Phase = Phase.observation
    phase_metric_recorded: bool = False
    step_count: int = 0


@dataclass(frozen=True)
class LearningSummary:
    """Aggregated learning summary for a story type or globally."""
    story_type: Optional[str] = None
    phase: Phase = Phase.observation
    metrics: PhaseMetrics = field(default_factory=PhaseMetrics)
    known_story_types: list = field(default_factory=list)


@dataclass(frozen=True)
class ExportedState:
    """Serialized state of the story-learning subsystem."""
    version: int = 1
    phase_metrics: dict = field(default_factory=dict)


TrainingExampleList = list[StoryTrainingExample]
OptionalRootConfig = Optional[RootConfig]
OptionalStoryType = Optional[str]
OptionalBatchSize = Optional[int]
OptionalJourneyScore = Optional[JourneyScore]


# ============================================================================
# ApprenticeStoryLearner
# ============================================================================

class ApprenticeStoryLearner:
    """Coordinates story-level learning across evaluator, phase tracker,
    and collector subsystems."""

    def __init__(
        self,
        evaluator: Any,
        phase_tracker: Any,
        collector: Any,
        config: OptionalRootConfig = None,
    ):
        # Validate evaluator protocol
        if not hasattr(evaluator, 'evaluate_detailed'):
            raise TypeError(
                "evaluator must satisfy JourneyEvaluatorProtocol (missing evaluate_detailed)"
            )

        # Validate phase tracker protocol
        required_tracker_methods = [
            'compute_phase', 'record_metric', 'get_phase_metrics',
            'get_known_story_types', 'normalize_story_type',
            'serialize_all_metrics', 'deserialize_all_metrics',
        ]
        for method in required_tracker_methods:
            if not hasattr(phase_tracker, method):
                raise TypeError(
                    f"phase_tracker must satisfy PhaseTrackerProtocol (missing required methods)"
                )

        # Validate collector protocol
        required_collector_methods = [
            'store_story', 'get_story_batch', 'stories_to_training_examples',
        ]
        for method in required_collector_methods:
            if not hasattr(collector, method):
                raise TypeError(
                    f"collector must satisfy StoryCollectorProtocol (missing required methods)"
                )

        if config is not None and not isinstance(config, RootConfig):
            raise TypeError("config must be a RootConfig instance or None")

        self._evaluator = evaluator
        self._phase_tracker = phase_tracker
        self._collector = collector
        self._config = config if config is not None else RootConfig()

    def process_story(self, story: Any) -> ProcessingResult:
        """Process a single Story through the full pipeline."""
        # Validate story
        if not hasattr(story, 'story_id') or not hasattr(story, 'story_type') or not hasattr(story, 'steps'):
            raise StoryProcessingError(
                "Expected a valid Story instance with story_id, story_type, and steps"
            )

        # Normalize story_type
        try:
            normalized_type = self._phase_tracker.normalize_story_type(story.story_type)
        except Exception:
            normalized_type = None

        if not normalized_type:
            raise StoryProcessingError("story_type must be non-empty after normalization")

        # Validate sanitized form is usable for filesystem storage
        try:
            sanitize_story_type(normalized_type)
        except ValueError:
            raise StoryProcessingError("story_type must be non-empty after normalization")

        step_count = len(story.steps) if hasattr(story, 'steps') and story.steps else 0

        # Step 1: Persist story
        stored = False
        try:
            # Convert data_models.Story to CollectorStory for storage
            collector_story = self._to_collector_story(story, normalized_type)
            self._collector.store_story(collector_story)
            stored = True
        except (OSError, TypeError) as e:
            raise StoryProcessingError(f"Failed to store story: {e}")
        except Exception as e:
            _log("warning", f"Story storage failed: {e}")

        # Step 2: Evaluate journey (if enough steps)
        journey_score = None
        if step_count >= self._config.min_steps_for_evaluation:
            try:
                je_story = self._to_evaluator_story(story)
                journey_score = self._evaluator.evaluate_detailed(je_story)
            except Exception as e:
                _log("warning", f"Journey evaluation failed: {e}")

        # Step 3: Record phase metric
        phase_metric_recorded = False
        if (self._config.auto_record_phase_metric and
                journey_score is not None):
            try:
                success = journey_score.goal_completion >= self._config.goal_threshold
                event = MetricEvent(success=success, story_type=normalized_type)
                self._phase_tracker.record_metric(event)
                phase_metric_recorded = True
            except Exception as e:
                _log("warning", f"Phase metric recording failed: {e}")

        # Step 4: Compute current phase
        try:
            result = self._phase_tracker.compute_phase(normalized_type)
            phase = result.phase
        except Exception:
            phase = Phase.observation

        return ProcessingResult(
            story_id=story.story_id,
            story_type=normalized_type,
            stored=stored,
            journey_score=journey_score,
            phase=phase,
            phase_metric_recorded=phase_metric_recorded,
            step_count=step_count,
        )

    def get_training_batch(
        self,
        story_type: str,
        split: str,
        batch_size: OptionalBatchSize = None,
    ) -> TrainingExampleList:
        """Retrieve training examples for a story type and split."""
        try:
            normalized = sanitize_story_type(story_type)
        except ValueError:
            raise ValueError("story_type must be non-empty after sanitization")

        valid_splits = {s.value for s in SplitName}
        if split not in valid_splits:
            raise ValueError(f"split must be one of {sorted(valid_splits)}")

        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be a positive integer or None")

        stories = self._collector.get_story_batch(normalized, split)

        examples = []
        for story in stories:
            examples.extend(self._collector.stories_to_training_examples(story))

        if batch_size is not None:
            examples = examples[:batch_size]

        return examples

    def get_learning_summary(
        self,
        story_type: OptionalStoryType = None,
    ) -> LearningSummary:
        """Get learning summary for a story type or globally."""
        if story_type is not None and not isinstance(story_type, str):
            raise TypeError("story_type must be a str or None")

        normalized = normalize_story_type(story_type)

        try:
            result = self._phase_tracker.compute_phase(normalized)
            phase = result.phase
        except Exception:
            phase = Phase.observation

        metrics = self._phase_tracker.get_phase_metrics(normalized)

        known_types = []
        if normalized is None:
            known_types = self._phase_tracker.get_known_story_types()

        return LearningSummary(
            story_type=normalized,
            phase=phase,
            metrics=metrics,
            known_story_types=known_types,
        )

    def export_state(self) -> ExportedState:
        """Serialize learning state for persistence."""
        try:
            phase_metrics = self._phase_tracker.serialize_all_metrics()
        except Exception as e:
            raise LearningStateError(f"Failed to serialize learning state: {e}")

        return ExportedState(version=1, phase_metrics=phase_metrics)

    def import_state(self, data: Any) -> None:
        """Restore learning state from exported data."""
        if isinstance(data, ExportedState):
            version = data.version
            phase_metrics = data.phase_metrics
        elif isinstance(data, dict):
            version = data.get('version')
            phase_metrics = data.get('phase_metrics')
        else:
            raise LearningStateError(
                "Invalid state data: expected ExportedState or compatible dict"
            )

        if version is None or version < 1:
            raise LearningStateError(f"Unsupported state version: {version}")

        if not phase_metrics or 'global' not in phase_metrics:
            raise LearningStateError(
                "Serialized state must contain phase_metrics with 'global' key"
            )

        try:
            self._phase_tracker.deserialize_all_metrics(phase_metrics)
        except Exception as e:
            raise LearningStateError(
                f"Invalid PhaseMetrics data in imported state: {e}"
            )

    # ---- Internal conversion helpers ----

    def _to_collector_story(self, story: Any, normalized_type: str) -> CollectorStory:
        """Convert a data_models.Story to a CollectorStory."""
        steps = []
        for i, step in enumerate(story.steps):
            input_data = step.input_data if isinstance(step.input_data, dict) else {"value": step.input_data}
            output_data = step.output_data if isinstance(step.output_data, dict) else {"value": step.output_data}
            metadata = step.metadata if hasattr(step, 'metadata') and isinstance(step.metadata, dict) else {}
            steps.append(CollectorStep(
                step_index=i,
                input_data=input_data,
                output_data=output_data,
                metadata=metadata,
            ))

        story_metadata = {}
        if hasattr(story, 'context') and isinstance(story.context, dict):
            story_metadata = story.context
        if hasattr(story, 'created_at'):
            story_metadata['created_at'] = str(story.created_at)
        if hasattr(story, 'closed_at') and story.closed_at is not None:
            story_metadata['closed_at'] = str(story.closed_at)

        return CollectorStory(
            story_id=story.story_id,
            story_type=normalized_type,
            steps=steps,
            metadata=story_metadata,
        )

    def _to_evaluator_story(self, story: Any) -> JEStory:
        """Convert a data_models.Story to a journey_evaluator.Story."""
        je_steps = []
        for step in story.steps:
            confidence = 0.0
            if hasattr(step, 'confidence_at_step') and step.confidence_at_step is not None:
                confidence = step.confidence_at_step

            context = {}
            if hasattr(step, 'output_data') and isinstance(step.output_data, dict):
                context = step.output_data
            elif hasattr(step, 'metadata') and isinstance(step.metadata, dict):
                context = step.metadata

            je_steps.append(JEStep(
                task_type=step.task_type,
                confidence_at_step=confidence,
                context=context,
            ))

        return JEStory(steps=je_steps)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'RootConfig',
    'Phase',
    'SplitName',
    'JourneyScore',
    'OptionalJourneyScore',
    'ProcessingResult',
    'PhaseMetrics',
    'LearningSummary',
    'StoryTrainingExample',
    'TrainingExampleList',
    'ExportedState',
    'OptionalRootConfig',
    'OptionalStoryType',
    'OptionalBatchSize',
    'StoryProcessingError',
    'LearningStateError',
    'ApprenticeStoryLearner',
]
