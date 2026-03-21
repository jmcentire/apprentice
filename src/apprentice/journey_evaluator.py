"""Journey Evaluator (journey_evaluator) v1

Evaluates a complete Story (sequence of Steps) and returns a composite score
in [0.0, 1.0] by computing four independent sub-scores: goal_completion,
step_efficiency, backtracking, and consistency.

Implements EvaluatorProtocol structurally (composition, no inheritance).
All scoring methods are pure functions with no side effects.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

_PACT_KEY = "PACT:journey_evaluator"
logger = logging.getLogger(__name__)


def _log(level: str, msg: str, **kwargs) -> None:
    getattr(logger, level)(f"[{_PACT_KEY}] {msg}", **kwargs)


# ============================================================================
# Types
# ============================================================================

UnitScore = float  # clamped to [0.0, 1.0]
PositiveFloat = float  # strictly positive


@dataclass(frozen=True)
class Weights:
    """4-tuple of positive floats for (goal_completion, step_efficiency,
    backtracking, consistency). Normalized at construction so they sum to 1.0."""
    goal_completion: float
    step_efficiency: float
    backtracking: float
    consistency: float


@dataclass(frozen=True)
class EvaluatorConfig:
    """Frozen config for JourneyEvaluator. Weights are validated and
    normalized at construction."""
    goal_threshold: float = 0.8
    terminal_task_types: frozenset = field(default_factory=frozenset)
    weights: Weights = field(default=None)
    exclude_consistency_keys: frozenset = field(default_factory=frozenset)

    def __post_init__(self):
        # Handle default weights
        if self.weights is None:
            w = Weights(0.25, 0.25, 0.25, 0.25)
            object.__setattr__(self, 'weights', w)

        # If weights is a tuple, convert to Weights
        if isinstance(self.weights, tuple):
            if len(self.weights) != 4:
                raise ValueError("weights tuple must have exactly 4 elements")
            w = Weights(*self.weights)
            object.__setattr__(self, 'weights', w)

        # Validate all weights are positive
        wt = self.weights
        for name, val in [('goal_completion', wt.goal_completion),
                          ('step_efficiency', wt.step_efficiency),
                          ('backtracking', wt.backtracking),
                          ('consistency', wt.consistency)]:
            if val <= 0.0:
                raise ValueError(f"All weight components must be strictly positive. {name}={val}")

        # Normalize weights to sum to 1.0
        total = wt.goal_completion + wt.step_efficiency + wt.backtracking + wt.consistency
        if total == 0:
            raise ValueError("All weight components must be strictly positive.")
        normalized = Weights(
            goal_completion=wt.goal_completion / total,
            step_efficiency=wt.step_efficiency / total,
            backtracking=wt.backtracking / total,
            consistency=wt.consistency / total,
        )
        object.__setattr__(self, 'weights', normalized)


class Step:
    """A single step within a Story, consumed read-only by JourneyEvaluator."""

    def __init__(self, task_type: str, confidence_at_step: float, context: dict = None):
        self.task_type = task_type
        self.confidence_at_step = confidence_at_step
        self.context = context if context is not None else {}


StepList = list[Step]


class Story:
    """A complete journey -- an ordered sequence of Steps."""

    def __init__(self, steps: list[Step]):
        self.steps = steps


@dataclass(frozen=True)
class JourneyScore:
    """Frozen dataclass holding the four sub-scores and their weighted composite."""
    goal_completion: float
    step_efficiency: float
    backtracking: float
    consistency: float
    composite: float


OptionalEvaluatorConfig = EvaluatorConfig | None


# ============================================================================
# JourneyEvaluator
# ============================================================================

class JourneyEvaluator:
    """Evaluates a complete Story and returns sub-scores and a composite score.
    Structurally satisfies EvaluatorProtocol without inheriting from it."""

    def __init__(self, config: OptionalEvaluatorConfig = None):
        if config is None:
            self.config = EvaluatorConfig()
        else:
            self.config = config

    def evaluate(self, story: Story) -> UnitScore:
        """EvaluatorProtocol-conforming method. Returns composite score in [0.0, 1.0]."""
        return self.evaluate_detailed(story).composite

    def evaluate_detailed(self, story: Story) -> JourneyScore:
        """Returns JourneyScore with all four sub-scores and composite."""
        if not story.steps:
            return JourneyScore(0.0, 0.0, 0.0, 0.0, 0.0)

        gc = self.score_goal_completion(story)
        se = self.score_step_efficiency(story)
        bt = self.score_backtracking(story)
        co = self.score_consistency(story)

        w = self.config.weights
        composite = (
            w.goal_completion * gc +
            w.step_efficiency * se +
            w.backtracking * bt +
            w.consistency * co
        )
        composite = max(0.0, min(1.0, composite))

        return JourneyScore(
            goal_completion=gc,
            step_efficiency=se,
            backtracking=bt,
            consistency=co,
            composite=composite,
        )

    def score_goal_completion(self, story: Story) -> UnitScore:
        """1.0 if last step exceeds goal_threshold or is a terminal task_type.
        Otherwise proportional to how close confidence is to the threshold."""
        if not story.steps:
            return 0.0

        last_step = story.steps[-1]

        # Check terminal task type
        if last_step.task_type in self.config.terminal_task_types:
            return 1.0

        # Check confidence threshold
        confidence = last_step.confidence_at_step
        if confidence is None:
            confidence = 0.0

        if confidence >= self.config.goal_threshold:
            return 1.0

        # Proportional score
        if self.config.goal_threshold > 0:
            return max(0.0, min(1.0, confidence / self.config.goal_threshold))
        return 0.0

    def score_step_efficiency(self, story: Story) -> UnitScore:
        """Ratio of unique task_types to total steps. Empty=0.0, single=1.0."""
        if not story.steps:
            return 0.0

        unique_types = set(s.task_type for s in story.steps)
        return len(unique_types) / len(story.steps)

    def score_backtracking(self, story: Story) -> UnitScore:
        """Penalizes repeated task_types after a different task_type intervenes.
        Consecutive identical types are not backtracking."""
        if not story.steps:
            return 0.0
        if len(story.steps) == 1:
            return 1.0

        backtracks = 0
        last_seen = {}  # task_type -> last index seen

        for i, step in enumerate(story.steps):
            tt = step.task_type
            if tt in last_seen:
                # Backtrack if something different appeared since last occurrence
                last_idx = last_seen[tt]
                # Check if any different task_type appeared between last_idx and i
                intervening = story.steps[last_idx + 1:i]
                if any(s.task_type != tt for s in intervening):
                    backtracks += 1
            last_seen[tt] = i

        denominator = len(story.steps) - 1
        if denominator <= 0:
            return 1.0

        return max(0.0, min(1.0, 1.0 - (backtracks / denominator)))

    def score_consistency(self, story: Story) -> UnitScore:
        """Checks shared context keys across steps for contradictions.
        1.0 = fully consistent. Empty stories = 0.0. Single step = 1.0."""
        if not story.steps:
            return 0.0
        if len(story.steps) == 1:
            return 1.0

        exclude = self.config.exclude_consistency_keys

        # Collect all context key-value pairs across steps
        key_values: dict[str, list] = {}
        for step in story.steps:
            for k, v in step.context.items():
                if k in exclude:
                    continue
                if k not in key_values:
                    key_values[k] = []
                key_values[k].append(v)

        # Find shared keys (appear in more than one step)
        shared_keys = {k: vs for k, vs in key_values.items() if len(vs) > 1}

        if not shared_keys:
            return 1.0

        consistent_count = 0
        total_shared = len(shared_keys)

        for k, values in shared_keys.items():
            # Consistent if all values are equal
            if all(v == values[0] for v in values):
                consistent_count += 1

        return consistent_count / total_shared


# ============================================================================
# Module-level convenience functions
# ============================================================================

_default_evaluator: JourneyEvaluator | None = None


def _get_evaluator() -> JourneyEvaluator:
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = JourneyEvaluator()
    return _default_evaluator


def evaluate(story: Story) -> UnitScore:
    return _get_evaluator().evaluate(story)


def evaluate_detailed(story: Story) -> JourneyScore:
    return _get_evaluator().evaluate_detailed(story)


def score_goal_completion(story: Story) -> UnitScore:
    return _get_evaluator().score_goal_completion(story)


def score_step_efficiency(story: Story) -> UnitScore:
    return _get_evaluator().score_step_efficiency(story)


def score_backtracking(story: Story) -> UnitScore:
    return _get_evaluator().score_backtracking(story)


def score_consistency(story: Story) -> UnitScore:
    return _get_evaluator().score_consistency(story)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'Weights',
    'EvaluatorConfig',
    'Step',
    'StepList',
    'Story',
    'JourneyScore',
    'OptionalEvaluatorConfig',
    'JourneyEvaluator',
    'evaluate',
    'evaluate_detailed',
    'score_goal_completion',
    'score_step_efficiency',
    'score_backtracking',
    'score_consistency',
]
