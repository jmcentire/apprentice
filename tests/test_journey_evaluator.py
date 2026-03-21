"""Tests for JourneyEvaluator."""

import pytest

from apprentice.journey_evaluator import (
    JourneyEvaluator,
    EvaluatorConfig,
    Weights,
    Step,
    Story,
    JourneyScore,
)


class TestEvaluatorConfig:
    def test_default_weights(self):
        cfg = EvaluatorConfig()
        w = cfg.weights
        assert abs(w.goal_completion - 0.25) < 1e-9
        assert abs(w.step_efficiency - 0.25) < 1e-9
        assert abs(w.backtracking - 0.25) < 1e-9
        assert abs(w.consistency - 0.25) < 1e-9

    def test_custom_weights_normalized(self):
        cfg = EvaluatorConfig(weights=(1.0, 1.0, 1.0, 1.0))
        assert abs(cfg.weights.goal_completion - 0.25) < 1e-9

    def test_unequal_weights_normalized(self):
        cfg = EvaluatorConfig(weights=(2.0, 1.0, 1.0, 1.0))
        assert abs(cfg.weights.goal_completion - 0.4) < 1e-9
        assert abs(cfg.weights.step_efficiency - 0.2) < 1e-9

    def test_zero_weight_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            EvaluatorConfig(weights=(0.0, 1.0, 1.0, 1.0))

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            EvaluatorConfig(weights=(-1.0, 1.0, 1.0, 1.0))

    def test_custom_goal_threshold(self):
        cfg = EvaluatorConfig(goal_threshold=0.9)
        assert cfg.goal_threshold == 0.9

    def test_terminal_task_types(self):
        cfg = EvaluatorConfig(terminal_task_types=frozenset({"submit", "deploy"}))
        assert "submit" in cfg.terminal_task_types


class TestJourneyEvaluator:
    def test_default_config(self):
        evaluator = JourneyEvaluator()
        assert evaluator.config.goal_threshold == 0.8

    def test_custom_config(self):
        cfg = EvaluatorConfig(goal_threshold=0.5)
        evaluator = JourneyEvaluator(config=cfg)
        assert evaluator.config.goal_threshold == 0.5


class TestEmptyStory:
    def test_evaluate_empty(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[])
        assert evaluator.evaluate(story) == 0.0

    def test_evaluate_detailed_empty(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[])
        score = evaluator.evaluate_detailed(story)
        assert score.goal_completion == 0.0
        assert score.step_efficiency == 0.0
        assert score.backtracking == 0.0
        assert score.consistency == 0.0
        assert score.composite == 0.0


class TestGoalCompletion:
    def test_high_confidence_completes(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[Step("task_a", 0.9)])
        assert evaluator.score_goal_completion(story) == 1.0

    def test_low_confidence_proportional(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[Step("task_a", 0.4)])
        score = evaluator.score_goal_completion(story)
        assert 0.0 < score < 1.0
        assert abs(score - 0.5) < 1e-9  # 0.4 / 0.8 = 0.5

    def test_terminal_task_type(self):
        cfg = EvaluatorConfig(terminal_task_types=frozenset({"submit"}))
        evaluator = JourneyEvaluator(config=cfg)
        story = Story(steps=[Step("submit", 0.1)])
        assert evaluator.score_goal_completion(story) == 1.0

    def test_exactly_at_threshold(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[Step("task_a", 0.8)])
        assert evaluator.score_goal_completion(story) == 1.0

    def test_zero_confidence(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[Step("task_a", 0.0)])
        assert evaluator.score_goal_completion(story) == 0.0


class TestStepEfficiency:
    def test_single_step(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[Step("a", 0.5)])
        assert evaluator.score_step_efficiency(story) == 1.0

    def test_all_unique(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[Step("a", 0.5), Step("b", 0.5), Step("c", 0.5)])
        assert evaluator.score_step_efficiency(story) == 1.0

    def test_all_repeated(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[Step("a", 0.5), Step("a", 0.5), Step("a", 0.5)])
        assert abs(evaluator.score_step_efficiency(story) - 1/3) < 1e-9

    def test_mixed(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[
            Step("a", 0.5), Step("b", 0.5),
            Step("a", 0.5), Step("c", 0.5),
        ])
        assert abs(evaluator.score_step_efficiency(story) - 3/4) < 1e-9


class TestBacktracking:
    def test_single_step(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[Step("a", 0.5)])
        assert evaluator.score_backtracking(story) == 1.0

    def test_no_backtracking(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[Step("a", 0.5), Step("b", 0.5), Step("c", 0.5)])
        assert evaluator.score_backtracking(story) == 1.0

    def test_consecutive_same_not_backtracking(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[Step("a", 0.5), Step("a", 0.5)])
        assert evaluator.score_backtracking(story) == 1.0

    def test_backtracking_detected(self):
        evaluator = JourneyEvaluator()
        # a -> b -> a is one backtrack
        story = Story(steps=[Step("a", 0.5), Step("b", 0.5), Step("a", 0.5)])
        score = evaluator.score_backtracking(story)
        assert score < 1.0
        # 1 backtrack / 2 transitions = 0.5
        assert abs(score - 0.5) < 1e-9

    def test_multiple_backtracks(self):
        evaluator = JourneyEvaluator()
        # a -> b -> a -> b: 2 backtracks
        story = Story(steps=[
            Step("a", 0.5), Step("b", 0.5),
            Step("a", 0.5), Step("b", 0.5),
        ])
        score = evaluator.score_backtracking(story)
        # 2 backtracks / 3 transitions
        assert abs(score - (1.0 - 2/3)) < 1e-9


class TestConsistency:
    def test_single_step(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[Step("a", 0.5, {"k": "v"})])
        assert evaluator.score_consistency(story) == 1.0

    def test_consistent_shared_keys(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[
            Step("a", 0.5, {"user": "alice"}),
            Step("b", 0.5, {"user": "alice"}),
        ])
        assert evaluator.score_consistency(story) == 1.0

    def test_inconsistent_shared_keys(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[
            Step("a", 0.5, {"user": "alice"}),
            Step("b", 0.5, {"user": "bob"}),
        ])
        assert evaluator.score_consistency(story) == 0.0

    def test_mixed_consistency(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[
            Step("a", 0.5, {"user": "alice", "project": "x"}),
            Step("b", 0.5, {"user": "alice", "project": "y"}),
        ])
        # user is consistent, project is not: 1/2 = 0.5
        assert abs(evaluator.score_consistency(story) - 0.5) < 1e-9

    def test_no_shared_keys(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[
            Step("a", 0.5, {"k1": "v1"}),
            Step("b", 0.5, {"k2": "v2"}),
        ])
        assert evaluator.score_consistency(story) == 1.0

    def test_excluded_keys(self):
        cfg = EvaluatorConfig(exclude_consistency_keys=frozenset({"timestamp"}))
        evaluator = JourneyEvaluator(config=cfg)
        story = Story(steps=[
            Step("a", 0.5, {"timestamp": "t1", "user": "alice"}),
            Step("b", 0.5, {"timestamp": "t2", "user": "alice"}),
        ])
        # timestamp excluded, user is consistent
        assert evaluator.score_consistency(story) == 1.0

    def test_empty_context(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[
            Step("a", 0.5, {}),
            Step("b", 0.5, {}),
        ])
        assert evaluator.score_consistency(story) == 1.0


class TestCompositeScore:
    def test_evaluate_equals_detailed_composite(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[
            Step("a", 0.9, {"user": "alice"}),
            Step("b", 0.8, {"user": "alice"}),
        ])
        assert evaluator.evaluate(story) == evaluator.evaluate_detailed(story).composite

    def test_perfect_story(self):
        cfg = EvaluatorConfig(terminal_task_types=frozenset({"done"}))
        evaluator = JourneyEvaluator(config=cfg)
        story = Story(steps=[
            Step("a", 0.9, {"user": "alice"}),
            Step("b", 0.9, {"user": "alice"}),
            Step("done", 0.9, {"user": "alice"}),
        ])
        score = evaluator.evaluate_detailed(story)
        assert score.goal_completion == 1.0
        assert score.consistency == 1.0
        assert score.composite > 0.8

    def test_composite_in_range(self):
        evaluator = JourneyEvaluator()
        story = Story(steps=[
            Step("a", 0.3, {"x": 1}),
            Step("b", 0.5, {"x": 2}),
            Step("a", 0.6, {"x": 1}),
        ])
        score = evaluator.evaluate(story)
        assert 0.0 <= score <= 1.0

    def test_weighted_composite(self):
        cfg = EvaluatorConfig(weights=(1.0, 0.0001, 0.0001, 0.0001))
        evaluator = JourneyEvaluator(config=cfg)
        story = Story(steps=[Step("a", 0.9)])
        score = evaluator.evaluate_detailed(story)
        # goal_completion dominates
        assert score.composite > 0.9
