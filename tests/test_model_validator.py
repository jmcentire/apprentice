"""
Contract tests for ModelValidator component.

Tests are organized into sections matching the contract functions:
- validate_candidate
- promote_candidate
- validate_and_promote
- get_validation_history
- invariants and property-based tests

All dependencies (TrainingDataStore, ModelServer, Evaluator, AuditLogger)
are mocked via dependency injection.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call, PropertyMock
from datetime import datetime, timezone
from typing import List, Dict, Optional
import time

# Component under test
from src.model_validator import (
    ModelValidator,
    ValidationConfig,
    ValidationResult,
    PromotionRecord,
    PromotionDecision,
    ThresholdComparison,
    PerMetricThreshold,
    TestSetReference,
    TestSample,
    TestSet,
    EvaluationScore,
    ModelServerStatus,
    SwapResult,
)


# ============================================================================
# FIXTURES (conftest-style, all in one file for contract test portability)
# ============================================================================


@pytest.fixture
def mock_training_data_store():
    """Mock TrainingDataStore dependency."""
    store = MagicMock()
    return store


@pytest.fixture
def mock_model_server():
    """Mock ModelServer dependency."""
    server = MagicMock()
    server.get_status.return_value = ModelServerStatus(
        is_healthy=True,
        current_model_id="model-v1",
        ready_to_swap=True,
    )
    server.swap_model.return_value = SwapResult(
        success=True,
        previous_model_id="model-v1",
        new_model_id="model-v2",
        error_message="",
    )
    server.infer.return_value = "generated output text"
    return server


@pytest.fixture
def mock_audit_logger():
    """Mock AuditLogger dependency."""
    logger = MagicMock()
    logger.log_validation_result.return_value = None
    logger.log_promotion_record.return_value = None
    return logger


@pytest.fixture
def make_evaluator():
    """Factory to create mock evaluators with configurable scores."""
    def _make(metric_name: str, score: float, should_fail: bool = False):
        evaluator = MagicMock()
        evaluator.metric_name = metric_name
        if should_fail:
            evaluator.evaluate.side_effect = Exception(f"Evaluator {metric_name} failed")
        else:
            evaluator.evaluate.return_value = EvaluationScore(
                metric_name=metric_name,
                score=score,
            )
        return evaluator
    return _make


@pytest.fixture
def make_test_set():
    """Factory to create TestSet instances."""
    def _make(test_set_id: str = "test-set-001", num_samples: int = 10):
        samples = [
            TestSample(
                sample_id=f"sample-{i}",
                input_prompt=f"prompt {i}",
                reference_output=f"reference {i}",
                metadata={"index": i},
            )
            for i in range(num_samples)
        ]
        reference = TestSetReference(
            test_set_id=test_set_id,
            content_hash=f"hash-{test_set_id}-{num_samples}",
            sample_count=num_samples,
        )
        return TestSet(reference=reference, samples=samples)
    return _make


@pytest.fixture
def default_config():
    """Default ValidationConfig for tests."""
    return ValidationConfig(
        promotion_threshold=0.7,
        min_test_samples=5,
        threshold_comparison=ThresholdComparison.gte,
        per_metric_thresholds=[],
        swap_timeout_seconds=30.0,
    )


@pytest.fixture
def make_validation_result():
    """Factory to create ValidationResult instances in various states."""
    def _make(
        decision: PromotionDecision = PromotionDecision.PROMOTED,
        candidate_model_id: str = "model-v2",
        aggregate_score: float = 0.85,
        per_metric_scores: Optional[Dict[str, float]] = None,
        threshold_used: float = 0.7,
        threshold_comparison_used: ThresholdComparison = ThresholdComparison.gte,
        test_set_id: str = "test-set-001",
        failed_per_metric_thresholds: Optional[List[str]] = None,
        evaluation_error_detail: str = "",
        samples_evaluated: int = 10,
    ):
        if per_metric_scores is None:
            per_metric_scores = {"accuracy": 0.85, "fluency": 0.85}
        if failed_per_metric_thresholds is None:
            failed_per_metric_thresholds = []
        return ValidationResult(
            candidate_model_id=candidate_model_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            per_metric_scores=per_metric_scores,
            aggregate_score=aggregate_score,
            threshold_used=threshold_used,
            threshold_comparison_used=threshold_comparison_used,
            test_set_reference=TestSetReference(
                test_set_id=test_set_id,
                content_hash="hash-test-set-001",
                sample_count=samples_evaluated,
            ),
            decision=decision,
            failed_per_metric_thresholds=failed_per_metric_thresholds,
            evaluation_error_detail=evaluation_error_detail,
            samples_evaluated=samples_evaluated,
        )
    return _make


@pytest.fixture
def make_validator(
    mock_training_data_store,
    mock_model_server,
    mock_audit_logger,
    default_config,
):
    """Factory to create ModelValidator instances with mock dependencies."""
    def _make(
        config=None,
        evaluators=None,
        training_data_store=None,
        model_server=None,
        audit_logger=None,
    ):
        return ModelValidator(
            config=config or default_config,
            evaluators=evaluators or [],
            training_data_store=training_data_store or mock_training_data_store,
            model_server=model_server or mock_model_server,
            audit_logger=audit_logger or mock_audit_logger,
        )
    return _make


# ============================================================================
# TEST: validate_candidate — Happy Paths
# ============================================================================


class TestValidateCandidateHappyPath:
    """Tests for validate_candidate happy path scenarios."""

    def test_vc_happy_path_gte_promoted(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, default_config,
    ):
        """validate_candidate returns PROMOTED when aggregate score meets threshold with gte comparison."""
        evaluators = [
            make_evaluator("accuracy", 0.85),
            make_evaluator("fluency", 0.90),
        ]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert result.decision == PromotionDecision.PROMOTED, \
            f"Expected PROMOTED but got {result.decision}"
        assert result.aggregate_score >= default_config.promotion_threshold, \
            f"Score {result.aggregate_score} should be >= threshold {default_config.promotion_threshold}"
        assert len(result.per_metric_scores) == 2, \
            f"Expected 2 metrics but got {len(result.per_metric_scores)}"
        assert result.candidate_model_id == "model-v2"

    def test_vc_happy_path_gt_promoted(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """validate_candidate returns PROMOTED when aggregate score strictly exceeds threshold with gt comparison."""
        config = ValidationConfig(
            promotion_threshold=0.7,
            min_test_samples=5,
            threshold_comparison=ThresholdComparison.gt,
            per_metric_thresholds=[],
            swap_timeout_seconds=30.0,
        )
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(config=config, evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert result.decision == PromotionDecision.PROMOTED, \
            f"Expected PROMOTED but got {result.decision}"
        assert result.aggregate_score > config.promotion_threshold, \
            f"Score {result.aggregate_score} should be > threshold {config.promotion_threshold}"

    def test_vc_happy_path_multiple_evaluators(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """validate_candidate aggregates scores from multiple evaluators correctly."""
        evaluators = [
            make_evaluator("accuracy", 0.80),
            make_evaluator("fluency", 0.90),
            make_evaluator("relevance", 0.70),
        ]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        expected_mean = sum(result.per_metric_scores.values()) / len(result.per_metric_scores)
        assert result.aggregate_score == pytest.approx(expected_mean), \
            f"Aggregate {result.aggregate_score} should equal mean {expected_mean}"
        assert len(result.per_metric_scores) == 3, \
            f"Expected 3 metrics, got {len(result.per_metric_scores)}"

    def test_vc_rejected_below_threshold_gte(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, default_config,
    ):
        """validate_candidate returns REJECTED_BELOW_THRESHOLD when score is below threshold."""
        evaluators = [make_evaluator("accuracy", 0.50)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert result.decision == PromotionDecision.REJECTED_BELOW_THRESHOLD, \
            f"Expected REJECTED_BELOW_THRESHOLD but got {result.decision}"
        assert result.aggregate_score < default_config.promotion_threshold, \
            f"Score {result.aggregate_score} should be < threshold {default_config.promotion_threshold}"

    def test_vc_skipped_no_test_data(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """validate_candidate returns SKIPPED_NO_TEST_DATA when test set is empty."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-empty", 0)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-empty")

        assert result.decision == PromotionDecision.SKIPPED_NO_TEST_DATA, \
            f"Expected SKIPPED_NO_TEST_DATA but got {result.decision}"


# ============================================================================
# TEST: validate_candidate — Edge Cases
# ============================================================================


class TestValidateCandidateEdgeCases:
    """Tests for validate_candidate edge/boundary scenarios."""

    def test_vc_edge_score_exactly_at_threshold_gte(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """Score exactly at threshold with gte comparison should be PROMOTED."""
        config = ValidationConfig(
            promotion_threshold=0.80,
            min_test_samples=5,
            threshold_comparison=ThresholdComparison.gte,
            per_metric_thresholds=[],
            swap_timeout_seconds=30.0,
        )
        evaluators = [make_evaluator("accuracy", 0.80)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(config=config, evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert result.decision == PromotionDecision.PROMOTED, \
            f"Score exactly at threshold with gte should be PROMOTED, got {result.decision}"
        assert result.aggregate_score == pytest.approx(config.promotion_threshold), \
            f"Aggregate score {result.aggregate_score} should equal threshold {config.promotion_threshold}"

    def test_vc_edge_score_exactly_at_threshold_gt(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """Score exactly at threshold with gt comparison should be REJECTED."""
        config = ValidationConfig(
            promotion_threshold=0.80,
            min_test_samples=5,
            threshold_comparison=ThresholdComparison.gt,
            per_metric_thresholds=[],
            swap_timeout_seconds=30.0,
        )
        evaluators = [make_evaluator("accuracy", 0.80)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(config=config, evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert result.decision == PromotionDecision.REJECTED_BELOW_THRESHOLD, \
            f"Score exactly at threshold with gt should be REJECTED, got {result.decision}"

    def test_vc_edge_per_metric_threshold_fails(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """Candidate rejected when aggregate passes but per-metric threshold fails."""
        config = ValidationConfig(
            promotion_threshold=0.7,
            min_test_samples=5,
            threshold_comparison=ThresholdComparison.gte,
            per_metric_thresholds=[
                PerMetricThreshold(metric_name="accuracy", min_score=0.90, comparison=ThresholdComparison.gte),
            ],
            swap_timeout_seconds=30.0,
        )
        # accuracy=0.75 fails per-metric (needs >=0.90), fluency=0.95 passes
        # aggregate=(0.75+0.95)/2=0.85 passes aggregate threshold of 0.7
        evaluators = [
            make_evaluator("accuracy", 0.75),
            make_evaluator("fluency", 0.95),
        ]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(config=config, evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert result.decision == PromotionDecision.REJECTED_BELOW_THRESHOLD, \
            f"Expected REJECTED due to per-metric failure, got {result.decision}"
        assert len(result.failed_per_metric_thresholds) > 0, \
            "failed_per_metric_thresholds should list failing metrics"
        assert "accuracy" in result.failed_per_metric_thresholds, \
            "accuracy should be in failed metrics"

    def test_vc_edge_one_evaluator_fails_other_succeeds(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """When one evaluator fails, score is computed from remaining evaluators."""
        evaluators = [
            make_evaluator("accuracy", 0.85),
            make_evaluator("broken_metric", 0.0, should_fail=True),
        ]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert len(result.per_metric_scores) == 1, \
            f"Expected 1 successful metric, got {len(result.per_metric_scores)}"
        assert result.aggregate_score == pytest.approx(
            list(result.per_metric_scores.values())[0]
        ), "Aggregate should be the single successful metric score"

    def test_vc_edge_minimum_sample_count_exact(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, default_config,
    ):
        """Validation proceeds when sample count exactly equals min_test_samples."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        # min_test_samples is 5 in default config
        test_set = make_test_set("test-set-min", default_config.min_test_samples)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-min")

        assert result.decision in (
            PromotionDecision.PROMOTED,
            PromotionDecision.REJECTED_BELOW_THRESHOLD,
        ), f"Should proceed with validation, got {result.decision}"
        assert result.samples_evaluated == default_config.min_test_samples, \
            f"Expected {default_config.min_test_samples} samples evaluated"

    def test_vc_edge_aggregate_zero_when_no_metrics(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """aggregate_score is 0.0 when no metrics completed successfully."""
        evaluators = [
            make_evaluator("metric_a", 0.0, should_fail=True),
            make_evaluator("metric_b", 0.0, should_fail=True),
        ]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert result.aggregate_score == 0.0, \
            f"Aggregate should be 0.0 when no metrics, got {result.aggregate_score}"
        assert result.decision == PromotionDecision.REJECTED_EVALUATION_ERROR, \
            f"Expected REJECTED_EVALUATION_ERROR, got {result.decision}"

    def test_vc_edge_timestamp_is_utc_iso8601(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """timestamp_utc is a valid ISO 8601 UTC string."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        parsed = datetime.fromisoformat(result.timestamp_utc)
        assert parsed is not None, "timestamp_utc should be valid ISO 8601"

    def test_vc_edge_test_set_reference_populated(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """test_set_reference accurately reflects the test set that was used."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert result.test_set_reference.test_set_id == "test-set-001", \
            f"test_set_id should be 'test-set-001', got '{result.test_set_reference.test_set_id}'"
        assert result.test_set_reference.sample_count > 0, \
            "sample_count should be > 0"
        assert result.test_set_reference.content_hash != "", \
            "content_hash should not be empty"


# ============================================================================
# TEST: validate_candidate — Error Cases
# ============================================================================


class TestValidateCandidateErrors:
    """Tests for validate_candidate error conditions."""

    def test_vc_error_test_set_not_found(
        self, make_validator, make_evaluator, mock_training_data_store,
    ):
        """Handles missing test set without raising exception."""
        mock_training_data_store.get_test_set.side_effect = Exception("Test set not found")
        evaluators = [make_evaluator("accuracy", 0.85)]

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "nonexistent-set")

        assert result.decision in (
            PromotionDecision.SKIPPED_NO_TEST_DATA,
            PromotionDecision.REJECTED_EVALUATION_ERROR,
        ), f"Expected graceful error decision, got {result.decision}"

    def test_vc_error_test_set_below_minimum(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, default_config,
    ):
        """Rejects when test set has fewer samples than min_test_samples."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        # min_test_samples is 5, give only 2
        test_set = make_test_set("test-set-small", 2)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-small")

        assert result.decision in (
            PromotionDecision.SKIPPED_NO_TEST_DATA,
            PromotionDecision.REJECTED_EVALUATION_ERROR,
        ), f"Expected rejection for too few samples, got {result.decision}"

    def test_vc_error_all_evaluators_failed(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """Returns REJECTED_EVALUATION_ERROR when all evaluators fail."""
        evaluators = [
            make_evaluator("metric_a", 0.0, should_fail=True),
            make_evaluator("metric_b", 0.0, should_fail=True),
        ]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert result.decision == PromotionDecision.REJECTED_EVALUATION_ERROR, \
            f"Expected REJECTED_EVALUATION_ERROR, got {result.decision}"
        assert result.evaluation_error_detail != "", \
            "evaluation_error_detail should describe the failure"

    def test_vc_error_candidate_inference_error(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, mock_model_server,
    ):
        """Returns REJECTED_EVALUATION_ERROR when candidate model cannot produce outputs."""
        mock_model_server.infer.side_effect = Exception("Model load failed")
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("broken-model", "test-set-001")

        assert result.decision == PromotionDecision.REJECTED_EVALUATION_ERROR, \
            f"Expected REJECTED_EVALUATION_ERROR, got {result.decision}"
        assert result.evaluation_error_detail != "", \
            "Should have error detail for inference failure"

    def test_vc_error_training_data_store_unavailable(
        self, make_validator, make_evaluator, mock_training_data_store,
    ):
        """Returns error decision when training data store is unreachable."""
        mock_training_data_store.get_test_set.side_effect = ConnectionError(
            "Training data store unreachable"
        )
        evaluators = [make_evaluator("accuracy", 0.85)]

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert result.decision == PromotionDecision.REJECTED_EVALUATION_ERROR, \
            f"Expected REJECTED_EVALUATION_ERROR, got {result.decision}"


# ============================================================================
# TEST: validate_candidate — Invariants
# ============================================================================


class TestValidateCandidateInvariants:
    """Invariant tests for validate_candidate."""

    def test_vc_invariant_never_swap_failed(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """validate_candidate never returns SWAP_FAILED decision."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert result.decision != PromotionDecision.SWAP_FAILED, \
            "validate_candidate must never return SWAP_FAILED"

    def test_vc_invariant_never_swap_failed_on_error(
        self, make_validator, make_evaluator, mock_training_data_store,
    ):
        """validate_candidate never returns SWAP_FAILED even on errors."""
        mock_training_data_store.get_test_set.side_effect = Exception("store error")
        evaluators = [make_evaluator("accuracy", 0.85)]

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert result.decision != PromotionDecision.SWAP_FAILED, \
            "validate_candidate must never return SWAP_FAILED, even on errors"

    def test_vc_invariant_no_serving_model_mutation(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, mock_model_server,
    ):
        """validate_candidate does not mutate the currently serving model."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        validator.validate_candidate("model-v2", "test-set-001")

        assert mock_model_server.swap_model.call_count == 0, \
            "validate_candidate must not call swap_model"

    def test_vc_invariant_result_logged(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, mock_audit_logger,
    ):
        """Validation result is logged to audit logger before returning."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        validator.validate_candidate("model-v2", "test-set-001")

        assert mock_audit_logger.log_validation_result.call_count >= 1, \
            "Validation result must be logged to audit logger"

    def test_vc_invariant_per_metric_keys_match_evaluators(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """per_metric_scores keys correspond to evaluator metric names."""
        evaluators = [
            make_evaluator("accuracy", 0.85),
            make_evaluator("fluency", 0.90),
        ]
        expected_metric_names = {"accuracy", "fluency"}
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert set(result.per_metric_scores.keys()).issubset(expected_metric_names), \
            f"Metric keys {result.per_metric_scores.keys()} should be subset of {expected_metric_names}"


# ============================================================================
# TEST: validate_candidate — Property-based (Hypothesis)
# ============================================================================


class TestValidateCandidateProperties:
    """Property-based tests for validate_candidate using Hypothesis."""

    @pytest.mark.parametrize("score,threshold,comparison,expected_decision", [
        (0.85, 0.70, ThresholdComparison.gte, PromotionDecision.PROMOTED),
        (0.70, 0.70, ThresholdComparison.gte, PromotionDecision.PROMOTED),
        (0.69, 0.70, ThresholdComparison.gte, PromotionDecision.REJECTED_BELOW_THRESHOLD),
        (0.85, 0.70, ThresholdComparison.gt, PromotionDecision.PROMOTED),
        (0.70, 0.70, ThresholdComparison.gt, PromotionDecision.REJECTED_BELOW_THRESHOLD),
        (0.0, 0.0, ThresholdComparison.gte, PromotionDecision.PROMOTED),
        (1.0, 1.0, ThresholdComparison.gt, PromotionDecision.REJECTED_BELOW_THRESHOLD),
        (1.0, 0.99, ThresholdComparison.gt, PromotionDecision.PROMOTED),
    ])
    def test_vc_property_decision_correctness(
        self, score, threshold, comparison, expected_decision,
        make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """Decision is deterministic and correct for given score/threshold/comparison combos."""
        config = ValidationConfig(
            promotion_threshold=threshold,
            min_test_samples=1,
            threshold_comparison=comparison,
            per_metric_thresholds=[],
            swap_timeout_seconds=30.0,
        )
        evaluators = [make_evaluator("accuracy", score)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(config=config, evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        assert result.decision == expected_decision, \
            f"score={score}, threshold={threshold}, comparison={comparison}: " \
            f"expected {expected_decision}, got {result.decision}"

    def test_vc_property_aggregate_is_mean(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """aggregate_score is always the arithmetic mean of per_metric_scores."""
        evaluators = [
            make_evaluator("m1", 0.60),
            make_evaluator("m2", 0.80),
            make_evaluator("m3", 0.70),
            make_evaluator("m4", 0.90),
        ]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        if len(result.per_metric_scores) > 0:
            expected_mean = sum(result.per_metric_scores.values()) / len(result.per_metric_scores)
        else:
            expected_mean = 0.0
        assert result.aggregate_score == pytest.approx(expected_mean), \
            f"Aggregate {result.aggregate_score} != mean {expected_mean}"

    def test_vc_property_decision_deterministic(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """Given same inputs, decision is always the same (determinism)."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result1 = validator.validate_candidate("model-v2", "test-set-001")
        result2 = validator.validate_candidate("model-v2", "test-set-001")

        assert result1.decision == result2.decision, \
            f"Decision should be deterministic: {result1.decision} != {result2.decision}"


# ============================================================================
# TEST: promote_candidate — Happy Paths
# ============================================================================


class TestPromoteCandidateHappyPath:
    """Tests for promote_candidate happy path scenarios."""

    def test_pc_happy_path_swap_success(
        self, make_validator, make_evaluator, make_validation_result,
        mock_model_server, mock_audit_logger,
    ):
        """promote_candidate swaps model successfully when decision is PROMOTED."""
        validation_result = make_validation_result(decision=PromotionDecision.PROMOTED)
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        record = validator.promote_candidate(validation_result)

        assert record.final_decision == PromotionDecision.PROMOTED, \
            f"Expected PROMOTED, got {record.final_decision}"
        assert record.promotion_attempted is True, \
            "promotion_attempted should be True for successful swap"
        assert record.swap_started_utc != "", \
            "swap_started_utc should be populated"
        assert record.swap_completed_utc != "", \
            "swap_completed_utc should be populated"
        assert record.swap_duration_seconds >= 0, \
            "swap_duration_seconds should be >= 0"

    def test_pc_happy_path_previous_model_tracked(
        self, make_validator, make_evaluator, make_validation_result,
        mock_model_server,
    ):
        """promote_candidate records the previous model ID from model server."""
        validation_result = make_validation_result(
            decision=PromotionDecision.PROMOTED,
            candidate_model_id="model-v2",
        )
        mock_model_server.swap_model.return_value = SwapResult(
            success=True,
            previous_model_id="model-v1",
            new_model_id="model-v2",
            error_message="",
        )
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        record = validator.promote_candidate(validation_result)

        assert record.previous_model_id == "model-v1", \
            f"Expected previous_model_id='model-v1', got '{record.previous_model_id}'"
        assert record.validation_result.candidate_model_id == "model-v2"

    def test_pc_not_promoted_rejected_below_threshold(
        self, make_validator, make_evaluator, make_validation_result,
        mock_model_server,
    ):
        """Does not attempt swap when decision is REJECTED_BELOW_THRESHOLD."""
        validation_result = make_validation_result(
            decision=PromotionDecision.REJECTED_BELOW_THRESHOLD,
        )
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        record = validator.promote_candidate(validation_result)

        assert record.final_decision == PromotionDecision.REJECTED_BELOW_THRESHOLD, \
            f"Expected REJECTED_BELOW_THRESHOLD, got {record.final_decision}"
        assert record.promotion_attempted is False, \
            "promotion_attempted should be False when not promoted"
        assert mock_model_server.swap_model.call_count == 0, \
            "swap_model should not be called"

    def test_pc_not_promoted_evaluation_error(
        self, make_validator, make_evaluator, make_validation_result,
        mock_model_server,
    ):
        """Does not attempt swap when decision is REJECTED_EVALUATION_ERROR."""
        validation_result = make_validation_result(
            decision=PromotionDecision.REJECTED_EVALUATION_ERROR,
            evaluation_error_detail="All evaluators failed",
        )
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        record = validator.promote_candidate(validation_result)

        assert record.final_decision == PromotionDecision.REJECTED_EVALUATION_ERROR
        assert record.promotion_attempted is False

    def test_pc_not_promoted_skipped_no_test_data(
        self, make_validator, make_evaluator, make_validation_result,
        mock_model_server,
    ):
        """Does not attempt swap when decision is SKIPPED_NO_TEST_DATA."""
        validation_result = make_validation_result(
            decision=PromotionDecision.SKIPPED_NO_TEST_DATA,
        )
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        record = validator.promote_candidate(validation_result)

        assert record.final_decision == PromotionDecision.SKIPPED_NO_TEST_DATA
        assert record.promotion_attempted is False


# ============================================================================
# TEST: promote_candidate — Error Cases
# ============================================================================


class TestPromoteCandidateErrors:
    """Tests for promote_candidate error conditions."""

    def test_pc_error_swap_timeout(
        self, make_validator, make_evaluator, make_validation_result,
        mock_model_server,
    ):
        """Records SWAP_FAILED with error detail when swap times out."""
        mock_model_server.swap_model.side_effect = TimeoutError("Swap timed out")
        validation_result = make_validation_result(decision=PromotionDecision.PROMOTED)
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        record = validator.promote_candidate(validation_result)

        assert record.final_decision == PromotionDecision.SWAP_FAILED, \
            f"Expected SWAP_FAILED, got {record.final_decision}"
        assert record.promotion_attempted is True
        assert record.swap_error_detail != "", \
            "swap_error_detail should describe the timeout"

    def test_pc_error_swap_connection_error(
        self, make_validator, make_evaluator, make_validation_result,
        mock_model_server,
    ):
        """Records SWAP_FAILED when model server is unreachable during swap."""
        mock_model_server.swap_model.side_effect = ConnectionError("Server unreachable")
        validation_result = make_validation_result(decision=PromotionDecision.PROMOTED)
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        record = validator.promote_candidate(validation_result)

        assert record.final_decision == PromotionDecision.SWAP_FAILED
        assert record.promotion_attempted is True
        assert record.swap_error_detail != ""

    def test_pc_error_swap_model_load_error(
        self, make_validator, make_evaluator, make_validation_result,
        mock_model_server,
    ):
        """Records SWAP_FAILED when model server fails to load candidate model."""
        mock_model_server.swap_model.return_value = SwapResult(
            success=False,
            previous_model_id="model-v1",
            new_model_id="model-v2",
            error_message="Failed to load model: corrupt weights",
        )
        validation_result = make_validation_result(decision=PromotionDecision.PROMOTED)
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        record = validator.promote_candidate(validation_result)

        assert record.final_decision == PromotionDecision.SWAP_FAILED
        assert record.promotion_attempted is True

    def test_pc_error_model_server_not_ready(
        self, make_validator, make_evaluator, make_validation_result,
        mock_model_server,
    ):
        """Records SWAP_FAILED when model server is not ready for swaps."""
        mock_model_server.get_status.return_value = ModelServerStatus(
            is_healthy=True,
            current_model_id="model-v1",
            ready_to_swap=False,
        )
        mock_model_server.swap_model.side_effect = Exception("Server not ready for swap")
        validation_result = make_validation_result(decision=PromotionDecision.PROMOTED)
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        record = validator.promote_candidate(validation_result)

        assert record.final_decision == PromotionDecision.SWAP_FAILED
        assert record.promotion_attempted is True

    def test_pc_error_validation_not_passed(
        self, make_validator, make_evaluator, make_validation_result,
        mock_model_server,
    ):
        """Handles non-PROMOTED validation result without swap attempt."""
        validation_result = make_validation_result(
            decision=PromotionDecision.REJECTED_BELOW_THRESHOLD,
        )
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        record = validator.promote_candidate(validation_result)

        assert record.promotion_attempted is False, \
            "Should not attempt swap for non-PROMOTED validation"
        assert record.final_decision != PromotionDecision.PROMOTED


# ============================================================================
# TEST: promote_candidate — Invariants
# ============================================================================


class TestPromoteCandidateInvariants:
    """Invariant tests for promote_candidate."""

    def test_pc_invariant_swap_fail_previous_model_serves(
        self, make_validator, make_evaluator, make_validation_result,
        mock_model_server,
    ):
        """When swap fails, previous model continues serving."""
        mock_model_server.swap_model.side_effect = Exception("Swap failed")
        validation_result = make_validation_result(decision=PromotionDecision.PROMOTED)
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        record = validator.promote_candidate(validation_result)

        assert record.final_decision == PromotionDecision.SWAP_FAILED
        # Verify model server still reports previous model
        status = mock_model_server.get_status()
        assert status.current_model_id == "model-v1", \
            "Previous model should continue serving after swap failure"

    def test_pc_invariant_promotion_logged(
        self, make_validator, make_evaluator, make_validation_result,
        mock_audit_logger,
    ):
        """PromotionRecord is logged to audit logger before returning."""
        validation_result = make_validation_result(decision=PromotionDecision.PROMOTED)
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        validator.promote_candidate(validation_result)

        assert mock_audit_logger.log_promotion_record.call_count >= 1, \
            "PromotionRecord must be logged to audit logger"

    def test_pc_invariant_timestamps_utc_iso8601(
        self, make_validator, make_evaluator, make_validation_result,
    ):
        """All timestamps in PromotionRecord are UTC ISO 8601."""
        validation_result = make_validation_result(decision=PromotionDecision.PROMOTED)
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        record = validator.promote_candidate(validation_result)

        if record.swap_started_utc:
            parsed = datetime.fromisoformat(record.swap_started_utc)
            assert parsed is not None, "swap_started_utc should be valid ISO 8601"
        if record.swap_completed_utc:
            parsed = datetime.fromisoformat(record.swap_completed_utc)
            assert parsed is not None, "swap_completed_utc should be valid ISO 8601"

    def test_pc_invariant_no_exception_raised_on_swap_failure(
        self, make_validator, make_evaluator, make_validation_result,
        mock_model_server,
    ):
        """promote_candidate never raises exceptions past public boundary."""
        mock_model_server.swap_model.side_effect = RuntimeError("Catastrophic swap failure")
        validation_result = make_validation_result(decision=PromotionDecision.PROMOTED)
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        # Should not raise — errors captured in return type
        record = validator.promote_candidate(validation_result)

        assert record is not None, "Should return PromotionRecord, not raise"
        assert record.final_decision == PromotionDecision.SWAP_FAILED


# ============================================================================
# TEST: validate_and_promote — Happy Paths
# ============================================================================


class TestValidateAndPromoteHappyPath:
    """Tests for validate_and_promote happy path scenarios."""

    def test_vp_happy_path_full_flow(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, mock_model_server,
    ):
        """End-to-end: validation passes and promotion succeeds."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        record = validator.validate_and_promote("model-v2", "test-set-001")

        assert record.final_decision == PromotionDecision.PROMOTED, \
            f"Expected PROMOTED, got {record.final_decision}"
        assert record.promotion_attempted is True
        assert record.validation_result is not None, \
            "validation_result should be populated"
        assert record.validation_result.decision == PromotionDecision.PROMOTED

    def test_vp_validation_fails_no_promotion(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, mock_model_server,
    ):
        """Promotion not attempted when validation score is below threshold."""
        evaluators = [make_evaluator("accuracy", 0.30)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        record = validator.validate_and_promote("model-v2", "test-set-001")

        assert record.final_decision == PromotionDecision.REJECTED_BELOW_THRESHOLD, \
            f"Expected REJECTED_BELOW_THRESHOLD, got {record.final_decision}"
        assert record.promotion_attempted is False
        assert mock_model_server.swap_model.call_count == 0

    def test_vp_validation_passes_promotion_fails(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, mock_model_server,
    ):
        """Returns SWAP_FAILED when validation passes but swap fails."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set
        mock_model_server.swap_model.side_effect = Exception("Swap failed")

        validator = make_validator(evaluators=evaluators)
        record = validator.validate_and_promote("model-v2", "test-set-001")

        assert record.final_decision == PromotionDecision.SWAP_FAILED, \
            f"Expected SWAP_FAILED, got {record.final_decision}"
        assert record.promotion_attempted is True
        assert record.validation_result.decision == PromotionDecision.PROMOTED, \
            "Validation should have passed even though swap failed"


# ============================================================================
# TEST: validate_and_promote — Error Cases
# ============================================================================


class TestValidateAndPromoteErrors:
    """Tests for validate_and_promote error conditions."""

    def test_vp_error_validation_phase_error(
        self, make_validator, make_evaluator,
        mock_training_data_store, mock_model_server,
    ):
        """Handles validation phase errors gracefully."""
        mock_training_data_store.get_test_set.side_effect = ConnectionError("Store down")
        evaluators = [make_evaluator("accuracy", 0.85)]

        validator = make_validator(evaluators=evaluators)
        record = validator.validate_and_promote("model-v2", "test-set-001")

        assert record.validation_result.decision == PromotionDecision.REJECTED_EVALUATION_ERROR, \
            f"Expected REJECTED_EVALUATION_ERROR in validation, got {record.validation_result.decision}"
        assert record.promotion_attempted is False, \
            "Should not attempt promotion when validation fails"

    def test_vp_error_promotion_phase_error(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, mock_model_server,
    ):
        """Handles promotion phase errors after successful validation."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set
        mock_model_server.swap_model.side_effect = TimeoutError("Swap timeout")

        validator = make_validator(evaluators=evaluators)
        record = validator.validate_and_promote("model-v2", "test-set-001")

        assert record.final_decision == PromotionDecision.SWAP_FAILED, \
            f"Expected SWAP_FAILED, got {record.final_decision}"


# ============================================================================
# TEST: validate_and_promote — Edge Cases
# ============================================================================


class TestValidateAndPromoteEdgeCases:
    """Edge case tests for validate_and_promote."""

    def test_vp_skipped_no_test_data_no_promotion(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """Returns SKIPPED_NO_TEST_DATA when test set has no data."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-empty", 0)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        record = validator.validate_and_promote("model-v2", "test-set-empty")

        assert record.final_decision == PromotionDecision.SKIPPED_NO_TEST_DATA
        assert record.promotion_attempted is False

    def test_vp_edge_candidate_model_id_consistent(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """candidate_model_id is consistent across ValidationResult and PromotionRecord."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        record = validator.validate_and_promote("model-v2", "test-set-001")

        assert record.validation_result.candidate_model_id == "model-v2", \
            "candidate_model_id should be consistent"

    def test_vp_edge_evaluation_error_stops_promotion(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, mock_model_server,
    ):
        """Promotion stops when all evaluators fail."""
        evaluators = [
            make_evaluator("m1", 0.0, should_fail=True),
            make_evaluator("m2", 0.0, should_fail=True),
        ]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        record = validator.validate_and_promote("model-v2", "test-set-001")

        assert record.final_decision == PromotionDecision.REJECTED_EVALUATION_ERROR
        assert record.promotion_attempted is False
        assert mock_model_server.swap_model.call_count == 0

    def test_vp_edge_per_metric_failure_stops_promotion(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, mock_model_server,
    ):
        """Promotion stops when per-metric threshold fails even if aggregate passes."""
        config = ValidationConfig(
            promotion_threshold=0.5,
            min_test_samples=5,
            threshold_comparison=ThresholdComparison.gte,
            per_metric_thresholds=[
                PerMetricThreshold(
                    metric_name="accuracy",
                    min_score=0.95,
                    comparison=ThresholdComparison.gte,
                ),
            ],
            swap_timeout_seconds=30.0,
        )
        evaluators = [
            make_evaluator("accuracy", 0.60),
            make_evaluator("fluency", 0.90),
        ]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(config=config, evaluators=evaluators)
        record = validator.validate_and_promote("model-v2", "test-set-001")

        assert record.final_decision == PromotionDecision.REJECTED_BELOW_THRESHOLD
        assert record.promotion_attempted is False
        assert len(record.validation_result.failed_per_metric_thresholds) > 0
        assert mock_model_server.swap_model.call_count == 0


# ============================================================================
# TEST: validate_and_promote — Invariants
# ============================================================================


class TestValidateAndPromoteInvariants:
    """Invariant tests for validate_and_promote."""

    def test_vp_invariant_both_logged(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, mock_audit_logger,
    ):
        """Both ValidationResult and PromotionRecord are logged to audit logger."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        validator.validate_and_promote("model-v2", "test-set-001")

        assert mock_audit_logger.log_validation_result.call_count >= 1, \
            "ValidationResult must be logged"
        assert mock_audit_logger.log_promotion_record.call_count >= 1, \
            "PromotionRecord must be logged"

    def test_vp_invariant_no_exceptions_raised(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, mock_model_server,
    ):
        """validate_and_promote never raises exceptions past public boundary."""
        mock_training_data_store.get_test_set.side_effect = RuntimeError("Catastrophic")
        evaluators = [make_evaluator("accuracy", 0.85)]

        validator = make_validator(evaluators=evaluators)
        # Should not raise
        record = validator.validate_and_promote("model-v2", "test-set-001")

        assert record is not None, "Should return a PromotionRecord, not raise"

    def test_vp_invariant_no_exceptions_on_swap_failure(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store, mock_model_server,
    ):
        """validate_and_promote does not raise even when swap raises unexpected errors."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set
        mock_model_server.swap_model.side_effect = RuntimeError("Unexpected swap error")

        validator = make_validator(evaluators=evaluators)
        record = validator.validate_and_promote("model-v2", "test-set-001")

        assert record is not None
        assert record.final_decision == PromotionDecision.SWAP_FAILED


# ============================================================================
# TEST: get_validation_history — Happy Paths
# ============================================================================


class TestGetValidationHistoryHappyPath:
    """Tests for get_validation_history happy path scenarios."""

    def test_gh_happy_path_returns_history(
        self, make_validator, make_evaluator, make_validation_result,
        mock_audit_logger,
    ):
        """Returns list of PromotionRecords for a given model."""
        # Setup mock audit logger to return history
        mock_records = [
            PromotionRecord(
                validation_result=make_validation_result(candidate_model_id="model-v2"),
                promotion_attempted=True,
                swap_started_utc="2024-01-01T00:00:00+00:00",
                swap_completed_utc="2024-01-01T00:00:05+00:00",
                swap_duration_seconds=5.0,
                final_decision=PromotionDecision.PROMOTED,
                swap_error_detail="",
                previous_model_id="model-v1",
            ),
            PromotionRecord(
                validation_result=make_validation_result(
                    candidate_model_id="model-v2",
                    decision=PromotionDecision.REJECTED_BELOW_THRESHOLD,
                ),
                promotion_attempted=False,
                swap_started_utc="",
                swap_completed_utc="",
                swap_duration_seconds=0.0,
                final_decision=PromotionDecision.REJECTED_BELOW_THRESHOLD,
                swap_error_detail="",
                previous_model_id="",
            ),
        ]
        mock_audit_logger.get_validation_history.return_value = mock_records

        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])
        history = validator.get_validation_history("model-v2", limit=10)

        assert len(history) > 0, "Should return non-empty history"
        assert all(
            r.validation_result.candidate_model_id == "model-v2" for r in history
        ), "All records should be for model-v2"

    def test_gh_happy_path_empty_history(
        self, make_validator, make_evaluator, mock_audit_logger,
    ):
        """Returns empty list when no history exists."""
        mock_audit_logger.get_validation_history.return_value = []

        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])
        history = validator.get_validation_history("model-never-tested", limit=10)

        assert history == [], f"Expected empty list, got {history}"


# ============================================================================
# TEST: get_validation_history — Edge Cases
# ============================================================================


class TestGetValidationHistoryEdgeCases:
    """Edge case tests for get_validation_history."""

    def test_gh_edge_limit_enforcement(
        self, make_validator, make_evaluator, make_validation_result,
        mock_audit_logger,
    ):
        """Returns at most 'limit' entries."""
        mock_records = [
            PromotionRecord(
                validation_result=make_validation_result(candidate_model_id="model-v2"),
                promotion_attempted=True,
                swap_started_utc=f"2024-01-0{i+1}T00:00:00+00:00",
                swap_completed_utc=f"2024-01-0{i+1}T00:00:05+00:00",
                swap_duration_seconds=5.0,
                final_decision=PromotionDecision.PROMOTED,
                swap_error_detail="",
                previous_model_id="model-v1",
            )
            for i in range(5)
        ]
        # Return only 3 when limit is 3
        mock_audit_logger.get_validation_history.return_value = mock_records[:3]

        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])
        history = validator.get_validation_history("model-v2", limit=3)

        assert len(history) <= 3, f"Expected at most 3 entries, got {len(history)}"

    def test_gh_edge_ordering_descending(
        self, make_validator, make_evaluator, make_validation_result,
        mock_audit_logger,
    ):
        """Returns records in descending timestamp order (most recent first)."""
        mock_records = [
            PromotionRecord(
                validation_result=make_validation_result(candidate_model_id="model-v2"),
                promotion_attempted=True,
                swap_started_utc="2024-01-03T00:00:00+00:00",
                swap_completed_utc="2024-01-03T00:00:05+00:00",
                swap_duration_seconds=5.0,
                final_decision=PromotionDecision.PROMOTED,
                swap_error_detail="",
                previous_model_id="model-v1",
            ),
            PromotionRecord(
                validation_result=make_validation_result(candidate_model_id="model-v2"),
                promotion_attempted=False,
                swap_started_utc="",
                swap_completed_utc="",
                swap_duration_seconds=0.0,
                final_decision=PromotionDecision.REJECTED_BELOW_THRESHOLD,
                swap_error_detail="",
                previous_model_id="",
            ),
        ]
        mock_audit_logger.get_validation_history.return_value = mock_records

        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])
        history = validator.get_validation_history("model-v2", limit=10)

        if len(history) >= 2:
            for i in range(len(history) - 1):
                assert history[i].validation_result.timestamp_utc >= history[i + 1].validation_result.timestamp_utc, \
                    f"Records should be in descending order at index {i}"

    def test_gh_edge_limit_zero(
        self, make_validator, make_evaluator, mock_audit_logger,
    ):
        """get_validation_history with limit=0 returns empty list."""
        mock_audit_logger.get_validation_history.return_value = []

        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])
        history = validator.get_validation_history("model-v2", limit=0)

        assert history == [], f"Expected empty list for limit=0, got {history}"


# ============================================================================
# TEST: get_validation_history — Error Cases
# ============================================================================


class TestGetValidationHistoryErrors:
    """Error case tests for get_validation_history."""

    def test_gh_error_audit_log_unavailable(
        self, make_validator, make_evaluator, mock_audit_logger,
    ):
        """Handles unavailable audit log gracefully."""
        mock_audit_logger.get_validation_history.side_effect = ConnectionError(
            "Audit log unavailable"
        )
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        # Should not raise — invariant says no exceptions past public boundary
        # The implementation may return empty list or handle differently
        try:
            history = validator.get_validation_history("model-v2", limit=10)
            assert isinstance(history, list), \
                f"Should return a list, got {type(history)}"
        except Exception:
            # If implementation does raise, that's a contract violation
            # but we document it rather than fail silently
            pytest.fail(
                "get_validation_history raised an exception, "
                "violating the no-exceptions-past-public-boundary invariant"
            )


# ============================================================================
# TEST: get_validation_history — Invariants
# ============================================================================


class TestGetValidationHistoryInvariants:
    """Invariant tests for get_validation_history."""

    def test_gh_invariant_no_exceptions(
        self, make_validator, make_evaluator, mock_audit_logger,
    ):
        """get_validation_history does not raise exceptions past public boundary."""
        mock_audit_logger.get_validation_history.side_effect = RuntimeError("Unexpected")
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        try:
            history = validator.get_validation_history("model-v2", limit=10)
            assert isinstance(history, list)
        except Exception:
            pytest.fail(
                "get_validation_history should not raise exceptions past public boundary"
            )


# ============================================================================
# TEST: Global Invariants
# ============================================================================


class TestGlobalInvariants:
    """Tests for cross-cutting contract invariants."""

    def test_inv_config_validated_at_init(self, mock_training_data_store, mock_model_server, mock_audit_logger, make_evaluator):
        """ValidationConfig is validated at initialization time — invalid config raises immediately."""
        with pytest.raises(Exception):
            ModelValidator(
                config=ValidationConfig(
                    promotion_threshold=-1.0,  # Invalid
                    min_test_samples=-5,  # Invalid
                    threshold_comparison=ThresholdComparison.gte,
                    per_metric_thresholds=[],
                    swap_timeout_seconds=-10.0,  # Invalid
                ),
                evaluators=[make_evaluator("accuracy", 0.85)],
                training_data_store=mock_training_data_store,
                model_server=mock_model_server,
                audit_logger=mock_audit_logger,
            )

    def test_inv_validate_candidate_does_not_affect_subsequent_calls(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """No global state: repeated calls don't affect each other."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result1 = validator.validate_candidate("model-v2", "test-set-001")
        result2 = validator.validate_candidate("model-v2", "test-set-001")

        assert result1.decision == result2.decision, \
            "Repeated calls with same inputs should produce same decision"
        assert result1.aggregate_score == pytest.approx(result2.aggregate_score), \
            "Repeated calls with same inputs should produce same aggregate score"

    def test_inv_promote_only_when_promoted(
        self, make_validator, make_evaluator, make_validation_result,
        mock_model_server,
    ):
        """promote_candidate only attempts a swap if decision == PROMOTED."""
        for decision in [
            PromotionDecision.REJECTED_BELOW_THRESHOLD,
            PromotionDecision.REJECTED_EVALUATION_ERROR,
            PromotionDecision.SKIPPED_NO_TEST_DATA,
        ]:
            mock_model_server.reset_mock()
            validation_result = make_validation_result(decision=decision)
            validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

            record = validator.promote_candidate(validation_result)

            assert record.promotion_attempted is False, \
                f"Should not attempt swap for decision={decision}"
            assert mock_model_server.swap_model.call_count == 0, \
                f"swap_model should not be called for decision={decision}"
            assert record.final_decision == decision, \
                f"final_decision should match input decision={decision}"

    def test_inv_validation_result_immutable(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """ValidationResult is immutable (frozen Pydantic model)."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        with pytest.raises((AttributeError, TypeError, Exception)):
            result.decision = PromotionDecision.REJECTED_BELOW_THRESHOLD

    def test_inv_all_validation_decisions_are_valid_enum_values(
        self, make_validator, make_evaluator, make_test_set,
        mock_training_data_store,
    ):
        """All returned decisions are valid PromotionDecision enum values."""
        evaluators = [make_evaluator("accuracy", 0.85)]
        test_set = make_test_set("test-set-001", 10)
        mock_training_data_store.get_test_set.return_value = test_set

        validator = make_validator(evaluators=evaluators)
        result = validator.validate_candidate("model-v2", "test-set-001")

        valid_decisions = {
            PromotionDecision.PROMOTED,
            PromotionDecision.REJECTED_BELOW_THRESHOLD,
            PromotionDecision.REJECTED_EVALUATION_ERROR,
            PromotionDecision.SWAP_FAILED,
            PromotionDecision.SKIPPED_NO_TEST_DATA,
        }
        assert result.decision in valid_decisions, \
            f"Decision {result.decision} is not a valid PromotionDecision"

    def test_inv_swap_failure_no_partial_state(
        self, make_validator, make_evaluator, make_validation_result,
        mock_model_server,
    ):
        """If a model swap fails, the previous model continues serving — no partial state."""
        mock_model_server.swap_model.return_value = SwapResult(
            success=False,
            previous_model_id="model-v1",
            new_model_id="model-v2",
            error_message="Load failure: corrupt weights",
        )
        validation_result = make_validation_result(decision=PromotionDecision.PROMOTED)
        validator = make_validator(evaluators=[make_evaluator("accuracy", 0.85)])

        record = validator.promote_candidate(validation_result)

        assert record.final_decision == PromotionDecision.SWAP_FAILED
        # Verify model server is still configured with previous model
        status = mock_model_server.get_status()
        assert status.current_model_id == "model-v1", \
            "Previous model should continue serving after failed swap"
