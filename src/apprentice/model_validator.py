"""
Model Validator & Promoter component.

Validates newly fine-tuned models before promoting them to serve live traffic.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
import hashlib
import time

from pydantic import BaseModel, Field, field_validator, ConfigDict


# ============================================================================
# Enums
# ============================================================================


class PromotionDecision(str, Enum):
    """StrEnum representing the outcome of a model validation and promotion attempt."""
    PROMOTED = "PROMOTED"
    REJECTED_BELOW_THRESHOLD = "REJECTED_BELOW_THRESHOLD"
    REJECTED_EVALUATION_ERROR = "REJECTED_EVALUATION_ERROR"
    SWAP_FAILED = "SWAP_FAILED"
    SKIPPED_NO_TEST_DATA = "SKIPPED_NO_TEST_DATA"


class ThresholdComparison(str, Enum):
    """Comparison operator used when checking aggregate score against the promotion threshold."""
    gte = "gte"
    gt = "gt"


# ============================================================================
# Data Models
# ============================================================================


class PerMetricThreshold(BaseModel):
    """A threshold constraint for an individual evaluation metric."""
    metric_name: str
    min_score: float = Field(ge=0.0, le=1.0)
    comparison: ThresholdComparison = ThresholdComparison.gte

    @field_validator('min_score')
    @classmethod
    def validate_min_score(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"min_score must be in [0.0, 1.0], got {v}")
        return v


class ValidationConfig(BaseModel):
    """Pydantic configuration model controlling validation and promotion behavior."""
    promotion_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    min_test_samples: int = Field(default=10, ge=1)
    threshold_comparison: ThresholdComparison = ThresholdComparison.gte
    per_metric_thresholds: List[PerMetricThreshold] = Field(default_factory=list)
    swap_timeout_seconds: float = Field(default=120.0, gt=0.0)

    @field_validator('promotion_threshold')
    @classmethod
    def validate_promotion_threshold(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"promotion_threshold must be in [0.0, 1.0], got {v}")
        return v

    @field_validator('min_test_samples')
    @classmethod
    def validate_min_test_samples(cls, v):
        if v < 1:
            raise ValueError(f"min_test_samples must be >= 1, got {v}")
        return v

    @field_validator('swap_timeout_seconds')
    @classmethod
    def validate_swap_timeout(cls, v):
        if v <= 0.0:
            raise ValueError(f"swap_timeout_seconds must be > 0.0, got {v}")
        return v


class TestSetReference(BaseModel):
    """Immutable reference to the held-out test set used during validation."""
    test_set_id: str = Field(min_length=1)
    content_hash: str
    sample_count: int = Field(ge=0)

    model_config = ConfigDict(frozen=True)


class TestSample(BaseModel):
    """A single sample from the held-out test set."""
    sample_id: str
    input_prompt: str
    reference_output: str
    metadata: Dict = Field(default_factory=dict)


class TestSet(BaseModel):
    """The held-out test set retrieved from the training data store."""
    reference: TestSetReference
    samples: List[TestSample]


class EvaluationScore(BaseModel):
    """Score result from a single evaluator for a single test sample."""
    metric_name: str
    score: float = Field(ge=0.0, le=1.0)


class ValidationResult(BaseModel):
    """Frozen (immutable) Pydantic model capturing the complete result of a candidate model validation."""
    candidate_model_id: str = Field(min_length=1)
    timestamp_utc: str
    per_metric_scores: Dict[str, float]
    aggregate_score: float = Field(ge=0.0, le=1.0)
    threshold_used: float = Field(ge=0.0, le=1.0)
    threshold_comparison_used: ThresholdComparison
    test_set_reference: TestSetReference
    decision: PromotionDecision
    failed_per_metric_thresholds: List[str] = Field(default_factory=list)
    evaluation_error_detail: str = ""
    samples_evaluated: int = Field(ge=0)

    model_config = ConfigDict(frozen=True)


class PromotionRecord(BaseModel):
    """Extends ValidationResult with promotion execution details."""
    validation_result: ValidationResult
    promotion_attempted: bool
    swap_started_utc: str = ""
    swap_completed_utc: str = ""
    swap_duration_seconds: float = 0.0
    final_decision: PromotionDecision
    swap_error_detail: str = ""
    previous_model_id: str = ""


class ModelServerStatus(BaseModel):
    """Status response from the local model server."""
    is_healthy: bool
    current_model_id: str
    ready_to_swap: bool


class SwapResult(BaseModel):
    """Result of a model swap operation from the ModelServer protocol."""
    success: bool
    previous_model_id: str
    new_model_id: str
    error_message: str = ""


# ============================================================================
# ModelValidator
# ============================================================================


class ModelValidator:
    """
    Validates a newly fine-tuned model before promoting it to serve live traffic.
    """

    def __init__(
        self,
        config: ValidationConfig,
        evaluators: List,
        training_data_store,
        model_server,
        audit_logger,
    ):
        """
        Initialize the ModelValidator with configuration and dependencies.

        Validates the config at initialization time - invalid config raises immediately.
        """
        # Validate config by accessing its fields (Pydantic will have already validated)
        if config.promotion_threshold < 0.0 or config.promotion_threshold > 1.0:
            raise ValueError(f"Invalid promotion_threshold: {config.promotion_threshold}")
        if config.min_test_samples < 1:
            raise ValueError(f"Invalid min_test_samples: {config.min_test_samples}")
        if config.swap_timeout_seconds <= 0.0:
            raise ValueError(f"Invalid swap_timeout_seconds: {config.swap_timeout_seconds}")

        self.config = config
        self.evaluators = evaluators
        self.training_data_store = training_data_store
        self.model_server = model_server
        self.audit_logger = audit_logger

    def validate_candidate(
        self,
        candidate_model_id: str,
        test_set_id: Optional[str] = None,
    ) -> ValidationResult:
        """
        Runs the candidate fine-tuned model against the held-out test set.

        This is a pure evaluation method - it performs no side effects on the serving model.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        # Default test set reference for error cases
        default_test_ref = TestSetReference(
            test_set_id=test_set_id or "unknown",
            content_hash="0" * 64,
            sample_count=0,
        )

        try:
            # Fetch test set from training data store
            test_set = self.training_data_store.get_test_set(test_set_id)

            # Check if test set has enough samples
            if len(test_set.samples) < self.config.min_test_samples:
                return ValidationResult(
                    candidate_model_id=candidate_model_id,
                    timestamp_utc=timestamp_utc,
                    per_metric_scores={},
                    aggregate_score=0.0,
                    threshold_used=self.config.promotion_threshold,
                    threshold_comparison_used=self.config.threshold_comparison,
                    test_set_reference=test_set.reference,
                    decision=PromotionDecision.SKIPPED_NO_TEST_DATA,
                    failed_per_metric_thresholds=[],
                    evaluation_error_detail=f"Test set has {len(test_set.samples)} samples, need at least {self.config.min_test_samples}",
                    samples_evaluated=0,
                )

            # Check if test set is empty
            if len(test_set.samples) == 0:
                return self._log_and_return_validation(ValidationResult(
                    candidate_model_id=candidate_model_id,
                    timestamp_utc=timestamp_utc,
                    per_metric_scores={},
                    aggregate_score=0.0,
                    threshold_used=self.config.promotion_threshold,
                    threshold_comparison_used=self.config.threshold_comparison,
                    test_set_reference=test_set.reference,
                    decision=PromotionDecision.SKIPPED_NO_TEST_DATA,
                    failed_per_metric_thresholds=[],
                    evaluation_error_detail="Test set is empty",
                    samples_evaluated=0,
                ))

            # Run evaluations
            per_metric_scores = {}
            evaluation_errors = []

            for evaluator in self.evaluators:
                try:
                    # Evaluate each sample
                    scores = []
                    for sample in test_set.samples:
                        try:
                            # Get model output
                            model_output = self.model_server.infer(sample.input_prompt)

                            # Evaluate
                            eval_score = evaluator.evaluate(
                                input_prompt=sample.input_prompt,
                                model_output=model_output,
                                reference_output=sample.reference_output,
                            )
                            scores.append(eval_score.score)
                        except Exception as e:
                            # Track inference errors
                            evaluation_errors.append(f"Inference error for sample {sample.sample_id}: {str(e)}")
                            continue

                    # Compute average score for this metric
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        per_metric_scores[evaluator.metric_name] = avg_score

                except Exception as e:
                    evaluation_errors.append(f"Evaluator {evaluator.metric_name} failed: {str(e)}")
                    continue

            # Check if all evaluators failed
            if not per_metric_scores:
                return self._log_and_return_validation(ValidationResult(
                    candidate_model_id=candidate_model_id,
                    timestamp_utc=timestamp_utc,
                    per_metric_scores={},
                    aggregate_score=0.0,
                    threshold_used=self.config.promotion_threshold,
                    threshold_comparison_used=self.config.threshold_comparison,
                    test_set_reference=test_set.reference,
                    decision=PromotionDecision.REJECTED_EVALUATION_ERROR,
                    failed_per_metric_thresholds=[],
                    evaluation_error_detail="; ".join(evaluation_errors) if evaluation_errors else "All evaluators failed",
                    samples_evaluated=len(test_set.samples),
                ))

            # Compute aggregate score (round to avoid float artifacts)
            aggregate_score = self._round_score(
                sum(per_metric_scores.values()) / len(per_metric_scores)
            )

            # Check aggregate threshold
            threshold_passed = self._check_threshold(
                aggregate_score,
                self.config.promotion_threshold,
                self.config.threshold_comparison,
            )

            # Check per-metric thresholds
            failed_per_metric_thresholds = []
            if self.config.per_metric_thresholds:
                for pmt in self.config.per_metric_thresholds:
                    if pmt.metric_name in per_metric_scores:
                        metric_score = per_metric_scores[pmt.metric_name]
                        if not self._check_threshold(metric_score, pmt.min_score, pmt.comparison):
                            failed_per_metric_thresholds.append(pmt.metric_name)

            # Determine decision
            if not threshold_passed or failed_per_metric_thresholds:
                decision = PromotionDecision.REJECTED_BELOW_THRESHOLD
            else:
                decision = PromotionDecision.PROMOTED

            result = ValidationResult(
                candidate_model_id=candidate_model_id,
                timestamp_utc=timestamp_utc,
                per_metric_scores=per_metric_scores,
                aggregate_score=aggregate_score,
                threshold_used=self.config.promotion_threshold,
                threshold_comparison_used=self.config.threshold_comparison,
                test_set_reference=test_set.reference,
                decision=decision,
                failed_per_metric_thresholds=failed_per_metric_thresholds,
                evaluation_error_detail="",
                samples_evaluated=len(test_set.samples),
            )

            return self._log_and_return_validation(result)

        except Exception as e:
            # Handle any unexpected errors gracefully
            result = ValidationResult(
                candidate_model_id=candidate_model_id,
                timestamp_utc=timestamp_utc,
                per_metric_scores={},
                aggregate_score=0.0,
                threshold_used=self.config.promotion_threshold,
                threshold_comparison_used=self.config.threshold_comparison,
                test_set_reference=default_test_ref,
                decision=PromotionDecision.REJECTED_EVALUATION_ERROR,
                failed_per_metric_thresholds=[],
                evaluation_error_detail=str(e),
                samples_evaluated=0,
            )
            return self._log_and_return_validation(result)

    def promote_candidate(
        self,
        validation_result: ValidationResult,
    ) -> PromotionRecord:
        """
        Attempts to promote a validated candidate model to serve live traffic.

        Only proceeds with the swap if the ValidationResult indicates the candidate passed validation.
        """
        # Only attempt swap if validation passed
        if validation_result.decision != PromotionDecision.PROMOTED:
            record = PromotionRecord(
                validation_result=validation_result,
                promotion_attempted=False,
                swap_started_utc="",
                swap_completed_utc="",
                swap_duration_seconds=0.0,
                final_decision=validation_result.decision,
                swap_error_detail="",
                previous_model_id="",
            )
            return self._log_and_return_promotion(record)

        # Attempt the swap
        swap_started_utc = datetime.now(timezone.utc).isoformat()
        swap_started_time = time.time()

        try:
            # Get current model status before swap
            status = self.model_server.get_status()
            previous_model_id = status.current_model_id

            # Perform the swap
            swap_result = self.model_server.swap_model(
                validation_result.candidate_model_id
            )

            swap_completed_utc = datetime.now(timezone.utc).isoformat()
            swap_duration_seconds = time.time() - swap_started_time

            # Check if swap succeeded
            if swap_result.success:
                record = PromotionRecord(
                    validation_result=validation_result,
                    promotion_attempted=True,
                    swap_started_utc=swap_started_utc,
                    swap_completed_utc=swap_completed_utc,
                    swap_duration_seconds=swap_duration_seconds,
                    final_decision=PromotionDecision.PROMOTED,
                    swap_error_detail="",
                    previous_model_id=swap_result.previous_model_id,
                )
            else:
                record = PromotionRecord(
                    validation_result=validation_result,
                    promotion_attempted=True,
                    swap_started_utc=swap_started_utc,
                    swap_completed_utc=swap_completed_utc,
                    swap_duration_seconds=swap_duration_seconds,
                    final_decision=PromotionDecision.SWAP_FAILED,
                    swap_error_detail=swap_result.error_message,
                    previous_model_id=swap_result.previous_model_id,
                )

            return self._log_and_return_promotion(record)

        except Exception as e:
            # Handle swap errors gracefully
            swap_completed_utc = datetime.now(timezone.utc).isoformat()
            swap_duration_seconds = time.time() - swap_started_time

            # Try to get previous model ID
            try:
                status = self.model_server.get_status()
                previous_model_id = status.current_model_id
            except:
                previous_model_id = ""

            record = PromotionRecord(
                validation_result=validation_result,
                promotion_attempted=True,
                swap_started_utc=swap_started_utc,
                swap_completed_utc=swap_completed_utc,
                swap_duration_seconds=swap_duration_seconds,
                final_decision=PromotionDecision.SWAP_FAILED,
                swap_error_detail=str(e),
                previous_model_id=previous_model_id,
            )
            return self._log_and_return_promotion(record)

    def validate_and_promote(
        self,
        candidate_model_id: str,
        test_set_id: Optional[str] = None,
    ) -> PromotionRecord:
        """
        Convenience method that runs validate_candidate followed by promote_candidate in sequence.
        """
        try:
            # Run validation
            validation_result = self.validate_candidate(candidate_model_id, test_set_id)

            # Run promotion
            promotion_record = self.promote_candidate(validation_result)

            return promotion_record

        except Exception as e:
            # Handle any unexpected errors gracefully
            timestamp_utc = datetime.now(timezone.utc).isoformat()
            default_test_ref = TestSetReference(
                test_set_id=test_set_id or "unknown",
                content_hash="0" * 64,
                sample_count=0,
            )

            validation_result = ValidationResult(
                candidate_model_id=candidate_model_id,
                timestamp_utc=timestamp_utc,
                per_metric_scores={},
                aggregate_score=0.0,
                threshold_used=self.config.promotion_threshold,
                threshold_comparison_used=self.config.threshold_comparison,
                test_set_reference=default_test_ref,
                decision=PromotionDecision.REJECTED_EVALUATION_ERROR,
                failed_per_metric_thresholds=[],
                evaluation_error_detail=str(e),
                samples_evaluated=0,
            )

            # Log validation result
            self._log_validation(validation_result)

            record = PromotionRecord(
                validation_result=validation_result,
                promotion_attempted=False,
                swap_started_utc="",
                swap_completed_utc="",
                swap_duration_seconds=0.0,
                final_decision=PromotionDecision.REJECTED_EVALUATION_ERROR,
                swap_error_detail="",
                previous_model_id="",
            )

            return self._log_and_return_promotion(record)

    def get_validation_history(
        self,
        candidate_model_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[PromotionRecord]:
        """
        Retrieves the history of validation results for a given model ID or all models.

        Returns records in descending timestamp order (most recent first).
        """
        try:
            # Handle limit=0 edge case
            if limit == 0:
                return []

            # Fetch from audit logger
            history = self.audit_logger.get_validation_history(
                candidate_model_id=candidate_model_id,
                limit=limit,
            )

            # Sort by timestamp descending (most recent first)
            sorted_history = sorted(
                history,
                key=lambda r: r.validation_result.timestamp_utc,
                reverse=True,
            )

            return sorted_history

        except Exception as e:
            # Return empty list on error (graceful degradation)
            return []

    # ========================================================================
    # Helper Methods
    # ========================================================================

    @staticmethod
    def _round_score(value: float, precision: int = 10) -> float:
        """Round a score to avoid floating-point comparison artifacts.

        Using 10 decimal places preserves meaningful precision while
        eliminating representation noise (e.g. 0.7999999999999999 → 0.8).
        """
        return round(value, precision)

    def _check_threshold(
        self,
        score: float,
        threshold: float,
        comparison: ThresholdComparison,
    ) -> bool:
        """Check if score passes threshold with given comparison operator.

        Both values are rounded to 10 decimal places before comparison
        to prevent floating-point representation artifacts from causing
        incorrect threshold decisions.
        """
        score = self._round_score(score)
        threshold = self._round_score(threshold)
        if comparison == ThresholdComparison.gte:
            return score >= threshold
        else:  # gt
            return score > threshold

    def _log_validation(self, result: ValidationResult) -> None:
        """Log validation result to audit logger."""
        try:
            self.audit_logger.log_validation_result(result)
        except:
            pass  # Gracefully handle logging failures

    def _log_promotion(self, record: PromotionRecord) -> None:
        """Log promotion record to audit logger."""
        try:
            self.audit_logger.log_promotion_record(record)
        except:
            pass  # Gracefully handle logging failures

    def _log_and_return_validation(self, result: ValidationResult) -> ValidationResult:
        """Log validation result and return it."""
        self._log_validation(result)
        return result

    def _log_and_return_promotion(self, record: PromotionRecord) -> PromotionRecord:
        """Log promotion record and return it."""
        self._log_promotion(record)
        return record


# ── Auto-injected export aliases (Pact export gate) ──
PerMetricThresholdList = PerMetricThreshold
TestSampleList = TestSample
PromotionRecordList = PromotionRecord
ValidationResultOptional = ValidationResult
