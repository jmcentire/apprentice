"""Tests for the PII detection evaluator."""

import pytest

from apprentice.pii_detection import PIIDetection
from apprentice.pii_evaluation import (
    EntityMetrics,
    PIIDetectionEvaluator,
    PIIEvaluationResult,
    _compute_f1,
    _span_iou,
)


# ============================================================================
# Span IoU
# ============================================================================


class TestSpanIoU:
    def test_identical_spans(self):
        assert _span_iou(0, 10, 0, 10) == 1.0

    def test_no_overlap(self):
        assert _span_iou(0, 5, 10, 15) == 0.0

    def test_partial_overlap(self):
        # [0, 10) and [5, 15) => intersection [5, 10) = 5, union = 15
        iou = _span_iou(0, 10, 5, 15)
        assert abs(iou - 5 / 15) < 1e-6

    def test_contained_span(self):
        # [0, 20) contains [5, 10) => intersection = 5, union = 20
        iou = _span_iou(0, 20, 5, 10)
        assert abs(iou - 5 / 20) < 1e-6

    def test_adjacent_spans(self):
        assert _span_iou(0, 5, 5, 10) == 0.0

    def test_zero_length_spans(self):
        assert _span_iou(5, 5, 5, 5) == 0.0


# ============================================================================
# F1 computation
# ============================================================================


class TestComputeF1:
    def test_perfect(self):
        assert _compute_f1(1.0, 1.0) == 1.0

    def test_zero_precision(self):
        assert _compute_f1(0.0, 1.0) == 0.0

    def test_zero_recall(self):
        assert _compute_f1(1.0, 0.0) == 0.0

    def test_both_zero(self):
        assert _compute_f1(0.0, 0.0) == 0.0

    def test_balanced(self):
        f1 = _compute_f1(0.5, 0.5)
        assert abs(f1 - 0.5) < 1e-6


# ============================================================================
# PIIEvaluationResult model
# ============================================================================


class TestPIIEvaluationResult:
    def test_creation(self):
        r = PIIEvaluationResult(
            micro_precision=0.8, micro_recall=0.7, micro_f1=0.75,
            macro_precision=0.8, macro_recall=0.7, macro_f1=0.75,
            total_predictions=10, total_ground_truth=12,
        )
        assert r.micro_precision == 0.8
        assert r.total_predictions == 10

    def test_frozen(self):
        r = PIIEvaluationResult()
        with pytest.raises(Exception):
            r.micro_f1 = 0.99


# ============================================================================
# EntityMetrics model
# ============================================================================


class TestEntityMetrics:
    def test_creation(self):
        m = EntityMetrics(
            entity_type="email",
            precision=1.0, recall=0.5, f1=0.667,
            true_positives=5, false_positives=0, false_negatives=5,
        )
        assert m.entity_type == "email"
        assert m.true_positives == 5


# ============================================================================
# PIIDetectionEvaluator
# ============================================================================


def _make_det(start: int, end: int, entity_type: str, source: str = "test") -> PIIDetection:
    return PIIDetection(
        start=start, end=end, value=f"v{start}_{end}",
        entity_type=entity_type, confidence=1.0, source=source,
    )


class TestPIIDetectionEvaluator:
    def test_empty_inputs(self):
        e = PIIDetectionEvaluator()
        result = e.evaluate([], [])
        assert result.micro_f1 == 0.0
        assert result.total_predictions == 0
        assert result.total_ground_truth == 0

    def test_perfect_detection(self):
        preds = [_make_det(0, 10, "email")]
        gt = [_make_det(0, 10, "email")]
        e = PIIDetectionEvaluator()
        result = e.evaluate(preds, gt)
        assert result.micro_precision == 1.0
        assert result.micro_recall == 1.0
        assert result.micro_f1 == 1.0
        assert result.entity_metrics[0].true_positives == 1

    def test_no_predictions(self):
        gt = [_make_det(0, 10, "email")]
        e = PIIDetectionEvaluator()
        result = e.evaluate([], gt)
        assert result.micro_precision == 0.0
        assert result.micro_recall == 0.0
        assert result.micro_f1 == 0.0
        assert result.entity_metrics[0].false_negatives == 1

    def test_no_ground_truth(self):
        preds = [_make_det(0, 10, "email")]
        e = PIIDetectionEvaluator()
        result = e.evaluate(preds, [])
        assert result.micro_precision == 0.0
        assert result.micro_recall == 0.0
        assert result.entity_metrics[0].false_positives == 1

    def test_partial_overlap_above_threshold(self):
        """Overlapping spans with IoU >= 0.5 count as TP."""
        preds = [_make_det(0, 12, "email")]
        gt = [_make_det(2, 12, "email")]
        e = PIIDetectionEvaluator(iou_threshold=0.5)
        result = e.evaluate(preds, gt)
        # IoU = 10/12 = 0.833 >= 0.5 → TP
        assert result.entity_metrics[0].true_positives == 1

    def test_partial_overlap_below_threshold(self):
        """Overlapping spans with IoU < 0.5 don't match."""
        preds = [_make_det(0, 20, "email")]
        gt = [_make_det(15, 20, "email")]
        e = PIIDetectionEvaluator(iou_threshold=0.5)
        result = e.evaluate(preds, gt)
        # IoU = 5/20 = 0.25 < 0.5 → FP + FN
        assert result.entity_metrics[0].false_positives == 1
        assert result.entity_metrics[0].false_negatives == 1

    def test_mismatched_entity_types(self):
        """Same span but different entity types → FP + FN."""
        preds = [_make_det(0, 10, "email")]
        gt = [_make_det(0, 10, "phone")]
        e = PIIDetectionEvaluator()
        result = e.evaluate(preds, gt)
        email_m = next(m for m in result.entity_metrics if m.entity_type == "email")
        phone_m = next(m for m in result.entity_metrics if m.entity_type == "phone")
        assert email_m.false_positives == 1
        assert phone_m.false_negatives == 1

    def test_multiple_entities(self):
        preds = [
            _make_det(0, 10, "email"),
            _make_det(20, 30, "phone"),
        ]
        gt = [
            _make_det(0, 10, "email"),
            _make_det(20, 30, "phone"),
            _make_det(40, 50, "ssn"),
        ]
        e = PIIDetectionEvaluator()
        result = e.evaluate(preds, gt)
        assert result.micro_recall == 2 / 3  # 2 TP out of 3 GT
        assert result.micro_precision == 1.0  # 2 TP, 0 FP

    def test_macro_averaging(self):
        """Macro averages should be the mean of per-entity metrics."""
        preds = [
            _make_det(0, 10, "email"),   # TP
            _make_det(20, 30, "phone"),  # FP (no matching GT)
        ]
        gt = [
            _make_det(0, 10, "email"),   # matched
        ]
        e = PIIDetectionEvaluator()
        result = e.evaluate(preds, gt)
        # email: P=1.0 R=1.0 F1=1.0
        # phone: P=0.0 R=0.0 F1=0.0 (1 FP, 0 TP, 0 FN)
        assert result.macro_precision == 0.5
        assert result.macro_recall == 0.5
        assert result.macro_f1 == 0.5

    def test_custom_iou_threshold(self):
        e = PIIDetectionEvaluator(iou_threshold=0.9)
        preds = [_make_det(0, 11, "email")]
        gt = [_make_det(0, 10, "email")]
        # IoU = 10/11 = 0.909 >= 0.9 → TP
        result = e.evaluate(preds, gt)
        assert result.entity_metrics[0].true_positives == 1

    def test_one_prediction_matches_one_gt_only(self):
        """Each prediction should match at most one ground truth."""
        preds = [_make_det(0, 10, "email")]
        gt = [
            _make_det(0, 10, "email"),
            _make_det(0, 10, "email"),  # duplicate GT
        ]
        e = PIIDetectionEvaluator()
        result = e.evaluate(preds, gt)
        email_m = result.entity_metrics[0]
        assert email_m.true_positives == 1
        assert email_m.false_negatives == 1  # second GT unmatched
