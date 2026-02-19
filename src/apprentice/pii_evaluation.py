"""
PII Detection Evaluator — measures detection quality against labeled data.

Provides span-level matching, per-entity-type precision/recall/F1, and
overall micro/macro F1 computation.
"""

from pydantic import BaseModel, ConfigDict, Field

from apprentice.pii_detection import PIIDetection


# ===========================================================================
# Evaluation result models
# ===========================================================================


class EntityMetrics(BaseModel):
    """Precision, recall, and F1 for a single entity type."""
    model_config = ConfigDict(frozen=True)

    entity_type: str
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


class PIIEvaluationResult(BaseModel):
    """Full evaluation result across all entity types."""
    model_config = ConfigDict(frozen=True)

    entity_metrics: list[EntityMetrics] = Field(default_factory=list)
    micro_precision: float = Field(ge=0.0, le=1.0, default=0.0)
    micro_recall: float = Field(ge=0.0, le=1.0, default=0.0)
    micro_f1: float = Field(ge=0.0, le=1.0, default=0.0)
    macro_precision: float = Field(ge=0.0, le=1.0, default=0.0)
    macro_recall: float = Field(ge=0.0, le=1.0, default=0.0)
    macro_f1: float = Field(ge=0.0, le=1.0, default=0.0)
    total_predictions: int = 0
    total_ground_truth: int = 0


# ===========================================================================
# Span matching
# ===========================================================================


def _span_iou(pred_start: int, pred_end: int, gt_start: int, gt_end: int) -> float:
    """Compute Intersection over Union for two character spans."""
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0, inter_end - inter_start)
    union = max(0, (pred_end - pred_start) + (gt_end - gt_start) - intersection)
    if union == 0:
        return 0.0
    return intersection / union


def _compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ===========================================================================
# Evaluator
# ===========================================================================


class PIIDetectionEvaluator:
    """Evaluates PII detection quality against ground-truth labeled data.

    Uses span-level matching with IoU threshold for determining true positives.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self._iou_threshold = iou_threshold

    def evaluate(
        self,
        predictions: list[PIIDetection],
        ground_truth: list[PIIDetection],
    ) -> PIIEvaluationResult:
        """Compare predicted detections against ground truth.

        Args:
            predictions: Detected PII spans from the detection pipeline.
            ground_truth: Labeled PII spans (the gold standard).

        Returns:
            PIIEvaluationResult with per-entity and overall metrics.
        """
        # Collect all entity types
        all_types: set[str] = set()
        for d in predictions:
            all_types.add(d.entity_type)
        for d in ground_truth:
            all_types.add(d.entity_type)

        # Per-entity-type matching
        per_entity: dict[str, dict[str, int]] = {}
        for et in all_types:
            per_entity[et] = {"tp": 0, "fp": 0, "fn": 0}

        # Track which ground truth spans have been matched
        gt_matched: set[int] = set()

        # For each prediction, try to find a matching ground truth span
        for pred in predictions:
            matched = False
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in gt_matched:
                    continue
                if pred.entity_type != gt.entity_type:
                    continue
                iou = _span_iou(pred.start, pred.end, gt.start, gt.end)
                if iou >= self._iou_threshold:
                    per_entity[pred.entity_type]["tp"] += 1
                    gt_matched.add(gt_idx)
                    matched = True
                    break
            if not matched:
                per_entity[pred.entity_type]["fp"] += 1

        # Unmatched ground truth = false negatives
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx not in gt_matched:
                per_entity[gt.entity_type]["fn"] += 1

        # Compute per-entity metrics
        entity_metrics: list[EntityMetrics] = []
        for et in sorted(all_types):
            counts = per_entity[et]
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = _compute_f1(precision, recall)
            entity_metrics.append(EntityMetrics(
                entity_type=et,
                precision=precision,
                recall=recall,
                f1=f1,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
            ))

        # Micro averages (sum tp/fp/fn across types)
        total_tp = sum(m.true_positives for m in entity_metrics)
        total_fp = sum(m.false_positives for m in entity_metrics)
        total_fn = sum(m.false_negatives for m in entity_metrics)
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = _compute_f1(micro_precision, micro_recall)

        # Macro averages (mean of per-entity metrics)
        n_types = len(entity_metrics)
        macro_precision = sum(m.precision for m in entity_metrics) / n_types if n_types > 0 else 0.0
        macro_recall = sum(m.recall for m in entity_metrics) / n_types if n_types > 0 else 0.0
        macro_f1 = sum(m.f1 for m in entity_metrics) / n_types if n_types > 0 else 0.0

        return PIIEvaluationResult(
            entity_metrics=entity_metrics,
            micro_precision=micro_precision,
            micro_recall=micro_recall,
            micro_f1=micro_f1,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            total_predictions=len(predictions),
            total_ground_truth=len(ground_truth),
        )
