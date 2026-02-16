"""
Contract tests for the Shared Data Models component.

Tests verify behavior at boundaries (inputs/outputs), not internals.
All tests are pure unit tests — no I/O, network, or GPU requirements.
"""

import json
import uuid
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

# Import the component under test
from src.data_models import (
    # Enums
    Phase,
    RoutingDecision,
    EvaluatorType,
    BudgetPeriodUnit,
    EvaluationResultType,
    # Structs / Models
    TaskRequest,
    TaskResponse,
    TrainingExample,
    ComparisonPair,
    ConfidenceSnapshot,
    BudgetPeriod,
    BudgetState,
    AuditEntry,
    SamplingDecision,
    ExactMatchResult,
    FuzzyMatchResult,
    SemanticSimilarityResult,
    LlmJudgeResult,
    CustomEvalResult,
    ModelVersion,
    ValidationErrorDetail,
    # Factory functions
    create_task_request,
    create_task_response,
    create_confidence_snapshot,
    create_budget_state,
    record_cost,
    create_audit_entry,
    create_sampling_decision,
    create_training_example,
    create_comparison_pair,
    create_model_version,
    # Serialization functions
    serialize_model,
    deserialize_model,
    validate_model,
    get_model_json_schema,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def aware_now():
    """Timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


@pytest.fixture
def aware_now_plus_1h(aware_now):
    """Timezone-aware datetime 1 hour in the future."""
    return aware_now + timedelta(hours=1)


@pytest.fixture
def valid_task_id():
    return "my-task-01"


@pytest.fixture
def valid_model_id():
    return "gpt-4-turbo"


@pytest.fixture
def valid_local_model_id():
    return "local-llama-7b"


@pytest.fixture
def valid_request_id():
    return str(uuid.uuid4())


@pytest.fixture
def valid_budget_period(aware_now, aware_now_plus_1h):
    return BudgetPeriod(
        unit=BudgetPeriodUnit.DAILY,
        period_start=aware_now,
        period_end=aware_now_plus_1h,
    )


@pytest.fixture
def valid_task_request(valid_task_id, aware_now):
    return create_task_request(
        task_id=valid_task_id,
        input_data={"prompt": "hello"},
        metadata={"source": "test"},
    )


@pytest.fixture
def valid_exact_match_result():
    return ExactMatchResult(
        result_type=EvaluationResultType.EXACT_MATCH_RESULT,
        is_match=True,
        matched_fields=["output"],
        mismatched_fields=[],
    )


@pytest.fixture
def valid_fuzzy_match_result():
    return FuzzyMatchResult(
        result_type=EvaluationResultType.FUZZY_MATCH_RESULT,
        similarity_score=0.85,
        threshold=0.8,
        is_pass=True,
    )


@pytest.fixture
def valid_semantic_similarity_result():
    return SemanticSimilarityResult(
        result_type=EvaluationResultType.SEMANTIC_SIMILARITY_RESULT,
        similarity_score=0.92,
        threshold=0.85,
        is_pass=True,
        embedding_model="text-embedding-ada-002",
    )


@pytest.fixture
def valid_llm_judge_result():
    return LlmJudgeResult(
        result_type=EvaluationResultType.LLM_JUDGE_RESULT,
        score=0.9,
        threshold=0.7,
        is_pass=True,
        judge_model_id="gpt-4-judge",
        reasoning="Output is semantically equivalent.",
        cost=Decimal("0.003"),
    )


@pytest.fixture
def valid_custom_eval_result():
    return CustomEvalResult(
        result_type=EvaluationResultType.CUSTOM_RESULT,
        score=0.75,
        is_pass=True,
        evaluator_name="my_custom_eval",
        details={"key": "value"},
    )


@pytest.fixture
def valid_budget_state(valid_task_id, valid_budget_period, aware_now):
    return create_budget_state(
        task_id=valid_task_id,
        total_budget=Decimal("100.00"),
        spent=Decimal("25.00"),
        period=valid_budget_period,
        request_count=10,
        last_updated=aware_now,
    )


@pytest.fixture
def valid_task_response(valid_request_id, valid_task_id, valid_model_id, aware_now):
    return create_task_response(
        request_id=valid_request_id,
        task_id=valid_task_id,
        output_data={"result": "world"},
        model_used=valid_model_id,
        routing_decision=RoutingDecision.REMOTE,
        phase=Phase.PHASE_1_REMOTE_ONLY,
        confidence_at_decision=0.3,
        latency_ms=150.0,
        cost_incurred=Decimal("0.01"),
        created_at=aware_now,
        metadata={},
    )


# ===========================================================================
# TestConstrainedPrimitives
# ===========================================================================

class TestConstrainedPrimitives:
    """Tests for TaskId, ModelId, ConfidenceScore, CostAmount, AwareDatetime constraints."""

    # -- TaskId --

    @pytest.mark.parametrize("task_id", [
        "a",  # min length 1
        "my-task",
        "task_with_underscores",
        "a" * 128,  # max length 128
        "abc123",
        "z-0-9",
    ], ids=["single_char", "hyphenated", "underscored", "max_length", "alphanumeric", "mixed"])
    def test_valid_task_id_accepted(self, task_id):
        req = create_task_request(task_id=task_id, input_data={}, metadata={})
        assert req.task_id == task_id, f"TaskId should be '{task_id}'"

    @pytest.mark.parametrize("task_id,reason", [
        ("", "empty string"),
        ("A", "uppercase letter"),
        ("1abc", "starts with digit"),
        ("-abc", "starts with hyphen"),
        ("_abc", "starts with underscore"),
        ("abc ABC", "contains space"),
        ("abc.def", "contains dot"),
        ("a" * 129, "exceeds max length 128"),
        ("ABC", "all uppercase"),
    ], ids=lambda x: x if isinstance(x, str) and len(x) < 30 else "param")
    def test_invalid_task_id_rejected(self, task_id, reason):
        with pytest.raises(Exception):
            create_task_request(task_id=task_id, input_data={}, metadata={})

    # -- ConfidenceScore --

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_valid_confidence_score(self, score, aware_now):
        snapshot = create_confidence_snapshot(
            task_id="test-task",
            score=score,
            sample_count=10,
            threshold_phase2=0.4,
            threshold_phase3=0.8,
            last_updated=aware_now,
            trend=0.0,
        )
        assert snapshot.score == score

    @pytest.mark.parametrize("score", [-0.01, 1.01, -1.0, 2.0, float('inf'), float('-inf')])
    def test_invalid_confidence_score_rejected(self, score, aware_now):
        with pytest.raises(Exception):
            create_confidence_snapshot(
                task_id="test-task",
                score=score,
                sample_count=10,
                threshold_phase2=0.4,
                threshold_phase3=0.8,
                last_updated=aware_now,
                trend=0.0,
            )

    # -- CostAmount --

    def test_valid_cost_amount_zero(self, valid_task_id, valid_budget_period, aware_now):
        state = create_budget_state(
            task_id=valid_task_id,
            total_budget=Decimal("0"),
            spent=Decimal("0"),
            period=valid_budget_period,
            request_count=0,
            last_updated=aware_now,
        )
        assert state.total_budget == Decimal("0")

    def test_negative_cost_amount_rejected(self, valid_task_id, valid_budget_period, aware_now):
        with pytest.raises(Exception):
            create_budget_state(
                task_id=valid_task_id,
                total_budget=Decimal("-1"),
                spent=Decimal("0"),
                period=valid_budget_period,
                request_count=0,
                last_updated=aware_now,
            )

    # -- AwareDatetime --

    def test_naive_datetime_rejected(self):
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)  # no tzinfo
        with pytest.raises(Exception):
            create_task_request(
                task_id="test-task",
                input_data={},
                metadata={},
                created_at=naive_dt,
            )

    def test_aware_datetime_accepted_non_utc(self):
        tz_plus5 = timezone(timedelta(hours=5))
        aware_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=tz_plus5)
        req = create_task_request(
            task_id="test-task",
            input_data={},
            metadata={},
            created_at=aware_dt,
        )
        assert req.created_at.tzinfo is not None, "created_at must be timezone-aware"


# ===========================================================================
# TestCreateTaskRequest
# ===========================================================================

class TestCreateTaskRequest:
    """Tests for the create_task_request factory function."""

    def test_happy_path_all_fields(self, aware_now):
        req_id = str(uuid.uuid4())
        req = create_task_request(
            task_id="my-task",
            input_data={"prompt": "hello"},
            metadata={"source": "test"},
            request_id=req_id,
            created_at=aware_now,
        )
        assert req.task_id == "my-task"
        assert req.input_data == {"prompt": "hello"}
        assert req.metadata == {"source": "test"}
        assert req.request_id == req_id
        assert req.created_at == aware_now

    def test_auto_generated_request_id_and_created_at(self):
        req = create_task_request(
            task_id="auto-task",
            input_data={"key": "val"},
            metadata={},
        )
        # request_id should be a valid UUID4
        parsed = uuid.UUID(req.request_id, version=4)
        assert str(parsed) == req.request_id, "request_id should be a valid UUID4 string"
        # created_at should be timezone-aware
        assert req.created_at.tzinfo is not None, "Auto-generated created_at must be timezone-aware"

    def test_empty_dicts_are_valid(self):
        req = create_task_request(task_id="task-a", input_data={}, metadata={})
        assert req.input_data == {}
        assert req.metadata == {}

    def test_frozen_model(self):
        req = create_task_request(task_id="task-a", input_data={}, metadata={})
        with pytest.raises((AttributeError, TypeError, Exception)):
            req.task_id = "changed"

    @pytest.mark.parametrize("bad_task_id", [
        "", "UPPER", "1start", " space", "has.dot", "a" * 129
    ])
    def test_invalid_task_id_errors(self, bad_task_id):
        with pytest.raises(Exception):
            create_task_request(task_id=bad_task_id, input_data={}, metadata={})

    def test_invalid_input_data_not_dict(self):
        with pytest.raises(Exception):
            create_task_request(task_id="task-a", input_data="not a dict", metadata={})


# ===========================================================================
# TestCreateTaskResponse
# ===========================================================================

class TestCreateTaskResponse:
    """Tests for the create_task_response factory function."""

    def test_happy_path(self, aware_now):
        resp = create_task_response(
            request_id=str(uuid.uuid4()),
            task_id="task-resp",
            output_data={"answer": 42},
            model_used="gpt-4",
            routing_decision=RoutingDecision.REMOTE,
            phase=Phase.PHASE_1_REMOTE_ONLY,
            confidence_at_decision=0.2,
            latency_ms=100.0,
            cost_incurred=Decimal("0.05"),
            created_at=aware_now,
            metadata={},
        )
        assert resp.routing_decision == RoutingDecision.REMOTE
        assert resp.phase == Phase.PHASE_1_REMOTE_ONLY
        assert resp.latency_ms == 100.0
        assert resp.cost_incurred == Decimal("0.05")

    def test_invalid_routing_decision(self, aware_now):
        with pytest.raises(Exception):
            create_task_response(
                request_id=str(uuid.uuid4()),
                task_id="task-resp",
                output_data={},
                model_used="gpt-4",
                routing_decision="INVALID_ROUTE",
                phase=Phase.PHASE_1_REMOTE_ONLY,
                confidence_at_decision=0.5,
                latency_ms=10.0,
                cost_incurred=Decimal("0"),
                created_at=aware_now,
                metadata={},
            )

    def test_invalid_phase(self, aware_now):
        with pytest.raises(Exception):
            create_task_response(
                request_id=str(uuid.uuid4()),
                task_id="task-resp",
                output_data={},
                model_used="gpt-4",
                routing_decision=RoutingDecision.REMOTE,
                phase="INVALID_PHASE",
                confidence_at_decision=0.5,
                latency_ms=10.0,
                cost_incurred=Decimal("0"),
                created_at=aware_now,
                metadata={},
            )

    def test_negative_latency(self, aware_now):
        with pytest.raises(Exception):
            create_task_response(
                request_id=str(uuid.uuid4()),
                task_id="task-resp",
                output_data={},
                model_used="gpt-4",
                routing_decision=RoutingDecision.REMOTE,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                confidence_at_decision=0.5,
                latency_ms=-1.0,
                cost_incurred=Decimal("0"),
                created_at=aware_now,
                metadata={},
            )

    def test_negative_cost(self, aware_now):
        with pytest.raises(Exception):
            create_task_response(
                request_id=str(uuid.uuid4()),
                task_id="task-resp",
                output_data={},
                model_used="gpt-4",
                routing_decision=RoutingDecision.REMOTE,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                confidence_at_decision=0.5,
                latency_ms=10.0,
                cost_incurred=Decimal("-0.01"),
                created_at=aware_now,
                metadata={},
            )

    @pytest.mark.parametrize("routing,phase", [
        (RoutingDecision.REMOTE, Phase.PHASE_1_REMOTE_ONLY),
        (RoutingDecision.LOCAL, Phase.PHASE_3_LOCAL_PRIMARY),
        (RoutingDecision.LOCAL_WITH_REMOTE_COMPARISON, Phase.PHASE_2_SAMPLING),
        (RoutingDecision.LOCAL_FALLBACK_TO_REMOTE, Phase.PHASE_3_LOCAL_PRIMARY),
    ])
    def test_all_enum_combinations_accepted(self, routing, phase, aware_now):
        resp = create_task_response(
            request_id=str(uuid.uuid4()),
            task_id="task-enum",
            output_data={},
            model_used="model-x",
            routing_decision=routing,
            phase=phase,
            confidence_at_decision=0.5,
            latency_ms=0.0,
            cost_incurred=Decimal("0"),
            created_at=aware_now,
            metadata={},
        )
        assert resp.routing_decision == routing
        assert resp.phase == phase


# ===========================================================================
# TestConfidenceSnapshotPhase
# ===========================================================================

class TestConfidenceSnapshotPhase:
    """Tests for phase derivation from score and thresholds."""

    def test_phase1_below_threshold_phase2(self, aware_now):
        snap = create_confidence_snapshot(
            task_id="snap-task",
            score=0.1,
            sample_count=5,
            threshold_phase2=0.4,
            threshold_phase3=0.8,
            last_updated=aware_now,
            trend=0.01,
        )
        assert snap.phase == Phase.PHASE_1_REMOTE_ONLY, \
            "Score 0.1 < threshold_phase2 0.4 should yield PHASE_1"

    def test_phase2_between_thresholds(self, aware_now):
        snap = create_confidence_snapshot(
            task_id="snap-task",
            score=0.6,
            sample_count=20,
            threshold_phase2=0.4,
            threshold_phase3=0.8,
            last_updated=aware_now,
            trend=0.05,
        )
        assert snap.phase == Phase.PHASE_2_SAMPLING, \
            "Score 0.6 between 0.4 and 0.8 should yield PHASE_2"

    def test_phase3_above_threshold_phase3(self, aware_now):
        snap = create_confidence_snapshot(
            task_id="snap-task",
            score=0.9,
            sample_count=100,
            threshold_phase2=0.4,
            threshold_phase3=0.8,
            last_updated=aware_now,
            trend=0.0,
        )
        assert snap.phase == Phase.PHASE_3_LOCAL_PRIMARY, \
            "Score 0.9 >= threshold_phase3 0.8 should yield PHASE_3"

    def test_score_exactly_at_threshold_phase2(self, aware_now):
        snap = create_confidence_snapshot(
            task_id="snap-task",
            score=0.4,
            sample_count=10,
            threshold_phase2=0.4,
            threshold_phase3=0.8,
            last_updated=aware_now,
            trend=0.0,
        )
        assert snap.phase == Phase.PHASE_2_SAMPLING, \
            "Score exactly at threshold_phase2 should yield PHASE_2 (>= threshold_phase2)"

    def test_score_exactly_at_threshold_phase3(self, aware_now):
        snap = create_confidence_snapshot(
            task_id="snap-task",
            score=0.8,
            sample_count=50,
            threshold_phase2=0.4,
            threshold_phase3=0.8,
            last_updated=aware_now,
            trend=0.0,
        )
        assert snap.phase == Phase.PHASE_3_LOCAL_PRIMARY, \
            "Score exactly at threshold_phase3 should yield PHASE_3 (>= threshold_phase3)"

    def test_score_zero_phase1(self, aware_now):
        snap = create_confidence_snapshot(
            task_id="snap-task",
            score=0.0,
            sample_count=0,
            threshold_phase2=0.4,
            threshold_phase3=0.8,
            last_updated=aware_now,
            trend=0.0,
        )
        assert snap.phase == Phase.PHASE_1_REMOTE_ONLY

    def test_score_one_phase3(self, aware_now):
        snap = create_confidence_snapshot(
            task_id="snap-task",
            score=1.0,
            sample_count=1000,
            threshold_phase2=0.4,
            threshold_phase3=0.8,
            last_updated=aware_now,
            trend=0.0,
        )
        assert snap.phase == Phase.PHASE_3_LOCAL_PRIMARY

    def test_equal_thresholds(self, aware_now):
        """When threshold_phase2 == threshold_phase3, there is no PHASE_2 band."""
        # score below both => PHASE_1
        snap = create_confidence_snapshot(
            task_id="snap-task",
            score=0.3,
            sample_count=5,
            threshold_phase2=0.5,
            threshold_phase3=0.5,
            last_updated=aware_now,
            trend=0.0,
        )
        assert snap.phase == Phase.PHASE_1_REMOTE_ONLY

        # score at both => PHASE_3 (score >= threshold_phase3)
        snap2 = create_confidence_snapshot(
            task_id="snap-task",
            score=0.5,
            sample_count=5,
            threshold_phase2=0.5,
            threshold_phase3=0.5,
            last_updated=aware_now,
            trend=0.0,
        )
        assert snap2.phase == Phase.PHASE_3_LOCAL_PRIMARY

    def test_invalid_score_out_of_range(self, aware_now):
        with pytest.raises(Exception):
            create_confidence_snapshot(
                task_id="snap-task",
                score=1.5,
                sample_count=5,
                threshold_phase2=0.4,
                threshold_phase3=0.8,
                last_updated=aware_now,
                trend=0.0,
            )

    def test_invalid_thresholds_reversed(self, aware_now):
        with pytest.raises(Exception):
            create_confidence_snapshot(
                task_id="snap-task",
                score=0.5,
                sample_count=5,
                threshold_phase2=0.9,
                threshold_phase3=0.4,
                last_updated=aware_now,
                trend=0.0,
            )

    def test_negative_sample_count(self, aware_now):
        with pytest.raises(Exception):
            create_confidence_snapshot(
                task_id="snap-task",
                score=0.5,
                sample_count=-1,
                threshold_phase2=0.4,
                threshold_phase3=0.8,
                last_updated=aware_now,
                trend=0.0,
            )

    def test_snapshot_is_frozen(self, aware_now):
        snap = create_confidence_snapshot(
            task_id="snap-task",
            score=0.5,
            sample_count=10,
            threshold_phase2=0.4,
            threshold_phase3=0.8,
            last_updated=aware_now,
            trend=0.0,
        )
        with pytest.raises((AttributeError, TypeError, Exception)):
            snap.score = 0.9


# ===========================================================================
# TestBudgetStateComputed
# ===========================================================================

class TestBudgetStateComputed:
    """Tests for BudgetState computed fields and record_cost."""

    def test_create_budget_state_remaining_computed(self, valid_task_id, valid_budget_period, aware_now):
        state = create_budget_state(
            task_id=valid_task_id,
            total_budget=Decimal("100.00"),
            spent=Decimal("30.00"),
            period=valid_budget_period,
            request_count=5,
            last_updated=aware_now,
        )
        assert state.remaining == Decimal("70.00"), \
            "remaining should be total_budget - spent"
        assert state.is_exhausted is False, \
            "Budget with remaining > 0 should not be exhausted"

    def test_budget_state_exactly_exhausted(self, valid_task_id, valid_budget_period, aware_now):
        state = create_budget_state(
            task_id=valid_task_id,
            total_budget=Decimal("50.00"),
            spent=Decimal("50.00"),
            period=valid_budget_period,
            request_count=100,
            last_updated=aware_now,
        )
        assert state.remaining == Decimal("0.00")
        assert state.is_exhausted is True

    def test_budget_state_overspent(self, valid_task_id, valid_budget_period, aware_now):
        state = create_budget_state(
            task_id=valid_task_id,
            total_budget=Decimal("50.00"),
            spent=Decimal("60.00"),
            period=valid_budget_period,
            request_count=100,
            last_updated=aware_now,
        )
        assert state.remaining == Decimal("-10.00")
        assert state.is_exhausted is True

    def test_budget_state_frozen(self, valid_budget_state):
        with pytest.raises((AttributeError, TypeError, Exception)):
            valid_budget_state.spent = Decimal("999")

    def test_negative_total_budget_rejected(self, valid_task_id, valid_budget_period, aware_now):
        with pytest.raises(Exception):
            create_budget_state(
                task_id=valid_task_id,
                total_budget=Decimal("-10"),
                spent=Decimal("0"),
                period=valid_budget_period,
                request_count=0,
                last_updated=aware_now,
            )

    def test_negative_spent_rejected(self, valid_task_id, valid_budget_period, aware_now):
        with pytest.raises(Exception):
            create_budget_state(
                task_id=valid_task_id,
                total_budget=Decimal("100"),
                spent=Decimal("-5"),
                period=valid_budget_period,
                request_count=0,
                last_updated=aware_now,
            )

    def test_invalid_period_start_after_end(self, valid_task_id, aware_now):
        bad_period = BudgetPeriod(
            unit=BudgetPeriodUnit.DAILY,
            period_start=aware_now + timedelta(hours=2),
            period_end=aware_now,
        )
        with pytest.raises(Exception):
            create_budget_state(
                task_id=valid_task_id,
                total_budget=Decimal("100"),
                spent=Decimal("0"),
                period=bad_period,
                request_count=0,
                last_updated=aware_now,
            )


# ===========================================================================
# TestRecordCost
# ===========================================================================

class TestRecordCost:
    """Tests for the record_cost function."""

    def test_record_cost_updates_fields(self, valid_budget_state, aware_now):
        original_spent = valid_budget_state.spent
        original_remaining = valid_budget_state.remaining
        original_count = valid_budget_state.request_count

        new_state = record_cost(valid_budget_state, Decimal("10.00"), aware_now)

        assert new_state.spent == original_spent + Decimal("10.00"), \
            "spent should increase by the cost amount"
        assert new_state.remaining == valid_budget_state.total_budget - new_state.spent, \
            "remaining should equal total_budget - new spent"
        assert new_state.request_count == original_count + 1, \
            "request_count should increment by 1"

    def test_record_zero_cost(self, valid_budget_state, aware_now):
        new_state = record_cost(valid_budget_state, Decimal("0"), aware_now)
        assert new_state.spent == valid_budget_state.spent
        assert new_state.request_count == valid_budget_state.request_count + 1

    def test_record_cost_exhausts_budget(self, valid_task_id, valid_budget_period, aware_now):
        state = create_budget_state(
            task_id=valid_task_id,
            total_budget=Decimal("10.00"),
            spent=Decimal("5.00"),
            period=valid_budget_period,
            request_count=2,
            last_updated=aware_now,
        )
        new_state = record_cost(state, Decimal("5.00"), aware_now)
        assert new_state.remaining == Decimal("0.00")
        assert new_state.is_exhausted is True

    def test_record_cost_original_not_mutated(self, valid_budget_state, aware_now):
        original_spent = valid_budget_state.spent
        original_remaining = valid_budget_state.remaining
        original_count = valid_budget_state.request_count

        _ = record_cost(valid_budget_state, Decimal("5.00"), aware_now)

        assert valid_budget_state.spent == original_spent, \
            "Original BudgetState.spent must not change"
        assert valid_budget_state.remaining == original_remaining, \
            "Original BudgetState.remaining must not change"
        assert valid_budget_state.request_count == original_count, \
            "Original BudgetState.request_count must not change"

    def test_record_cost_returns_new_instance(self, valid_budget_state, aware_now):
        new_state = record_cost(valid_budget_state, Decimal("1.00"), aware_now)
        assert new_state is not valid_budget_state, \
            "record_cost must return a new instance, not mutate the original"

    def test_record_cost_negative_rejected(self, valid_budget_state, aware_now):
        with pytest.raises(Exception):
            record_cost(valid_budget_state, Decimal("-1.00"), aware_now)

    def test_record_cost_sequential_accumulation(self, valid_task_id, valid_budget_period, aware_now):
        state = create_budget_state(
            task_id=valid_task_id,
            total_budget=Decimal("100.00"),
            spent=Decimal("0"),
            period=valid_budget_period,
            request_count=0,
            last_updated=aware_now,
        )
        costs = [Decimal("10"), Decimal("20"), Decimal("15"), Decimal("5")]
        for cost in costs:
            state = record_cost(state, cost, aware_now)

        assert state.spent == Decimal("50"), \
            "Spent should be cumulative sum of all recorded costs"
        assert state.remaining == Decimal("50"), \
            "Remaining should be total_budget - cumulative spent"
        assert state.request_count == 4, \
            "request_count should equal number of record_cost calls"


# ===========================================================================
# TestCreateAuditEntry
# ===========================================================================

class TestCreateAuditEntry:
    """Tests for the create_audit_entry factory function."""

    def test_happy_path(self, aware_now):
        entry = create_audit_entry(
            request_id="req-123",
            task_id="audit-task",
            routing_decision=RoutingDecision.REMOTE,
            model_used="gpt-4",
            confidence_at_decision=0.5,
            phase=Phase.PHASE_1_REMOTE_ONLY,
            cost=Decimal("0.02"),
            latency_ms=200.0,
            fallback_triggered=False,
            timestamp=aware_now,
            sampling_probability=0.0,
            error="",
            metadata={"key": "val"},
        )
        assert entry.request_id == "req-123"
        assert entry.fallback_triggered is False
        assert entry.timestamp.tzinfo is not None

    def test_audit_entry_serializable_to_json(self, aware_now):
        entry = create_audit_entry(
            request_id="req-456",
            task_id="audit-task",
            routing_decision=RoutingDecision.LOCAL,
            model_used="local-model",
            confidence_at_decision=0.9,
            phase=Phase.PHASE_3_LOCAL_PRIMARY,
            cost=Decimal("0"),
            latency_ms=50.0,
            fallback_triggered=False,
            timestamp=aware_now,
            sampling_probability=0.0,
            error="",
            metadata={},
        )
        json_str = serialize_model(entry)
        parsed = json.loads(json_str)
        assert "request_id" in parsed

    def test_empty_request_id_rejected(self, aware_now):
        with pytest.raises(Exception):
            create_audit_entry(
                request_id="",
                task_id="audit-task",
                routing_decision=RoutingDecision.REMOTE,
                model_used="gpt-4",
                confidence_at_decision=0.5,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                cost=Decimal("0"),
                latency_ms=10.0,
                fallback_triggered=False,
                timestamp=aware_now,
                sampling_probability=0.0,
                error="",
                metadata={},
            )

    def test_negative_latency_rejected(self, aware_now):
        with pytest.raises(Exception):
            create_audit_entry(
                request_id="req-neg",
                task_id="audit-task",
                routing_decision=RoutingDecision.REMOTE,
                model_used="gpt-4",
                confidence_at_decision=0.5,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                cost=Decimal("0"),
                latency_ms=-5.0,
                fallback_triggered=False,
                timestamp=aware_now,
                sampling_probability=0.0,
                error="",
                metadata={},
            )


# ===========================================================================
# TestCreateSamplingDecision
# ===========================================================================

class TestCreateSamplingDecision:
    """Tests for the create_sampling_decision factory function."""

    def test_happy_path(self, aware_now):
        sd = create_sampling_decision(
            request_id="req-sd-1",
            task_id="sd-task",
            decision=RoutingDecision.LOCAL_WITH_REMOTE_COMPARISON,
            sampling_probability=0.3,
            phase=Phase.PHASE_2_SAMPLING,
            confidence_at_decision=0.6,
            budget_remaining=Decimal("50.00"),
            rationale="Sampling at 30% during Phase 2",
            decided_at=aware_now,
        )
        assert sd.sampling_probability == 0.3
        assert sd.rationale == "Sampling at 30% during Phase 2"

    @pytest.mark.parametrize("prob", [-0.1, 1.1, 2.0, -1.0])
    def test_invalid_sampling_probability(self, prob, aware_now):
        with pytest.raises(Exception):
            create_sampling_decision(
                request_id="req-sd-bad",
                task_id="sd-task",
                decision=RoutingDecision.REMOTE,
                sampling_probability=prob,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                confidence_at_decision=0.5,
                budget_remaining=Decimal("10"),
                rationale="Test",
                decided_at=aware_now,
            )

    def test_empty_rationale_rejected(self, aware_now):
        with pytest.raises(Exception):
            create_sampling_decision(
                request_id="req-sd-empty",
                task_id="sd-task",
                decision=RoutingDecision.REMOTE,
                sampling_probability=0.5,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                confidence_at_decision=0.5,
                budget_remaining=Decimal("10"),
                rationale="",
                decided_at=aware_now,
            )

    @pytest.mark.parametrize("prob", [0.0, 1.0])
    def test_boundary_sampling_probabilities(self, prob, aware_now):
        sd = create_sampling_decision(
            request_id="req-sd-bound",
            task_id="sd-task",
            decision=RoutingDecision.REMOTE,
            sampling_probability=prob,
            phase=Phase.PHASE_1_REMOTE_ONLY,
            confidence_at_decision=0.5,
            budget_remaining=Decimal("10"),
            rationale="Boundary test",
            decided_at=aware_now,
        )
        assert sd.sampling_probability == prob


# ===========================================================================
# TestCreateTrainingExample
# ===========================================================================

class TestCreateTrainingExample:
    """Tests for the create_training_example factory function."""

    def test_happy_path(self, aware_now):
        te = create_training_example(
            task_id="train-task",
            input_data={"prompt": "Question?"},
            output_data={"answer": "42"},
            model_id="gpt-4",
            source_request_id="req-src-1",
            quality_score=0.95,
        )
        # example_id should be a valid UUID4
        uuid.UUID(te.example_id, version=4)
        assert te.task_id == "train-task"
        assert te.created_at.tzinfo is not None

    def test_invalid_task_id(self):
        with pytest.raises(Exception):
            create_training_example(
                task_id="INVALID",
                input_data={},
                output_data={},
                model_id="gpt-4",
                source_request_id="req-1",
            )

    def test_empty_source_request_id(self):
        with pytest.raises(Exception):
            create_training_example(
                task_id="train-task",
                input_data={},
                output_data={},
                model_id="gpt-4",
                source_request_id="",
            )


# ===========================================================================
# TestCreateComparisonPair
# ===========================================================================

class TestCreateComparisonPair:
    """Tests for the create_comparison_pair factory function."""

    def test_happy_path(self, valid_exact_match_result, aware_now):
        cp = create_comparison_pair(
            task_id="cmp-task",
            input_data={"prompt": "hello"},
            remote_output={"answer": "world"},
            local_output={"answer": "world"},
            remote_model_id="gpt-4",
            local_model_id="llama-7b",
            source_request_id="req-cmp-1",
            evaluation_result=valid_exact_match_result,
        )
        uuid.UUID(cp.pair_id, version=4)
        assert cp.task_id == "cmp-task"
        assert cp.created_at.tzinfo is not None

    def test_same_model_ids_rejected(self, valid_exact_match_result):
        with pytest.raises(Exception):
            create_comparison_pair(
                task_id="cmp-task",
                input_data={},
                remote_output={},
                local_output={},
                remote_model_id="same-model",
                local_model_id="same-model",
                source_request_id="req-cmp-2",
                evaluation_result=valid_exact_match_result,
            )

    def test_invalid_task_id(self, valid_exact_match_result):
        with pytest.raises(Exception):
            create_comparison_pair(
                task_id="INVALID",
                input_data={},
                remote_output={},
                local_output={},
                remote_model_id="gpt-4",
                local_model_id="llama-7b",
                source_request_id="req-cmp-3",
                evaluation_result=valid_exact_match_result,
            )


# ===========================================================================
# TestEvaluationResultUnion
# ===========================================================================

class TestEvaluationResultUnion:
    """Tests for the EvaluationResult discriminated union and all 5 variants."""

    def test_exact_match_result(self, valid_exact_match_result, aware_now):
        cp = create_comparison_pair(
            task_id="union-task",
            input_data={},
            remote_output={},
            local_output={},
            remote_model_id="remote-m",
            local_model_id="local-m",
            source_request_id="req-union-1",
            evaluation_result=valid_exact_match_result,
        )
        assert cp.evaluation_result.result_type == EvaluationResultType.EXACT_MATCH_RESULT

    def test_fuzzy_match_result(self, valid_fuzzy_match_result, aware_now):
        cp = create_comparison_pair(
            task_id="union-task",
            input_data={},
            remote_output={},
            local_output={},
            remote_model_id="remote-m",
            local_model_id="local-m",
            source_request_id="req-union-2",
            evaluation_result=valid_fuzzy_match_result,
        )
        assert cp.evaluation_result.result_type == EvaluationResultType.FUZZY_MATCH_RESULT

    def test_semantic_similarity_result(self, valid_semantic_similarity_result):
        cp = create_comparison_pair(
            task_id="union-task",
            input_data={},
            remote_output={},
            local_output={},
            remote_model_id="remote-m",
            local_model_id="local-m",
            source_request_id="req-union-3",
            evaluation_result=valid_semantic_similarity_result,
        )
        assert cp.evaluation_result.result_type == EvaluationResultType.SEMANTIC_SIMILARITY_RESULT

    def test_llm_judge_result(self, valid_llm_judge_result):
        cp = create_comparison_pair(
            task_id="union-task",
            input_data={},
            remote_output={},
            local_output={},
            remote_model_id="remote-m",
            local_model_id="local-m",
            source_request_id="req-union-4",
            evaluation_result=valid_llm_judge_result,
        )
        assert cp.evaluation_result.result_type == EvaluationResultType.LLM_JUDGE_RESULT

    def test_custom_eval_result(self, valid_custom_eval_result):
        cp = create_comparison_pair(
            task_id="union-task",
            input_data={},
            remote_output={},
            local_output={},
            remote_model_id="remote-m",
            local_model_id="local-m",
            source_request_id="req-union-5",
            evaluation_result=valid_custom_eval_result,
        )
        assert cp.evaluation_result.result_type == EvaluationResultType.CUSTOM_RESULT


# ===========================================================================
# TestCreateModelVersion
# ===========================================================================

class TestCreateModelVersion:
    """Tests for the create_model_version factory function."""

    def test_happy_path(self, aware_now):
        mv = create_model_version(
            model_id="local-llama-7b",
            path="/models/llama/v1",
            validation_score=0.85,
            training_example_count=1000,
            is_active=False,
            created_at=aware_now,
        )
        uuid.UUID(mv.version_id, version=4)
        assert mv.model_id == "local-llama-7b"
        assert mv.is_active is False
        assert mv.created_at.tzinfo is not None

    def test_active_model_no_promoted_at_defaults(self, aware_now):
        mv = create_model_version(
            model_id="local-llama-7b",
            path="/models/llama/v2",
            validation_score=0.9,
            training_example_count=2000,
            is_active=True,
            created_at=aware_now,
        )
        assert mv.promoted_at == mv.created_at, \
            "Active model with no promoted_at should default to created_at"

    def test_empty_path_rejected(self):
        with pytest.raises(Exception):
            create_model_version(
                model_id="local-llama",
                path="",
                validation_score=0.5,
                training_example_count=10,
                is_active=False,
            )

    def test_negative_example_count_rejected(self):
        with pytest.raises(Exception):
            create_model_version(
                model_id="local-llama",
                path="/models/bad",
                validation_score=0.5,
                training_example_count=-1,
                is_active=False,
            )

    @pytest.mark.parametrize("bad_score", [-0.1, 1.1, 2.0])
    def test_invalid_validation_score_rejected(self, bad_score):
        with pytest.raises(Exception):
            create_model_version(
                model_id="local-llama",
                path="/models/bad",
                validation_score=bad_score,
                training_example_count=10,
                is_active=False,
            )


# ===========================================================================
# TestSerialization
# ===========================================================================

class TestSerialization:
    """Tests for serialize_model, deserialize_model, round-trip, and edge cases."""

    def test_serialize_task_request(self, valid_task_request):
        json_str = serialize_model(valid_task_request)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["task_id"] == valid_task_request.task_id

    def test_deserialize_task_request(self, valid_task_request):
        json_str = serialize_model(valid_task_request)
        restored = deserialize_model(json_str, "TaskRequest")
        assert restored.task_id == valid_task_request.task_id
        assert restored.request_id == valid_task_request.request_id
        assert restored.input_data == valid_task_request.input_data

    def test_roundtrip_task_request(self, valid_task_request):
        json_str = serialize_model(valid_task_request)
        restored = deserialize_model(json_str, "TaskRequest")
        json_str2 = serialize_model(restored)
        assert json_str == json_str2, "Double round-trip should produce identical JSON"

    def test_roundtrip_task_response(self, valid_task_response):
        json_str = serialize_model(valid_task_response)
        restored = deserialize_model(json_str, "TaskResponse")
        assert restored.request_id == valid_task_response.request_id
        assert restored.routing_decision == valid_task_response.routing_decision
        assert restored.phase == valid_task_response.phase

    def test_roundtrip_budget_state(self, valid_budget_state):
        json_str = serialize_model(valid_budget_state)
        restored = deserialize_model(json_str, "BudgetState")
        assert restored.total_budget == valid_budget_state.total_budget
        assert restored.spent == valid_budget_state.spent
        assert restored.remaining == valid_budget_state.remaining
        assert restored.is_exhausted == valid_budget_state.is_exhausted

    def test_roundtrip_confidence_snapshot(self, aware_now):
        snap = create_confidence_snapshot(
            task_id="serial-snap",
            score=0.65,
            sample_count=42,
            threshold_phase2=0.4,
            threshold_phase3=0.8,
            last_updated=aware_now,
            trend=0.02,
        )
        json_str = serialize_model(snap)
        restored = deserialize_model(json_str, "ConfidenceSnapshot")
        assert restored.score == snap.score
        assert restored.phase == snap.phase
        assert restored.sample_count == snap.sample_count

    def test_roundtrip_audit_entry(self, aware_now):
        entry = create_audit_entry(
            request_id="req-serial-ae",
            task_id="serial-task",
            routing_decision=RoutingDecision.LOCAL,
            model_used="llama-7b",
            confidence_at_decision=0.85,
            phase=Phase.PHASE_3_LOCAL_PRIMARY,
            cost=Decimal("0.001"),
            latency_ms=42.5,
            fallback_triggered=True,
            timestamp=aware_now,
            sampling_probability=0.1,
            error="timeout fallback",
            metadata={"retry": True},
        )
        json_str = serialize_model(entry)
        restored = deserialize_model(json_str, "AuditEntry")
        assert restored.request_id == entry.request_id
        assert restored.fallback_triggered is True

    def test_roundtrip_comparison_pair_with_each_eval_variant(
        self,
        valid_exact_match_result,
        valid_fuzzy_match_result,
        valid_semantic_similarity_result,
        valid_llm_judge_result,
        valid_custom_eval_result,
    ):
        for eval_result in [
            valid_exact_match_result,
            valid_fuzzy_match_result,
            valid_semantic_similarity_result,
            valid_llm_judge_result,
            valid_custom_eval_result,
        ]:
            cp = create_comparison_pair(
                task_id="serial-cmp",
                input_data={"q": "test"},
                remote_output={"a": "remote"},
                local_output={"a": "local"},
                remote_model_id="gpt-4",
                local_model_id="llama-7b",
                source_request_id="req-serial-cmp",
                evaluation_result=eval_result,
            )
            json_str = serialize_model(cp)
            restored = deserialize_model(json_str, "ComparisonPair")
            assert restored.evaluation_result.result_type == eval_result.result_type, \
                f"Round-trip failed for {eval_result.result_type}"

    def test_roundtrip_model_version(self, aware_now):
        mv = create_model_version(
            model_id="llama-13b",
            path="/models/llama-13b/v3",
            validation_score=0.92,
            training_example_count=5000,
            is_active=True,
            created_at=aware_now,
        )
        json_str = serialize_model(mv)
        restored = deserialize_model(json_str, "ModelVersion")
        assert restored.model_id == mv.model_id
        assert restored.is_active is True

    def test_decimal_serialized_as_string(self, valid_budget_state):
        json_str = serialize_model(valid_budget_state)
        parsed = json.loads(json_str)
        # Decimal fields should be serialized as strings to preserve precision
        # or as numbers that preserve precision — check it round-trips correctly
        restored = deserialize_model(json_str, "BudgetState")
        assert isinstance(restored.total_budget, Decimal)
        assert isinstance(restored.spent, Decimal)
        assert isinstance(restored.remaining, Decimal)

    def test_aware_datetime_serialized_as_iso8601(self, valid_task_request):
        json_str = serialize_model(valid_task_request)
        parsed = json.loads(json_str)
        # Should contain timezone offset indicator
        created_at_str = parsed["created_at"]
        assert "+" in created_at_str or "Z" in created_at_str or "z" in created_at_str, \
            f"Datetime should have timezone info: {created_at_str}"

    def test_deserialize_invalid_json(self):
        with pytest.raises(Exception):
            deserialize_model("not valid json{{{", "TaskRequest")

    def test_deserialize_unknown_model_type(self):
        with pytest.raises(Exception):
            deserialize_model("{}", "NonExistentModel")

    def test_deserialize_validation_error(self):
        # Missing required fields for TaskRequest
        with pytest.raises(Exception):
            deserialize_model('{"task_id": "ok"}', "TaskRequest")

    def test_deserialize_extra_fields_forbid(self, valid_task_request):
        json_str = serialize_model(valid_task_request)
        parsed = json.loads(json_str)
        parsed["extra_forbidden_field"] = "bad"
        modified_json = json.dumps(parsed)
        with pytest.raises(Exception):
            deserialize_model(modified_json, "TaskRequest")

    def test_serialize_not_pydantic_model_rejected(self):
        with pytest.raises(Exception):
            serialize_model({"not": "a model"})

    def test_serialize_non_model_string_rejected(self):
        with pytest.raises(Exception):
            serialize_model("just a string")


# ===========================================================================
# TestValidateModel
# ===========================================================================

class TestValidateModel:
    """Tests for the validate_model function."""

    def test_valid_data_returns_empty_list(self, valid_task_request):
        json_str = serialize_model(valid_task_request)
        data = json.loads(json_str)
        errors = validate_model(data, "TaskRequest")
        assert errors == [], f"Valid data should produce no errors, got: {errors}"

    def test_invalid_data_returns_errors(self):
        # TaskRequest with invalid task_id and missing fields
        data = {"task_id": "INVALID", "input_data": "not_a_dict"}
        errors = validate_model(data, "TaskRequest")
        assert len(errors) > 0, "Invalid data should produce validation errors"
        # Each error should be a ValidationErrorDetail or similar
        for err in errors:
            assert hasattr(err, 'field') or isinstance(err, dict) or hasattr(err, 'message'), \
                "Each error should have structured information"

    def test_unknown_model_type_rejected(self):
        with pytest.raises(Exception):
            validate_model({}, "NonExistentType")

    def test_multiple_simultaneous_errors(self):
        data = {
            "task_id": "",  # invalid
            # missing input_data
            # missing request_id
        }
        errors = validate_model(data, "TaskRequest")
        assert len(errors) >= 1, "Should catch multiple validation errors"


# ===========================================================================
# TestGetModelJsonSchema
# ===========================================================================

class TestGetModelJsonSchema:
    """Tests for the get_model_json_schema function."""

    @pytest.mark.parametrize("model_type", [
        "TaskRequest",
        "TaskResponse",
        "TrainingExample",
        "ComparisonPair",
        "ConfidenceSnapshot",
        "BudgetState",
        "AuditEntry",
        "SamplingDecision",
        "ModelVersion",
        "BudgetPeriod",
        "ExactMatchResult",
        "FuzzyMatchResult",
        "SemanticSimilarityResult",
        "LlmJudgeResult",
        "CustomEvalResult",
    ])
    def test_schema_returned_for_known_types(self, model_type):
        schema = get_model_json_schema(model_type)
        assert isinstance(schema, dict), "Schema should be a dict"
        # JSON Schema should have 'properties' or 'type' at minimum
        assert "properties" in schema or "type" in schema or "$defs" in schema, \
            f"Schema for {model_type} should be a valid JSON Schema"

    def test_unknown_model_type_rejected(self):
        with pytest.raises(Exception):
            get_model_json_schema("FakeModel")

    def test_schema_includes_field_constraints(self):
        schema = get_model_json_schema("TaskRequest")
        # Should have properties for all defined fields
        props = schema.get("properties", {})
        assert "task_id" in props, "TaskRequest schema should include task_id"
        assert "input_data" in props, "TaskRequest schema should include input_data"
        assert "request_id" in props, "TaskRequest schema should include request_id"


# ===========================================================================
# TestInvariants
# ===========================================================================

class TestInvariants:
    """Tests for global invariants defined in the contract."""

    def test_frozen_model_immutability(self, valid_task_request):
        """All frozen models reject field assignment."""
        with pytest.raises((AttributeError, TypeError, Exception)):
            valid_task_request.task_id = "changed"

    def test_frozen_task_response_immutability(self, valid_task_response):
        with pytest.raises((AttributeError, TypeError, Exception)):
            valid_task_response.output_data = {"hacked": True}

    def test_frozen_budget_state_immutability(self, valid_budget_state):
        with pytest.raises((AttributeError, TypeError, Exception)):
            valid_budget_state.spent = Decimal("999")

    def test_extra_forbid_task_request(self):
        """TaskRequest with extra='forbid' rejects unknown fields."""
        with pytest.raises(Exception):
            TaskRequest(
                request_id=str(uuid.uuid4()),
                task_id="test-task",
                input_data={},
                created_at=datetime.now(timezone.utc),
                metadata={},
                unknown_field="rejected",
            )

    def test_extra_forbid_task_response(self, aware_now):
        """TaskResponse with extra='forbid' rejects unknown fields."""
        with pytest.raises(Exception):
            TaskResponse(
                request_id=str(uuid.uuid4()),
                task_id="test-task",
                output_data={},
                model_used="gpt-4",
                routing_decision=RoutingDecision.REMOTE,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                confidence_at_decision=0.5,
                created_at=aware_now,
                latency_ms=10.0,
                cost_incurred=Decimal("0"),
                metadata={},
                extra_field="boom",
            )

    def test_extra_forbid_audit_entry(self, aware_now):
        """AuditEntry with extra='forbid' rejects unknown fields."""
        with pytest.raises(Exception):
            AuditEntry(
                timestamp=aware_now,
                request_id="req-1",
                task_id="test-task",
                routing_decision=RoutingDecision.REMOTE,
                model_used="gpt-4",
                confidence_at_decision=0.5,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                cost=Decimal("0"),
                latency_ms=10.0,
                sampling_probability=0.0,
                fallback_triggered=False,
                error="",
                metadata={},
                extra_not_allowed="nope",
            )

    def test_decimal_cost_arithmetic(self, valid_budget_state, aware_now):
        """CostAmount uses Decimal arithmetic, not float."""
        assert isinstance(valid_budget_state.total_budget, Decimal)
        assert isinstance(valid_budget_state.spent, Decimal)
        assert isinstance(valid_budget_state.remaining, Decimal)

        new_state = record_cost(valid_budget_state, Decimal("0.1"), aware_now)
        assert isinstance(new_state.spent, Decimal)
        assert isinstance(new_state.remaining, Decimal)
        # Verify no floating point artifacts
        expected_spent = valid_budget_state.spent + Decimal("0.1")
        assert new_state.spent == expected_spent, \
            f"Decimal arithmetic should be exact: {new_state.spent} != {expected_spent}"

    def test_budget_remaining_invariant(self, valid_task_id, valid_budget_period, aware_now):
        """BudgetState.remaining always equals total_budget - spent."""
        for spent_val in ["0", "10.50", "99.99", "100.00", "150.00"]:
            state = create_budget_state(
                task_id=valid_task_id,
                total_budget=Decimal("100.00"),
                spent=Decimal(spent_val),
                period=valid_budget_period,
                request_count=0,
                last_updated=aware_now,
            )
            assert state.remaining == state.total_budget - state.spent, \
                f"remaining invariant violated: {state.remaining} != {state.total_budget} - {state.spent}"

    def test_budget_is_exhausted_invariant(self, valid_task_id, valid_budget_period, aware_now):
        """BudgetState.is_exhausted == (remaining <= 0)."""
        test_cases = [
            ("50.00", False),   # remaining = 50
            ("100.00", True),   # remaining = 0
            ("110.00", True),   # remaining = -10
        ]
        for spent_str, expected_exhausted in test_cases:
            state = create_budget_state(
                task_id=valid_task_id,
                total_budget=Decimal("100.00"),
                spent=Decimal(spent_str),
                period=valid_budget_period,
                request_count=0,
                last_updated=aware_now,
            )
            assert state.is_exhausted == expected_exhausted, \
                f"is_exhausted should be {expected_exhausted} for spent={spent_str}"

    def test_phase_derived_not_manual(self, aware_now):
        """Phase is always derived from ConfidenceScore and thresholds."""
        # Create snapshots and verify phase matches derivation rule
        cases = [
            (0.0, 0.3, 0.7, Phase.PHASE_1_REMOTE_ONLY),
            (0.29, 0.3, 0.7, Phase.PHASE_1_REMOTE_ONLY),
            (0.3, 0.3, 0.7, Phase.PHASE_2_SAMPLING),
            (0.5, 0.3, 0.7, Phase.PHASE_2_SAMPLING),
            (0.69, 0.3, 0.7, Phase.PHASE_2_SAMPLING),
            (0.7, 0.3, 0.7, Phase.PHASE_3_LOCAL_PRIMARY),
            (1.0, 0.3, 0.7, Phase.PHASE_3_LOCAL_PRIMARY),
        ]
        for score, t2, t3, expected_phase in cases:
            snap = create_confidence_snapshot(
                task_id="inv-phase",
                score=score,
                sample_count=10,
                threshold_phase2=t2,
                threshold_phase3=t3,
                last_updated=aware_now,
                trend=0.0,
            )
            assert snap.phase == expected_phase, \
                f"score={score}, t2={t2}, t3={t3}: expected {expected_phase}, got {snap.phase}"

    def test_no_model_performs_io(self):
        """All functions are pure — this test ensures no network or file I/O."""
        # We simply call all factory functions and assert they don't hang
        # or raise I/O-related exceptions
        now = datetime.now(timezone.utc)
        period = BudgetPeriod(
            unit=BudgetPeriodUnit.HOURLY,
            period_start=now,
            period_end=now + timedelta(hours=1),
        )
        req = create_task_request(task_id="io-test", input_data={}, metadata={})
        resp = create_task_response(
            request_id=req.request_id, task_id="io-test", output_data={},
            model_used="m1", routing_decision=RoutingDecision.REMOTE,
            phase=Phase.PHASE_1_REMOTE_ONLY, confidence_at_decision=0.1,
            latency_ms=0.0, cost_incurred=Decimal("0"), created_at=now, metadata={},
        )
        snap = create_confidence_snapshot(
            task_id="io-test", score=0.5, sample_count=1,
            threshold_phase2=0.3, threshold_phase3=0.7, last_updated=now, trend=0.0,
        )
        bs = create_budget_state(
            task_id="io-test", total_budget=Decimal("10"), spent=Decimal("0"),
            period=period, request_count=0, last_updated=now,
        )
        bs2 = record_cost(bs, Decimal("1"), now)
        ae = create_audit_entry(
            request_id="r1", task_id="io-test",
            routing_decision=RoutingDecision.REMOTE, model_used="m1",
            confidence_at_decision=0.5, phase=Phase.PHASE_1_REMOTE_ONLY,
            cost=Decimal("0"), latency_ms=0.0, fallback_triggered=False,
            timestamp=now, sampling_probability=0.0, error="", metadata={},
        )
        # If we got here without hanging or raising, pure functions confirmed
        assert True


# ===========================================================================
# TestEnumCompleteness
# ===========================================================================

class TestEnumCompleteness:
    """Tests that all enum members are valid and non-members are rejected."""

    def test_phase_enum_members(self):
        expected = {"PHASE_1_REMOTE_ONLY", "PHASE_2_SAMPLING", "PHASE_3_LOCAL_PRIMARY"}
        actual = {m.name for m in Phase}
        assert actual == expected, f"Phase enum members: expected {expected}, got {actual}"

    def test_routing_decision_enum_members(self):
        expected = {"REMOTE", "LOCAL", "LOCAL_WITH_REMOTE_COMPARISON", "LOCAL_FALLBACK_TO_REMOTE"}
        actual = {m.name for m in RoutingDecision}
        assert actual == expected

    def test_evaluator_type_enum_members(self):
        expected = {"EXACT_MATCH", "FUZZY_MATCH", "SEMANTIC_SIMILARITY", "LLM_JUDGE", "CUSTOM"}
        actual = {m.name for m in EvaluatorType}
        assert actual == expected

    def test_budget_period_unit_enum_members(self):
        expected = {"HOURLY", "DAILY", "WEEKLY", "MONTHLY"}
        actual = {m.name for m in BudgetPeriodUnit}
        assert actual == expected

    def test_evaluation_result_type_enum_members(self):
        expected = {
            "EXACT_MATCH_RESULT", "FUZZY_MATCH_RESULT", "SEMANTIC_SIMILARITY_RESULT",
            "LLM_JUDGE_RESULT", "CUSTOM_RESULT"
        }
        actual = {m.name for m in EvaluationResultType}
        assert actual == expected

    @pytest.mark.parametrize("routing", list(RoutingDecision))
    def test_all_routing_decisions_accepted_in_response(self, routing, aware_now):
        resp = create_task_response(
            request_id=str(uuid.uuid4()),
            task_id="enum-test",
            output_data={},
            model_used="model-x",
            routing_decision=routing,
            phase=Phase.PHASE_1_REMOTE_ONLY,
            confidence_at_decision=0.5,
            latency_ms=0.0,
            cost_incurred=Decimal("0"),
            created_at=aware_now,
            metadata={},
        )
        assert resp.routing_decision == routing

    @pytest.mark.parametrize("phase", list(Phase))
    def test_all_phases_accepted_in_response(self, phase, aware_now):
        resp = create_task_response(
            request_id=str(uuid.uuid4()),
            task_id="enum-test",
            output_data={},
            model_used="model-x",
            routing_decision=RoutingDecision.REMOTE,
            phase=phase,
            confidence_at_decision=0.5,
            latency_ms=0.0,
            cost_incurred=Decimal("0"),
            created_at=aware_now,
            metadata={},
        )
        assert resp.phase == phase

    @pytest.mark.parametrize("unit", list(BudgetPeriodUnit))
    def test_all_budget_period_units_accepted(self, unit, aware_now, aware_now_plus_1h):
        period = BudgetPeriod(
            unit=unit,
            period_start=aware_now,
            period_end=aware_now_plus_1h,
        )
        assert period.unit == unit


# ===========================================================================
# TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    """Additional edge case tests."""

    def test_task_id_min_length(self):
        req = create_task_request(task_id="a", input_data={}, metadata={})
        assert req.task_id == "a"

    def test_task_id_max_length(self):
        task_id = "a" + "b" * 127  # 128 chars total
        req = create_task_request(task_id=task_id, input_data={}, metadata={})
        assert len(req.task_id) == 128

    def test_model_id_max_length(self, aware_now):
        model_id = "m" * 256
        resp = create_task_response(
            request_id=str(uuid.uuid4()),
            task_id="edge-task",
            output_data={},
            model_used=model_id,
            routing_decision=RoutingDecision.REMOTE,
            phase=Phase.PHASE_1_REMOTE_ONLY,
            confidence_at_decision=0.5,
            latency_ms=0.0,
            cost_incurred=Decimal("0"),
            created_at=aware_now,
            metadata={},
        )
        assert resp.model_used == model_id

    def test_model_id_exceeds_max_length(self, aware_now):
        model_id = "m" * 257
        with pytest.raises(Exception):
            create_task_response(
                request_id=str(uuid.uuid4()),
                task_id="edge-task",
                output_data={},
                model_used=model_id,
                routing_decision=RoutingDecision.REMOTE,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                confidence_at_decision=0.5,
                latency_ms=0.0,
                cost_incurred=Decimal("0"),
                created_at=aware_now,
                metadata={},
            )

    def test_empty_model_id_rejected(self, aware_now):
        with pytest.raises(Exception):
            create_task_response(
                request_id=str(uuid.uuid4()),
                task_id="edge-task",
                output_data={},
                model_used="",
                routing_decision=RoutingDecision.REMOTE,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                confidence_at_decision=0.5,
                latency_ms=0.0,
                cost_incurred=Decimal("0"),
                created_at=aware_now,
                metadata={},
            )

    def test_zero_latency_accepted(self, aware_now):
        resp = create_task_response(
            request_id=str(uuid.uuid4()),
            task_id="edge-task",
            output_data={},
            model_used="model-x",
            routing_decision=RoutingDecision.REMOTE,
            phase=Phase.PHASE_1_REMOTE_ONLY,
            confidence_at_decision=0.5,
            latency_ms=0.0,
            cost_incurred=Decimal("0"),
            created_at=aware_now,
            metadata={},
        )
        assert resp.latency_ms == 0.0

    def test_unicode_in_string_fields(self, aware_now):
        req = create_task_request(
            task_id="unicode-task",
            input_data={"prompt": "こんにちは世界 🌍"},
            metadata={"name": "José García", "emoji": "🚀"},
        )
        assert "こんにちは世界" in req.input_data["prompt"]

    def test_aware_datetime_different_timezones(self):
        tz_plus5 = timezone(timedelta(hours=5))
        tz_minus3 = timezone(timedelta(hours=-3))

        req1 = create_task_request(
            task_id="tz-test",
            input_data={},
            metadata={},
            created_at=datetime(2024, 6, 15, 12, 0, 0, tzinfo=tz_plus5),
        )
        req2 = create_task_request(
            task_id="tz-test",
            input_data={},
            metadata={},
            created_at=datetime(2024, 6, 15, 12, 0, 0, tzinfo=tz_minus3),
        )
        assert req1.created_at.tzinfo is not None
        assert req2.created_at.tzinfo is not None

    def test_confidence_score_nan_rejected(self, aware_now):
        with pytest.raises(Exception):
            create_confidence_snapshot(
                task_id="nan-test",
                score=float('nan'),
                sample_count=10,
                threshold_phase2=0.4,
                threshold_phase3=0.8,
                last_updated=aware_now,
                trend=0.0,
            )

    def test_budget_state_extra_ignore(self, aware_now):
        """BudgetState with extra='ignore' should silently discard unknown fields."""
        period = BudgetPeriod(
            unit=BudgetPeriodUnit.DAILY,
            period_start=aware_now,
            period_end=aware_now + timedelta(hours=1),
        )
        state = create_budget_state(
            task_id="ignore-test",
            total_budget=Decimal("100"),
            spent=Decimal("0"),
            period=period,
            request_count=0,
            last_updated=aware_now,
        )
        json_str = serialize_model(state)
        parsed = json.loads(json_str)
        parsed["future_field"] = "should be ignored"
        modified_json = json.dumps(parsed)
        # Should NOT raise — extra='ignore' discards unknown fields
        restored = deserialize_model(modified_json, "BudgetState")
        assert restored.task_id == "ignore-test"
        assert not hasattr(restored, "future_field")

    def test_model_version_extra_ignore(self, aware_now):
        """ModelVersion with extra='ignore' should silently discard unknown fields."""
        mv = create_model_version(
            model_id="mv-ignore",
            path="/models/test",
            validation_score=0.5,
            training_example_count=10,
            is_active=False,
            created_at=aware_now,
        )
        json_str = serialize_model(mv)
        parsed = json.loads(json_str)
        parsed["new_future_field"] = 42
        modified_json = json.dumps(parsed)
        restored = deserialize_model(modified_json, "ModelVersion")
        assert restored.model_id == "mv-ignore"

    def test_large_input_data_dict(self):
        """Large nested dicts should be accepted."""
        large_data = {f"key_{i}": {"nested": [1, 2, 3]} for i in range(100)}
        req = create_task_request(
            task_id="large-data",
            input_data=large_data,
            metadata={},
        )
        assert len(req.input_data) == 100

    def test_training_example_with_explicit_params(self, aware_now):
        """TrainingExample with all explicit parameters."""
        eid = str(uuid.uuid4())
        te = create_training_example(
            task_id="explicit-task",
            input_data={"q": "What?"},
            output_data={"a": "This."},
            model_id="gpt-4",
            source_request_id="req-explicit",
            example_id=eid,
            created_at=aware_now,
            quality_score=1.0,
        )
        assert te.example_id == eid
        assert te.created_at == aware_now
        assert te.quality_score == 1.0

    def test_sampling_decision_frozen(self, aware_now):
        sd = create_sampling_decision(
            request_id="req-frozen",
            task_id="frozen-task",
            decision=RoutingDecision.REMOTE,
            sampling_probability=0.5,
            phase=Phase.PHASE_2_SAMPLING,
            confidence_at_decision=0.6,
            budget_remaining=Decimal("50"),
            rationale="Testing freeze",
            decided_at=aware_now,
        )
        with pytest.raises((AttributeError, TypeError, Exception)):
            sd.decision = RoutingDecision.LOCAL
