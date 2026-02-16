from pydantic import model_validator
from pydantic import field_validator
from pydantic import ConfigDict
"""
Shared Data Models (data_models) v1

Pydantic v2 immutable data models forming the inter-component contract for the entire Apprentice system.
All models use frozen=True and strict=True as baseline.
Contract models use extra='forbid'; persisted models use extra='ignore' for forward compatibility.
"""

import json
import uuid
import math
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Annotated, Union, Optional, Any

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)


# ===========================================================================
# Enums
# ===========================================================================

class Phase(str, Enum):
    """StrEnum representing the current operational phase."""
    PHASE_1_REMOTE_ONLY = "PHASE_1_REMOTE_ONLY"
    PHASE_2_SAMPLING = "PHASE_2_SAMPLING"
    PHASE_3_LOCAL_PRIMARY = "PHASE_3_LOCAL_PRIMARY"


class RoutingDecision(str, Enum):
    """StrEnum representing where a request is routed."""
    REMOTE = "REMOTE"
    LOCAL = "LOCAL"
    LOCAL_WITH_REMOTE_COMPARISON = "LOCAL_WITH_REMOTE_COMPARISON"
    LOCAL_FALLBACK_TO_REMOTE = "LOCAL_FALLBACK_TO_REMOTE"


class EvaluatorType(str, Enum):
    """StrEnum representing the type of evaluator used for comparing responses."""
    EXACT_MATCH = "EXACT_MATCH"
    FUZZY_MATCH = "FUZZY_MATCH"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    LLM_JUDGE = "LLM_JUDGE"
    CUSTOM = "CUSTOM"


class BudgetPeriodUnit(str, Enum):
    """StrEnum for budget period granularity."""
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


class EvaluationResultType(str, Enum):
    """StrEnum discriminator for the EvaluationResult discriminated union."""
    EXACT_MATCH_RESULT = "EXACT_MATCH_RESULT"
    FUZZY_MATCH_RESULT = "FUZZY_MATCH_RESULT"
    SEMANTIC_SIMILARITY_RESULT = "SEMANTIC_SIMILARITY_RESULT"
    LLM_JUDGE_RESULT = "LLM_JUDGE_RESULT"
    CUSTOM_RESULT = "CUSTOM_RESULT"


# ===========================================================================
# Type Aliases for Domain Primitives
# ===========================================================================

TaskId = Annotated[str, Field(min_length=1, max_length=128, pattern=r'^[a-z][a-z0-9_-]*$')]
ModelId = Annotated[str, Field(min_length=1, max_length=256)]
ConfidenceScore = Annotated[float, Field(ge=0.0, le=1.0)]
CostAmount = Annotated[Decimal, Field(ge=0)]
AwareDatetime = datetime  # We'll validate timezone-awareness separately


# ===========================================================================
# Data Models
# ===========================================================================

class TaskRequest(BaseModel):
    """Frozen Pydantic model representing an incoming task request."""
    model_config = ConfigDict(frozen=True, extra='forbid', strict=True)

    request_id: str
    task_id: TaskId
    input_data: dict
    created_at: AwareDatetime
    metadata: dict = Field(default_factory=dict)

    @field_validator('created_at')
    @classmethod
    def validate_aware_datetime(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return v

    @field_validator('input_data', 'metadata')
    @classmethod
    def validate_dict(cls, v: Any) -> dict:
        if not isinstance(v, dict):
            raise ValueError("Must be a dict")
        return v


class TaskResponse(BaseModel):
    """Frozen Pydantic model representing the response to a task request."""
    model_config = ConfigDict(frozen=True, extra='forbid', strict=True)

    request_id: str
    task_id: TaskId
    output_data: dict
    model_used: ModelId
    routing_decision: RoutingDecision
    phase: Phase
    confidence_at_decision: ConfidenceScore
    created_at: AwareDatetime
    latency_ms: Annotated[float, Field(ge=0.0)]
    cost_incurred: CostAmount
    metadata: dict = Field(default_factory=dict)

    @field_validator('created_at')
    @classmethod
    def validate_aware_datetime(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return v

    @field_validator('output_data', 'metadata')
    @classmethod
    def validate_dict(cls, v: Any) -> dict:
        if not isinstance(v, dict):
            raise ValueError("Must be a dict")
        return v


class BudgetPeriod(BaseModel):
    """Frozen Pydantic model tracking a single budget period window."""
    model_config = ConfigDict(frozen=True, strict=True)

    unit: BudgetPeriodUnit
    period_start: AwareDatetime
    period_end: AwareDatetime

    @field_validator('period_start', 'period_end')
    @classmethod
    def validate_aware_datetime(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("period datetime must be timezone-aware")
        return v


class BudgetState(BaseModel):
    """Frozen Pydantic model representing current budget state (extra='ignore' for persistence)."""
    model_config = ConfigDict(frozen=True, extra='ignore', strict=True)

    task_id: TaskId
    total_budget: CostAmount
    spent: CostAmount
    remaining: Decimal  # Can be negative when overspent
    period: BudgetPeriod
    is_exhausted: bool
    last_updated: AwareDatetime
    request_count: Annotated[int, Field(ge=0)]

    @field_validator('last_updated')
    @classmethod
    def validate_aware_datetime(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("last_updated must be timezone-aware")
        return v


class ConfidenceSnapshot(BaseModel):
    """Frozen Pydantic model capturing the confidence state for a task at a point in time."""
    model_config = ConfigDict(frozen=True, strict=True)

    task_id: TaskId
    score: ConfidenceScore
    phase: Phase
    sample_count: Annotated[int, Field(ge=0)]
    last_updated: AwareDatetime
    threshold_phase2: ConfidenceScore
    threshold_phase3: ConfidenceScore
    trend: float = 0.0

    @field_validator('last_updated')
    @classmethod
    def validate_aware_datetime(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("last_updated must be timezone-aware")
        return v

    @field_validator('score')
    @classmethod
    def validate_score_not_nan(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("score cannot be NaN or infinite")
        return v


class AuditEntry(BaseModel):
    """Frozen Pydantic model representing a single auditable decision."""
    model_config = ConfigDict(frozen=True, extra='forbid', strict=True)

    timestamp: AwareDatetime
    request_id: str
    task_id: TaskId
    routing_decision: RoutingDecision
    model_used: ModelId
    confidence_at_decision: ConfidenceScore
    phase: Phase
    cost: CostAmount
    latency_ms: Annotated[float, Field(ge=0.0)]
    sampling_probability: Optional[float] = None
    fallback_triggered: bool
    error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    @field_validator('timestamp')
    @classmethod
    def validate_aware_datetime(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return v

    @field_validator('request_id')
    @classmethod
    def validate_request_id(cls, v: str) -> str:
        if not v:
            raise ValueError("request_id cannot be empty")
        return v

    @field_validator('metadata')
    @classmethod
    def validate_dict(cls, v: Any) -> dict:
        if not isinstance(v, dict):
            raise ValueError("Must be a dict")
        return v


class SamplingDecision(BaseModel):
    """Frozen Pydantic model representing a routing/sampling decision made by the Router."""
    model_config = ConfigDict(frozen=True, strict=True)

    request_id: str
    task_id: TaskId
    decision: RoutingDecision
    sampling_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    phase: Phase
    confidence_at_decision: ConfidenceScore
    budget_remaining: CostAmount
    rationale: str
    decided_at: AwareDatetime

    @field_validator('decided_at')
    @classmethod
    def validate_aware_datetime(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")
        return v

    @field_validator('rationale')
    @classmethod
    def validate_rationale(cls, v: str) -> str:
        if not v:
            raise ValueError("rationale cannot be empty")
        return v


class TrainingExample(BaseModel):
    """Frozen Pydantic model representing a single training example."""
    model_config = ConfigDict(frozen=True, strict=True)

    example_id: str
    task_id: TaskId
    input_data: dict
    output_data: dict
    model_id: ModelId
    created_at: AwareDatetime
    quality_score: Optional[ConfidenceScore] = None
    source_request_id: str

    @field_validator('created_at')
    @classmethod
    def validate_aware_datetime(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return v

    @field_validator('source_request_id')
    @classmethod
    def validate_source_request_id(cls, v: str) -> str:
        if not v:
            raise ValueError("source_request_id cannot be empty")
        return v

    @field_validator('input_data', 'output_data')
    @classmethod
    def validate_dict(cls, v: Any) -> dict:
        if not isinstance(v, dict):
            raise ValueError("Must be a dict")
        return v


class ExactMatchResult(BaseModel):
    """Evaluation result for exact match comparison."""
    model_config = ConfigDict(frozen=True, strict=True)

    result_type: EvaluationResultType
    is_match: bool
    matched_fields: list = Field(default_factory=list)
    mismatched_fields: list = Field(default_factory=list)


class FuzzyMatchResult(BaseModel):
    """Evaluation result for fuzzy/approximate match comparison."""
    model_config = ConfigDict(frozen=True, strict=True)

    result_type: EvaluationResultType
    similarity_score: ConfidenceScore
    threshold: ConfidenceScore
    is_pass: bool


class SemanticSimilarityResult(BaseModel):
    """Evaluation result for semantic similarity comparison."""
    model_config = ConfigDict(frozen=True, strict=True)

    result_type: EvaluationResultType
    similarity_score: ConfidenceScore
    threshold: ConfidenceScore
    is_pass: bool
    embedding_model: str


class LlmJudgeResult(BaseModel):
    """Evaluation result from an LLM-as-judge comparison."""
    model_config = ConfigDict(frozen=True, strict=True)

    result_type: EvaluationResultType
    score: ConfidenceScore
    threshold: ConfidenceScore
    is_pass: bool
    judge_model_id: ModelId
    reasoning: str
    cost: CostAmount


class CustomEvalResult(BaseModel):
    """Evaluation result from a custom/plugin evaluator."""
    model_config = ConfigDict(frozen=True, strict=True)

    result_type: EvaluationResultType
    score: ConfidenceScore
    is_pass: bool
    evaluator_name: str
    details: dict = Field(default_factory=dict)

    @field_validator('details')
    @classmethod
    def validate_dict(cls, v: Any) -> dict:
        if not isinstance(v, dict):
            raise ValueError("Must be a dict")
        return v


# Discriminated union type
EvaluationResult = Union[
    ExactMatchResult,
    FuzzyMatchResult,
    SemanticSimilarityResult,
    LlmJudgeResult,
    CustomEvalResult
]


class ComparisonPair(BaseModel):
    """Frozen Pydantic model representing a side-by-side comparison of remote vs. local model outputs."""
    model_config = ConfigDict(frozen=True, strict=True)

    pair_id: str
    task_id: TaskId
    input_data: dict
    remote_output: dict
    local_output: dict
    remote_model_id: ModelId
    local_model_id: ModelId
    evaluation_result: Optional[EvaluationResult] = None
    created_at: AwareDatetime
    source_request_id: str

    @field_validator('created_at')
    @classmethod
    def validate_aware_datetime(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return v

    @field_validator('input_data', 'remote_output', 'local_output')
    @classmethod
    def validate_dict(cls, v: Any) -> dict:
        if not isinstance(v, dict):
            raise ValueError("Must be a dict")
        return v


class ModelVersion(BaseModel):
    """Frozen Pydantic model representing a versioned local model (extra='ignore' for persistence)."""
    model_config = ConfigDict(frozen=True, extra='ignore', strict=True)

    version_id: str
    model_id: ModelId
    path: str
    validation_score: ConfidenceScore
    promoted_at: Optional[AwareDatetime] = None
    created_at: AwareDatetime
    training_example_count: Annotated[int, Field(ge=0)]
    parent_version_id: Optional[str] = None
    is_active: bool

    @field_validator('created_at')
    @classmethod
    def validate_aware_datetime(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return v

    @field_validator('promoted_at')
    @classmethod
    def validate_promoted_at(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is not None and v.tzinfo is None:
            raise ValueError("promoted_at must be timezone-aware")
        return v

    @field_validator('path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v:
            raise ValueError("path cannot be empty")
        return v


class ValidationErrorDetail(BaseModel):
    """Structured detail about a single Pydantic validation error."""
    model_config = ConfigDict(strict=True)

    field: str
    message: str
    value: Optional[str] = None
    error_type: str


# ===========================================================================
# Factory Functions
# ===========================================================================

def create_task_request(
    task_id: str,
    input_data: dict,
    metadata: dict = None,
    request_id: str = None,
    created_at: datetime = None,
) -> TaskRequest:
    """Factory function to construct and validate a TaskRequest."""
    if metadata is None:
        metadata = {}
    if request_id is None:
        request_id = str(uuid.uuid4())
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    return TaskRequest(
        request_id=request_id,
        task_id=task_id,
        input_data=input_data,
        created_at=created_at,
        metadata=metadata,
    )


def create_task_response(
    request_id: str,
    task_id: str,
    output_data: dict,
    model_used: str,
    routing_decision: RoutingDecision,
    phase: Phase,
    confidence_at_decision: float,
    latency_ms: float,
    cost_incurred: Decimal,
    created_at: datetime = None,
    metadata: dict = None,
) -> TaskResponse:
    """Factory function to construct and validate a TaskResponse."""
    if metadata is None:
        metadata = {}
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    return TaskResponse(
        request_id=request_id,
        task_id=task_id,
        output_data=output_data,
        model_used=model_used,
        routing_decision=routing_decision,
        phase=phase,
        confidence_at_decision=confidence_at_decision,
        created_at=created_at,
        latency_ms=latency_ms,
        cost_incurred=cost_incurred,
        metadata=metadata,
    )


def create_confidence_snapshot(
    task_id: str,
    score: float,
    sample_count: int,
    threshold_phase2: float,
    threshold_phase3: float,
    last_updated: datetime = None,
    trend: float = 0.0,
) -> ConfidenceSnapshot:
    """
    Constructs a ConfidenceSnapshot.
    Derives the phase from the score and the two thresholds.
    """
    if last_updated is None:
        last_updated = datetime.now(timezone.utc)

    # Validate thresholds
    if not (0.0 <= threshold_phase2 <= 1.0 and 0.0 <= threshold_phase3 <= 1.0):
        raise ValueError("threshold_phase2 must be <= threshold_phase3 and both in [0.0, 1.0]")
    if threshold_phase2 > threshold_phase3:
        raise ValueError("threshold_phase2 must be <= threshold_phase3 and both in [0.0, 1.0]")

    # Derive phase from score and thresholds
    if score < threshold_phase2:
        phase = Phase.PHASE_1_REMOTE_ONLY
    elif score < threshold_phase3:
        phase = Phase.PHASE_2_SAMPLING
    else:
        phase = Phase.PHASE_3_LOCAL_PRIMARY

    return ConfidenceSnapshot(
        task_id=task_id,
        score=score,
        phase=phase,
        sample_count=sample_count,
        last_updated=last_updated,
        threshold_phase2=threshold_phase2,
        threshold_phase3=threshold_phase3,
        trend=trend,
    )


def create_budget_state(
    task_id: str,
    total_budget: Decimal,
    spent: Decimal,
    period: BudgetPeriod,
    request_count: int,
    last_updated: datetime = None,
) -> BudgetState:
    """
    Constructs a BudgetState.
    Computes remaining = total_budget - spent and is_exhausted = remaining <= 0.
    """
    if last_updated is None:
        last_updated = datetime.now(timezone.utc)

    # Validate period
    if period.period_start >= period.period_end:
        raise ValueError("period_start must be before period_end")

    # Compute derived fields
    remaining = total_budget - spent
    is_exhausted = remaining <= 0

    return BudgetState(
        task_id=task_id,
        total_budget=total_budget,
        spent=spent,
        remaining=remaining,
        period=period,
        is_exhausted=is_exhausted,
        last_updated=last_updated,
        request_count=request_count,
    )


def record_cost(
    budget_state: BudgetState,
    cost: Decimal,
    timestamp: datetime = None,
) -> BudgetState:
    """
    Returns a new BudgetState with cost added to spent.
    Uses model_copy for immutable update.
    """
    if cost < 0:
        raise ValueError("Cost to record must be non-negative")

    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    new_spent = budget_state.spent + cost
    new_remaining = budget_state.total_budget - new_spent
    new_is_exhausted = new_remaining <= 0
    new_request_count = budget_state.request_count + 1

    return budget_state.model_copy(
        update={
            'spent': new_spent,
            'remaining': new_remaining,
            'is_exhausted': new_is_exhausted,
            'last_updated': timestamp,
            'request_count': new_request_count,
        }
    )


def create_audit_entry(
    request_id: str,
    task_id: str,
    routing_decision: RoutingDecision,
    model_used: str,
    confidence_at_decision: float,
    phase: Phase,
    cost: Decimal,
    latency_ms: float,
    fallback_triggered: bool,
    timestamp: datetime = None,
    sampling_probability: float = None,
    error: str = None,
    metadata: dict = None,
) -> AuditEntry:
    """Factory function to construct and validate an AuditEntry."""
    if metadata is None:
        metadata = {}
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    return AuditEntry(
        timestamp=timestamp,
        request_id=request_id,
        task_id=task_id,
        routing_decision=routing_decision,
        model_used=model_used,
        confidence_at_decision=confidence_at_decision,
        phase=phase,
        cost=cost,
        latency_ms=latency_ms,
        sampling_probability=sampling_probability,
        fallback_triggered=fallback_triggered,
        error=error,
        metadata=metadata,
    )


def create_sampling_decision(
    request_id: str,
    task_id: str,
    decision: RoutingDecision,
    sampling_probability: float,
    phase: Phase,
    confidence_at_decision: float,
    budget_remaining: Decimal,
    rationale: str,
    decided_at: datetime = None,
) -> SamplingDecision:
    """Factory function to construct and validate a SamplingDecision."""
    if decided_at is None:
        decided_at = datetime.now(timezone.utc)

    return SamplingDecision(
        request_id=request_id,
        task_id=task_id,
        decision=decision,
        sampling_probability=sampling_probability,
        phase=phase,
        confidence_at_decision=confidence_at_decision,
        budget_remaining=budget_remaining,
        rationale=rationale,
        decided_at=decided_at,
    )


def create_training_example(
    task_id: str,
    input_data: dict,
    output_data: dict,
    model_id: str,
    source_request_id: str,
    example_id: str = None,
    created_at: datetime = None,
    quality_score: float = None,
) -> TrainingExample:
    """Factory function to construct and validate a TrainingExample."""
    if example_id is None:
        example_id = str(uuid.uuid4())
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    return TrainingExample(
        example_id=example_id,
        task_id=task_id,
        input_data=input_data,
        output_data=output_data,
        model_id=model_id,
        created_at=created_at,
        quality_score=quality_score,
        source_request_id=source_request_id,
    )


def create_comparison_pair(
    task_id: str,
    input_data: dict,
    remote_output: dict,
    local_output: dict,
    remote_model_id: str,
    local_model_id: str,
    source_request_id: str,
    pair_id: str = None,
    created_at: datetime = None,
    evaluation_result: EvaluationResult = None,
) -> ComparisonPair:
    """Factory function to construct and validate a ComparisonPair."""
    if remote_model_id == local_model_id:
        raise ValueError("Remote and local model IDs must be different for a valid comparison")

    if pair_id is None:
        pair_id = str(uuid.uuid4())
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    return ComparisonPair(
        pair_id=pair_id,
        task_id=task_id,
        input_data=input_data,
        remote_output=remote_output,
        local_output=local_output,
        remote_model_id=remote_model_id,
        local_model_id=local_model_id,
        evaluation_result=evaluation_result,
        created_at=created_at,
        source_request_id=source_request_id,
    )


def create_model_version(
    model_id: str,
    path: str,
    validation_score: float,
    training_example_count: int,
    is_active: bool,
    version_id: str = None,
    created_at: datetime = None,
    promoted_at: datetime = None,
    parent_version_id: str = None,
) -> ModelVersion:
    """Factory function to construct and validate a ModelVersion record."""
    if version_id is None:
        version_id = str(uuid.uuid4())
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    # If is_active is True and promoted_at was not provided, set it to created_at
    if is_active and promoted_at is None:
        promoted_at = created_at

    return ModelVersion(
        version_id=version_id,
        model_id=model_id,
        path=path,
        validation_score=validation_score,
        promoted_at=promoted_at,
        created_at=created_at,
        training_example_count=training_example_count,
        parent_version_id=parent_version_id,
        is_active=is_active,
    )


# ===========================================================================
# Serialization Functions
# ===========================================================================

def serialize_model(model_instance: Any) -> str:
    """
    Serializes any data model to a JSON string using Pydantic's model_dump_json().
    """
    if not isinstance(model_instance, BaseModel):
        raise TypeError("model_instance must be a Pydantic BaseModel")

    try:
        return model_instance.model_dump_json()
    except Exception as e:
        raise RuntimeError(f"Serialization error: {e}") from e


def deserialize_model(json_data: str, model_type: str) -> Any:
    """
    Deserializes a JSON string back into a typed Pydantic model instance.
    """
    # Map of model type names to classes
    MODEL_MAP = {
        'TaskRequest': TaskRequest,
        'TaskResponse': TaskResponse,
        'TrainingExample': TrainingExample,
        'ComparisonPair': ComparisonPair,
        'ConfidenceSnapshot': ConfidenceSnapshot,
        'BudgetPeriod': BudgetPeriod,
        'BudgetState': BudgetState,
        'AuditEntry': AuditEntry,
        'SamplingDecision': SamplingDecision,
        'ExactMatchResult': ExactMatchResult,
        'FuzzyMatchResult': FuzzyMatchResult,
        'SemanticSimilarityResult': SemanticSimilarityResult,
        'LlmJudgeResult': LlmJudgeResult,
        'CustomEvalResult': CustomEvalResult,
        'ModelVersion': ModelVersion,
    }

    if model_type not in MODEL_MAP:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class = MODEL_MAP[model_type]

    try:
        json.loads(json_data)  # Validate JSON
    except json.JSONDecodeError as e:
        raise ValueError("Input is not valid JSON") from e

    try:
        return model_class.model_validate_json(json_data)
    except Exception as e:
        raise


def validate_model(data: dict, model_type: str) -> list:
    """
    Validates raw data (dict) against a model type without constructing the full model.
    Returns a list of validation errors or an empty list if valid.
    """
    MODEL_MAP = {
        'TaskRequest': TaskRequest,
        'TaskResponse': TaskResponse,
        'TrainingExample': TrainingExample,
        'ComparisonPair': ComparisonPair,
        'ConfidenceSnapshot': ConfidenceSnapshot,
        'BudgetPeriod': BudgetPeriod,
        'BudgetState': BudgetState,
        'AuditEntry': AuditEntry,
        'SamplingDecision': SamplingDecision,
        'ExactMatchResult': ExactMatchResult,
        'FuzzyMatchResult': FuzzyMatchResult,
        'SemanticSimilarityResult': SemanticSimilarityResult,
        'LlmJudgeResult': LlmJudgeResult,
        'CustomEvalResult': CustomEvalResult,
        'ModelVersion': ModelVersion,
    }

    if model_type not in MODEL_MAP:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class = MODEL_MAP[model_type]

    try:
        # Use model_validate with mode='json' to handle serialized data properly
        # This allows datetime strings and other JSON-serialized formats
        model_class.model_validate(data, strict=False)
        return []
    except Exception as e:
        # Convert Pydantic validation errors to structured list
        error_list = []
        if hasattr(e, 'errors'):
            for err in e.errors():
                field_path = '.'.join(str(loc) for loc in err.get('loc', []))
                error_detail = ValidationErrorDetail(
                    field=field_path,
                    message=err.get('msg', 'Validation error'),
                    value=str(err.get('input', '')) if 'input' in err else None,
                    error_type=err.get('type', 'unknown'),
                )
                error_list.append(error_detail)
        else:
            # Generic error
            error_detail = ValidationErrorDetail(
                field='unknown',
                message=str(e),
                value=None,
                error_type='validation_error',
            )
            error_list.append(error_detail)
        return error_list


def get_model_json_schema(model_type: str) -> dict:
    """
    Returns the JSON Schema for a model type.
    """
    MODEL_MAP = {
        'TaskRequest': TaskRequest,
        'TaskResponse': TaskResponse,
        'TrainingExample': TrainingExample,
        'ComparisonPair': ComparisonPair,
        'ConfidenceSnapshot': ConfidenceSnapshot,
        'BudgetPeriod': BudgetPeriod,
        'BudgetState': BudgetState,
        'AuditEntry': AuditEntry,
        'SamplingDecision': SamplingDecision,
        'ExactMatchResult': ExactMatchResult,
        'FuzzyMatchResult': FuzzyMatchResult,
        'SemanticSimilarityResult': SemanticSimilarityResult,
        'LlmJudgeResult': LlmJudgeResult,
        'CustomEvalResult': CustomEvalResult,
        'ModelVersion': ModelVersion,
    }

    if model_type not in MODEL_MAP:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class = MODEL_MAP[model_type]
    return model_class.model_json_schema()


# ── Auto-injected export aliases (Pact export gate) ──
ValidationError = ValidationErrorDetail
