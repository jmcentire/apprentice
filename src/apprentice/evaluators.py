"""Response Evaluators (evaluators) v1

Pluggable evaluator implementations behind an Evaluator protocol (@runtime_checkable typing.Protocol).
Each evaluator takes a local TaskResponse and a remote TaskResponse (ground truth) plus task-specific
evaluator config, and returns a frozen EvaluationResult (score 0.0–1.0, per-field breakdown, evaluator
type used, optional error, metadata, timestamp).
"""

import logging
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict

_PACT_KEY = "PACT:95c95f:evaluators"
logger = logging.getLogger(__name__)


class PactFormatter(logging.Formatter):
    """Formatter that injects the PACT log key into every record."""

    def format(self, record):
        record.pact_key = _PACT_KEY
        return super().format(record)


def _log(level: str, msg: str, **kwargs) -> None:
    """Log with PACT key embedded for production traceability."""
    getattr(logger, level)(f"[{_PACT_KEY}] {msg}", **kwargs)


# ============================================================================
# Type Definitions
# ============================================================================


class EvaluatorType(Enum):
    """Enumeration of supported evaluator type identifiers."""
    structured_match = "structured_match"
    exact_match = "exact_match"
    semantic_similarity = "semantic_similarity"
    custom = "custom"


class TaskResponse(BaseModel):
    """A task response (either local model output or remote ground truth)."""
    model_config = ConfigDict(frozen=False)
    
    task_id: str
    output: str
    fields: dict[str, Any] = Field(default_factory=dict)
    model_id: str = ""
    timestamp: str


class TaskEvaluatorConfig(BaseModel):
    """Per-task evaluator configuration."""
    model_config = ConfigDict(frozen=True)
    
    evaluator_type: EvaluatorType
    match_fields: list[str] = Field(default_factory=list)
    model_name: str = "all-MiniLM-L6-v2"
    custom_evaluator: Optional[Any] = None
    
    @field_validator('match_fields')
    @classmethod
    def validate_match_fields(cls, v, info):
        """Validate match_fields is non-empty when evaluator_type is structured_match."""
        # Access evaluator_type from info.data
        evaluator_type = info.data.get('evaluator_type')
        if evaluator_type == EvaluatorType.structured_match:
            if not v or len(v) == 0:
                raise ValueError(
                    "match_fields must be non-empty when evaluator_type is 'structured_match'"
                )
        return v
    
    @field_validator('custom_evaluator')
    @classmethod
    def validate_custom_evaluator(cls, v, info):
        """Validate custom_evaluator is not None when evaluator_type is custom."""
        evaluator_type = info.data.get('evaluator_type')
        if evaluator_type == EvaluatorType.custom:
            if v is None:
                raise ValueError(
                    "custom_evaluator must be provided when evaluator_type is 'custom'"
                )
        return v


class FieldEvaluation(BaseModel):
    """Per-field evaluation result for StructuredMatchEvaluator."""
    model_config = ConfigDict(frozen=True)
    
    field_name: str
    matched: bool
    local_value: Any
    remote_value: Any
    similarity: Optional[float] = None


class EvaluationResult(BaseModel):
    """Frozen Pydantic model representing the outcome of a single evaluation."""
    model_config = ConfigDict(frozen=True)
    
    score: float = Field(ge=0.0, le=1.0)
    evaluator_type: str
    field_breakdown: dict[str, FieldEvaluation] = Field(default_factory=dict)
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str
    
    @field_validator('score')
    @classmethod
    def clamp_score_validator(cls, v):
        """Ensure score is clamped to [0.0, 1.0]."""
        if math.isnan(v):
            raise ValueError("Score cannot be NaN")
        return max(0.0, min(1.0, v))


# Type alias for embedding function
EmbedFunction = Callable[[list[str]], list[list[float]]]


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Protocol defining the Evaluator interface."""
    
    def evaluate(
        self,
        local: TaskResponse,
        remote: TaskResponse,
        task_config: TaskEvaluatorConfig,
    ) -> EvaluationResult:
        """Evaluate local vs remote TaskResponse."""
        ...


# ============================================================================
# Utility Functions
# ============================================================================


def clamp_score(score: float) -> float:
    """Clamp a raw score to the [0.0, 1.0] range.
    
    Args:
        score: Raw score value
        
    Returns:
        Clamped score in [0.0, 1.0]
        
    Raises:
        ValueError: If score is NaN
    """
    if math.isnan(score):
        raise ValueError("NaN is not a valid score")
    return max(0.0, min(1.0, score))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.
    
    Args:
        vec_a: First embedding vector
        vec_b: Second embedding vector
        
    Returns:
        Cosine similarity in [-1.0, 1.0]
        
    Raises:
        ValueError: If vectors are empty or have different dimensions
    """
    if not vec_a or not vec_b:
        raise ValueError("Embedding vectors must be non-empty")
    
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vectors must have same dimensionality: {len(vec_a)} vs {len(vec_b)}"
        )
    
    # Compute dot product and magnitudes
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    
    # Handle zero magnitude
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    
    similarity = dot_product / (magnitude_a * magnitude_b)
    
    # Clamp to [-1.0, 1.0] to handle floating point errors
    return max(-1.0, min(1.0, similarity))


# ============================================================================
# Evaluator Implementations
# ============================================================================


class ExactMatchEvaluator:
    """Evaluates local vs remote TaskResponse by exact output string equality."""
    
    def evaluate(
        self,
        local: TaskResponse,
        remote: TaskResponse,
        task_config: TaskEvaluatorConfig,
    ) -> EvaluationResult:
        """Evaluate by comparing full output strings for exact equality.
        
        Returns score 1.0 if outputs match exactly, else 0.0.
        """
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Compare outputs
            matched = local.output == remote.output
            score = 1.0 if matched else 0.0
            
            _log("debug", f"ExactMatchEvaluator: score={score}, matched={matched}")
            
            return EvaluationResult(
                score=score,
                evaluator_type="exact_match",
                field_breakdown={},
                error=None,
                metadata={},
                timestamp=timestamp,
            )
        except Exception as e:
            _log("error", f"ExactMatchEvaluator error: {e}")
            timestamp = datetime.now(timezone.utc).isoformat()
            return EvaluationResult(
                score=0.0,
                evaluator_type="exact_match",
                field_breakdown={},
                error=f"Output access error: {str(e)}",
                metadata={},
                timestamp=timestamp,
            )


class StructuredMatchEvaluator:
    """Evaluates local vs remote TaskResponse by per-field exact equality."""
    
    def evaluate(
        self,
        local: TaskResponse,
        remote: TaskResponse,
        task_config: TaskEvaluatorConfig,
    ) -> EvaluationResult:
        """Evaluate by comparing each field in match_fields using exact equality.
        
        Returns average match rate across all configured fields.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            match_fields = task_config.match_fields
            if not match_fields:
                return EvaluationResult(
                    score=0.0,
                    evaluator_type="structured_match",
                    field_breakdown={},
                    error="match_fields is empty",
                    metadata={},
                    timestamp=timestamp,
                )
            
            field_breakdown = {}
            match_count = 0
            
            for field_name in match_fields:
                try:
                    # Get field values, treating missing fields as None
                    local_value = local.fields.get(field_name)
                    remote_value = remote.fields.get(field_name)
                    
                    # Check for exact match
                    matched = local_value == remote_value
                    
                    if matched:
                        match_count += 1
                    
                    # Create field evaluation
                    field_eval = FieldEvaluation(
                        field_name=field_name,
                        matched=matched,
                        local_value=local_value,
                        remote_value=remote_value,
                        similarity=1.0 if matched else 0.0,
                    )
                    field_breakdown[field_name] = field_eval
                    
                except Exception as e:
                    _log("error", f"Field access error for {field_name}: {e}")
                    # Treat field access error as non-matching
                    field_eval = FieldEvaluation(
                        field_name=field_name,
                        matched=False,
                        local_value=None,
                        remote_value=None,
                        similarity=0.0,
                    )
                    field_breakdown[field_name] = field_eval
            
            # Compute average match rate
            score = match_count / len(match_fields) if match_fields else 0.0
            score = clamp_score(score)
            
            _log(
                "debug",
                f"StructuredMatchEvaluator: score={score}, "
                f"matched={match_count}/{len(match_fields)}"
            )
            
            return EvaluationResult(
                score=score,
                evaluator_type="structured_match",
                field_breakdown=field_breakdown,
                error=None,
                metadata={},
                timestamp=timestamp,
            )
            
        except Exception as e:
            _log("error", f"StructuredMatchEvaluator error: {e}")
            return EvaluationResult(
                score=0.0,
                evaluator_type="structured_match",
                field_breakdown={},
                error=f"Field access error: {str(e)}",
                metadata={},
                timestamp=timestamp,
            )


class SemanticSimilarityEvaluator:
    """Evaluates local vs remote TaskResponse using semantic similarity."""
    
    def __init__(self, embed_fn: Optional[EmbedFunction] = None):
        """Construct SemanticSimilarityEvaluator with optional embed_fn.
        
        Args:
            embed_fn: Optional embedding function. If None, will lazy-load
                     sentence-transformers model on first evaluate() call.
        """
        self._embed_fn = embed_fn
        self._model = None
    
    def _get_embed_fn(self, model_name: str) -> EmbedFunction:
        """Get or create embedding function, lazy-loading model if needed."""
        if self._embed_fn is not None:
            return self._embed_fn
        
        # Lazy-load sentence-transformers model
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                _log("info", f"Loaded sentence-transformers model: {model_name}")
            except Exception as e:
                raise ImportError(
                    f"Failed to load sentence-transformers model '{model_name}': {e}"
                )
        
        def embed_fn(texts: list[str]) -> list[list[float]]:
            embeddings = self._model.encode(texts)
            return embeddings.tolist()
        
        return embed_fn
    
    def evaluate(
        self,
        local: TaskResponse,
        remote: TaskResponse,
        task_config: TaskEvaluatorConfig,
    ) -> EvaluationResult:
        """Evaluate by computing cosine similarity of output embeddings.
        
        Returns cosine similarity clamped to [0.0, 1.0].
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        model_name = task_config.model_name
        
        try:
            # Check for empty outputs
            if not local.output or not remote.output:
                _log("error", "Empty output detected")
                return EvaluationResult(
                    score=0.0,
                    evaluator_type="semantic_similarity",
                    field_breakdown={},
                    error="Empty output cannot be embedded",
                    metadata={"model_name": model_name},
                    timestamp=timestamp,
                )
            
            # Get embedding function
            try:
                embed_fn = self._get_embed_fn(model_name)
            except Exception as e:
                _log("error", f"Embedding model load failure: {e}")
                return EvaluationResult(
                    score=0.0,
                    evaluator_type="semantic_similarity",
                    field_breakdown={},
                    error=f"Embedding model load failure: {str(e)}",
                    metadata={"model_name": model_name},
                    timestamp=timestamp,
                )
            
            # Compute embeddings
            try:
                embeddings = embed_fn([local.output, remote.output])
                local_embedding = embeddings[0]
                remote_embedding = embeddings[1]
            except Exception as e:
                _log("error", f"Embedding computation error: {e}")
                return EvaluationResult(
                    score=0.0,
                    evaluator_type="semantic_similarity",
                    field_breakdown={},
                    error=f"Embedding computation error: {str(e)}",
                    metadata={"model_name": model_name},
                    timestamp=timestamp,
                )
            
            # Compute cosine similarity
            try:
                raw_similarity = cosine_similarity(local_embedding, remote_embedding)
            except Exception as e:
                _log("error", f"Cosine similarity error: {e}")
                return EvaluationResult(
                    score=0.0,
                    evaluator_type="semantic_similarity",
                    field_breakdown={},
                    error=f"Cosine similarity error: {str(e)}",
                    metadata={"model_name": model_name},
                    timestamp=timestamp,
                )
            
            # Map negative similarity to 0.0, clamp to [0.0, 1.0]
            score = max(0.0, raw_similarity)
            score = clamp_score(score)
            
            _log(
                "debug",
                f"SemanticSimilarityEvaluator: score={score}, "
                f"raw_cosine={raw_similarity}"
            )
            
            return EvaluationResult(
                score=score,
                evaluator_type="semantic_similarity",
                field_breakdown={},
                error=None,
                metadata={
                    "model_name": model_name,
                    "raw_cosine_similarity": raw_similarity,
                },
                timestamp=timestamp,
            )
            
        except Exception as e:
            _log("error", f"SemanticSimilarityEvaluator error: {e}")
            return EvaluationResult(
                score=0.0,
                evaluator_type="semantic_similarity",
                field_breakdown={},
                error=f"Evaluation error: {str(e)}",
                metadata={"model_name": model_name},
                timestamp=timestamp,
            )


class FunctionEvaluatorAdapter:
    """Wraps a plain callable into an object satisfying the Evaluator protocol."""
    
    def __init__(self, fn: Callable):
        """Wrap a callable as an Evaluator.
        
        Args:
            fn: Callable accepting (local, remote, task_config) and returning
                either a float score or an EvaluationResult.
                
        Raises:
            TypeError: If fn is not callable.
        """
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn).__name__}")
        self._fn = fn
    
    def evaluate(
        self,
        local: TaskResponse,
        remote: TaskResponse,
        task_config: TaskEvaluatorConfig,
    ) -> EvaluationResult:
        """Delegate to wrapped callable, wrapping result as needed.
        
        Returns EvaluationResult with clamped score. Catches all exceptions.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            result = self._fn(local, remote, task_config)
            
            # Handle different return types
            if isinstance(result, EvaluationResult):
                # Pass through EvaluationResult, but ensure score is clamped
                score = clamp_score(result.score)
                return EvaluationResult(
                    score=score,
                    evaluator_type="custom",
                    field_breakdown=result.field_breakdown,
                    error=result.error,
                    metadata=result.metadata,
                    timestamp=timestamp,
                )
            elif isinstance(result, (int, float)):
                # Wrap numeric score
                score = clamp_score(float(result))
                return EvaluationResult(
                    score=score,
                    evaluator_type="custom",
                    field_breakdown={},
                    error=None,
                    metadata={},
                    timestamp=timestamp,
                )
            else:
                # Invalid return type
                _log(
                    "error",
                    f"Custom evaluator returned invalid type: {type(result).__name__}"
                )
                return EvaluationResult(
                    score=0.0,
                    evaluator_type="custom",
                    field_breakdown={},
                    error=f"Invalid return type: {type(result).__name__}",
                    metadata={},
                    timestamp=timestamp,
                )
                
        except Exception as e:
            _log("error", f"Custom evaluator exception: {e}")
            return EvaluationResult(
                score=0.0,
                evaluator_type="custom",
                field_breakdown={},
                error=f"Custom function exception: {str(e)}",
                metadata={},
                timestamp=timestamp,
            )


# ============================================================================
# Factory Function
# ============================================================================


def create_evaluator(
    task_config: TaskEvaluatorConfig,
    embed_fn: Optional[EmbedFunction] = None,
) -> EvaluatorProtocol:
    """Factory function to create evaluator from config.
    
    Args:
        task_config: Validated TaskEvaluatorConfig
        embed_fn: Optional embedding function for SemanticSimilarityEvaluator
        
    Returns:
        Evaluator instance satisfying EvaluatorProtocol
        
    Raises:
        ValueError: If evaluator_type is unknown
        ImportError: If custom evaluator import fails
    """
    evaluator_type = task_config.evaluator_type
    
    if evaluator_type == EvaluatorType.exact_match:
        return ExactMatchEvaluator()
    elif evaluator_type == EvaluatorType.structured_match:
        return StructuredMatchEvaluator()
    elif evaluator_type == EvaluatorType.semantic_similarity:
        return SemanticSimilarityEvaluator(embed_fn=embed_fn)
    elif evaluator_type == EvaluatorType.custom:
        custom_evaluator = task_config.custom_evaluator
        if custom_evaluator is None:
            raise ValueError(
                "custom_evaluator must be provided when evaluator_type is 'custom'"
            )
        # If custom_evaluator is already a callable, wrap it
        if callable(custom_evaluator):
            return FunctionEvaluatorAdapter(custom_evaluator)
        # Otherwise, try to import it
        try:
            if isinstance(custom_evaluator, str):
                # Import from dotted path
                module_path, attr_name = custom_evaluator.rsplit('.', 1)
                import importlib
                module = importlib.import_module(module_path)
                fn = getattr(module, attr_name)
                return FunctionEvaluatorAdapter(fn)
            else:
                raise ImportError(
                    f"custom_evaluator must be callable or import string, "
                    f"got {type(custom_evaluator).__name__}"
                )
        except Exception as e:
            raise ImportError(
                f"Failed to import custom evaluator '{custom_evaluator}': {e}"
            )
    else:
        raise ValueError(f"Unknown evaluator_type: {evaluator_type}")


# ============================================================================
# Exports
# ============================================================================

# For compatibility with tests that import ValidationError
from pydantic import ValidationError
from builtins import ImportError

__all__ = [
    'EvaluatorType',
    'TaskResponse',
    'TaskEvaluatorConfig',
    'FieldEvaluation',
    'EvaluationResult',
    'EvaluatorProtocol',
    'ValidationError',
    'create_evaluator',
    'ImportError',
    'clamp_score',
    'cosine_similarity',
    'ExactMatchEvaluator',
    'StructuredMatchEvaluator',
    'SemanticSimilarityEvaluator',
    'FunctionEvaluatorAdapter',
    'EmbedFunction',
]
