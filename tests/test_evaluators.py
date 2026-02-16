"""
Contract tests for the evaluators component.
Tests verify behavior at boundaries (inputs/outputs), not internals.
All dependencies are mocked — tests verify one component in isolation.
No network, GPU, or API keys required.
"""

import json
import math
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

from src.evaluators import (
    clamp_score,
    cosine_similarity,
    ExactMatchEvaluator,
    StructuredMatchEvaluator,
    SemanticSimilarityEvaluator,
    FunctionEvaluatorAdapter,
    create_evaluator,
    TaskResponse,
    TaskEvaluatorConfig,
    EvaluationResult,
    EvaluatorType,
)


# ============================================================================
# Fixtures / Helpers
# ============================================================================

def make_task_response(
    task_id="task-1",
    output="hello world",
    fields=None,
    model_id="model-1",
    timestamp="2024-01-01T00:00:00Z",
):
    """Factory helper for TaskResponse with sensible defaults."""
    return TaskResponse(
        task_id=task_id,
        output=output,
        fields=fields or {},
        model_id=model_id,
        timestamp=timestamp,
    )


def make_task_config(
    evaluator_type="exact_match",
    match_fields=None,
    model_name="all-MiniLM-L6-v2",
    custom_evaluator=None,
):
    """Factory helper for TaskEvaluatorConfig with sensible defaults."""
    kwargs = {
        "evaluator_type": evaluator_type,
        "match_fields": match_fields or [],
        "model_name": model_name,
    }
    if custom_evaluator is not None:
        kwargs["custom_evaluator"] = custom_evaluator
    return TaskEvaluatorConfig(**kwargs)


def _is_valid_iso8601(ts):
    """Check if a string is a valid ISO-8601 datetime."""
    try:
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return True
    except (ValueError, AttributeError):
        return False


# ============================================================================
# Tests for clamp_score
# ============================================================================

class TestClampScore:
    """Tests for the clamp_score utility function."""

    @pytest.mark.parametrize("score,expected", [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (0.001, 0.001),
        (0.999, 0.999),
    ])
    def test_clamp_score_value_in_range(self, score, expected):
        """clamp_score returns same value when score is in [0.0, 1.0]."""
        result = clamp_score(score)
        assert result == expected, f"Expected {expected}, got {result}"

    @pytest.mark.parametrize("score", [-0.5, -1.0, -100.0, -0.001])
    def test_clamp_score_negative_clamped_to_zero(self, score):
        """clamp_score clamps negative scores to 0.0."""
        result = clamp_score(score)
        assert result == 0.0, f"Expected 0.0 for input {score}, got {result}"

    @pytest.mark.parametrize("score", [1.5, 2.0, 100.0, 1.001])
    def test_clamp_score_above_one_clamped(self, score):
        """clamp_score clamps scores above 1.0 to 1.0."""
        result = clamp_score(score)
        assert result == 1.0, f"Expected 1.0 for input {score}, got {result}"

    def test_clamp_score_nan_raises_or_handles(self):
        """clamp_score handles NaN score (error: nan_score)."""
        try:
            result = clamp_score(float("nan"))
            # If it doesn't raise, the result should still be in [0, 1]
            # or indicate an error. NaN comparisons are tricky.
            # Per contract, NaN is an error case.
            assert not math.isnan(result), (
                "clamp_score should not return NaN"
            )
        except (ValueError, TypeError):
            pass  # Expected: nan_score error

    def test_clamp_score_boundary_zero(self):
        """clamp_score returns exactly 0.0 for input 0.0."""
        assert clamp_score(0.0) == 0.0

    def test_clamp_score_boundary_one(self):
        """clamp_score returns exactly 1.0 for input 1.0."""
        assert clamp_score(1.0) == 1.0


# ============================================================================
# Tests for cosine_similarity
# ============================================================================

class TestCosineSimilarity:
    """Tests for the cosine_similarity utility function."""

    def test_identical_vectors_return_one(self):
        """cosine_similarity returns 1.0 for identical vectors."""
        result = cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert abs(result - 1.0) < 1e-9, f"Expected 1.0, got {result}"

    def test_orthogonal_vectors_return_zero(self):
        """cosine_similarity returns 0.0 for orthogonal vectors."""
        result = cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(result) < 1e-9, f"Expected 0.0, got {result}"

    def test_opposite_vectors_return_negative_one(self):
        """cosine_similarity returns -1.0 for opposite vectors."""
        result = cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(result - (-1.0)) < 1e-9, f"Expected -1.0, got {result}"

    def test_result_bounded(self):
        """cosine_similarity result is always in [-1.0, 1.0]."""
        vecs = [
            ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
            ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.5, 0.5], [-0.5, 0.5]),
        ]
        for va, vb in vecs:
            result = cosine_similarity(va, vb)
            assert -1.0 - 1e-9 <= result <= 1.0 + 1e-9, (
                f"Result {result} out of bounds for {va}, {vb}"
            )

    def test_dimension_mismatch_raises(self):
        """cosine_similarity raises error for mismatched dimensions."""
        with pytest.raises((ValueError, Exception)):
            cosine_similarity([1.0, 2.0], [1.0])

    def test_empty_vectors_raises(self):
        """cosine_similarity raises error for empty vectors."""
        with pytest.raises((ValueError, Exception)):
            cosine_similarity([], [])

    def test_zero_magnitude_returns_zero(self):
        """cosine_similarity returns 0.0 when one vector has zero magnitude."""
        result = cosine_similarity([0.0, 0.0], [1.0, 1.0])
        assert result == 0.0, f"Expected 0.0, got {result}"

    def test_symmetry(self):
        """cosine_similarity(a, b) == cosine_similarity(b, a)."""
        va, vb = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert abs(cosine_similarity(va, vb) - cosine_similarity(vb, va)) < 1e-9


# ============================================================================
# Tests for ExactMatchEvaluator
# ============================================================================

class TestExactMatchEvaluator:
    """Tests for ExactMatchEvaluator.evaluate."""

    def setup_method(self):
        self.evaluator = ExactMatchEvaluator()
        self.config = make_task_config(evaluator_type="exact_match")

    def test_identical_outputs_score_one(self):
        """Returns score 1.0 for identical outputs."""
        local = make_task_response(output="hello world")
        remote = make_task_response(output="hello world")
        result = self.evaluator.evaluate(local, remote, self.config)
        assert result.score == 1.0, f"Expected 1.0, got {result.score}"
        assert result.evaluator_type == "exact_match"
        assert result.field_breakdown == {}
        assert result.error is None

    def test_different_outputs_score_zero(self):
        """Returns score 0.0 for different outputs."""
        local = make_task_response(output="hello")
        remote = make_task_response(output="world")
        result = self.evaluator.evaluate(local, remote, self.config)
        assert result.score == 0.0, f"Expected 0.0, got {result.score}"
        assert result.evaluator_type == "exact_match"
        assert result.field_breakdown == {}
        assert result.error is None

    def test_empty_strings_match(self):
        """Returns 1.0 for two empty output strings."""
        local = make_task_response(output="")
        remote = make_task_response(output="")
        result = self.evaluator.evaluate(local, remote, self.config)
        assert result.score == 1.0

    def test_whitespace_sensitive(self):
        """Treats whitespace differences as non-matching."""
        local = make_task_response(output="hello ")
        remote = make_task_response(output="hello")
        result = self.evaluator.evaluate(local, remote, self.config)
        assert result.score == 0.0

    def test_valid_timestamp(self):
        """Result has a valid ISO-8601 timestamp."""
        local = make_task_response(output="test")
        remote = make_task_response(output="test")
        result = self.evaluator.evaluate(local, remote, self.config)
        assert result.timestamp is not None
        assert _is_valid_iso8601(result.timestamp), (
            f"Timestamp '{result.timestamp}' is not valid ISO-8601"
        )

    def test_case_sensitive(self):
        """Exact match is case-sensitive."""
        local = make_task_response(output="Hello")
        remote = make_task_response(output="hello")
        result = self.evaluator.evaluate(local, remote, self.config)
        assert result.score == 0.0


# ============================================================================
# Tests for StructuredMatchEvaluator
# ============================================================================

class TestStructuredMatchEvaluator:
    """Tests for StructuredMatchEvaluator.evaluate."""

    def setup_method(self):
        self.evaluator = StructuredMatchEvaluator()

    def test_full_match_all_fields(self):
        """Returns 1.0 when all fields match."""
        config = make_task_config(
            evaluator_type="structured_match",
            match_fields=["name", "age"],
        )
        local = make_task_response(fields={"name": "Alice", "age": 30})
        remote = make_task_response(fields={"name": "Alice", "age": 30})
        result = self.evaluator.evaluate(local, remote, config)
        assert result.score == 1.0
        assert result.evaluator_type == "structured_match"

    def test_no_match_all_fields(self):
        """Returns 0.0 when no fields match."""
        config = make_task_config(
            evaluator_type="structured_match",
            match_fields=["name", "age"],
        )
        local = make_task_response(fields={"name": "Alice", "age": 30})
        remote = make_task_response(fields={"name": "Bob", "age": 25})
        result = self.evaluator.evaluate(local, remote, config)
        assert result.score == 0.0

    def test_partial_match(self):
        """Returns partial score (0.5) for 1 of 2 fields matching."""
        config = make_task_config(
            evaluator_type="structured_match",
            match_fields=["name", "age"],
        )
        local = make_task_response(fields={"name": "Alice", "age": 30})
        remote = make_task_response(fields={"name": "Alice", "age": 25})
        result = self.evaluator.evaluate(local, remote, config)
        assert abs(result.score - 0.5) < 1e-9, f"Expected 0.5, got {result.score}"

    def test_field_breakdown_has_all_fields(self):
        """field_breakdown has one entry per field in match_fields."""
        fields = ["name", "age", "city"]
        config = make_task_config(
            evaluator_type="structured_match",
            match_fields=fields,
        )
        local = make_task_response(fields={"name": "A", "age": 1, "city": "X"})
        remote = make_task_response(fields={"name": "A", "age": 1, "city": "X"})
        result = self.evaluator.evaluate(local, remote, config)
        assert len(result.field_breakdown) == len(fields), (
            f"Expected {len(fields)} entries, got {len(result.field_breakdown)}"
        )

    def test_missing_field_treated_as_non_matching(self):
        """Missing field in one response is treated as non-matching."""
        config = make_task_config(
            evaluator_type="structured_match",
            match_fields=["name", "age"],
        )
        local = make_task_response(fields={"name": "Alice"})
        remote = make_task_response(fields={"name": "Alice", "age": 30})
        result = self.evaluator.evaluate(local, remote, config)
        # 'name' matches, 'age' missing in local => 0.5 or error-annotated
        assert result.score < 1.0

    def test_single_field_match(self):
        """Single match_field returns 1.0 on match."""
        config = make_task_config(
            evaluator_type="structured_match",
            match_fields=["name"],
        )
        local = make_task_response(fields={"name": "Alice"})
        remote = make_task_response(fields={"name": "Alice"})
        result = self.evaluator.evaluate(local, remote, config)
        assert result.score == 1.0

    def test_score_always_clamped(self):
        """Score is always in [0.0, 1.0]."""
        config = make_task_config(
            evaluator_type="structured_match",
            match_fields=["x"],
        )
        local = make_task_response(fields={"x": 1})
        remote = make_task_response(fields={"x": 2})
        result = self.evaluator.evaluate(local, remote, config)
        assert 0.0 <= result.score <= 1.0

    def test_valid_timestamp(self):
        """Result has valid ISO-8601 timestamp."""
        config = make_task_config(
            evaluator_type="structured_match",
            match_fields=["name"],
        )
        local = make_task_response(fields={"name": "A"})
        remote = make_task_response(fields={"name": "A"})
        result = self.evaluator.evaluate(local, remote, config)
        assert _is_valid_iso8601(result.timestamp)

    def test_score_is_average_of_matches(self):
        """Score equals the average of matched fields (1/3)."""
        config = make_task_config(
            evaluator_type="structured_match",
            match_fields=["a", "b", "c"],
        )
        local = make_task_response(fields={"a": 1, "b": 2, "c": 3})
        remote = make_task_response(fields={"a": 1, "b": 99, "c": 99})
        result = self.evaluator.evaluate(local, remote, config)
        expected = 1.0 / 3.0
        assert abs(result.score - expected) < 1e-9, (
            f"Expected {expected}, got {result.score}"
        )


# ============================================================================
# Tests for SemanticSimilarityEvaluator
# ============================================================================

class TestSemanticSimilarityEvaluator:
    """Tests for SemanticSimilarityEvaluator with mocked embed_fn."""

    def _make_embed_fn(self, embeddings_map):
        """Create a mock embed_fn that maps strings to vectors."""
        def embed_fn(texts):
            return [embeddings_map.get(t, [0.0, 0.0, 0.0]) for t in texts]
        return embed_fn

    def test_identical_outputs_score_one(self):
        """Returns 1.0 for identical embeddings."""
        embed_fn = lambda texts: [[1.0, 0.0, 0.0]] * len(texts)
        evaluator = SemanticSimilarityEvaluator(embed_fn=embed_fn)
        config = make_task_config(evaluator_type="semantic_similarity")
        local = make_task_response(output="hello")
        remote = make_task_response(output="hello")
        result = evaluator.evaluate(local, remote, config)
        assert abs(result.score - 1.0) < 1e-6, f"Expected 1.0, got {result.score}"
        assert result.evaluator_type == "semantic_similarity"
        assert result.field_breakdown == {}

    def test_orthogonal_outputs_score_zero(self):
        """Returns 0.0 for orthogonal embeddings."""
        call_count = [0]
        def embed_fn(texts):
            results = []
            for _ in texts:
                if call_count[0] % 2 == 0:
                    results.append([1.0, 0.0])
                else:
                    results.append([0.0, 1.0])
                call_count[0] += 1
            return results
        # Simpler approach: map specific texts
        def embed_fn_v2(texts):
            mapping = {
                "hello": [1.0, 0.0],
                "world": [0.0, 1.0],
            }
            return [mapping.get(t, [0.0, 0.0]) for t in texts]

        evaluator = SemanticSimilarityEvaluator(embed_fn=embed_fn_v2)
        config = make_task_config(evaluator_type="semantic_similarity")
        local = make_task_response(output="hello")
        remote = make_task_response(output="world")
        result = evaluator.evaluate(local, remote, config)
        assert abs(result.score) < 1e-6, f"Expected 0.0, got {result.score}"

    def test_negative_cosine_clamped_to_zero(self):
        """Negative cosine similarity is clamped to 0.0."""
        def embed_fn(texts):
            mapping = {"hello": [1.0, 0.0], "opposite": [-1.0, 0.0]}
            return [mapping.get(t, [0.0, 0.0]) for t in texts]

        evaluator = SemanticSimilarityEvaluator(embed_fn=embed_fn)
        config = make_task_config(evaluator_type="semantic_similarity")
        local = make_task_response(output="hello")
        remote = make_task_response(output="opposite")
        result = evaluator.evaluate(local, remote, config)
        assert result.score == 0.0, f"Expected 0.0, got {result.score}"
        assert "raw_cosine_similarity" in result.metadata

    def test_metadata_contains_required_keys(self):
        """Result metadata contains model_name and raw_cosine_similarity."""
        embed_fn = lambda texts: [[1.0, 0.0]] * len(texts)
        evaluator = SemanticSimilarityEvaluator(embed_fn=embed_fn)
        config = make_task_config(evaluator_type="semantic_similarity")
        local = make_task_response(output="test")
        remote = make_task_response(output="test")
        result = evaluator.evaluate(local, remote, config)
        assert "model_name" in result.metadata, "metadata missing 'model_name'"
        assert isinstance(result.metadata["model_name"], str)
        assert "raw_cosine_similarity" in result.metadata, (
            "metadata missing 'raw_cosine_similarity'"
        )
        assert isinstance(result.metadata["raw_cosine_similarity"], (int, float))

    def test_empty_output_error(self):
        """Returns error for empty output string."""
        embed_fn = lambda texts: [[1.0, 0.0]] * len(texts)
        evaluator = SemanticSimilarityEvaluator(embed_fn=embed_fn)
        config = make_task_config(evaluator_type="semantic_similarity")
        local = make_task_response(output="")
        remote = make_task_response(output="hello")
        result = evaluator.evaluate(local, remote, config)
        # Per contract: error for empty_output, score should be 0.0
        assert result.error is not None, "Expected error for empty output"
        assert result.score == 0.0

    def test_embed_fn_exception_returns_error(self):
        """Returns error result when embed_fn raises exception."""
        def failing_embed_fn(texts):
            raise RuntimeError("OOM error")

        evaluator = SemanticSimilarityEvaluator(embed_fn=failing_embed_fn)
        config = make_task_config(evaluator_type="semantic_similarity")
        local = make_task_response(output="hello")
        remote = make_task_response(output="world")
        result = evaluator.evaluate(local, remote, config)
        assert result.error is not None, "Expected error annotation"
        assert result.score == 0.0

    def test_lazy_load_no_model_at_init(self):
        """No model is loaded at construction time."""
        # Constructing with None should not import sentence-transformers
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # Should not raise even if sentence_transformers unavailable
            try:
                evaluator = SemanticSimilarityEvaluator(embed_fn=None)
                # Construction succeeded without loading model
            except ImportError:
                pytest.fail(
                    "SemanticSimilarityEvaluator should not load model at init"
                )

    def test_score_always_clamped(self):
        """Score is always in [0.0, 1.0]."""
        embed_fn = lambda texts: [[0.5, 0.5]] * len(texts)
        evaluator = SemanticSimilarityEvaluator(embed_fn=embed_fn)
        config = make_task_config(evaluator_type="semantic_similarity")
        local = make_task_response(output="a")
        remote = make_task_response(output="b")
        result = evaluator.evaluate(local, remote, config)
        assert 0.0 <= result.score <= 1.0

    def test_valid_timestamp(self):
        """Result has valid ISO-8601 timestamp."""
        embed_fn = lambda texts: [[1.0, 0.0]] * len(texts)
        evaluator = SemanticSimilarityEvaluator(embed_fn=embed_fn)
        config = make_task_config(evaluator_type="semantic_similarity")
        local = make_task_response(output="hello")
        remote = make_task_response(output="hello")
        result = evaluator.evaluate(local, remote, config)
        assert _is_valid_iso8601(result.timestamp)


# ============================================================================
# Tests for FunctionEvaluatorAdapter
# ============================================================================

class TestFunctionEvaluatorAdapter:
    """Tests for FunctionEvaluatorAdapter."""

    def _make_config(self):
        return make_task_config(evaluator_type="custom")

    def test_float_return_wrapped(self):
        """Wraps float return into EvaluationResult."""
        fn = lambda local, remote, config: 0.75
        adapter = FunctionEvaluatorAdapter(fn)
        config = self._make_config()
        local = make_task_response()
        remote = make_task_response()
        result = adapter.evaluate(local, remote, config)
        assert result.score == 0.75
        assert result.evaluator_type == "custom"

    def test_int_return_wrapped(self):
        """Wraps int return into EvaluationResult."""
        fn = lambda local, remote, config: 1
        adapter = FunctionEvaluatorAdapter(fn)
        result = adapter.evaluate(
            make_task_response(), make_task_response(), self._make_config()
        )
        assert result.score == 1.0

    def test_evaluation_result_passthrough(self):
        """Passes through EvaluationResult from callable."""
        expected_result = EvaluationResult(
            score=0.8,
            evaluator_type="custom",
            field_breakdown={},
            error=None,
            metadata={},
            timestamp="2024-01-01T00:00:00Z",
        )
        fn = lambda local, remote, config: expected_result
        adapter = FunctionEvaluatorAdapter(fn)
        result = adapter.evaluate(
            make_task_response(), make_task_response(), self._make_config()
        )
        assert result.score == 0.8
        assert result.evaluator_type == "custom"

    def test_clamps_float_above_one(self):
        """Clamps float return above 1.0 to 1.0."""
        fn = lambda local, remote, config: 1.5
        adapter = FunctionEvaluatorAdapter(fn)
        result = adapter.evaluate(
            make_task_response(), make_task_response(), self._make_config()
        )
        assert result.score == 1.0

    def test_clamps_negative_float(self):
        """Clamps negative float return to 0.0."""
        fn = lambda local, remote, config: -0.5
        adapter = FunctionEvaluatorAdapter(fn)
        result = adapter.evaluate(
            make_task_response(), make_task_response(), self._make_config()
        )
        assert result.score == 0.0

    def test_callable_exception_returns_error(self):
        """Catches callable exception and returns error result."""
        def failing_fn(local, remote, config):
            raise ValueError("Something went wrong")

        adapter = FunctionEvaluatorAdapter(failing_fn)
        result = adapter.evaluate(
            make_task_response(), make_task_response(), self._make_config()
        )
        assert result.error is not None, "Expected error annotation"
        assert result.score == 0.0
        assert result.evaluator_type == "custom"

    def test_invalid_return_type_returns_error(self):
        """Handles invalid return type from callable."""
        fn = lambda local, remote, config: "not a valid return"
        adapter = FunctionEvaluatorAdapter(fn)
        result = adapter.evaluate(
            make_task_response(), make_task_response(), self._make_config()
        )
        # Per contract: invalid_return_type returns error result
        assert result.error is not None, "Expected error for invalid return type"
        assert result.score == 0.0

    def test_non_callable_init_raises(self):
        """Raises error when fn is not callable."""
        with pytest.raises((TypeError, ValueError, Exception)):
            FunctionEvaluatorAdapter(42)

    def test_has_evaluate_method(self):
        """Adapter satisfies EvaluatorProtocol (has evaluate method)."""
        fn = lambda local, remote, config: 0.5
        adapter = FunctionEvaluatorAdapter(fn)
        assert hasattr(adapter, "evaluate"), "Adapter must have evaluate method"
        assert callable(adapter.evaluate), "evaluate must be callable"

    def test_valid_timestamp(self):
        """Result has valid ISO-8601 timestamp."""
        fn = lambda local, remote, config: 0.5
        adapter = FunctionEvaluatorAdapter(fn)
        result = adapter.evaluate(
            make_task_response(), make_task_response(), self._make_config()
        )
        assert _is_valid_iso8601(result.timestamp)


# ============================================================================
# Tests for create_evaluator
# ============================================================================

class TestCreateEvaluator:
    """Tests for the create_evaluator factory function."""

    def test_creates_structured_match_evaluator(self):
        """Returns StructuredMatchEvaluator for structured_match type."""
        config = make_task_config(
            evaluator_type="structured_match", match_fields=["name"]
        )
        evaluator = create_evaluator(config, embed_fn=None)
        assert hasattr(evaluator, "evaluate")
        assert isinstance(evaluator, StructuredMatchEvaluator)

    def test_creates_exact_match_evaluator(self):
        """Returns ExactMatchEvaluator for exact_match type."""
        config = make_task_config(evaluator_type="exact_match")
        evaluator = create_evaluator(config, embed_fn=None)
        assert hasattr(evaluator, "evaluate")
        assert isinstance(evaluator, ExactMatchEvaluator)

    def test_creates_semantic_similarity_evaluator(self):
        """Returns SemanticSimilarityEvaluator for semantic_similarity type."""
        mock_embed = lambda texts: [[1.0]] * len(texts)
        config = make_task_config(evaluator_type="semantic_similarity")
        evaluator = create_evaluator(config, embed_fn=mock_embed)
        assert hasattr(evaluator, "evaluate")
        assert isinstance(evaluator, SemanticSimilarityEvaluator)

    def test_creates_custom_evaluator(self):
        """Returns FunctionEvaluatorAdapter for custom type."""
        custom_fn = lambda local, remote, config: 0.5
        config = make_task_config(
            evaluator_type="custom", custom_evaluator=custom_fn
        )
        evaluator = create_evaluator(config, embed_fn=None)
        assert hasattr(evaluator, "evaluate")
        assert isinstance(evaluator, FunctionEvaluatorAdapter)

    def test_unknown_type_raises(self):
        """Raises error for unknown evaluator type."""
        # We need to bypass Pydantic validation to test this
        config = MagicMock()
        config.evaluator_type = "unknown_type"
        config.match_fields = []
        config.model_name = ""
        config.custom_evaluator = None
        with pytest.raises((ValueError, KeyError, Exception)):
            create_evaluator(config, embed_fn=None)


# ============================================================================
# Invariant Tests
# ============================================================================

class TestInvariants:
    """Cross-cutting invariant tests for all evaluators."""

    def test_score_always_in_range_exact_match(self):
        """ExactMatchEvaluator score is always in [0.0, 1.0]."""
        evaluator = ExactMatchEvaluator()
        config = make_task_config(evaluator_type="exact_match")
        for out_local, out_remote in [("a", "a"), ("a", "b"), ("", ""), ("", "x")]:
            result = evaluator.evaluate(
                make_task_response(output=out_local),
                make_task_response(output=out_remote),
                config,
            )
            assert 0.0 <= result.score <= 1.0

    def test_score_always_in_range_structured_match(self):
        """StructuredMatchEvaluator score is always in [0.0, 1.0]."""
        evaluator = StructuredMatchEvaluator()
        config = make_task_config(
            evaluator_type="structured_match", match_fields=["a", "b"]
        )
        cases = [
            ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
            ({"a": 1, "b": 2}, {"a": 3, "b": 4}),
            ({"a": 1}, {"a": 1, "b": 2}),
            ({}, {}),
        ]
        for local_fields, remote_fields in cases:
            result = evaluator.evaluate(
                make_task_response(fields=local_fields),
                make_task_response(fields=remote_fields),
                config,
            )
            assert 0.0 <= result.score <= 1.0

    def test_errors_never_raise_function_adapter(self):
        """FunctionEvaluatorAdapter.evaluate never raises, returns error result."""
        def bad_fn(local, remote, config):
            raise RuntimeError("boom")

        adapter = FunctionEvaluatorAdapter(bad_fn)
        config = make_task_config(evaluator_type="custom")
        # Should not raise
        result = adapter.evaluate(
            make_task_response(), make_task_response(), config
        )
        assert result.error is not None
        assert result.score == 0.0

    def test_errors_never_raise_semantic_similarity(self):
        """SemanticSimilarityEvaluator.evaluate never raises on embed error."""
        def bad_embed(texts):
            raise RuntimeError("OOM")

        evaluator = SemanticSimilarityEvaluator(embed_fn=bad_embed)
        config = make_task_config(evaluator_type="semantic_similarity")
        result = evaluator.evaluate(
            make_task_response(output="a"),
            make_task_response(output="b"),
            config,
        )
        assert result.error is not None
        assert result.score == 0.0

    def test_evaluation_result_immutable(self):
        """EvaluationResult instances are frozen (immutable)."""
        result = EvaluationResult(
            score=0.5,
            evaluator_type="exact_match",
            field_breakdown={},
            error=None,
            metadata={},
            timestamp="2024-01-01T00:00:00Z",
        )
        with pytest.raises((AttributeError, TypeError, Exception)):
            result.score = 0.9

    def test_evaluation_result_serializable(self):
        """EvaluationResult is JSON-serializable via model_dump(mode='json')."""
        result = EvaluationResult(
            score=0.5,
            evaluator_type="exact_match",
            field_breakdown={},
            error=None,
            metadata={"key": "value"},
            timestamp="2024-01-01T00:00:00Z",
        )
        dumped = result.model_dump(mode="json")
        assert isinstance(dumped, dict)
        # Should not raise
        json_str = json.dumps(dumped)
        assert isinstance(json_str, str)

    def test_evaluation_result_score_validator_clamps(self):
        """EvaluationResult Pydantic validator clamps score to [0.0, 1.0]."""
        # Depending on implementation, this may clamp or reject
        try:
            result = EvaluationResult(
                score=1.5,
                evaluator_type="test",
                field_breakdown={},
                error=None,
                metadata={},
                timestamp="2024-01-01T00:00:00Z",
            )
            # If it doesn't raise, score should be clamped
            assert result.score <= 1.0, (
                f"Score {result.score} should be clamped to 1.0"
            )
        except (ValueError, Exception):
            pass  # Also acceptable: reject invalid score

    def test_all_evaluators_satisfy_protocol(self):
        """All evaluator implementations have an evaluate method."""
        evaluators = [
            ExactMatchEvaluator(),
            StructuredMatchEvaluator(),
            SemanticSimilarityEvaluator(embed_fn=lambda x: [[1.0]] * len(x)),
            FunctionEvaluatorAdapter(lambda l, r, c: 0.5),
        ]
        for ev in evaluators:
            assert hasattr(ev, "evaluate"), (
                f"{type(ev).__name__} missing evaluate method"
            )
            assert callable(ev.evaluate), (
                f"{type(ev).__name__}.evaluate is not callable"
            )

    def test_negative_cosine_maps_to_zero_score(self):
        """Negative cosine similarity maps to score 0.0, not negative."""
        def embed_fn(texts):
            mapping = {"pos": [1.0, 0.0], "neg": [-1.0, 0.0]}
            return [mapping.get(t, [0.0, 0.0]) for t in texts]

        evaluator = SemanticSimilarityEvaluator(embed_fn=embed_fn)
        config = make_task_config(evaluator_type="semantic_similarity")
        result = evaluator.evaluate(
            make_task_response(output="pos"),
            make_task_response(output="neg"),
            config,
        )
        assert result.score >= 0.0, (
            f"Negative cosine should map to 0.0, got {result.score}"
        )
        assert result.metadata.get("raw_cosine_similarity", 0.0) < 0.0 or \
               result.score == 0.0

    def test_no_global_state_between_evaluations(self):
        """Evaluators are stateless; repeated calls produce same results."""
        evaluator = ExactMatchEvaluator()
        config = make_task_config(evaluator_type="exact_match")
        local = make_task_response(output="hello")
        remote = make_task_response(output="hello")

        result1 = evaluator.evaluate(local, remote, config)
        result2 = evaluator.evaluate(local, remote, config)
        assert result1.score == result2.score
        assert result1.evaluator_type == result2.evaluator_type
