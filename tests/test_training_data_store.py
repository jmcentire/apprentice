"""
Contract tests for TrainingDataStore component.

Organized into logical groups:
1. Initialization tests
2. Add example tests
3. Query/iterator tests
4. Split logic tests
5. Batch readiness tests
6. Persistence tests
7. Property-based / invariant tests
"""

import json
import os
import hashlib
import re
import struct
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

# Try importing with common module paths
try:
    from src.training_data_store import (
        TrainingDataStore,
        TrainingDataStoreConfig,
        TrainingExample,
        AddExampleResult,
        ExampleCount,
    )
except ImportError:
    try:
        from training_data_store import (
            TrainingDataStore,
            TrainingDataStoreConfig,
            TrainingExample,
            AddExampleResult,
            ExampleCount,
        )
    except ImportError:
        pass

# Try importing Hypothesis for property-based tests; skip if unavailable
try:
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

if not HAS_HYPOTHESIS:
    def given(*a, **kw):
        return lambda f: f
    class st:
        @staticmethod
        def floats(*a, **kw): return None
        @staticmethod
        def integers(*a, **kw): return None
        @staticmethod
        def text(*a, **kw): return None
        @staticmethod
        def booleans(*a, **kw): return None
        @staticmethod
        def lists(*a, **kw): return None
        @staticmethod
        def one_of(*a, **kw): return None
        @staticmethod
        def sampled_from(*a, **kw): return None
        @staticmethod
        def just(*a, **kw): return None
        @staticmethod
        def none(*a, **kw): return None
        @staticmethod
        def dictionaries(*a, **kw): return None
        @staticmethod
        def from_regex(*a, **kw): return None
        @staticmethod
        def datetimes(*a, **kw): return None
        @staticmethod
        def timedeltas(*a, **kw): return None
        @staticmethod
        def characters(*a, **kw): return None
        @staticmethod
        def decimals(*a, **kw): return None
    def assume(*a, **kw): pass
    def settings(*a, **kw):
        return lambda f: f


# ---------------------------------------------------------------------------
# Helper: reference split computation from the contract invariant
# ---------------------------------------------------------------------------
def reference_split(input_str: str, holdout_percentage: float) -> str:
    """
    Compute the split assignment per contract invariant:
    SHA-256(input.encode('utf-8')), last 4 bytes as unsigned int,
    modulo 10000, compared against holdout_percentage * 10000.
    Values < threshold are validation; values >= threshold are training.
    """
    digest = hashlib.sha256(input_str.encode("utf-8")).digest()
    value = struct.unpack(">I", digest[-4:])[0] % 10000
    threshold = holdout_percentage * 10000
    return "validation" if value < threshold else "training"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def store_config(tmp_path):
    """Returns a valid TrainingDataStoreConfig with sensible defaults."""
    return TrainingDataStoreConfig(
        base_dir=str(tmp_path / "training_data"),
        holdout_percentage=0.2,
        fine_tune_batch_size=10,
    )


@pytest.fixture
def initialized_store(store_config):
    """Returns a TrainingDataStore that has already been initialized."""
    store = TrainingDataStore()
    store.initialize(store_config)
    return store


@pytest.fixture
def sample_example():
    """Factory fixture that creates TrainingExample instances with controlled fields."""
    def _make(
        input_text="Summarize the following article about AI.",
        output_text="AI is transforming many industries.",
        task_type="summarization",
        timestamp="2024-01-15T10:30:00Z",
        metadata=None,
    ):
        return TrainingExample(
            input=input_text,
            output=output_text,
            task_type=task_type,
            timestamp=timestamp,
            metadata=metadata if metadata is not None else {},
        )
    return _make


# ===========================================================================
# 1. INITIALIZATION TESTS (test_training_data_store_init)
# ===========================================================================
class TestInitialize:
    """Tests for the initialize() function."""

    def test_init_fresh_directory(self, tmp_path):
        """initialize() creates base_dir, _counts.json, and prepares store."""
        base_dir = str(tmp_path / "new_training_data")
        config = TrainingDataStoreConfig(
            base_dir=base_dir,
            holdout_percentage=0.2,
            fine_tune_batch_size=10,
        )
        store = TrainingDataStore()
        store.initialize(config)

        assert os.path.isdir(base_dir), "base_dir should have been created"
        counts_path = os.path.join(base_dir, "_counts.json")
        assert os.path.isfile(counts_path), "_counts.json should exist after init"

    def test_init_existing_directory(self, tmp_path):
        """initialize() succeeds when base_dir and _counts.json already exist."""
        base_dir = str(tmp_path / "existing")
        os.makedirs(base_dir)
        counts_path = os.path.join(base_dir, "_counts.json")
        with open(counts_path, "w") as f:
            json.dump({}, f)

        config = TrainingDataStoreConfig(
            base_dir=base_dir,
            holdout_percentage=0.2,
            fine_tune_batch_size=10,
        )
        store = TrainingDataStore()
        store.initialize(config)

        assert os.path.isdir(base_dir)
        assert os.path.isfile(counts_path)

    def test_init_reconciles_missing_counts(self, tmp_path):
        """initialize() rebuilds _counts.json entries from existing JSONL files."""
        base_dir = str(tmp_path / "reconcile")
        os.makedirs(base_dir)

        # Write a JSONL file with 3 lines
        jsonl_path = os.path.join(base_dir, "summarization.jsonl")
        for i in range(3):
            example = {
                "input": f"input {i}",
                "output": f"output {i}",
                "task_type": "summarization",
                "timestamp": "2024-01-15T10:30:00Z",
                "metadata": {},
            }
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(example) + "\n")

        # Write _counts.json missing this task_type
        with open(os.path.join(base_dir, "_counts.json"), "w") as f:
            json.dump({}, f)

        config = TrainingDataStoreConfig(
            base_dir=base_dir,
            holdout_percentage=0.2,
            fine_tune_batch_size=10,
        )
        store = TrainingDataStore()
        store.initialize(config)

        # After reconciliation, counts should include summarization
        count = store.get_example_count("summarization")
        assert count.total == 3, (
            f"After reconciliation, total should be 3, got {count.total}"
        )

    def test_init_reconciles_corrupt_counts(self, tmp_path):
        """initialize() rebuilds _counts.json from JSONL when counts are corrupt."""
        base_dir = str(tmp_path / "corrupt_counts")
        os.makedirs(base_dir)

        # Write valid JSONL
        jsonl_path = os.path.join(base_dir, "qa.jsonl")
        example = {
            "input": "What is AI?",
            "output": "Artificial Intelligence",
            "task_type": "qa",
            "timestamp": "2024-01-15T10:30:00Z",
            "metadata": {},
        }
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(example) + "\n")

        # Write corrupt _counts.json
        with open(os.path.join(base_dir, "_counts.json"), "w") as f:
            f.write("{invalid json content!!!}")

        config = TrainingDataStoreConfig(
            base_dir=base_dir,
            holdout_percentage=0.2,
            fine_tune_batch_size=10,
        )
        store = TrainingDataStore()
        store.initialize(config)

        count = store.get_example_count("qa")
        assert count.total == 1, (
            "After rebuilding from corrupt _counts.json, total should be 1"
        )

    def test_init_config_validation_error_holdout_negative(self, tmp_path):
        """initialize() raises error when holdout_percentage is negative."""
        with pytest.raises(Exception) as exc_info:
            config = TrainingDataStoreConfig(
                base_dir=str(tmp_path / "data"),
                holdout_percentage=-0.1,
                fine_tune_batch_size=10,
            )
            store = TrainingDataStore()
            store.initialize(config)
        # Accept any validation error type
        assert exc_info.value is not None

    def test_init_config_validation_error_holdout_over_one(self, tmp_path):
        """initialize() raises error when holdout_percentage > 1.0."""
        with pytest.raises(Exception) as exc_info:
            config = TrainingDataStoreConfig(
                base_dir=str(tmp_path / "data"),
                holdout_percentage=1.5,
                fine_tune_batch_size=10,
            )
            store = TrainingDataStore()
            store.initialize(config)
        assert exc_info.value is not None

    def test_init_config_validation_error_batch_size_zero(self, tmp_path):
        """initialize() raises error when fine_tune_batch_size <= 0."""
        with pytest.raises(Exception) as exc_info:
            config = TrainingDataStoreConfig(
                base_dir=str(tmp_path / "data"),
                holdout_percentage=0.2,
                fine_tune_batch_size=0,
            )
            store = TrainingDataStore()
            store.initialize(config)
        assert exc_info.value is not None

    def test_init_config_validation_error_batch_size_negative(self, tmp_path):
        """initialize() raises error when fine_tune_batch_size is negative."""
        with pytest.raises(Exception) as exc_info:
            config = TrainingDataStoreConfig(
                base_dir=str(tmp_path / "data"),
                holdout_percentage=0.2,
                fine_tune_batch_size=-5,
            )
            store = TrainingDataStore()
            store.initialize(config)
        assert exc_info.value is not None

    def test_init_io_error_mkdir(self, tmp_path):
        """initialize() raises io_error_mkdir when directory creation fails."""
        # Use an invalid path or mock os.makedirs to raise PermissionError
        config = TrainingDataStoreConfig(
            base_dir=str(tmp_path / "data"),
            holdout_percentage=0.2,
            fine_tune_batch_size=10,
        )
        store = TrainingDataStore()
        with patch("os.makedirs", side_effect=PermissionError("Permission denied")):
            with pytest.raises(Exception) as exc_info:
                store.initialize(config)
            assert exc_info.value is not None


# ===========================================================================
# 2. ADD EXAMPLE TESTS (test_training_data_store_add)
# ===========================================================================
class TestAddExample:
    """Tests for the add_example() function."""

    def test_add_single_example(self, initialized_store, sample_example):
        """add_example() appends one example and returns correct result with count=1."""
        example = sample_example()
        result = initialized_store.add_example(example)

        assert result.task_type == "summarization", (
            f"Expected task_type='summarization', got '{result.task_type}'"
        )
        assert result.example_count == 1, (
            f"Expected example_count=1, got {result.example_count}"
        )
        assert result.training_count + result.validation_count == result.example_count, (
            "training_count + validation_count must equal example_count"
        )

    def test_add_multiple_examples_count_tracking(self, initialized_store, sample_example):
        """add_example() correctly increments counts across multiple additions."""
        for i in range(5):
            result = initialized_store.add_example(
                sample_example(input_text=f"Input number {i}")
            )
            assert result.example_count == i + 1, (
                f"After {i+1} adds, example_count should be {i+1}, got {result.example_count}"
            )
            assert result.training_count + result.validation_count == result.example_count

    def test_add_example_creates_jsonl_file(self, initialized_store, sample_example, store_config):
        """add_example() creates the JSONL file for the task type if it doesn't exist."""
        jsonl_path = os.path.join(store_config.base_dir, "summarization.jsonl")
        assert not os.path.exists(jsonl_path), "JSONL file should not exist before first add"

        initialized_store.add_example(sample_example())

        assert os.path.isfile(jsonl_path), "JSONL file should exist after add_example()"

    def test_add_example_appends_without_rewrite(self, initialized_store, sample_example, store_config):
        """add_example() appends new line without modifying existing JSONL lines."""
        # Add first example
        initialized_store.add_example(sample_example(input_text="First input"))

        jsonl_path = os.path.join(store_config.base_dir, "summarization.jsonl")
        with open(jsonl_path, "r") as f:
            first_line = f.readline()

        # Add second example
        initialized_store.add_example(sample_example(input_text="Second input"))

        with open(jsonl_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
        assert lines[0] == first_line, "First line should not have been modified"

    def test_add_example_validation_error_empty_input(self, initialized_store, sample_example):
        """add_example() raises validation_error when example.input is empty."""
        with pytest.raises(Exception):
            initialized_store.add_example(sample_example(input_text=""))

    def test_add_example_validation_error_empty_output(self, initialized_store, sample_example):
        """add_example() raises validation_error when example.output is empty."""
        with pytest.raises(Exception):
            initialized_store.add_example(sample_example(output_text=""))

    @pytest.mark.parametrize("bad_task_type", [
        "",
        "has space",
        "has.dot",
        "has/slash",
        "has@symbol",
        "a" * 129,
        "task type!",
    ])
    def test_add_example_validation_error_invalid_task_type(self, initialized_store, sample_example, bad_task_type):
        """add_example() raises validation_error when task_type has invalid characters."""
        with pytest.raises(Exception):
            initialized_store.add_example(sample_example(task_type=bad_task_type))

    def test_add_example_validation_error_invalid_timestamp(self, initialized_store, sample_example):
        """add_example() raises validation_error when timestamp is not valid ISO 8601."""
        with pytest.raises(Exception):
            initialized_store.add_example(sample_example(timestamp="not-a-timestamp"))

    def test_add_example_serialization_error(self, initialized_store, sample_example):
        """add_example() raises serialization_error when metadata has non-JSON-serializable values."""
        with pytest.raises(Exception):
            initialized_store.add_example(
                sample_example(metadata={"bad_value": object()})
            )

    def test_add_example_io_error_write(self, initialized_store, sample_example, store_config):
        """add_example() raises io_error_write when JSONL file write fails."""
        example = sample_example()
        # Mock the open/write to fail
        original_open = open
        def failing_open(path, *args, **kwargs):
            if str(path).endswith(".jsonl"):
                raise IOError("Disk full")
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=failing_open):
            with pytest.raises(Exception):
                initialized_store.add_example(example)

    def test_add_example_io_error_counts(self, initialized_store, sample_example, store_config):
        """add_example() raises io_error_counts when _counts.json cannot be updated."""
        counts_path = os.path.join(store_config.base_dir, "_counts.json")
        example = sample_example()

        # Make _counts.json unwritable after JSONL write
        original_open = open
        call_count = {"n": 0}

        def selective_fail(path, *args, **kwargs):
            path_str = str(path)
            if "_counts.json" in path_str:
                # Allow reads but fail on writes
                mode = args[0] if args else kwargs.get("mode", "r")
                if "w" in mode or "a" in mode:
                    raise IOError("Cannot write _counts.json")
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=selective_fail):
            with pytest.raises(Exception):
                initialized_store.add_example(example)

    @pytest.mark.parametrize("valid_task_type", [
        "a",
        "my-task",
        "TASK_123",
        "A" * 128,
        "simple",
        "with-dashes",
        "with_underscores",
        "MixedCase123",
    ])
    def test_task_type_valid_patterns(self, initialized_store, sample_example, valid_task_type):
        """Valid task_type patterns are accepted by add_example()."""
        example = sample_example(task_type=valid_task_type)
        result = initialized_store.add_example(example)
        assert result.task_type == valid_task_type
        assert result.example_count == 1


# ===========================================================================
# 3. QUERY / ITERATOR TESTS (test_training_data_store_query)
# ===========================================================================
class TestIterTrainingExamples:
    """Tests for iter_training_examples()."""

    def test_iter_training_examples_happy(self, initialized_store, sample_example):
        """iter_training_examples() returns only training-split examples."""
        # Add several examples
        for i in range(20):
            initialized_store.add_example(
                sample_example(input_text=f"training test input {i}")
            )

        training_examples = list(initialized_store.iter_training_examples("summarization"))

        for ex in training_examples:
            split = reference_split(ex.input, 0.2)
            assert split == "training", (
                f"Example with input '{ex.input}' should be in training split, "
                f"but reference_split returned '{split}'"
            )

    def test_iter_training_examples_empty_file(self, initialized_store):
        """iter_training_examples() returns empty iterator when no JSONL file exists."""
        result = list(initialized_store.iter_training_examples("nonexistent_task"))
        assert result == [], "Should return empty list for non-existent task type"

    def test_iter_training_examples_skips_corrupt_lines(self, initialized_store, sample_example, store_config):
        """iter_training_examples() skips corrupt JSONL lines and logs warnings."""
        # Add valid example
        initialized_store.add_example(sample_example(input_text="valid input 1"))

        # Manually inject a corrupt line
        jsonl_path = os.path.join(store_config.base_dir, "summarization.jsonl")
        with open(jsonl_path, "a") as f:
            f.write("{this is not valid json\n")

        # Add another valid example
        initialized_store.add_example(sample_example(input_text="valid input 2"))

        # Iterate all: training + validation should only include the 2 valid ones
        training = list(initialized_store.iter_training_examples("summarization"))
        validation = list(initialized_store.iter_validation_examples("summarization"))

        # Total valid examples should be 2 (the corrupt line is skipped)
        total_valid = len(training) + len(validation)
        assert total_valid == 2, (
            f"Expected 2 valid examples (corrupt line skipped), got {total_valid}"
        )

    @pytest.mark.parametrize("bad_task_type", [
        "has space",
        "has.dot",
        "",
        "bad!type",
    ])
    def test_iter_training_examples_invalid_task_type(self, initialized_store, bad_task_type):
        """iter_training_examples() raises error for invalid task_type format."""
        with pytest.raises(Exception):
            list(initialized_store.iter_training_examples(bad_task_type))

    def test_iter_training_examples_io_error_read(self, initialized_store, sample_example, store_config):
        """iter_training_examples() raises io_error_read when file cannot be read."""
        initialized_store.add_example(sample_example())

        jsonl_path = os.path.join(store_config.base_dir, "summarization.jsonl")
        # Make the file unreadable
        os.chmod(jsonl_path, 0o000)
        try:
            with pytest.raises(Exception):
                list(initialized_store.iter_training_examples("summarization"))
        finally:
            # Restore permissions for cleanup
            os.chmod(jsonl_path, 0o644)


class TestIterValidationExamples:
    """Tests for iter_validation_examples()."""

    def test_iter_validation_examples_happy(self, initialized_store, sample_example):
        """iter_validation_examples() returns only validation-split examples."""
        for i in range(20):
            initialized_store.add_example(
                sample_example(input_text=f"validation test input {i}")
            )

        validation_examples = list(initialized_store.iter_validation_examples("summarization"))

        for ex in validation_examples:
            split = reference_split(ex.input, 0.2)
            assert split == "validation", (
                f"Example with input '{ex.input}' should be in validation split, "
                f"but reference_split returned '{split}'"
            )

    def test_iter_validation_examples_empty_file(self, initialized_store):
        """iter_validation_examples() returns empty iterator when no JSONL file exists."""
        result = list(initialized_store.iter_validation_examples("nonexistent_task"))
        assert result == []

    @pytest.mark.parametrize("bad_task_type", [
        "has space",
        "has.dot",
        "",
    ])
    def test_iter_validation_examples_invalid_task_type(self, initialized_store, bad_task_type):
        """iter_validation_examples() raises error for invalid task_type format."""
        with pytest.raises(Exception):
            list(initialized_store.iter_validation_examples(bad_task_type))

    def test_iter_validation_examples_io_error_read(self, initialized_store, sample_example, store_config):
        """iter_validation_examples() raises io_error_read when file cannot be read."""
        initialized_store.add_example(sample_example())

        jsonl_path = os.path.join(store_config.base_dir, "summarization.jsonl")
        os.chmod(jsonl_path, 0o000)
        try:
            with pytest.raises(Exception):
                list(initialized_store.iter_validation_examples("summarization"))
        finally:
            os.chmod(jsonl_path, 0o644)


class TestGetExampleCount:
    """Tests for get_example_count()."""

    def test_get_example_count_happy(self, initialized_store, sample_example):
        """get_example_count() returns correct total, training, and validation counts."""
        for i in range(10):
            initialized_store.add_example(
                sample_example(input_text=f"count test {i}")
            )

        count = initialized_store.get_example_count("summarization")

        assert count.task_type == "summarization"
        assert count.total == 10, f"Expected total=10, got {count.total}"
        assert count.training + count.validation == count.total, (
            f"training ({count.training}) + validation ({count.validation}) "
            f"!= total ({count.total})"
        )
        assert count.training >= 0
        assert count.validation >= 0

    def test_get_example_count_unknown_task_type(self, initialized_store):
        """get_example_count() returns all zeros for unknown task type."""
        count = initialized_store.get_example_count("never_seen_task")

        assert count.task_type == "never_seen_task"
        assert count.total == 0
        assert count.training == 0
        assert count.validation == 0

    @pytest.mark.parametrize("bad_task_type", [
        "has space",
        "bad!type",
        "",
    ])
    def test_get_example_count_invalid_task_type(self, initialized_store, bad_task_type):
        """get_example_count() raises error for invalid task_type format."""
        with pytest.raises(Exception):
            initialized_store.get_example_count(bad_task_type)

    def test_get_example_count_io_error_counts(self, initialized_store, store_config):
        """get_example_count() raises io_error_counts when _counts.json cannot be read."""
        counts_path = os.path.join(store_config.base_dir, "_counts.json")
        os.chmod(counts_path, 0o000)
        try:
            with pytest.raises(Exception):
                initialized_store.get_example_count("summarization")
        finally:
            os.chmod(counts_path, 0o644)


class TestGetTaskTypes:
    """Tests for get_task_types()."""

    def test_get_task_types_happy(self, initialized_store, sample_example):
        """get_task_types() returns sorted list of task types with examples."""
        initialized_store.add_example(sample_example(task_type="zebra"))
        initialized_store.add_example(sample_example(task_type="alpha"))
        initialized_store.add_example(sample_example(task_type="middle"))

        task_types = initialized_store.get_task_types()

        assert task_types == ["alpha", "middle", "zebra"], (
            f"Expected sorted list ['alpha', 'middle', 'zebra'], got {task_types}"
        )

    def test_get_task_types_empty_store(self, initialized_store):
        """get_task_types() returns empty list when no examples have been added."""
        task_types = initialized_store.get_task_types()
        assert task_types == [], f"Expected empty list, got {task_types}"

    def test_get_task_types_io_error_read(self, initialized_store, store_config):
        """get_task_types() raises error when base_dir or _counts.json cannot be read."""
        counts_path = os.path.join(store_config.base_dir, "_counts.json")
        os.chmod(counts_path, 0o000)
        try:
            with pytest.raises(Exception):
                initialized_store.get_task_types()
        finally:
            os.chmod(counts_path, 0o644)

    def test_task_types_sorted_no_duplicates(self, initialized_store, sample_example):
        """get_task_types() always returns sorted list with no duplicates."""
        # Add multiple examples to same task types
        for i in range(3):
            initialized_store.add_example(sample_example(task_type="beta", input_text=f"b{i}"))
            initialized_store.add_example(sample_example(task_type="alpha", input_text=f"a{i}"))

        task_types = initialized_store.get_task_types()

        # Check sorted
        assert task_types == sorted(task_types), "Task types should be sorted alphabetically"
        # Check no duplicates
        assert len(task_types) == len(set(task_types)), "Task types should have no duplicates"
        # Check valid patterns
        pattern = re.compile(r"^[a-zA-Z0-9_\-]+$")
        for tt in task_types:
            assert pattern.match(tt), f"Task type '{tt}' does not match required pattern"


# ===========================================================================
# 4. SPLIT LOGIC TESTS (test_training_data_store_split)
# ===========================================================================
class TestSplitLogic:
    """Tests for the deterministic split logic."""

    def test_split_disjoint_union(self, initialized_store, sample_example):
        """Training and validation splits are disjoint and their union equals all examples."""
        for i in range(50):
            initialized_store.add_example(
                sample_example(input_text=f"split disjoint test {i}")
            )

        training = list(initialized_store.iter_training_examples("summarization"))
        validation = list(initialized_store.iter_validation_examples("summarization"))

        training_inputs = {ex.input for ex in training}
        validation_inputs = {ex.input for ex in validation}

        # Disjoint
        overlap = training_inputs & validation_inputs
        assert len(overlap) == 0, f"Splits are not disjoint, overlap: {overlap}"

        # Union equals total
        total = len(training_inputs) + len(validation_inputs)
        assert total == 50, (
            f"Training ({len(training_inputs)}) + Validation ({len(validation_inputs)}) "
            f"should equal 50, got {total}"
        )

    def test_split_deterministic(self, initialized_store, sample_example):
        """Same input string always maps to the same split."""
        for i in range(20):
            initialized_store.add_example(
                sample_example(input_text=f"deterministic test {i}")
            )

        training1 = {ex.input for ex in initialized_store.iter_training_examples("summarization")}
        training2 = {ex.input for ex in initialized_store.iter_training_examples("summarization")}

        assert training1 == training2, "Training split should be deterministic across calls"

    def test_split_count_invariant(self, initialized_store, sample_example):
        """training_count + validation_count == example_count for every AddExampleResult."""
        for i in range(30):
            result = initialized_store.add_example(
                sample_example(input_text=f"count invariant {i}")
            )
            assert result.training_count + result.validation_count == result.example_count, (
                f"At step {i}: training({result.training_count}) + "
                f"validation({result.validation_count}) != "
                f"total({result.example_count})"
            )

    def test_split_holdout_zero(self, tmp_path, sample_example):
        """With holdout_percentage=0.0, all examples are in training split."""
        config = TrainingDataStoreConfig(
            base_dir=str(tmp_path / "holdout_zero"),
            holdout_percentage=0.0,
            fine_tune_batch_size=10,
        )
        store = TrainingDataStore()
        store.initialize(config)

        for i in range(20):
            store.add_example(sample_example(input_text=f"holdout zero {i}"))

        validation = list(store.iter_validation_examples("summarization"))
        training = list(store.iter_training_examples("summarization"))

        assert len(validation) == 0, (
            f"With holdout_percentage=0.0, validation should be empty, got {len(validation)}"
        )
        assert len(training) == 20, (
            f"With holdout_percentage=0.0, all 20 should be training, got {len(training)}"
        )

    def test_split_holdout_one(self, tmp_path, sample_example):
        """With holdout_percentage=1.0, all examples are in validation split."""
        config = TrainingDataStoreConfig(
            base_dir=str(tmp_path / "holdout_one"),
            holdout_percentage=1.0,
            fine_tune_batch_size=10,
        )
        store = TrainingDataStore()
        store.initialize(config)

        for i in range(20):
            store.add_example(sample_example(input_text=f"holdout one {i}"))

        training = list(store.iter_training_examples("summarization"))
        validation = list(store.iter_validation_examples("summarization"))

        assert len(training) == 0, (
            f"With holdout_percentage=1.0, training should be empty, got {len(training)}"
        )
        assert len(validation) == 20, (
            f"With holdout_percentage=1.0, all 20 should be validation, got {len(validation)}"
        )

    def test_split_approximate_ratio(self, tmp_path, sample_example):
        """With many examples, the split ratio approximately matches holdout_percentage."""
        holdout = 0.2
        config = TrainingDataStoreConfig(
            base_dir=str(tmp_path / "approx_ratio"),
            holdout_percentage=holdout,
            fine_tune_batch_size=1000,
        )
        store = TrainingDataStore()
        store.initialize(config)

        n = 500
        for i in range(n):
            store.add_example(sample_example(input_text=f"ratio test {i}"))

        training = list(store.iter_training_examples("summarization"))
        validation = list(store.iter_validation_examples("summarization"))

        actual_holdout_ratio = len(validation) / n
        # With 500 samples and 20% holdout, expect roughly 100 validation
        # Allow generous tolerance for hash distribution
        assert 0.05 <= actual_holdout_ratio <= 0.40, (
            f"Holdout ratio {actual_holdout_ratio:.3f} is too far from {holdout} "
            f"(training={len(training)}, validation={len(validation)})"
        )


# ===========================================================================
# 5. BATCH READINESS TESTS (test_training_data_store_batch)
# ===========================================================================
class TestBatchReadiness:
    """Tests for batch_ready flag in AddExampleResult."""

    def test_batch_ready_transitions_at_threshold(self, initialized_store, sample_example):
        """batch_ready becomes True exactly when training_count >= fine_tune_batch_size (10)."""
        batch_size = 10  # from store_config fixture
        results = []

        # Add enough examples to guarantee we cross the threshold
        for i in range(50):
            result = initialized_store.add_example(
                sample_example(input_text=f"batch test {i}")
            )
            results.append(result)

        # Verify batch_ready is correct for each step
        for r in results:
            expected_batch_ready = r.training_count >= batch_size
            assert r.batch_ready == expected_batch_ready, (
                f"batch_ready should be {expected_batch_ready} when "
                f"training_count={r.training_count}, batch_size={batch_size}, "
                f"but got {r.batch_ready}"
            )

    def test_batch_ready_independent_per_task_type(self, initialized_store, sample_example):
        """batch_ready is computed independently for each task_type."""
        # Fill up one task type past batch threshold
        for i in range(15):
            initialized_store.add_example(
                sample_example(task_type="task_a", input_text=f"task_a input {i}")
            )

        # Add one example to a different task type
        result = initialized_store.add_example(
            sample_example(task_type="task_b", input_text="task_b input 0")
        )

        # task_b should not be batch_ready (only 1 example, or very few training)
        assert result.training_count <= 1, (
            f"task_b should have at most 1 training example, got {result.training_count}"
        )
        # batch_ready should reflect task_b's training_count, not task_a's
        expected = result.training_count >= 10
        assert result.batch_ready == expected

    def test_batch_ready_batch_size_one(self, tmp_path, sample_example):
        """With batch_size=1, batch_ready is True on the very first training-split example."""
        config = TrainingDataStoreConfig(
            base_dir=str(tmp_path / "batch_one"),
            holdout_percentage=0.0,  # all training
            fine_tune_batch_size=1,
        )
        store = TrainingDataStore()
        store.initialize(config)

        result = store.add_example(sample_example(input_text="first example"))

        assert result.training_count >= 1, "With holdout=0.0, first example should be training"
        assert result.batch_ready is True, (
            "With batch_size=1 and holdout=0.0, batch_ready should be True on first add"
        )


# ===========================================================================
# 6. PERSISTENCE TESTS (test_training_data_store_persistence)
# ===========================================================================
class TestPersistence:
    """Tests for durability and persistence across reload cycles."""

    def test_persistence_write_reload_cycle(self, tmp_path, sample_example):
        """Data survives destroy-and-reload cycle with re-initialization."""
        base_dir = str(tmp_path / "persist")
        config = TrainingDataStoreConfig(
            base_dir=base_dir,
            holdout_percentage=0.2,
            fine_tune_batch_size=10,
        )

        # First session: add examples
        store1 = TrainingDataStore()
        store1.initialize(config)
        for i in range(5):
            store1.add_example(sample_example(input_text=f"persist test {i}"))

        # Destroy store object
        del store1

        # Second session: reload
        store2 = TrainingDataStore()
        store2.initialize(config)

        # Data should still be there
        training = list(store2.iter_training_examples("summarization"))
        validation = list(store2.iter_validation_examples("summarization"))
        total = len(training) + len(validation)

        assert total == 5, (
            f"After reload, expected 5 total examples, got {total}"
        )

    def test_persistence_counts_consistent_after_reload(self, tmp_path, sample_example):
        """Counts are consistent after re-initialization from persisted JSONL files."""
        base_dir = str(tmp_path / "counts_persist")
        config = TrainingDataStoreConfig(
            base_dir=base_dir,
            holdout_percentage=0.2,
            fine_tune_batch_size=10,
        )

        # First session
        store1 = TrainingDataStore()
        store1.initialize(config)
        for i in range(7):
            store1.add_example(sample_example(input_text=f"counts persist {i}"))
        count_before = store1.get_example_count("summarization")
        del store1

        # Second session
        store2 = TrainingDataStore()
        store2.initialize(config)
        count_after = store2.get_example_count("summarization")

        assert count_after.total == count_before.total, (
            f"Total count mismatch after reload: {count_after.total} != {count_before.total}"
        )
        assert count_after.training + count_after.validation == count_after.total, (
            "Count invariant violated after reload"
        )

    def test_persistence_task_types_after_reload(self, tmp_path, sample_example):
        """Task types list is consistent after reload."""
        base_dir = str(tmp_path / "task_persist")
        config = TrainingDataStoreConfig(
            base_dir=base_dir,
            holdout_percentage=0.2,
            fine_tune_batch_size=10,
        )

        store1 = TrainingDataStore()
        store1.initialize(config)
        store1.add_example(sample_example(task_type="alpha"))
        store1.add_example(sample_example(task_type="beta"))
        types_before = store1.get_task_types()
        del store1

        store2 = TrainingDataStore()
        store2.initialize(config)
        types_after = store2.get_task_types()

        assert types_after == types_before, (
            f"Task types mismatch after reload: {types_after} != {types_before}"
        )


# ===========================================================================
# 7. PROPERTY-BASED / INVARIANT TESTS (test_training_data_store_properties)
# ===========================================================================
class TestInvariants:
    """Invariant tests and property-based tests."""

    def test_jsonl_append_only(self, initialized_store, sample_example, store_config):
        """JSONL files are append-only: existing lines remain unchanged."""
        lines_snapshots = []
        for i in range(10):
            initialized_store.add_example(
                sample_example(input_text=f"append only test {i}")
            )
            jsonl_path = os.path.join(store_config.base_dir, "summarization.jsonl")
            with open(jsonl_path, "r") as f:
                lines_snapshots.append(f.readlines())

        # Each snapshot should be a prefix of the next
        for i in range(len(lines_snapshots) - 1):
            current = lines_snapshots[i]
            next_snapshot = lines_snapshots[i + 1]
            assert len(next_snapshot) == len(current) + 1, (
                f"After add {i+1}, expected {len(current)+1} lines, got {len(next_snapshot)}"
            )
            for j, line in enumerate(current):
                assert line == next_snapshot[j], (
                    f"Line {j} changed after add {i+1}: "
                    f"'{line.strip()}' != '{next_snapshot[j].strip()}'"
                )

    def test_property_training_validation_total(self, initialized_store, sample_example):
        """For any set of examples, training + validation == total."""
        for i in range(40):
            result = initialized_store.add_example(
                sample_example(input_text=f"property total {i}")
            )
            assert result.training_count + result.validation_count == result.example_count, (
                f"Invariant violated at step {i}: "
                f"{result.training_count} + {result.validation_count} != {result.example_count}"
            )

        # Also verify via get_example_count
        count = initialized_store.get_example_count("summarization")
        assert count.training + count.validation == count.total

    def test_property_no_cross_task_leakage(self, initialized_store, sample_example):
        """Examples added to one task_type never appear in another task_type's iterators."""
        # Add examples to two different task types with distinct inputs
        for i in range(10):
            initialized_store.add_example(
                sample_example(task_type="task_alpha", input_text=f"alpha input {i}")
            )
            initialized_store.add_example(
                sample_example(task_type="task_beta", input_text=f"beta input {i}")
            )

        alpha_training = list(initialized_store.iter_training_examples("task_alpha"))
        alpha_validation = list(initialized_store.iter_validation_examples("task_alpha"))
        beta_training = list(initialized_store.iter_training_examples("task_beta"))
        beta_validation = list(initialized_store.iter_validation_examples("task_beta"))

        alpha_inputs = {ex.input for ex in alpha_training + alpha_validation}
        beta_inputs = {ex.input for ex in beta_training + beta_validation}

        # No overlap
        assert alpha_inputs & beta_inputs == set(), (
            f"Cross-task leakage detected: {alpha_inputs & beta_inputs}"
        )

        # All alpha inputs start with "alpha"
        for inp in alpha_inputs:
            assert inp.startswith("alpha"), f"Unexpected input in alpha: {inp}"

        # All beta inputs start with "beta"
        for inp in beta_inputs:
            assert inp.startswith("beta"), f"Unexpected input in beta: {inp}"

    @pytest.mark.parametrize("holdout", [0.0, 0.1, 0.2, 0.5, 0.8, 1.0])
    def test_split_count_invariant_various_holdout(self, tmp_path, sample_example, holdout):
        """training + validation == total holds for various holdout percentages."""
        config = TrainingDataStoreConfig(
            base_dir=str(tmp_path / f"holdout_{holdout}"),
            holdout_percentage=holdout,
            fine_tune_batch_size=10,
        )
        store = TrainingDataStore()
        store.initialize(config)

        for i in range(25):
            result = store.add_example(
                sample_example(input_text=f"holdout {holdout} input {i}")
            )
            assert result.training_count + result.validation_count == result.example_count, (
                f"Invariant violated with holdout={holdout} at step {i}"
            )

    def test_split_reference_implementation_agreement(self, initialized_store, sample_example):
        """Split assignments agree with the reference SHA-256 computation."""
        inputs_added = []
        for i in range(30):
            inp = f"reference check {i}"
            initialized_store.add_example(sample_example(input_text=inp))
            inputs_added.append(inp)

        training_inputs = {
            ex.input
            for ex in initialized_store.iter_training_examples("summarization")
        }
        validation_inputs = {
            ex.input
            for ex in initialized_store.iter_validation_examples("summarization")
        }

        for inp in inputs_added:
            expected_split = reference_split(inp, 0.2)
            if expected_split == "training":
                assert inp in training_inputs, (
                    f"Input '{inp}' should be training per reference, but not found"
                )
                assert inp not in validation_inputs
            else:
                assert inp in validation_inputs, (
                    f"Input '{inp}' should be validation per reference, but not found"
                )
                assert inp not in training_inputs

    def test_batch_ready_invariant(self, initialized_store, sample_example):
        """batch_ready is True iff training_count >= fine_tune_batch_size for all adds."""
        batch_size = 10
        for i in range(30):
            result = initialized_store.add_example(
                sample_example(input_text=f"batch invariant {i}")
            )
            expected = result.training_count >= batch_size
            assert result.batch_ready == expected, (
                f"batch_ready invariant violated: training_count={result.training_count}, "
                f"batch_size={batch_size}, batch_ready={result.batch_ready}, expected={expected}"
            )


# ===========================================================================
# 8. HYPOTHESIS PROPERTY-BASED TESTS (conditional on hypothesis availability)
# ===========================================================================
@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="Hypothesis not installed")
class TestHypothesisProperties:
    """Property-based tests using Hypothesis."""

    @given(
        inputs=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
                min_size=1,
                max_size=50,
            ),
            min_size=1,
            max_size=30,
        )
    )
    @settings(max_examples=20, deadline=10000)
    def test_hypothesis_training_validation_total(self, inputs, tmp_path_factory):
        """Property: training + validation == total for any set of inputs."""
        tmp_path = tmp_path_factory.mktemp("hyp_total")
        config = TrainingDataStoreConfig(
            base_dir=str(tmp_path),
            holdout_percentage=0.2,
            fine_tune_batch_size=100,
        )
        store = TrainingDataStore()
        store.initialize(config)

        for inp in inputs:
            try:
                example = TrainingExample(
                    input=inp,
                    output="test output",
                    task_type="hyp_task",
                    timestamp="2024-01-15T10:30:00Z",
                    metadata={},
                )
                result = store.add_example(example)
                assert result.training_count + result.validation_count == result.example_count
            except Exception:
                # Skip invalid inputs (Hypothesis might generate empty strings etc.)
                pass

    @given(
        holdout=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=10, deadline=10000)
    def test_hypothesis_split_deterministic(self, holdout, tmp_path_factory):
        """Property: split is deterministic for any holdout percentage."""
        tmp_path = tmp_path_factory.mktemp("hyp_det")
        config = TrainingDataStoreConfig(
            base_dir=str(tmp_path),
            holdout_percentage=holdout,
            fine_tune_batch_size=100,
        )
        store = TrainingDataStore()
        store.initialize(config)

        for i in range(10):
            example = TrainingExample(
                input=f"deterministic hyp {i}",
                output="output",
                task_type="det_task",
                timestamp="2024-01-15T10:30:00Z",
                metadata={},
            )
            store.add_example(example)

        training1 = {ex.input for ex in store.iter_training_examples("det_task")}
        training2 = {ex.input for ex in store.iter_training_examples("det_task")}
        assert training1 == training2

    @given(
        task_types=st.lists(
            st.from_regex(r"[a-zA-Z0-9_\-]{1,20}", fullmatch=True),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    @settings(max_examples=10, deadline=10000)
    def test_hypothesis_no_cross_task_leakage(self, task_types, tmp_path_factory):
        """Property: no cross-task leakage for any set of task types."""
        tmp_path = tmp_path_factory.mktemp("hyp_leak")
        config = TrainingDataStoreConfig(
            base_dir=str(tmp_path),
            holdout_percentage=0.2,
            fine_tune_batch_size=100,
        )
        store = TrainingDataStore()
        store.initialize(config)

        # Add unique examples per task type
        for tt in task_types:
            for i in range(3):
                example = TrainingExample(
                    input=f"{tt}_input_{i}",
                    output="output",
                    task_type=tt,
                    timestamp="2024-01-15T10:30:00Z",
                    metadata={},
                )
                store.add_example(example)

        # Check no leakage
        for tt in task_types:
            training = list(store.iter_training_examples(tt))
            validation = list(store.iter_validation_examples(tt))
            all_inputs = {ex.input for ex in training + validation}

            for inp in all_inputs:
                assert inp.startswith(f"{tt}_input_"), (
                    f"Task '{tt}' contains foreign input: {inp}"
                )
