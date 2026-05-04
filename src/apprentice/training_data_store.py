"""Training Data Store implementation.

Collects and persists training examples (input + remote response pairs) per task type.
Stores as JSON-lines files in a configurable directory.
"""

import hashlib
import json
import logging
import os
import re
import struct
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, List

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

_PACT_KEY = "PACT:339426:training_data_store"
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
# DATA MODELS
# ============================================================================


class TrainingExample(BaseModel):
    """A single training example record."""

    model_config = ConfigDict(extra="forbid")

    input: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)
    task_type: str = Field(..., min_length=1, max_length=128)
    timestamp: str
    metadata: dict = Field(default_factory=dict)

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        """Validate task_type matches required pattern."""
        if not re.match(r"^[a-zA-Z0-9_:\-]+$", v):
            raise ValueError(
                f"task_type must match ^[a-zA-Z0-9_:\\-]+$, got '{v}'"
            )
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate timestamp is ISO 8601 format."""
        if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", v):
            raise ValueError(
                f"timestamp must be ISO 8601 format (YYYY-MM-DDTHH:MM:SS), got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_metadata_serializable(self):
        """Ensure metadata is JSON-serializable."""
        try:
            json.dumps(self.metadata)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"metadata contains non-JSON-serializable values: {e}"
            )
        return self


class TrainingDataStoreConfig(BaseModel):
    """Configuration for the filesystem-backed TrainingDataStore."""

    model_config = ConfigDict(extra="forbid")

    base_dir: str = Field(..., min_length=1)
    holdout_percentage: float = Field(default=0.10, ge=0.0, le=1.0)
    fine_tune_batch_size: int = Field(default=100, gt=0)


class AddExampleResult(BaseModel):
    """Return value from add_example()."""

    model_config = ConfigDict(extra="forbid")

    task_type: str
    example_count: int = Field(..., ge=1)
    training_count: int = Field(..., ge=0)
    validation_count: int = Field(..., ge=0)
    batch_ready: bool


class ExampleCount(BaseModel):
    """Summary of example counts for a specific task type."""

    model_config = ConfigDict(extra="forbid")

    task_type: str
    total: int = Field(..., ge=0)
    training: int = Field(..., ge=0)
    validation: int = Field(..., ge=0)


class SplitKind(Enum):
    """Enumeration of dataset split types."""

    training = "training"
    validation = "validation"


# Type aliases
TrainingExampleList = List[TrainingExample]
TaskTypeList = List[str]


# ============================================================================
# IMPLEMENTATION
# ============================================================================


class TrainingDataStore:
    """Filesystem-backed implementation of the Training Data Store."""

    def __init__(self):
        """Initialize an empty store. Must call initialize() before use."""
        self._config: TrainingDataStoreConfig | None = None
        self._counts: dict[str, int] = {}  # task_type -> total count

    def initialize(self, config: TrainingDataStoreConfig) -> None:
        """Initialize the store with the given configuration.

        Creates base_dir if it doesn't exist, loads or reconciles _counts.json.
        """
        try:
            # Validate config
            self._config = config

            # Create base_dir
            try:
                os.makedirs(config.base_dir, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise IOError(
                    f"io_error_mkdir: Cannot create base_dir '{config.base_dir}': {e}"
                )

            # Load or create _counts.json
            counts_path = self._get_counts_path()
            self._counts = self._load_or_reconcile_counts(counts_path)

            _log("info", f"Initialized training data store at {config.base_dir}")

        except ValueError as e:
            # Re-raise validation errors
            raise ValueError(f"config_validation_error: {e}")

    def add_example(self, example: TrainingExample) -> AddExampleResult:
        """Append a single training example to the JSONL file for its task type."""
        if self._config is None:
            raise RuntimeError("Store not initialized")

        # Validate the example (Pydantic will raise ValueError on invalid data)
        try:
            # Ensure metadata is JSON-serializable
            json.dumps(example.metadata)
        except (TypeError, ValueError) as e:
            raise ValueError(f"serialization_error: {e}")

        task_type = example.task_type
        jsonl_path = self._get_jsonl_path(task_type)

        # Ensure base_dir exists (in case it was deleted after init)
        try:
            os.makedirs(self._config.base_dir, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise IOError(
                f"io_error_write: Cannot create base_dir '{self._config.base_dir}': {e}"
            )

        # Serialize example
        try:
            line = json.dumps(example.model_dump()) + "\n"
        except (TypeError, ValueError) as e:
            raise ValueError(f"serialization_error: {e}")

        # Append to JSONL file
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(line)
        except (OSError, IOError, PermissionError) as e:
            raise IOError(f"io_error_write: Cannot write to '{jsonl_path}': {e}")

        # Update counts
        try:
            self._counts[task_type] = self._counts.get(task_type, 0) + 1
            self._save_counts()
        except (OSError, IOError, PermissionError) as e:
            counts_path = self._get_counts_path()
            raise IOError(
                f"io_error_counts: Cannot update '{counts_path}': {e}"
            )

        # Compute split counts
        total = self._counts[task_type]
        training_count, validation_count = self._compute_split_counts(task_type)

        # Check batch readiness
        batch_ready = training_count >= self._config.fine_tune_batch_size

        return AddExampleResult(
            task_type=task_type,
            example_count=total,
            training_count=training_count,
            validation_count=validation_count,
            batch_ready=batch_ready,
        )

    def add_examples(self, examples: list[TrainingExample]) -> list[AddExampleResult]:
        """Bulk-add a list of training examples. More efficient than calling add_example() in a loop
        because it groups writes by task_type and flushes once per file."""
        if self._config is None:
            raise RuntimeError("Store not initialized")
        if not examples:
            return []

        os.makedirs(self._config.base_dir, exist_ok=True)

        # Group by task_type for batched file writes
        by_task: dict[str, list[TrainingExample]] = {}
        for ex in examples:
            by_task.setdefault(ex.task_type, []).append(ex)

        results = []
        for task_type, task_examples in by_task.items():
            jsonl_path = self._get_jsonl_path(task_type)
            lines = []
            for ex in task_examples:
                json.dumps(ex.metadata)  # validate serializable
                lines.append(json.dumps(ex.model_dump()) + "\n")

            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.writelines(lines)

            self._counts[task_type] = self._counts.get(task_type, 0) + len(task_examples)
            self._save_counts()

            total = self._counts[task_type]
            training_count, validation_count = self._compute_split_counts(task_type)
            batch_ready = training_count >= self._config.fine_tune_batch_size

            results.append(AddExampleResult(
                task_type=task_type,
                example_count=total,
                training_count=training_count,
                validation_count=validation_count,
                batch_ready=batch_ready,
            ))

        return results

    def iter_training_examples(self, task_type: str) -> TrainingExampleList:
        """Returns an iterator over all training-split examples for the task type."""
        self._validate_task_type_format(task_type)
        return list(self._iter_examples(task_type, SplitKind.training))

    def iter_validation_examples(self, task_type: str) -> TrainingExampleList:
        """Returns an iterator over all validation-split examples for the task type."""
        self._validate_task_type_format(task_type)
        return list(self._iter_examples(task_type, SplitKind.validation))

    def get_example_count(self, task_type: str) -> ExampleCount:
        """Returns the total, training, and validation example counts for a task type."""
        self._validate_task_type_format(task_type)

        if self._config is None:
            raise RuntimeError("Store not initialized")

        # Load counts
        try:
            self._load_counts()
        except (OSError, IOError, PermissionError) as e:
            counts_path = self._get_counts_path()
            raise IOError(
                f"io_error_counts: Cannot read '{counts_path}': {e}"
            )

        total = self._counts.get(task_type, 0)

        if total == 0:
            return ExampleCount(
                task_type=task_type,
                total=0,
                training=0,
                validation=0,
            )

        training_count, validation_count = self._compute_split_counts(task_type)

        return ExampleCount(
            task_type=task_type,
            total=total,
            training=training_count,
            validation=validation_count,
        )

    def get_task_types(self) -> TaskTypeList:
        """Returns a list of all task type identifiers that have at least one example."""
        if self._config is None:
            raise RuntimeError("Store not initialized")

        try:
            self._load_counts()
        except (OSError, IOError, PermissionError) as e:
            counts_path = self._get_counts_path()
            raise IOError(
                f"io_error_read: Cannot read '{counts_path}': {e}"
            )

        # Return sorted list of task types with count > 0
        task_types = sorted([tt for tt, count in self._counts.items() if count > 0])
        return task_types

    # ------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------

    def _get_counts_path(self) -> str:
        """Return path to _counts.json."""
        if self._config is None:
            raise RuntimeError("Store not initialized")
        return os.path.join(self._config.base_dir, "_counts.json")

    def _get_jsonl_path(self, task_type: str) -> str:
        """Return path to JSONL file for the given task_type."""
        if self._config is None:
            raise RuntimeError("Store not initialized")
        return os.path.join(self._config.base_dir, f"{task_type}.jsonl")

    def _load_or_reconcile_counts(self, counts_path: str) -> dict[str, int]:
        """Load _counts.json or reconcile from JSONL files if missing/corrupt."""
        counts = {}

        # Try to load existing counts
        if os.path.exists(counts_path):
            try:
                with open(counts_path, "r", encoding="utf-8") as f:
                    counts = json.load(f)
                    if not isinstance(counts, dict):
                        raise ValueError("Invalid counts format")
            except (json.JSONDecodeError, ValueError) as e:
                _log("warning", f"Corrupt _counts.json, rebuilding: {e}")
                counts = {}

        # Scan for JSONL files and reconcile
        if self._config is None:
            raise RuntimeError("Store not initialized")

        try:
            if os.path.exists(self._config.base_dir):
                for filename in os.listdir(self._config.base_dir):
                    if filename.endswith(".jsonl"):
                        task_type = filename[:-6]  # Remove .jsonl extension
                        jsonl_path = os.path.join(self._config.base_dir, filename)

                        # Count lines in JSONL file
                        try:
                            actual_count = self._count_valid_lines(jsonl_path)
                            if task_type not in counts or counts[task_type] != actual_count:
                                _log(
                                    "info",
                                    f"Reconciling {task_type}: counted {actual_count} examples",
                                )
                                counts[task_type] = actual_count
                        except (OSError, IOError) as e:
                            raise IOError(
                                f"io_error_reconcile: Cannot read '{jsonl_path}': {e}"
                            )
        except (OSError, PermissionError) as e:
            raise IOError(
                f"io_error_read: Cannot list directory '{self._config.base_dir}': {e}"
            )

        # Save reconciled counts
        self._counts = counts
        self._save_counts()

        return counts

    def _count_valid_lines(self, jsonl_path: str) -> int:
        """Count the number of valid JSON lines in a JSONL file."""
        count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError as e:
                    _log(
                        "warning",
                        f"Skipping corrupt line {line_num} in {jsonl_path}: {e}",
                    )
        return count

    def _save_counts(self) -> None:
        """Save counts to _counts.json."""
        counts_path = self._get_counts_path()
        try:
            with open(counts_path, "w", encoding="utf-8") as f:
                json.dump(self._counts, f, indent=2)
        except (OSError, IOError, PermissionError) as e:
            raise IOError(
                f"io_error_counts: Cannot write '{counts_path}': {e}"
            )

    def _load_counts(self) -> None:
        """Reload counts from _counts.json."""
        counts_path = self._get_counts_path()
        if os.path.exists(counts_path):
            try:
                with open(counts_path, "r", encoding="utf-8") as f:
                    self._counts = json.load(f)
            except (json.JSONDecodeError, OSError, IOError, PermissionError) as e:
                raise IOError(
                    f"io_error_counts: Cannot read '{counts_path}': {e}"
                )
        else:
            self._counts = {}

    def _iter_examples(
        self, task_type: str, split: SplitKind
    ) -> Iterator[TrainingExample]:
        """Internal iterator over examples for a given task_type and split."""
        if self._config is None:
            raise RuntimeError("Store not initialized")

        jsonl_path = self._get_jsonl_path(task_type)

        # If file doesn't exist, return empty iterator
        if not os.path.exists(jsonl_path):
            return

        # Read and yield examples
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        example = TrainingExample(**data)

                        # Check split assignment
                        example_split = self._compute_split(example.input)
                        if example_split == split:
                            yield example

                    except (json.JSONDecodeError, ValueError) as e:
                        _log(
                            "warning",
                            f"Skipping corrupt line {line_num} in {jsonl_path}: {e}",
                        )
                        continue
        except (OSError, IOError, PermissionError) as e:
            raise IOError(
                f"io_error_read: Cannot read '{jsonl_path}': {e}"
            )

    def _compute_split(self, input_str: str) -> SplitKind:
        """Compute the split assignment for an input string using SHA-256 hash."""
        if self._config is None:
            raise RuntimeError("Store not initialized")

        digest = hashlib.sha256(input_str.encode("utf-8")).digest()
        value = struct.unpack(">I", digest[-4:])[0] % 10000
        threshold = self._config.holdout_percentage * 10000

        if value < threshold:
            return SplitKind.validation
        else:
            return SplitKind.training

    def _compute_split_counts(self, task_type: str) -> tuple[int, int]:
        """Compute training and validation counts for a task type."""
        training_count = 0
        validation_count = 0

        for example in self._iter_examples(task_type, SplitKind.training):
            training_count += 1

        for example in self._iter_examples(task_type, SplitKind.validation):
            validation_count += 1

        return training_count, validation_count

    def _validate_task_type_format(self, task_type: str) -> None:
        """Validate that task_type matches required format."""
        if not re.match(r"^[a-zA-Z0-9_:\-]+$", task_type):
            raise ValueError(
                f"invalid_task_type: task_type must match ^[a-zA-Z0-9_:\\-]+$, got '{task_type}'"
            )


# ============================================================================
# MODULE-LEVEL FUNCTIONS (per contract requirements)
# ============================================================================

# Global store instance for module-level API
_global_store: TrainingDataStore | None = None


def initialize(config: TrainingDataStoreConfig) -> None:
    """Initialize the Training Data Store with the given configuration."""
    global _global_store
    _global_store = TrainingDataStore()
    _global_store.initialize(config)


def add_example(example: TrainingExample) -> AddExampleResult:
    """Appends a single training example to the JSONL file for its task type."""
    if _global_store is None:
        raise RuntimeError("Store not initialized, call initialize() first")
    return _global_store.add_example(example)


def iter_training_examples(task_type: str) -> TrainingExampleList:
    """Returns an iterator over all training-split examples for the task type."""
    if _global_store is None:
        raise RuntimeError("Store not initialized, call initialize() first")
    return _global_store.iter_training_examples(task_type)


def iter_validation_examples(task_type: str) -> TrainingExampleList:
    """Returns an iterator over all validation-split examples for the task type."""
    if _global_store is None:
        raise RuntimeError("Store not initialized, call initialize() first")
    return _global_store.iter_validation_examples(task_type)


def get_example_count(task_type: str) -> ExampleCount:
    """Returns the total, training, and validation example counts for a task type."""
    if _global_store is None:
        raise RuntimeError("Store not initialized, call initialize() first")
    return _global_store.get_example_count(task_type)


def get_task_types() -> TaskTypeList:
    """Returns a list of all task type identifiers that have at least one example."""
    if _global_store is None:
        raise RuntimeError("Store not initialized, call initialize() first")
    return _global_store.get_task_types()


# ============================================================================
# STORY STORAGE
# ============================================================================


class SplitName(Enum):
    """Named data split for partitioning stories into train/val/test sets.
    Split assignment is deterministic via SHA-256 of story_id."""
    train = "train"
    val = "val"
    test = "test"


class StoryStep:
    """A single step within a multi-step story for collector storage."""

    def __init__(self, step_index: int, input_data: dict, output_data: dict, metadata: dict = None):
        if step_index < 0:
            raise ValueError("step_index must be >= 0")
        self.step_index = step_index
        self.input_data = input_data
        self.output_data = output_data
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'StoryStep':
        return cls(
            step_index=data["step_index"],
            input_data=data["input_data"],
            output_data=data["output_data"],
            metadata=data.get("metadata", {}),
        )

    def __eq__(self, other):
        if not isinstance(other, StoryStep):
            return NotImplemented
        return (self.step_index == other.step_index and
                self.input_data == other.input_data and
                self.output_data == other.output_data and
                self.metadata == other.metadata)


StoryStepList = List[StoryStep]


class CollectorStory:
    """A multi-step interaction story for the collector subsystem.
    Supports to_dict()/from_dict() round-trip serialization."""

    def __init__(self, story_id: str, story_type: str, steps: List[StoryStep], metadata: dict = None):
        if not story_id:
            raise ValueError("story_id must be non-empty")
        if not story_type:
            raise ValueError("story_type must be non-empty")
        self.story_id = story_id
        self.story_type = story_type
        self.steps = steps
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> dict:
        return {
            "story_id": self.story_id,
            "story_type": self.story_type,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CollectorStory':
        steps = [StoryStep.from_dict(s) for s in data["steps"]]
        return cls(
            story_id=data["story_id"],
            story_type=data["story_type"],
            steps=steps,
            metadata=data.get("metadata", {}),
        )

    def __eq__(self, other):
        if not isinstance(other, CollectorStory):
            return NotImplemented
        return (self.story_id == other.story_id and
                self.story_type == other.story_type and
                self.steps == other.steps and
                self.metadata == other.metadata)


StoryList = List[CollectorStory]


class StoryTrainingExample:
    """A training example derived from two consecutive story steps."""

    def __init__(self, story_id: str, step_index: int, context: dict,
                 input_data: dict, expected_output: dict):
        self.story_id = story_id
        self.step_index = step_index
        self.context = context
        self.input_data = input_data
        self.expected_output = expected_output

    def __eq__(self, other):
        if not isinstance(other, StoryTrainingExample):
            return NotImplemented
        return (self.story_id == other.story_id and
                self.step_index == other.step_index and
                self.context == other.context and
                self.input_data == other.input_data and
                self.expected_output == other.expected_output)


TrainingExampleList_Story = List[StoryTrainingExample]


def sanitize_story_type(story_type: str) -> str:
    """Sanitize story_type for filesystem safety by keeping only
    alphanumeric, underscore, and hyphen characters."""
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', story_type)
    if not sanitized:
        raise ValueError("story_type is empty after sanitization")
    return sanitized


def compute_split(story_id: str) -> SplitName:
    """Deterministically assign a story_id to a data split using SHA-256.
    0-79 = train, 80-89 = val, 90-99 = test."""
    digest = hashlib.sha256(story_id.encode('utf-8')).digest()
    bucket = int.from_bytes(digest[:8], 'big') % 100
    if bucket < 80:
        return SplitName.train
    elif bucket < 90:
        return SplitName.val
    else:
        return SplitName.test


class StoryCollector:
    """Manages story storage and retrieval in JSONL format."""

    def __init__(self, base_dir: str):
        self._base_dir = base_dir

    def _stories_dir(self) -> str:
        return os.path.join(self._base_dir, "stories")

    def _jsonl_path(self, story_type: str) -> str:
        sanitized = sanitize_story_type(story_type)
        return os.path.join(self._stories_dir(), f"{sanitized}.jsonl")

    def store_story(self, story: CollectorStory) -> None:
        """Persist a single Story as one JSONL line."""
        sanitized = sanitize_story_type(story.story_type)
        stories_dir = self._stories_dir()
        os.makedirs(stories_dir, exist_ok=True)
        jsonl_path = os.path.join(stories_dir, f"{sanitized}.jsonl")
        try:
            line = json.dumps(story.to_dict(), sort_keys=True) + "\n"
        except (TypeError, ValueError) as e:
            raise TypeError(f"Story contains non-JSON-serializable data: {e}")
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            raise OSError(f"Failed to write story to JSONL file: {e}")

    def get_story_batch(self, story_type: str, split: str) -> StoryList:
        """Read all stories of a given story_type filtered by split."""
        sanitized = sanitize_story_type(story_type)

        # Validate split
        valid_splits = {s.value for s in SplitName}
        if split not in valid_splits:
            raise ValueError(f"split must be one of {sorted(valid_splits)}, got '{split}'")

        split_enum = SplitName(split)
        jsonl_path = os.path.join(self._stories_dir(), f"{sanitized}.jsonl")

        if not os.path.exists(jsonl_path):
            return []

        stories = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        story = CollectorStory.from_dict(data)
                        if compute_split(story.story_id) == split_enum:
                            stories.append(story)
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        _log("warning", f"Corrupt JSONL line {line_num} in {jsonl_path}: {e}")
                        continue
        except OSError as e:
            raise OSError(f"Failed to read story JSONL file: {e}")

        return stories

    def stories_to_training_examples(self, story: CollectorStory) -> TrainingExampleList_Story:
        """Convert an N-step story into N-1 sequential training examples."""
        if len(story.steps) < 2:
            return []

        examples = []
        for i in range(1, len(story.steps)):
            prev_step = story.steps[i - 1]
            curr_step = story.steps[i]
            examples.append(StoryTrainingExample(
                story_id=story.story_id,
                step_index=curr_step.step_index,
                context=prev_step.output_data,
                input_data=curr_step.input_data,
                expected_output=curr_step.output_data,
            ))
        return examples


# Module-level story collector instance
_global_collector: StoryCollector | None = None


def store_story(story: CollectorStory) -> None:
    """Module-level store_story using the global collector."""
    if _global_collector is None:
        raise RuntimeError("Story collector not initialized, call initialize() first")
    _global_collector.store_story(story)


def get_story_batch(story_type: str, split: str) -> StoryList:
    """Module-level get_story_batch using the global collector."""
    if _global_collector is None:
        raise RuntimeError("Story collector not initialized, call initialize() first")
    return _global_collector.get_story_batch(story_type, split)


def stories_to_training_examples(story: CollectorStory) -> TrainingExampleList_Story:
    """Module-level stories_to_training_examples."""
    if _global_collector is None:
        raise RuntimeError("Story collector not initialized, call initialize() first")
    return _global_collector.stories_to_training_examples(story)


# Patch initialize to also set up the story collector
_original_initialize = initialize


def _patched_initialize(config: TrainingDataStoreConfig) -> None:
    """Initialize both the training data store and the story collector."""
    global _global_collector
    _original_initialize(config)
    _global_collector = StoryCollector(config.base_dir)


initialize = _patched_initialize


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "TrainingExample",
    "TrainingDataStoreConfig",
    "AddExampleResult",
    "ExampleCount",
    "TrainingExampleList",
    "TaskTypeList",
    "SplitKind",
    "add_example",
    "iter_training_examples",
    "iter_validation_examples",
    "get_example_count",
    "get_task_types",
    "initialize",
    "TrainingDataStore",
    # Story exports
    "SplitName",
    "StoryStep",
    "StoryStepList",
    "CollectorStory",
    "StoryList",
    "StoryTrainingExample",
    "TrainingExampleList_Story",
    "StoryCollector",
    "store_story",
    "get_story_batch",
    "stories_to_training_examples",
    "sanitize_story_type",
    "compute_split",
]
