"""Task Registry module.

Holds the set of configured task types loaded from config. Each task entry includes:
name, description, input/output schema, system_prompt and prompt_template, evaluator
type, match_fields, and confidence thresholds. Provides lookup by task name. Built as
a frozen Pydantic v2 model containing a validated, immutable mapping of task names to
TaskConfig objects.
"""

import logging
import string
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

_PACT_KEY = "PACT:e7215b:task_registry"
logger = logging.getLogger(__name__)


class PactFormatter(logging.Formatter):
    """Formatter that injects the PACT log key into every record."""

    def format(self, record):
        record.pact_key = _PACT_KEY
        return super().format(record)


def _log(level: str, msg: str, **kwargs) -> None:
    """Log with PACT key embedded for production traceability."""
    getattr(logger, level)(f"[{_PACT_KEY}] {msg}", **kwargs)


# ── Type Definitions ──────────────────────────────────────────────────────


class EvaluatorType(str, Enum):
    """Enum of supported evaluator types for task output evaluation.
    
    Implemented as Python StrEnum (via str, Enum).
    """
    structured_match = "structured_match"
    semantic_similarity = "semantic_similarity"
    exact_match = "exact_match"
    custom = "custom"


JsonSchemaDict = dict[str, Any]
"""A dict representing a JSON Schema object.

Must be a valid JSON Schema with at minimum a 'type' key.
Used for input_schema and output_schema on TaskConfig.
"""


class ConfidenceThresholds(BaseModel):
    """Confidence thresholds governing phase transitions and alerting for a task.
    
    Frozen Pydantic v2 BaseModel. Cross-field invariant:
    emergency_threshold < coaching_trigger < phase2_to_phase3_correlation <= 1.0,
    and phase1_to_phase2_example_count >= 1.
    All float thresholds are in [0.0, 1.0].
    """
    model_config = ConfigDict(frozen=True, extra='forbid')

    phase1_to_phase2_example_count: int = Field(
        ...,
        ge=1,
        description="Minimum number of validated training examples required before "
                    "transitioning from Phase 1 (remote-only) to Phase 2 (coaching/shadow). "
                    "Must be >= 1."
    )
    phase2_to_phase3_correlation: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="Minimum correlation score between local and remote outputs required "
                    "to transition from Phase 2 to Phase 3 (local-primary). "
                    "Must be in (0.0, 1.0]."
    )
    coaching_trigger: float = Field(
        ...,
        gt=0.0,
        lt=1.0,
        description="Confidence score below which the system triggers a coaching/retraining "
                    "cycle. Must be in (0.0, 1.0) and strictly less than "
                    "phase2_to_phase3_correlation."
    )
    emergency_threshold: float = Field(
        ...,
        ge=0.0,
        lt=1.0,
        description="Confidence score below which the system raises an emergency alert "
                    "and falls back to remote. Must be in [0.0, 1.0) and strictly less than "
                    "coaching_trigger."
    )

    @model_validator(mode='after')
    def validate_threshold_ordering(self) -> 'ConfidenceThresholds':
        """Verify the logical ordering: emergency < coaching < phase2_to_phase3.
        
        Ensures the confidence score boundaries make semantic sense for phase transitions.
        """
        e = self.emergency_threshold
        c = self.coaching_trigger
        p = self.phase2_to_phase3_correlation
        
        if not (e < c < p):
            raise ValueError(
                f"Confidence thresholds must satisfy: "
                f"emergency_threshold ({e}) < coaching_trigger ({c}) < "
                f"phase2_to_phase3_correlation ({p}). Got: {e}, {c}, {p}"
            )
        
        return self


class TaskConfig(BaseModel):
    """Full configuration for a single task type.
    
    Frozen Pydantic v2 BaseModel with extra='forbid'.
    Model validators ensure:
    (1) all match_fields reference keys present in output_schema properties,
    (2) all placeholders in prompt_template (Python str.format style, e.g. {field_name})
        reference keys present in input_schema properties,
    (3) system_prompt is non-empty,
    (4) name is a non-empty lowercase_snake identifier.
    """
    model_config = ConfigDict(frozen=True, extra='forbid')

    name: str = Field(
        ...,
        pattern=r'^[a-z][a-z0-9_]*$',
        min_length=1,
        max_length=128,
        description="Unique task identifier. Must be a non-empty snake_case string "
                    "matching ^[a-z][a-z0-9_]*$."
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of what this task does."
    )
    input_schema: JsonSchemaDict = Field(
        ...,
        description="JSON Schema defining the expected input structure for this task. "
                    "Must have 'type': 'object' and a 'properties' key."
    )
    output_schema: JsonSchemaDict = Field(
        ...,
        description="JSON Schema defining the expected output structure for this task. "
                    "Must have 'type': 'object' and a 'properties' key."
    )
    system_prompt: str = Field(
        ...,
        min_length=1,
        description="System prompt sent to the LLM for this task. Must be non-empty."
    )
    prompt_template: str = Field(
        ...,
        min_length=1,
        description="Python str.format-style template for constructing the user prompt. "
                    "All placeholders (e.g. {field_name}) must correspond to keys in "
                    "input_schema.properties. Must be non-empty."
    )
    evaluator_type: EvaluatorType = Field(
        ...,
        description="Which evaluator strategy to use when scoring local model outputs "
                    "against remote reference outputs for this task."
    )
    match_fields: list[str] = Field(
        ...,
        min_length=1,
        description="List of output field names used by the evaluator to compare outputs. "
                    "Each field must be a key in output_schema.properties. Non-empty list."
    )
    confidence_thresholds: ConfidenceThresholds = Field(
        ...,
        description="Confidence thresholds controlling phase transitions and alerting "
                    "for this task."
    )

    @field_validator('input_schema', 'output_schema')
    @classmethod
    def validate_schema_structure(cls, v: JsonSchemaDict) -> JsonSchemaDict:
        """Validate that schema is a valid JSON Schema object with required fields."""
        if not isinstance(v, dict):
            raise ValueError("Schema must be a dict")
        if v.get('type') != 'object':
            raise ValueError("Schema must have 'type': 'object'")
        if 'properties' not in v:
            raise ValueError("Schema must have a 'properties' key")
        return v

    @model_validator(mode='after')
    def validate_match_fields_against_output_schema(self) -> 'TaskConfig':
        """Check every entry in match_fields is a key in output_schema['properties'].
        
        Ensures evaluator can actually find the fields it needs in the task output.
        """
        output_props = self.output_schema.get('properties', {})
        missing = [f for f in self.match_fields if f not in output_props]
        
        if missing:
            available = sorted(output_props.keys())
            raise ValueError(
                f"match_fields references unknown output_schema properties: {missing}. "
                f"Available properties: {available}"
            )
        
        return self

    @model_validator(mode='after')
    def validate_prompt_template_against_input_schema(self) -> 'TaskConfig':
        """Extract all {placeholder} names from prompt_template and verify each one
        is a key in input_schema['properties'].
        
        Uses string.Formatter to parse the template.
        """
        input_props = self.input_schema.get('properties', {})
        formatter = string.Formatter()
        
        try:
            parsed = formatter.parse(self.prompt_template)
            placeholders = [
                field_name
                for _, field_name, _, _ in parsed
                if field_name is not None
            ]
        except (ValueError, KeyError) as e:
            raise ValueError(
                f"prompt_template has invalid format string syntax: {e}"
            ) from e
        
        missing = [p for p in placeholders if p not in input_props]
        
        if missing:
            available = sorted(input_props.keys())
            raise ValueError(
                f"prompt_template references unknown input_schema properties: {missing}. "
                f"Available properties: {available}"
            )
        
        return self


TaskConfigList = list[TaskConfig]
"""A list of TaskConfig objects, used as input to TaskRegistry construction from parsed YAML."""


class TaskNotFoundError(Exception):
    """Domain exception raised when a task name lookup fails in the registry.
    
    Contains the requested task name and the set of available task names for
    diagnostic purposes. Inherits from Exception (not KeyError) to distinguish
    domain errors from programming errors.
    """

    def __init__(self, task_name: str, available_tasks: list[str]):
        self.task_name = task_name
        self.available_tasks = available_tasks
        super().__init__(
            f"Task '{task_name}' not found. Available tasks: {sorted(available_tasks)}"
        )


@runtime_checkable
class TaskRegistryProtocol(Protocol):
    """typing.Protocol defining the read-only interface for the Task Registry.
    
    Downstream components type-annotate their dependency as TaskRegistryProtocol,
    enabling test doubles (e.g. simple dict-backed stubs) without inheritance from
    the concrete TaskRegistry class.
    
    Methods:
      - get_task(name: str) -> TaskConfig
      - __contains__(name: str) -> bool
      - task_names -> frozenset[str]
      - __len__() -> int
    """

    def get_task(self, name: str) -> TaskConfig:
        ...

    def __contains__(self, name: str) -> bool:
        ...

    @property
    def task_names(self) -> frozenset[str]:
        ...

    def __len__(self) -> int:
        ...


class TaskRegistry(BaseModel):
    """Frozen Pydantic v2 BaseModel containing a validated, immutable mapping of
    task names to TaskConfig objects.
    
    Constructed from a list of TaskConfig objects (typically parsed from the 'tasks'
    key in apprentice.yaml by config_loader). A model_validator detects duplicate
    task names at construction time. After construction, the registry is fully
    immutable — no tasks can be added, removed, or modified. Exposes read-only
    lookup methods.
    """
    model_config = ConfigDict(frozen=True, extra='forbid')

    tasks: dict[str, TaskConfig] = Field(
        default_factory=dict,
        description="Internal mapping from task name (str) to TaskConfig. Built from "
                    "the validated task list at construction. Keys are task names, "
                    "values are TaskConfig instances. This field is populated by the "
                    "model_validator from the input task list and should not be set "
                    "directly by callers."
    )

    def __init__(self, task_list: TaskConfigList, **data):
        """Construct a TaskRegistry from a list of TaskConfig objects.
        
        Validates all tasks, detects duplicate names, builds the internal immutable
        mapping. This is the only way to create a TaskRegistry — it is frozen after
        construction. Typically called by config_loader after parsing apprentice.yaml.
        
        Args:
            task_list: Non-empty list of TaskConfig objects
        
        Raises:
            ValidationError: If task_list is empty, contains duplicates, or any
                           TaskConfig fails validation
        """
        # Validate non-empty
        if not task_list:
            raise ValueError("At least one task must be configured.")
        
        # Check for duplicates
        names = [tc.name for tc in task_list]
        duplicates = [name for name in set(names) if names.count(name) > 1]
        if duplicates:
            raise ValueError(f"Duplicate task name(s) found: {sorted(set(duplicates))}")
        
        # Build tasks mapping
        tasks_dict = {tc.name: tc for tc in task_list}
        
        # Initialize via parent with computed tasks dict
        super().__init__(tasks=tasks_dict, **data)
        
        _log('info', f"TaskRegistry initialized with {len(self.tasks)} task(s): "
                     f"{sorted(self.tasks.keys())}")

    def get_task(self, name: str) -> TaskConfig:
        """Look up a task configuration by its unique name.
        
        Returns the frozen TaskConfig for the given task name.
        
        Args:
            name: Non-empty task name string
        
        Returns:
            TaskConfig: The task configuration for the given name
        
        Raises:
            TaskNotFoundError: If name is not in the registry
        """
        if name not in self.tasks:
            raise TaskNotFoundError(
                task_name=name,
                available_tasks=sorted(self.tasks.keys())
            )
        return self.tasks[name]

    def __contains__(self, name: str) -> bool:
        """Check whether a task name exists in the registry.
        
        Supports the 'in' operator: `'my_task' in registry`.
        
        Args:
            name: Task name string
        
        Returns:
            bool: True if the task name is registered, False otherwise
        """
        return name in self.tasks

    @property
    def task_names(self) -> frozenset[str]:
        """Property returning the frozenset of all registered task names.
        
        Useful for diagnostics, logging, and iteration. Returns an immutable set —
        callers cannot modify the registry through this property.
        
        Returns:
            frozenset[str]: Immutable set of task names
        """
        return frozenset(self.tasks.keys())

    def __len__(self) -> int:
        """Return the number of tasks in the registry.
        
        Supports the len() builtin: `len(registry)`.
        
        Returns:
            int: Number of registered tasks (always >= 1)
        """
        return len(self.tasks)


# ── Exports ───────────────────────────────────────────────────────────────

# Re-export ValidationError from pydantic for convenience
from pydantic import ValidationError

# Module-level convenience functions that delegate to registry instance methods
# These are exported to satisfy the contract's REQUIRED EXPORTS list
def get_task(registry: TaskRegistry, name: str) -> TaskConfig:
    """Module-level convenience function for registry.get_task().
    
    NOTE: The contract shows get_task as a standalone function, but the natural
    implementation is as a method on TaskRegistry. This wrapper satisfies the
    contract while keeping the implementation clean.
    
    Args:
        registry: TaskRegistry instance
        name: Task name to look up
    
    Returns:
        TaskConfig: The task configuration
    
    Raises:
        TaskNotFoundError: If the task name is not found
    """
    return registry.get_task(name)


def task_names(registry: TaskRegistry) -> frozenset[str]:
    """Module-level convenience function for registry.task_names property.
    
    Args:
        registry: TaskRegistry instance
    
    Returns:
        frozenset[str]: Set of all registered task names
    """
    return registry.task_names


__all__ = [
    'EvaluatorType',
    'ConfidenceThresholds',
    'TaskConfig',
    'TaskRegistry',
    'TaskRegistryProtocol',
    'TaskNotFoundError',
    'TaskConfigList',
    'ValidationError',
    'get_task',
    'task_names',
]
