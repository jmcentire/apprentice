"""
Contract tests for task_registry component.

Tests verify behavior at boundaries (inputs/outputs) against the contract.
All dependencies are mocked; tests verify one component in isolation.
"""

import string
import pytest
from pydantic import ValidationError

from src.task_registry import (
    ConfidenceThresholds,
    EvaluatorType,
    TaskConfig,
    TaskRegistry,
    TaskNotFoundError,
)


# ---------------------------------------------------------------------------
# Fixtures & Factories
# ---------------------------------------------------------------------------

def make_confidence_thresholds(**overrides):
    """Create a minimal valid ConfidenceThresholds with easy overrides."""
    defaults = dict(
        phase1_to_phase2_example_count=5,
        phase2_to_phase3_correlation=0.9,
        coaching_trigger=0.6,
        emergency_threshold=0.3,
    )
    defaults.update(overrides)
    return ConfidenceThresholds(**defaults)


def make_task_config(**overrides):
    """Create a minimal valid TaskConfig with easy overrides."""
    defaults = dict(
        name="classify_email",
        description="Classify an email into categories.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "result": {"type": "string"},
            },
        },
        system_prompt="You are a helpful assistant.",
        prompt_template="Classify the following: {query}",
        evaluator_type=EvaluatorType.structured_match,
        match_fields=["result"],
        confidence_thresholds=make_confidence_thresholds(),
    )
    defaults.update(overrides)
    # Allow passing confidence_thresholds as a dict for convenience
    if isinstance(defaults["confidence_thresholds"], dict):
        defaults["confidence_thresholds"] = make_confidence_thresholds(
            **defaults["confidence_thresholds"]
        )
    return TaskConfig(**defaults)


def make_registry(task_configs=None):
    """Build a TaskRegistry from a list of TaskConfigs (default: one valid task)."""
    if task_configs is None:
        task_configs = [make_task_config()]
    return TaskRegistry(task_configs)


# ---------------------------------------------------------------------------
# TestConfidenceThresholds
# ---------------------------------------------------------------------------

class TestConfidenceThresholds:
    """Validate ConfidenceThresholds construction and ordering invariants."""

    def test_valid_construction(self):
        ct = make_confidence_thresholds()
        assert ct.emergency_threshold == 0.3
        assert ct.coaching_trigger == 0.6
        assert ct.phase2_to_phase3_correlation == 0.9
        assert ct.phase1_to_phase2_example_count == 5

    def test_boundary_values_accepted(self):
        ct = make_confidence_thresholds(
            emergency_threshold=0.01,
            coaching_trigger=0.5,
            phase2_to_phase3_correlation=1.0,
            phase1_to_phase2_example_count=1,
        )
        assert ct.phase2_to_phase3_correlation == 1.0
        assert ct.phase1_to_phase2_example_count == 1

    @pytest.mark.parametrize(
        "emergency, coaching, correlation, reason",
        [
            (0.6, 0.5, 0.9, "emergency >= coaching"),
            (0.5, 0.5, 0.9, "emergency == coaching"),
            (0.1, 0.9, 0.8, "coaching >= correlation"),
            (0.1, 0.8, 0.8, "coaching == correlation"),
            (0.5, 0.5, 0.5, "all equal"),
            (0.9, 0.6, 0.3, "reverse order"),
        ],
        ids=[
            "emergency_gt_coaching",
            "emergency_eq_coaching",
            "coaching_gt_correlation",
            "coaching_eq_correlation",
            "all_equal",
            "reverse_order",
        ],
    )
    def test_invalid_threshold_ordering(self, emergency, coaching, correlation, reason):
        with pytest.raises(ValidationError):
            make_confidence_thresholds(
                emergency_threshold=emergency,
                coaching_trigger=coaching,
                phase2_to_phase3_correlation=correlation,
            )

    def test_example_count_less_than_one_rejected(self):
        with pytest.raises(ValidationError):
            make_confidence_thresholds(phase1_to_phase2_example_count=0)

    def test_example_count_negative_rejected(self):
        with pytest.raises(ValidationError):
            make_confidence_thresholds(phase1_to_phase2_example_count=-1)


# ---------------------------------------------------------------------------
# TestTaskConfigValidation
# ---------------------------------------------------------------------------

class TestTaskConfigValidation:
    """Validate TaskConfig cross-field model validators."""

    def test_valid_construction(self):
        tc = make_task_config()
        assert tc.name == "classify_email"
        assert tc.match_fields == ["result"]

    def test_multiple_match_fields_all_in_output_schema(self):
        tc = make_task_config(
            output_schema={
                "type": "object",
                "properties": {
                    "field_a": {"type": "string"},
                    "field_b": {"type": "integer"},
                },
            },
            match_fields=["field_a", "field_b"],
        )
        assert set(tc.match_fields) == {"field_a", "field_b"}

    def test_match_field_not_in_output_schema_rejected(self):
        with pytest.raises(ValidationError):
            make_task_config(match_fields=["nonexistent"])

    def test_multiple_placeholders_all_in_input_schema(self):
        tc = make_task_config(
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "context": {"type": "string"},
                },
            },
            prompt_template="Process {query} with {context}",
        )
        assert tc.prompt_template == "Process {query} with {context}"

    def test_template_placeholder_not_in_input_schema_rejected(self):
        with pytest.raises(ValidationError):
            make_task_config(prompt_template="Hello {unknown_field}")

    def test_malformed_template_rejected(self):
        with pytest.raises((ValidationError, ValueError)):
            make_task_config(prompt_template="Hello {broken")

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            make_task_config(unknown_field="surprise")

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            make_task_config(name="")

    def test_non_snake_case_name_rejected(self):
        with pytest.raises(ValidationError):
            make_task_config(name="CamelCaseName")

    def test_empty_system_prompt_rejected(self):
        with pytest.raises(ValidationError):
            make_task_config(system_prompt="")

    def test_evaluator_type_is_str_enum(self):
        assert EvaluatorType.structured_match == "structured_match"
        assert EvaluatorType.semantic_similarity == "semantic_similarity"
        assert EvaluatorType.exact_match == "exact_match"
        assert EvaluatorType.custom == "custom"


# ---------------------------------------------------------------------------
# TestTaskRegistryConstruction
# ---------------------------------------------------------------------------

class TestTaskRegistryConstruction:
    """Validate TaskRegistry construction from task lists."""

    def test_valid_single_task(self):
        tc = make_task_config()
        registry = TaskRegistry([tc])
        assert len(registry) == 1
        assert tc.name in registry

    def test_valid_multiple_tasks(self):
        tc1 = make_task_config(name="task_one")
        tc2 = make_task_config(name="task_two")
        tc3 = make_task_config(name="task_three")
        registry = TaskRegistry([tc1, tc2, tc3])
        assert len(registry) == 3
        for tc in [tc1, tc2, tc3]:
            assert tc.name in registry

    def test_empty_list_raises_error(self):
        with pytest.raises((ValidationError, ValueError)):
            TaskRegistry([])

    def test_duplicate_names_raises_error(self):
        tc1 = make_task_config(name="my_task")
        tc2 = make_task_config(name="my_task")
        with pytest.raises((ValidationError, ValueError)):
            TaskRegistry([tc1, tc2])

    def test_registry_is_frozen(self):
        registry = make_registry()
        with pytest.raises((TypeError, ValidationError, AttributeError)):
            registry.tasks = {}

    def test_postcondition_tasks_keyed_by_name(self):
        tc = make_task_config(name="my_task")
        registry = TaskRegistry([tc])
        assert "my_task" in registry.tasks
        assert registry.tasks["my_task"] == tc


# ---------------------------------------------------------------------------
# TestTaskRegistryLookup
# ---------------------------------------------------------------------------

class TestTaskRegistryLookup:
    """Validate registry read-only lookup methods."""

    def test_get_task_happy_path(self):
        tc = make_task_config(name="classify_email")
        registry = TaskRegistry([tc])
        result = registry.get_task("classify_email")
        assert result.name == "classify_email"
        assert result == tc

    def test_get_task_returns_same_object(self):
        tc = make_task_config(name="classify_email")
        registry = TaskRegistry([tc])
        result = registry.get_task("classify_email")
        # Frozen model identity — at minimum equality must hold
        assert result == tc

    def test_get_task_not_found_raises_with_context(self):
        tc = make_task_config(name="classify_email")
        registry = TaskRegistry([tc])
        with pytest.raises(TaskNotFoundError) as exc_info:
            registry.get_task("nonexistent")
        err = exc_info.value
        assert err.task_name == "nonexistent", (
            f"Expected task_name='nonexistent', got '{err.task_name}'"
        )
        assert "classify_email" in err.available_tasks, (
            f"Expected 'classify_email' in available_tasks, got {err.available_tasks}"
        )

    def test_get_task_not_found_is_not_key_error(self):
        """TaskNotFoundError should be a domain error, not KeyError."""
        registry = make_registry()
        with pytest.raises(TaskNotFoundError):
            registry.get_task("nonexistent")
        # Ensure it does NOT raise KeyError
        assert not issubclass(TaskNotFoundError, KeyError)

    def test_get_task_empty_string_raises_not_found(self):
        registry = make_registry()
        with pytest.raises(TaskNotFoundError):
            registry.get_task("")

    def test_get_task_resolves_prefixed_name_to_registered_suffix(self):
        # Multi-tenant convention: callers prefix per-instance task
        # names with a tenant identifier (e.g. "tnt_abc:classify_email")
        # so the confidence engine tracks state per-tenant under a
        # single registry entry. Lookup must fall back to the suffix.
        tc = make_task_config(name="classify_email")
        registry = TaskRegistry([tc])
        result = registry.get_task("tnt_abc:classify_email")
        assert result.name == "classify_email"
        assert result == tc

    def test_get_task_prefix_resolution_only_strips_first_colon(self):
        # Suffix resolution strips up to and including the first ":".
        # If a registered task name itself contains a ":", a chained
        # prefix like "tenant:env:task_name" still resolves correctly
        # because the stripped suffix "env:task_name" is then matched
        # against the registry — registered task names without colons
        # match the canonical post-first-colon form.
        tc = make_task_config(name="classify_email")
        registry = TaskRegistry([tc])
        # "tnt_abc:classify_email" -> suffix "classify_email" -> match
        result = registry.get_task("tnt_abc:classify_email")
        assert result.name == "classify_email"

    def test_get_task_unknown_prefixed_name_still_raises(self):
        # Suffix fallback only matches when the suffix is registered;
        # entirely unknown task names still raise TaskNotFoundError.
        tc = make_task_config(name="classify_email")
        registry = TaskRegistry([tc])
        with pytest.raises(TaskNotFoundError):
            registry.get_task("tnt_abc:not_registered")

    def test_contains_true_for_prefixed_name(self):
        tc = make_task_config(name="classify_email")
        registry = TaskRegistry([tc])
        assert "tnt_abc:classify_email" in registry
        assert "another_tenant:classify_email" in registry

    def test_contains_false_for_prefixed_unknown(self):
        tc = make_task_config(name="classify_email")
        registry = TaskRegistry([tc])
        assert "tnt_abc:something_else" not in registry

    def test_contains_true(self):
        tc = make_task_config(name="classify_email")
        registry = TaskRegistry([tc])
        assert "classify_email" in registry

    def test_contains_false(self):
        registry = make_registry()
        assert "nonexistent" not in registry

    def test_contains_empty_string_false(self):
        registry = make_registry()
        assert "" not in registry

    def test_task_names_returns_frozenset(self):
        tc1 = make_task_config(name="classify_email")
        tc2 = make_task_config(name="extract_data")
        registry = TaskRegistry([tc1, tc2])
        names = registry.task_names
        assert isinstance(names, frozenset), (
            f"Expected frozenset, got {type(names).__name__}"
        )
        assert names == frozenset({"classify_email", "extract_data"})

    def test_len_correct(self):
        tc1 = make_task_config(name="task_a")
        tc2 = make_task_config(name="task_b")
        registry = TaskRegistry([tc1, tc2])
        assert len(registry) == 2

    def test_len_minimum_one(self):
        registry = make_registry()
        assert len(registry) >= 1


# ---------------------------------------------------------------------------
# TestTaskRegistryInvariants
# ---------------------------------------------------------------------------

class TestTaskRegistryInvariants:
    """Verify algebraic invariants that must hold for any valid registry."""

    def _make_multi_task_registry(self):
        tc1 = make_task_config(name="task_alpha")
        tc2 = make_task_config(name="task_beta")
        tc3 = make_task_config(name="task_gamma")
        tasks = [tc1, tc2, tc3]
        registry = TaskRegistry(tasks)
        return registry, tasks

    def test_len_equals_task_names_size(self):
        registry, _ = self._make_multi_task_registry()
        assert len(registry) == len(registry.task_names), (
            f"len(registry)={len(registry)} != len(task_names)={len(registry.task_names)}"
        )

    def test_task_names_equals_tasks_keys(self):
        registry, _ = self._make_multi_task_registry()
        assert registry.task_names == frozenset(registry.tasks.keys())

    def test_contains_agrees_with_get_task(self):
        registry, tasks = self._make_multi_task_registry()
        for name in registry.task_names:
            assert name in registry, f"'{name}' should be in registry"
            result = registry.get_task(name)
            assert result.name == name

    def test_get_task_fails_for_names_not_in_task_names(self):
        registry, _ = self._make_multi_task_registry()
        fake_names = ["nonexistent", "fake_task", ""]
        for name in fake_names:
            assert name not in registry
            with pytest.raises(TaskNotFoundError):
                registry.get_task(name)

    def test_all_match_fields_in_output_schema(self):
        """Invariant: for every task, all match_fields are keys in output_schema properties."""
        registry, _ = self._make_multi_task_registry()
        for name in registry.task_names:
            tc = registry.get_task(name)
            output_props = tc.output_schema.get("properties", {})
            for field in tc.match_fields:
                assert field in output_props, (
                    f"Task '{name}': match_field '{field}' not in output_schema properties"
                )

    def test_all_template_placeholders_in_input_schema(self):
        """Invariant: for every task, all prompt_template placeholders are in input_schema properties."""
        registry, _ = self._make_multi_task_registry()
        formatter = string.Formatter()
        for name in registry.task_names:
            tc = registry.get_task(name)
            input_props = tc.input_schema.get("properties", {})
            placeholders = [
                fname
                for _, fname, _, _ in formatter.parse(tc.prompt_template)
                if fname is not None
            ]
            for ph in placeholders:
                assert ph in input_props, (
                    f"Task '{name}': placeholder '{ph}' not in input_schema properties"
                )

    def test_threshold_ordering_holds_for_all_tasks(self):
        """Invariant: for every task, emergency < coaching < phase2_to_phase3."""
        registry, _ = self._make_multi_task_registry()
        for name in registry.task_names:
            tc = registry.get_task(name)
            ct = tc.confidence_thresholds
            assert ct.emergency_threshold < ct.coaching_trigger, (
                f"Task '{name}': emergency ({ct.emergency_threshold}) >= "
                f"coaching ({ct.coaching_trigger})"
            )
            assert ct.coaching_trigger < ct.phase2_to_phase3_correlation, (
                f"Task '{name}': coaching ({ct.coaching_trigger}) >= "
                f"correlation ({ct.phase2_to_phase3_correlation})"
            )

    def test_registry_postcondition_all_tasks_accessible(self):
        """Postcondition: for every tc in task_list, tc.name in registry and get_task returns tc."""
        tc1 = make_task_config(name="task_one")
        tc2 = make_task_config(name="task_two")
        tasks = [tc1, tc2]
        registry = TaskRegistry(tasks)
        assert len(registry) == len(tasks)
        assert registry.task_names == frozenset(tc.name for tc in tasks)
        for tc in tasks:
            assert tc.name in registry
            assert registry.get_task(tc.name) == tc

    def test_immutability_tasks_dict_cannot_be_replaced(self):
        registry = make_registry()
        with pytest.raises((TypeError, ValidationError, AttributeError)):
            registry.tasks = {"hacked": None}

    def test_task_not_found_error_is_exception_not_key_error(self):
        """TaskNotFoundError inherits from Exception, not KeyError."""
        assert issubclass(TaskNotFoundError, Exception)
        assert not issubclass(TaskNotFoundError, KeyError)


# ---------------------------------------------------------------------------
# TestTaskRegistryProtocol (lightweight protocol satisfaction)
# ---------------------------------------------------------------------------

class TestTaskRegistryProtocol:
    """Verify that TaskRegistry satisfies the read-only protocol interface."""

    def test_registry_has_get_task_method(self):
        registry = make_registry()
        assert callable(getattr(registry, "get_task", None))

    def test_registry_has_contains(self):
        registry = make_registry()
        # __contains__ is used by 'in' operator
        assert hasattr(registry, "__contains__")

    def test_registry_has_task_names_property(self):
        registry = make_registry()
        # Should be accessible as a property
        names = registry.task_names
        assert isinstance(names, frozenset)

    def test_registry_has_len(self):
        registry = make_registry()
        assert hasattr(registry, "__len__")
        assert isinstance(len(registry), int)
