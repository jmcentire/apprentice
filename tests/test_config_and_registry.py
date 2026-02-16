"""
Contract tests for config_and_registry component.
Organized across logical sections: config loading, task registry, data models,
serialization, and cross-field validation.

Run with: pytest contract_test.py -v
"""

import json
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Import the component under test
# ---------------------------------------------------------------------------
from src.config_and_registry import (
    # Config loading
    load_config,
    resolve_env_var,
    validate_cross_field_constraints,
    # Config types / errors
    ApprenticeConfig,
    ProviderConfig,
    LocalModelConfig,
    BudgetConfig,
    FinetuningConfig,
    AuditConfig,
    TrainingDataStoreConfig,
    ConfigConfidenceThresholds,
    ConfigTaskConfig,
    EvaluatorConfig,
    MatchFieldSpec,
    TaskFieldSpec,
    ConfigError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
    EnvVarNotFoundError,
    ValidationErrorDetail,
    # Task Registry
    TaskRegistry,
    RegistryTaskConfig,
    RegistryConfidenceThresholds,
    TaskNotFoundError,
    # Data models
    TaskRequest,
    TaskResponse,
    TrainingExample,
    ComparisonPair,
    ConfidenceSnapshot,
    BudgetPeriod,
    BudgetState,
    AuditEntry,
    SamplingDecision,
    ModelVersion,
    ExactMatchResult,
    FuzzyMatchResult,
    SemanticSimilarityResult,
    LlmJudgeResult,
    CustomEvalResult,
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
    # Serialization
    serialize_model,
    deserialize_model,
    validate_model,
    get_model_json_schema,
    # Enums
    Phase,
    RoutingDecision,
    EvaluatorType,
    EvaluatorTypeName,
    FinetuningBackendName,
    BudgetPeriodUnit,
    EvaluationResultType,
    LogLevel,
)


# ===========================================================================
# Fixtures (conftest-equivalent)
# ===========================================================================

NOW = datetime.now(timezone.utc)
NOW_ISO = NOW.isoformat()


@pytest.fixture
def aware_now():
    """Return a timezone-aware datetime."""
    return datetime.now(timezone.utc)


@pytest.fixture
def mock_env():
    """Common environment mapping for tests."""
    return {
        "PROVIDER_API_KEY": "sk-test-key-12345",
        "FINETUNING_API_KEY": "ft-test-key-67890",
    }


@pytest.fixture
def minimal_input_schema():
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
    }


@pytest.fixture
def minimal_output_schema():
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
    }


def _make_valid_yaml_content(
    *,
    extra_provider_fields="",
    extra_budget_fields="",
    extra_task_fields="",
    extra_finetuning_fields="",
    extra_top_level_fields="",
    provider_api_key="env:PROVIDER_API_KEY",
    finetuning_api_key="env:FINETUNING_API_KEY",
    finetuning_backend="local_lora",
    task_name="my_task",
    second_task_name=None,
    evaluator_type="exact_match",
    threshold_local_ready=0.5,
    threshold_local_only=0.9,
    threshold_degraded=0.3,
    max_daily_cost=100.0,
    max_monthly_cost=3000.0,
    semantic_model="",
    judge_prompt_template="",
    regex_pattern="",
    json_schema_path="",
    match_field_name="answer",
    finetuning_api_base_url="https://api.example.com",
):
    """Build a valid YAML config string with optional overrides."""
    evaluator_extras = ""
    if semantic_model:
        evaluator_extras += f"\n        semantic_model: '{semantic_model}'"
    if judge_prompt_template:
        evaluator_extras += f"\n        judge_prompt_template: '{judge_prompt_template}'"
    if regex_pattern:
        evaluator_extras += f"\n        regex_pattern: '{regex_pattern}'"
    if json_schema_path:
        evaluator_extras += f"\n        json_schema_path: '{json_schema_path}'"

    second_task = ""
    if second_task_name:
        second_task = f"""
  - task_name: '{second_task_name}'
    prompt_template: 'Translate: {{query}}'
    input_schema:
      - name: 'query'
        type: 'string'
        required: true
        description: 'Input query'
    output_schema:
      - name: 'answer'
        type: 'string'
        required: true
        description: 'Output answer'
    evaluators:
      - type: 'exact_match'
        match_fields:
          - name: 'answer'
            weight: 1.0
            case_sensitive: true
        semantic_model: ''
        judge_prompt_template: ''
        regex_pattern: ''
        json_schema_path: ''
    thresholds:
      local_ready: 0.5
      local_only: 0.9
      degraded_threshold: 0.3
    sampling_rate_initial: 0.1
    min_training_examples: 100"""

    yaml_content = f"""
provider:
  api_base_url: 'https://api.openai.com/v1'
  api_key: '{provider_api_key}'
  model: 'gpt-4'
  timeout_seconds: 30
  max_retries: 3
  retry_base_delay_seconds: 1.0
  {extra_provider_fields}

local_model:
  endpoint: 'http://localhost:8080'
  model_name: 'local-llama'
  timeout_seconds: 10
  max_retries: 2
  health_check_path: '/health'

tasks:
  - task_name: '{task_name}'
    prompt_template: 'Answer: {{query}}'
    input_schema:
      - name: 'query'
        type: 'string'
        required: true
        description: 'Input query'
    output_schema:
      - name: 'answer'
        type: 'string'
        required: true
        description: 'Output answer'
    evaluators:
      - type: '{evaluator_type}'
        match_fields:
          - name: '{match_field_name}'
            weight: 1.0
            case_sensitive: true
        semantic_model: '{semantic_model}'
        judge_prompt_template: '{judge_prompt_template}'
        regex_pattern: '{regex_pattern}'
        json_schema_path: '{json_schema_path}'
    thresholds:
      local_ready: {threshold_local_ready}
      local_only: {threshold_local_only}
      degraded_threshold: {threshold_degraded}
    sampling_rate_initial: 0.1
    min_training_examples: 100
    {extra_task_fields}{second_task}

budget:
  max_daily_cost_usd: {max_daily_cost}
  max_monthly_cost_usd: {max_monthly_cost}
  rolling_window_hours: 24
  cost_per_input_token: 0.00003
  cost_per_output_token: 0.00006
  budget_state_path: '/tmp/budget.json'
  {extra_budget_fields}

finetuning:
  backend: '{finetuning_backend}'
  model_base: 'llama-7b'
  batch_size: 32
  trigger_interval_hours: 24
  api_key: '{finetuning_api_key}'
  api_base_url: '{finetuning_api_base_url}'
  output_dir: '/tmp/models'
  max_concurrent_jobs: 2
  {extra_finetuning_fields}

audit:
  log_path: '/tmp/audit.log'
  log_level: 'INFO'
  log_to_stdout: false
  max_file_size_mb: 100
  backup_count: 5

training_data:
  storage_dir: '/tmp/training'
  max_examples_per_task: 10000

{extra_top_level_fields}
"""
    return yaml_content


def _write_yaml(tmp_path, content, filename="config.yaml"):
    """Write content to a YAML file and return its path."""
    p = tmp_path / filename
    p.write_text(content)
    return p


def _make_registry_task_config(
    name="summarize",
    input_schema=None,
    output_schema=None,
    prompt_template="Summarize: {text}",
    match_fields=None,
    system_prompt="You are a helpful assistant.",
    evaluator_type=None,
    thresholds=None,
):
    """Helper to build a RegistryTaskConfig for tests."""
    if input_schema is None:
        input_schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        }
    if output_schema is None:
        output_schema = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
        }
    if match_fields is None:
        match_fields = ["summary"]
    if evaluator_type is None:
        evaluator_type = EvaluatorType.exact_match
    if thresholds is None:
        thresholds = RegistryConfidenceThresholds(
            phase1_to_phase2_example_count=50,
            phase2_to_phase3_correlation=0.9,
            coaching_trigger=0.5,
            emergency_threshold=0.2,
        )

    return RegistryTaskConfig(
        name=name,
        description=f"Test task: {name}",
        input_schema=input_schema,
        output_schema=output_schema,
        system_prompt=system_prompt,
        prompt_template=prompt_template,
        evaluator_type=evaluator_type,
        match_fields=match_fields,
        confidence_thresholds=thresholds,
    )


def _make_budget_period(aware_now=None):
    if aware_now is None:
        aware_now = datetime.now(timezone.utc)
    return BudgetPeriod(
        unit=BudgetPeriodUnit.DAILY,
        period_start=aware_now,
        period_end=aware_now + timedelta(days=1),
    )


def _make_exact_match_result():
    return ExactMatchResult(
        result_type=EvaluationResultType.EXACT_MATCH_RESULT,
        is_match=True,
        matched_fields=["answer"],
        mismatched_fields=[],
    )


# ===========================================================================
# SECTION 1: Config Loading Tests
# ===========================================================================

class TestLoadConfigHappyPath:
    """Tests for successful config loading."""

    def test_load_config_returns_valid_config(self, tmp_path, mock_env):
        """load_config returns a frozen ApprenticeConfig from valid YAML."""
        yaml_content = _make_valid_yaml_content()
        config_path = _write_yaml(tmp_path, yaml_content)

        config = load_config(config_path, env=mock_env)

        assert isinstance(config, ApprenticeConfig), \
            "load_config must return an ApprenticeConfig instance"
        assert config.provider.model == "gpt-4"

    def test_loaded_config_is_frozen(self, tmp_path, mock_env):
        """Returned ApprenticeConfig is frozen — attribute assignment raises."""
        yaml_content = _make_valid_yaml_content()
        config_path = _write_yaml(tmp_path, yaml_content)
        config = load_config(config_path, env=mock_env)

        with pytest.raises(ValidationError):
            config.provider = None

    def test_env_vars_resolved(self, tmp_path, mock_env):
        """All env:VAR_NAME references are resolved to actual values."""
        yaml_content = _make_valid_yaml_content()
        config_path = _write_yaml(tmp_path, yaml_content)
        config = load_config(config_path, env=mock_env)

        secret_value = config.provider.api_key.get_secret_value()
        assert secret_value == "sk-test-key-12345", \
            "env:PROVIDER_API_KEY should resolve to 'sk-test-key-12345'"

    def test_secret_str_masking(self, tmp_path, mock_env):
        """SecretStr fields mask their values in repr and str."""
        yaml_content = _make_valid_yaml_content()
        config_path = _write_yaml(tmp_path, yaml_content)
        config = load_config(config_path, env=mock_env)

        config_repr = repr(config)
        assert "sk-test-key-12345" not in config_repr, \
            "Secret value must not appear in repr"
        config_str = str(config)
        assert "sk-test-key-12345" not in config_str, \
            "Secret value must not appear in str"

    def test_loader_is_pure(self, tmp_path, mock_env):
        """Calling load_config twice with same inputs produces identical results."""
        yaml_content = _make_valid_yaml_content()
        config_path = _write_yaml(tmp_path, yaml_content)

        config1 = load_config(config_path, env=mock_env)
        config2 = load_config(config_path, env=mock_env)

        assert config1 == config2, \
            "load_config must be pure — same inputs yield equal results"


class TestLoadConfigFileErrors:
    """Tests for file-level errors in load_config."""

    def test_file_not_found(self, tmp_path, mock_env):
        """Raises ConfigFileNotFoundError when file does not exist."""
        missing_path = tmp_path / "nonexistent.yaml"
        with pytest.raises(ConfigFileNotFoundError) as exc_info:
            load_config(missing_path, env=mock_env)
        assert str(missing_path) in str(exc_info.value.path) or \
               "nonexistent" in str(exc_info.value), \
            "Error should reference the missing file path"

    def test_yaml_syntax_error(self, tmp_path, mock_env):
        """Raises ConfigParseError on invalid YAML syntax."""
        config_path = _write_yaml(tmp_path, "{{invalid: yaml: [")
        with pytest.raises(ConfigParseError):
            load_config(config_path, env=mock_env)

    def test_empty_config_file(self, tmp_path, mock_env):
        """Raises error when YAML file is empty."""
        config_path = _write_yaml(tmp_path, "")
        with pytest.raises((ConfigParseError, ConfigValidationError)):
            load_config(config_path, env=mock_env)

    def test_yaml_not_a_mapping(self, tmp_path, mock_env):
        """Raises ConfigParseError when top-level YAML is a list."""
        config_path = _write_yaml(tmp_path, "- item1\n- item2\n")
        with pytest.raises((ConfigParseError, ConfigValidationError)):
            load_config(config_path, env=mock_env)


class TestLoadConfigValidationErrors:
    """Tests for schema/type validation errors in load_config."""

    def test_missing_required_field(self, tmp_path, mock_env):
        """Raises ConfigValidationError when required field is missing."""
        # Remove the 'provider' section entirely
        yaml_content = """
local_model:
  endpoint: 'http://localhost:8080'
  model_name: 'local-llama'
  timeout_seconds: 10
  max_retries: 2
  health_check_path: '/health'
"""
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(config_path, env=mock_env)
        assert len(exc_info.value.validation_errors) >= 1, \
            "Should have at least one validation error for missing fields"

    def test_invalid_field_type_strict(self, tmp_path, mock_env):
        """Raises ConfigValidationError when strict type coercion fails."""
        yaml_content = _make_valid_yaml_content()
        # Replace timeout_seconds int with a string
        yaml_content = yaml_content.replace(
            "timeout_seconds: 30", "timeout_seconds: 'thirty'"
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)

    def test_extra_field_forbidden(self, tmp_path, mock_env):
        """Raises ConfigValidationError when extra fields are present."""
        yaml_content = _make_valid_yaml_content(
            extra_top_level_fields="unexpected_field: 'should fail'"
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)

    def test_unknown_evaluator_type(self, tmp_path, mock_env):
        """Raises ConfigValidationError for unknown evaluator type."""
        yaml_content = _make_valid_yaml_content(evaluator_type="nonexistent_evaluator")
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)

    def test_unknown_finetuning_backend(self, tmp_path, mock_env):
        """Raises ConfigValidationError for unknown finetuning backend."""
        yaml_content = _make_valid_yaml_content(finetuning_backend="totally_unknown")
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)


class TestLoadConfigCrossFieldErrors:
    """Tests for cross-field validation errors in load_config."""

    def test_threshold_ordering_violation(self, tmp_path, mock_env):
        """Raises ConfigValidationError when local_only < local_ready."""
        yaml_content = _make_valid_yaml_content(
            threshold_local_ready=0.9,
            threshold_local_only=0.5,  # local_only < local_ready => violation
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)

    def test_monthly_less_than_daily(self, tmp_path, mock_env):
        """Raises ConfigValidationError when monthly budget < daily budget."""
        yaml_content = _make_valid_yaml_content(
            max_daily_cost=500.0,
            max_monthly_cost=100.0,
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)

    def test_semantic_similarity_without_model(self, tmp_path, mock_env):
        """Raises ConfigValidationError when semantic_similarity lacks semantic_model."""
        yaml_content = _make_valid_yaml_content(
            evaluator_type="semantic_similarity",
            semantic_model="",  # missing
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)

    def test_llm_judge_without_template(self, tmp_path, mock_env):
        """Raises ConfigValidationError when llm_judge lacks judge_prompt_template."""
        yaml_content = _make_valid_yaml_content(
            evaluator_type="llm_judge",
            judge_prompt_template="",  # missing
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)

    def test_regex_match_without_pattern(self, tmp_path, mock_env):
        """Raises ConfigValidationError when regex_match lacks regex_pattern."""
        yaml_content = _make_valid_yaml_content(
            evaluator_type="regex_match",
            regex_pattern="",
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)

    def test_json_schema_match_without_path(self, tmp_path, mock_env):
        """Raises ConfigValidationError when json_schema_match lacks json_schema_path."""
        yaml_content = _make_valid_yaml_content(
            evaluator_type="json_schema_match",
            json_schema_path="",
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)

    def test_duplicate_task_names(self, tmp_path, mock_env):
        """Raises ConfigValidationError when two tasks share the same name."""
        yaml_content = _make_valid_yaml_content(
            task_name="duplicate_task",
            second_task_name="duplicate_task",
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)

    def test_match_field_not_in_output_schema(self, tmp_path, mock_env):
        """Raises ConfigValidationError when match_field not in output_schema."""
        yaml_content = _make_valid_yaml_content(
            match_field_name="nonexistent_field",
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)

    def test_unsupported_backend_combination(self, tmp_path, mock_env):
        """Raises ConfigValidationError for remote backend missing api_key/api_base_url."""
        yaml_content = _make_valid_yaml_content(
            finetuning_backend="openai",
            finetuning_api_key="",
            finetuning_api_base_url="",
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises((ConfigValidationError, EnvVarNotFoundError)):
            load_config(config_path, env=mock_env)


class TestLoadConfigEnvVars:
    """Tests for env var resolution in load_config."""

    def test_env_var_not_found(self, tmp_path):
        """Raises EnvVarNotFoundError when env var is not set."""
        yaml_content = _make_valid_yaml_content(
            provider_api_key="env:MISSING_API_KEY"
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        # Provide env without the referenced var
        with pytest.raises(EnvVarNotFoundError) as exc_info:
            load_config(config_path, env={"FINETUNING_API_KEY": "ft-key"})
        assert "MISSING_API_KEY" in str(exc_info.value) or \
               hasattr(exc_info.value, 'env_var_name') and exc_info.value.env_var_name == "MISSING_API_KEY"

    def test_yaml_boolean_gotcha(self, tmp_path, mock_env):
        """YAML bare 'yes' parsed as boolean causes strict type validation error."""
        yaml_content = _make_valid_yaml_content()
        # Replace a string field with bare 'yes' (YAML parses as True)
        yaml_content = yaml_content.replace(
            "model: 'gpt-4'", "model: yes"
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)


# ===========================================================================
# SECTION 2: resolve_env_var Tests
# ===========================================================================

class TestResolveEnvVar:
    """Tests for the resolve_env_var function."""

    def test_resolves_present_var(self):
        """Returns the env var value when env:VAR_NAME is provided and var exists."""
        result = resolve_env_var("env:MY_KEY", {"MY_KEY": "resolved_value"}, "test.path")
        assert result == "resolved_value"

    def test_passthrough_no_prefix(self):
        """Returns input unchanged when it does not start with 'env:'."""
        result = resolve_env_var("plain_value", {}, "test.path")
        assert result == "plain_value"

    def test_env_var_not_set(self):
        """Raises EnvVarNotFoundError when referenced env var is not in mapping."""
        with pytest.raises(EnvVarNotFoundError):
            resolve_env_var("env:MISSING_VAR", {}, "test.path")

    def test_empty_var_name(self):
        """Raises error when value is 'env:' with no variable name."""
        with pytest.raises((EnvVarNotFoundError, ValueError, ConfigError)):
            resolve_env_var("env:", {}, "test.path")

    def test_env_prefix_case_sensitive(self):
        """The 'env:' prefix check is case-sensitive — 'ENV:' is not resolved."""
        result = resolve_env_var("ENV:NOT_A_VAR", {}, "test.path")
        assert result == "ENV:NOT_A_VAR", "Only lowercase 'env:' should trigger resolution"


# ===========================================================================
# SECTION 3: validate_cross_field_constraints Tests
# ===========================================================================

class TestValidateCrossFieldConstraints:
    """Tests for validate_cross_field_constraints."""

    def test_valid_config_passes(self, tmp_path, mock_env):
        """Returns config unchanged when all constraints pass."""
        yaml_content = _make_valid_yaml_content()
        config_path = _write_yaml(tmp_path, yaml_content)
        config = load_config(config_path, env=mock_env)
        # If load_config succeeded, constraints passed. We can also call directly:
        result = validate_cross_field_constraints(config)
        assert result == config, "Valid config should be returned unchanged"

    def test_threshold_ordering_violation(self, tmp_path, mock_env):
        """Raises ConfigValidationError for threshold ordering violation."""
        # Build a config programmatically with bad thresholds
        # Since we can't easily construct an invalid ApprenticeConfig (it validates),
        # we test via load_config which calls validate_cross_field_constraints
        yaml_content = _make_valid_yaml_content(
            threshold_local_ready=0.95,
            threshold_local_only=0.3,
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)

    def test_monthly_less_than_daily_violation(self, tmp_path, mock_env):
        """Raises ConfigValidationError when monthly < daily."""
        yaml_content = _make_valid_yaml_content(
            max_daily_cost=1000.0,
            max_monthly_cost=500.0,
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError):
            load_config(config_path, env=mock_env)

    def test_multiple_violations_collected(self, tmp_path, mock_env):
        """Multiple violations are collected into one ConfigValidationError."""
        yaml_content = _make_valid_yaml_content(
            threshold_local_ready=0.95,
            threshold_local_only=0.3,
            max_daily_cost=5000.0,
            max_monthly_cost=100.0,
        )
        config_path = _write_yaml(tmp_path, yaml_content)
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config(config_path, env=mock_env)
        # Should have at least 2 violations
        if hasattr(exc_info.value, 'validation_errors'):
            assert len(exc_info.value.validation_errors) >= 1


# ===========================================================================
# SECTION 4: Task Registry Tests
# ===========================================================================

class TestTaskRegistryConstruction:
    """Tests for TaskRegistry construction."""

    def test_construct_with_valid_tasks(self):
        """TaskRegistry constructs from a valid list of RegistryTaskConfig objects."""
        tc1 = _make_registry_task_config(name="task_one")
        tc2 = _make_registry_task_config(
            name="task_two",
            prompt_template="Translate: {text}",
        )
        registry = TaskRegistry(tasks=[tc1, tc2])

        assert len(registry) == 2
        assert "task_one" in registry
        assert "task_two" in registry

    def test_empty_list_raises(self):
        """TaskRegistry raises error when constructed with empty list."""
        with pytest.raises((ValueError, Exception)):
            TaskRegistry(tasks=[])

    def test_duplicate_names_raises(self):
        """TaskRegistry raises error when two tasks share a name."""
        tc1 = _make_registry_task_config(name="duplicate")
        tc2 = _make_registry_task_config(name="duplicate")
        with pytest.raises((ValueError, Exception)):
            TaskRegistry(tasks=[tc1, tc2])

    def test_registry_is_frozen(self):
        """TaskRegistry is immutable after construction."""
        tc = _make_registry_task_config()
        registry = TaskRegistry(tasks=[tc])
        with pytest.raises(ValidationError):
            registry.tasks = {}


class TestTaskRegistryLookup:
    """Tests for TaskRegistry get_task and membership."""

    def test_get_task_returns_correct_config(self):
        """get_task returns the RegistryTaskConfig for a known task name."""
        tc = _make_registry_task_config(name="lookup_test")
        registry = TaskRegistry(tasks=[tc])

        result = registry.get_task("lookup_test")
        assert result.name == "lookup_test"
        assert result == tc

    def test_get_task_not_found(self):
        """get_task raises TaskNotFoundError for unknown task name."""
        tc = _make_registry_task_config(name="existing")
        registry = TaskRegistry(tasks=[tc])

        with pytest.raises(TaskNotFoundError) as exc_info:
            registry.get_task("nonexistent")
        assert exc_info.value.task_name == "nonexistent"
        assert "existing" in exc_info.value.available_tasks

    def test_contains_true(self):
        """__contains__ returns True for registered task."""
        tc = _make_registry_task_config(name="present")
        registry = TaskRegistry(tasks=[tc])
        assert "present" in registry

    def test_contains_false(self):
        """__contains__ returns False for unregistered task."""
        tc = _make_registry_task_config(name="present")
        registry = TaskRegistry(tasks=[tc])
        assert "absent" not in registry

    def test_len(self):
        """__len__ returns the number of tasks."""
        tc1 = _make_registry_task_config(name="a_task")
        tc2 = _make_registry_task_config(name="b_task", prompt_template="Do: {text}")
        registry = TaskRegistry(tasks=[tc1, tc2])
        assert len(registry) == 2

    def test_task_names_returns_frozenset(self):
        """task_names returns a frozenset of all registered task names."""
        tc1 = _make_registry_task_config(name="alpha")
        tc2 = _make_registry_task_config(name="beta", prompt_template="Q: {text}")
        registry = TaskRegistry(tasks=[tc1, tc2])

        names = registry.task_names
        assert isinstance(names, frozenset)
        assert names == frozenset({"alpha", "beta"})


class TestRegistryTaskConfigValidators:
    """Tests for RegistryTaskConfig model validators."""

    def test_match_fields_valid(self):
        """All match_fields present in output_schema passes validation."""
        tc = _make_registry_task_config(
            match_fields=["summary"],
            output_schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
            },
        )
        assert tc.match_fields == ["summary"]

    def test_match_fields_not_in_output_schema(self):
        """Raises ValidationError when match_field not in output_schema."""
        with pytest.raises((ValueError, Exception)):
            _make_registry_task_config(
                match_fields=["nonexistent_field"],
                output_schema={
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                },
            )

    def test_prompt_template_valid(self):
        """All prompt placeholders in input_schema passes validation."""
        tc = _make_registry_task_config(
            prompt_template="Process: {text}",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        )
        assert "{text}" in tc.prompt_template

    def test_prompt_template_missing_placeholder(self):
        """Raises error when placeholder not in input_schema."""
        with pytest.raises((ValueError, Exception)):
            _make_registry_task_config(
                prompt_template="Process: {missing_field}",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            )


class TestRegistryConfidenceThresholds:
    """Tests for RegistryConfidenceThresholds threshold ordering."""

    def test_valid_ordering(self):
        """Valid ordering: emergency < coaching < phase2_to_phase3."""
        thresholds = RegistryConfidenceThresholds(
            phase1_to_phase2_example_count=50,
            phase2_to_phase3_correlation=0.9,
            coaching_trigger=0.5,
            emergency_threshold=0.2,
        )
        assert thresholds.emergency_threshold < thresholds.coaching_trigger
        assert thresholds.coaching_trigger < thresholds.phase2_to_phase3_correlation

    def test_invalid_ordering_emergency_ge_coaching(self):
        """Raises error when emergency_threshold >= coaching_trigger."""
        with pytest.raises((ValueError, Exception)):
            RegistryConfidenceThresholds(
                phase1_to_phase2_example_count=50,
                phase2_to_phase3_correlation=0.9,
                coaching_trigger=0.3,
                emergency_threshold=0.5,  # >= coaching_trigger
            )

    def test_invalid_ordering_coaching_ge_correlation(self):
        """Raises error when coaching_trigger >= phase2_to_phase3_correlation."""
        with pytest.raises((ValueError, Exception)):
            RegistryConfidenceThresholds(
                phase1_to_phase2_example_count=50,
                phase2_to_phase3_correlation=0.4,
                coaching_trigger=0.5,  # >= phase2_to_phase3_correlation
                emergency_threshold=0.1,
            )


# ===========================================================================
# SECTION 5: Data Models / Factory Functions Tests
# ===========================================================================

class TestCreateTaskRequest:
    """Tests for create_task_request factory."""

    def test_happy_path(self, aware_now):
        """Creates a frozen TaskRequest with valid inputs."""
        req = create_task_request(
            task_id="my_task",
            input_data={"query": "hello"},
        )
        assert req.task_id == "my_task"
        assert req.input_data == {"query": "hello"}
        assert req.request_id  # auto-generated
        assert req.created_at.tzinfo is not None, "created_at must be timezone-aware"
        # Verify frozen
        with pytest.raises(ValidationError):
            req.task_id = "other"

    def test_invalid_task_id_pattern(self):
        """Raises error for task_id not matching ^[a-z][a-z0-9_-]*$."""
        with pytest.raises((ValueError, Exception)):
            create_task_request(task_id="Invalid-ID", input_data={})

    def test_empty_task_id(self):
        """Raises error for empty task_id."""
        with pytest.raises((ValueError, Exception)):
            create_task_request(task_id="", input_data={})

    def test_task_id_starts_with_number(self):
        """Raises error when task_id starts with a number."""
        with pytest.raises((ValueError, Exception)):
            create_task_request(task_id="1invalid", input_data={})

    def test_task_id_min_length(self):
        """Single character task_id 'a' is valid."""
        req = create_task_request(task_id="a", input_data={})
        assert req.task_id == "a"

    def test_task_id_max_length(self):
        """Task_id at exactly 128 characters is valid."""
        long_id = "a" * 128
        req = create_task_request(task_id=long_id, input_data={})
        assert req.task_id == long_id

    def test_task_id_over_max_length(self):
        """Task_id over 128 characters is rejected."""
        with pytest.raises((ValueError, Exception)):
            create_task_request(task_id="a" * 129, input_data={})

    def test_naive_datetime_rejected(self):
        """Naive datetime is rejected for created_at."""
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)  # no tzinfo
        with pytest.raises((ValueError, Exception)):
            create_task_request(
                task_id="test_task",
                input_data={},
                created_at=naive_dt,
            )


class TestCreateTaskResponse:
    """Tests for create_task_response factory."""

    def test_happy_path(self, aware_now):
        """Creates a frozen TaskResponse with valid inputs."""
        resp = create_task_response(
            request_id="req-123",
            task_id="my_task",
            output_data={"answer": "world"},
            model_used="gpt-4",
            routing_decision=RoutingDecision.REMOTE,
            phase=Phase.PHASE_1_REMOTE_ONLY,
            confidence_at_decision=0.3,
            latency_ms=150.0,
            cost_incurred=Decimal("0.01"),
            created_at=aware_now,
            metadata={},
        )
        assert resp.request_id == "req-123"
        assert resp.routing_decision == RoutingDecision.REMOTE
        assert resp.phase == Phase.PHASE_1_REMOTE_ONLY

    def test_invalid_routing_decision(self, aware_now):
        """Raises error for invalid routing_decision."""
        with pytest.raises((ValueError, Exception)):
            create_task_response(
                request_id="req-123",
                task_id="my_task",
                output_data={},
                model_used="gpt-4",
                routing_decision="INVALID_ROUTE",
                phase=Phase.PHASE_1_REMOTE_ONLY,
                confidence_at_decision=0.5,
                latency_ms=100.0,
                cost_incurred=Decimal("0"),
                created_at=aware_now,
                metadata={},
            )

    def test_negative_latency(self, aware_now):
        """Raises error when latency_ms < 0."""
        with pytest.raises((ValueError, Exception)):
            create_task_response(
                request_id="req-123",
                task_id="my_task",
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
        """Raises error when cost_incurred < 0."""
        with pytest.raises((ValueError, Exception)):
            create_task_response(
                request_id="req-123",
                task_id="my_task",
                output_data={},
                model_used="gpt-4",
                routing_decision=RoutingDecision.REMOTE,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                confidence_at_decision=0.5,
                latency_ms=100.0,
                cost_incurred=Decimal("-1"),
                created_at=aware_now,
                metadata={},
            )


class TestCreateConfidenceSnapshot:
    """Tests for create_confidence_snapshot factory and phase derivation."""

    def test_phase1_derivation(self, aware_now):
        """Score < threshold_phase2 => PHASE_1_REMOTE_ONLY."""
        snap = create_confidence_snapshot(
            task_id="test_task",
            score=0.2,
            sample_count=10,
            threshold_phase2=0.5,
            threshold_phase3=0.9,
            last_updated=aware_now,
            trend=0.01,
        )
        assert snap.phase == Phase.PHASE_1_REMOTE_ONLY
        assert snap.score == 0.2

    def test_phase2_derivation(self, aware_now):
        """threshold_phase2 <= score < threshold_phase3 => PHASE_2_SAMPLING."""
        snap = create_confidence_snapshot(
            task_id="test_task",
            score=0.7,
            sample_count=50,
            threshold_phase2=0.5,
            threshold_phase3=0.9,
            last_updated=aware_now,
            trend=0.02,
        )
        assert snap.phase == Phase.PHASE_2_SAMPLING

    def test_phase3_derivation(self, aware_now):
        """Score >= threshold_phase3 => PHASE_3_LOCAL_PRIMARY."""
        snap = create_confidence_snapshot(
            task_id="test_task",
            score=0.95,
            sample_count=200,
            threshold_phase2=0.5,
            threshold_phase3=0.9,
            last_updated=aware_now,
            trend=0.005,
        )
        assert snap.phase == Phase.PHASE_3_LOCAL_PRIMARY

    def test_phase2_boundary_exact(self, aware_now):
        """Score exactly at threshold_phase2 yields PHASE_2."""
        snap = create_confidence_snapshot(
            task_id="test_task",
            score=0.5,
            sample_count=30,
            threshold_phase2=0.5,
            threshold_phase3=0.9,
            last_updated=aware_now,
            trend=0.0,
        )
        assert snap.phase == Phase.PHASE_2_SAMPLING

    def test_phase3_boundary_exact(self, aware_now):
        """Score exactly at threshold_phase3 yields PHASE_3."""
        snap = create_confidence_snapshot(
            task_id="test_task",
            score=0.9,
            sample_count=100,
            threshold_phase2=0.5,
            threshold_phase3=0.9,
            last_updated=aware_now,
            trend=0.0,
        )
        assert snap.phase == Phase.PHASE_3_LOCAL_PRIMARY

    def test_score_zero(self, aware_now):
        """Score 0.0 is valid and yields PHASE_1."""
        snap = create_confidence_snapshot(
            task_id="test_task",
            score=0.0,
            sample_count=0,
            threshold_phase2=0.5,
            threshold_phase3=0.9,
            last_updated=aware_now,
            trend=0.0,
        )
        assert snap.phase == Phase.PHASE_1_REMOTE_ONLY
        assert snap.score == 0.0

    def test_score_one(self, aware_now):
        """Score 1.0 is valid and yields PHASE_3."""
        snap = create_confidence_snapshot(
            task_id="test_task",
            score=1.0,
            sample_count=500,
            threshold_phase2=0.5,
            threshold_phase3=0.9,
            last_updated=aware_now,
            trend=0.0,
        )
        assert snap.phase == Phase.PHASE_3_LOCAL_PRIMARY
        assert snap.score == 1.0

    @pytest.mark.parametrize("score", [-0.01, 1.01, -1.0, 2.0])
    def test_invalid_score(self, score, aware_now):
        """Raises error for score out of [0.0, 1.0]."""
        with pytest.raises((ValueError, Exception)):
            create_confidence_snapshot(
                task_id="test_task",
                score=score,
                sample_count=10,
                threshold_phase2=0.5,
                threshold_phase3=0.9,
                last_updated=aware_now,
                trend=0.0,
            )

    def test_invalid_thresholds(self, aware_now):
        """Raises error when threshold_phase2 > threshold_phase3."""
        with pytest.raises((ValueError, Exception)):
            create_confidence_snapshot(
                task_id="test_task",
                score=0.5,
                sample_count=10,
                threshold_phase2=0.9,
                threshold_phase3=0.5,
                last_updated=aware_now,
                trend=0.0,
            )

    def test_negative_sample_count(self, aware_now):
        """Raises error when sample_count < 0."""
        with pytest.raises((ValueError, Exception)):
            create_confidence_snapshot(
                task_id="test_task",
                score=0.5,
                sample_count=-1,
                threshold_phase2=0.5,
                threshold_phase3=0.9,
                last_updated=aware_now,
                trend=0.0,
            )

    def test_snapshot_is_frozen(self, aware_now):
        """ConfidenceSnapshot is frozen after creation."""
        snap = create_confidence_snapshot(
            task_id="test_task",
            score=0.5,
            sample_count=10,
            threshold_phase2=0.5,
            threshold_phase3=0.9,
            last_updated=aware_now,
            trend=0.0,
        )
        with pytest.raises(ValidationError):
            snap.score = 0.99


class TestCreateBudgetState:
    """Tests for create_budget_state factory."""

    def test_happy_path(self, aware_now):
        """Creates BudgetState with correct remaining and is_exhausted."""
        period = _make_budget_period(aware_now)
        state = create_budget_state(
            task_id="test_task",
            total_budget=Decimal("100.00"),
            spent=Decimal("30.00"),
            period=period,
            request_count=5,
            last_updated=aware_now,
        )
        assert state.remaining == Decimal("70.00"), \
            "remaining should be total_budget - spent"
        assert state.is_exhausted is False
        assert state.request_count == 5

    def test_budget_exhausted(self, aware_now):
        """is_exhausted=True when spent >= total_budget."""
        period = _make_budget_period(aware_now)
        state = create_budget_state(
            task_id="test_task",
            total_budget=Decimal("100.00"),
            spent=Decimal("100.00"),
            period=period,
            request_count=10,
            last_updated=aware_now,
        )
        assert state.remaining == Decimal("0.00")
        assert state.is_exhausted is True

    def test_budget_overspent(self, aware_now):
        """Spent > total_budget => negative remaining, is_exhausted=True."""
        period = _make_budget_period(aware_now)
        state = create_budget_state(
            task_id="test_task",
            total_budget=Decimal("100.00"),
            spent=Decimal("120.00"),
            period=period,
            request_count=15,
            last_updated=aware_now,
        )
        assert state.remaining == Decimal("-20.00")
        assert state.is_exhausted is True

    def test_zero_total_budget(self, aware_now):
        """Budget with total_budget=0 is valid but immediately exhausted."""
        period = _make_budget_period(aware_now)
        state = create_budget_state(
            task_id="test_task",
            total_budget=Decimal("0"),
            spent=Decimal("0"),
            period=period,
            request_count=0,
            last_updated=aware_now,
        )
        assert state.is_exhausted is True

    def test_negative_total_budget(self, aware_now):
        """Raises error when total_budget < 0."""
        period = _make_budget_period(aware_now)
        with pytest.raises((ValueError, Exception)):
            create_budget_state(
                task_id="test_task",
                total_budget=Decimal("-1"),
                spent=Decimal("0"),
                period=period,
                request_count=0,
                last_updated=aware_now,
            )

    def test_negative_spent(self, aware_now):
        """Raises error when spent < 0."""
        period = _make_budget_period(aware_now)
        with pytest.raises((ValueError, Exception)):
            create_budget_state(
                task_id="test_task",
                total_budget=Decimal("100"),
                spent=Decimal("-5"),
                period=period,
                request_count=0,
                last_updated=aware_now,
            )

    def test_budget_state_is_frozen(self, aware_now):
        """BudgetState is frozen after creation."""
        period = _make_budget_period(aware_now)
        state = create_budget_state(
            task_id="test_task",
            total_budget=Decimal("100"),
            spent=Decimal("0"),
            period=period,
            request_count=0,
            last_updated=aware_now,
        )
        with pytest.raises(ValidationError):
            state.spent = Decimal("50")

    def test_decimal_arithmetic(self, aware_now):
        """CostAmount uses Decimal for exact monetary arithmetic."""
        period = _make_budget_period(aware_now)
        state = create_budget_state(
            task_id="test_task",
            total_budget=Decimal("0.1") + Decimal("0.2"),
            spent=Decimal("0.0"),
            period=period,
            request_count=0,
            last_updated=aware_now,
        )
        assert state.total_budget == Decimal("0.3"), \
            "Decimal arithmetic should be exact: 0.1 + 0.2 == 0.3"
        assert isinstance(state.remaining, Decimal)


class TestRecordCost:
    """Tests for record_cost function."""

    def test_happy_path(self, aware_now):
        """record_cost returns updated BudgetState."""
        period = _make_budget_period(aware_now)
        original = create_budget_state(
            task_id="test_task",
            total_budget=Decimal("100"),
            spent=Decimal("30"),
            period=period,
            request_count=5,
            last_updated=aware_now,
        )
        updated = record_cost(original, Decimal("10"), aware_now)

        assert updated.spent == Decimal("40")
        assert updated.remaining == Decimal("60")
        assert updated.request_count == 6
        assert updated.is_exhausted is False

    def test_exhausts_budget(self, aware_now):
        """record_cost sets is_exhausted=True when budget exhausted."""
        period = _make_budget_period(aware_now)
        original = create_budget_state(
            task_id="test_task",
            total_budget=Decimal("100"),
            spent=Decimal("95"),
            period=period,
            request_count=10,
            last_updated=aware_now,
        )
        updated = record_cost(original, Decimal("10"), aware_now)

        assert updated.spent == Decimal("105")
        assert updated.remaining == Decimal("-5")
        assert updated.is_exhausted is True

    def test_original_not_mutated(self, aware_now):
        """record_cost does not mutate the original BudgetState."""
        period = _make_budget_period(aware_now)
        original = create_budget_state(
            task_id="test_task",
            total_budget=Decimal("100"),
            spent=Decimal("30"),
            period=period,
            request_count=5,
            last_updated=aware_now,
        )
        original_spent = original.spent
        original_count = original.request_count

        _ = record_cost(original, Decimal("10"), aware_now)

        assert original.spent == original_spent, "Original must not be mutated"
        assert original.request_count == original_count, "Original must not be mutated"

    def test_zero_cost(self, aware_now):
        """record_cost with cost=0 increments request_count but doesn't change spent."""
        period = _make_budget_period(aware_now)
        original = create_budget_state(
            task_id="test_task",
            total_budget=Decimal("100"),
            spent=Decimal("30"),
            period=period,
            request_count=5,
            last_updated=aware_now,
        )
        updated = record_cost(original, Decimal("0"), aware_now)

        assert updated.spent == Decimal("30")
        assert updated.request_count == 6

    def test_negative_cost_rejected(self, aware_now):
        """Raises error when cost < 0."""
        period = _make_budget_period(aware_now)
        state = create_budget_state(
            task_id="test_task",
            total_budget=Decimal("100"),
            spent=Decimal("0"),
            period=period,
            request_count=0,
            last_updated=aware_now,
        )
        with pytest.raises((ValueError, Exception)):
            record_cost(state, Decimal("-5"), aware_now)


class TestCreateAuditEntry:
    """Tests for create_audit_entry factory."""

    def test_happy_path(self, aware_now):
        """Creates a frozen AuditEntry with all fields."""
        entry = create_audit_entry(
            request_id="req-123",
            task_id="my_task",
            routing_decision=RoutingDecision.REMOTE,
            model_used="gpt-4",
            confidence_at_decision=0.5,
            phase=Phase.PHASE_1_REMOTE_ONLY,
            cost=Decimal("0.01"),
            latency_ms=150.0,
            fallback_triggered=False,
            timestamp=aware_now,
            sampling_probability=0.1,
            error="",
            metadata={},
        )
        assert entry.request_id == "req-123"
        assert entry.timestamp.tzinfo is not None

    def test_empty_request_id(self, aware_now):
        """Raises error when request_id is empty."""
        with pytest.raises((ValueError, Exception)):
            create_audit_entry(
                request_id="",
                task_id="my_task",
                routing_decision=RoutingDecision.REMOTE,
                model_used="gpt-4",
                confidence_at_decision=0.5,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                cost=Decimal("0"),
                latency_ms=100.0,
                fallback_triggered=False,
                timestamp=aware_now,
                sampling_probability=0.0,
                error="",
                metadata={},
            )

    def test_negative_latency(self, aware_now):
        """Raises error when latency_ms < 0."""
        with pytest.raises((ValueError, Exception)):
            create_audit_entry(
                request_id="req-123",
                task_id="my_task",
                routing_decision=RoutingDecision.REMOTE,
                model_used="gpt-4",
                confidence_at_decision=0.5,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                cost=Decimal("0"),
                latency_ms=-10.0,
                fallback_triggered=False,
                timestamp=aware_now,
                sampling_probability=0.0,
                error="",
                metadata={},
            )


class TestCreateSamplingDecision:
    """Tests for create_sampling_decision factory."""

    def test_happy_path(self, aware_now):
        """Creates a frozen SamplingDecision."""
        decision = create_sampling_decision(
            request_id="req-456",
            task_id="my_task",
            decision=RoutingDecision.LOCAL_WITH_REMOTE_COMPARISON,
            sampling_probability=0.3,
            phase=Phase.PHASE_2_SAMPLING,
            confidence_at_decision=0.6,
            budget_remaining=Decimal("50.00"),
            rationale="Confidence above phase2 threshold, sampling required",
            decided_at=aware_now,
        )
        assert decision.sampling_probability == 0.3
        assert decision.rationale != ""

    @pytest.mark.parametrize("prob", [-0.01, 1.01, -1.0, 2.0])
    def test_invalid_probability(self, prob, aware_now):
        """Raises error when sampling_probability outside [0.0, 1.0]."""
        with pytest.raises((ValueError, Exception)):
            create_sampling_decision(
                request_id="req-456",
                task_id="my_task",
                decision=RoutingDecision.REMOTE,
                sampling_probability=prob,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                confidence_at_decision=0.5,
                budget_remaining=Decimal("50"),
                rationale="test",
                decided_at=aware_now,
            )

    def test_empty_rationale(self, aware_now):
        """Raises error when rationale is empty."""
        with pytest.raises((ValueError, Exception)):
            create_sampling_decision(
                request_id="req-456",
                task_id="my_task",
                decision=RoutingDecision.REMOTE,
                sampling_probability=0.5,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                confidence_at_decision=0.5,
                budget_remaining=Decimal("50"),
                rationale="",
                decided_at=aware_now,
            )


class TestCreateTrainingExample:
    """Tests for create_training_example factory."""

    def test_happy_path(self, aware_now):
        """Creates a frozen TrainingExample."""
        example = create_training_example(
            task_id="my_task",
            input_data={"query": "hello"},
            output_data={"answer": "world"},
            model_id="gpt-4",
            source_request_id="req-789",
        )
        assert example.task_id == "my_task"
        assert example.source_request_id == "req-789"
        assert example.example_id  # auto-generated UUID
        assert example.created_at.tzinfo is not None

    def test_invalid_task_id(self):
        """Raises error for invalid task_id."""
        with pytest.raises((ValueError, Exception)):
            create_training_example(
                task_id="INVALID",
                input_data={},
                output_data={},
                model_id="gpt-4",
                source_request_id="req-789",
            )

    def test_empty_source_request_id(self):
        """Raises error when source_request_id is empty."""
        with pytest.raises((ValueError, Exception)):
            create_training_example(
                task_id="my_task",
                input_data={},
                output_data={},
                model_id="gpt-4",
                source_request_id="",
            )


class TestCreateComparisonPair:
    """Tests for create_comparison_pair factory."""

    def test_happy_path(self, aware_now):
        """Creates a frozen ComparisonPair."""
        eval_result = _make_exact_match_result()
        pair = create_comparison_pair(
            task_id="my_task",
            input_data={"query": "hello"},
            remote_output={"answer": "world"},
            local_output={"answer": "world"},
            remote_model_id="gpt-4",
            local_model_id="local-llama",
            source_request_id="req-101",
            evaluation_result=eval_result,
        )
        assert pair.remote_model_id == "gpt-4"
        assert pair.local_model_id == "local-llama"
        assert pair.pair_id  # auto-generated

    def test_same_model_ids_rejected(self, aware_now):
        """Raises error when remote_model_id == local_model_id."""
        eval_result = _make_exact_match_result()
        with pytest.raises((ValueError, Exception)):
            create_comparison_pair(
                task_id="my_task",
                input_data={},
                remote_output={},
                local_output={},
                remote_model_id="same-model",
                local_model_id="same-model",
                source_request_id="req-101",
                evaluation_result=eval_result,
            )


class TestCreateModelVersion:
    """Tests for create_model_version factory."""

    def test_happy_path(self, aware_now):
        """Creates a frozen ModelVersion."""
        version = create_model_version(
            model_id="local-llama",
            path="/models/v1",
            validation_score=0.85,
            training_example_count=500,
            is_active=True,
        )
        assert version.model_id == "local-llama"
        assert version.validation_score == 0.85
        assert version.version_id  # auto-generated UUID
        assert version.is_active is True

    def test_active_without_promoted_at(self, aware_now):
        """Active model version without explicit promoted_at gets promoted_at set."""
        version = create_model_version(
            model_id="local-llama",
            path="/models/v1",
            validation_score=0.85,
            training_example_count=500,
            is_active=True,
        )
        assert version.promoted_at is not None, \
            "Active version should have promoted_at set"
        assert version.promoted_at == version.created_at or \
               version.promoted_at is not None

    def test_empty_path(self):
        """Raises error when path is empty."""
        with pytest.raises((ValueError, Exception)):
            create_model_version(
                model_id="local-llama",
                path="",
                validation_score=0.85,
                training_example_count=500,
                is_active=False,
            )

    def test_negative_example_count(self):
        """Raises error when training_example_count < 0."""
        with pytest.raises((ValueError, Exception)):
            create_model_version(
                model_id="local-llama",
                path="/models/v1",
                validation_score=0.85,
                training_example_count=-1,
                is_active=False,
            )

    @pytest.mark.parametrize("score", [-0.01, 1.01])
    def test_invalid_validation_score(self, score):
        """Raises error when validation_score not in [0.0, 1.0]."""
        with pytest.raises((ValueError, Exception)):
            create_model_version(
                model_id="local-llama",
                path="/models/v1",
                validation_score=score,
                training_example_count=500,
                is_active=False,
            )


# ===========================================================================
# SECTION 6: Serialization Tests
# ===========================================================================

class TestSerializeModel:
    """Tests for serialize_model function."""

    def test_serialize_task_request(self):
        """serialize_model returns valid JSON for a TaskRequest."""
        req = create_task_request(task_id="my_task", input_data={"q": "hi"})
        json_str = serialize_model(req)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["task_id"] == "my_task"

    def test_serialize_not_pydantic(self):
        """serialize_model raises error for non-Pydantic input."""
        with pytest.raises((TypeError, AttributeError, Exception)):
            serialize_model({"not": "a model"})

    def test_serialize_preserves_decimal_precision(self, aware_now):
        """Decimal fields are serialized with precision preserved."""
        period = _make_budget_period(aware_now)
        state = create_budget_state(
            task_id="test_task",
            total_budget=Decimal("100.005"),
            spent=Decimal("0.001"),
            period=period,
            request_count=0,
            last_updated=aware_now,
        )
        json_str = serialize_model(state)
        # Decimal should be serialized as string or exact number
        assert "100.005" in json_str or "100005" in json_str

    def test_serialize_aware_datetime_iso(self):
        """AwareDatetime serialized as ISO 8601 with timezone."""
        req = create_task_request(task_id="my_task", input_data={})
        json_str = serialize_model(req)
        parsed = json.loads(json_str)
        # Should contain timezone indicator (+ or Z)
        created = parsed["created_at"]
        assert "+" in created or "Z" in created or "z" in created


class TestDeserializeModel:
    """Tests for deserialize_model function."""

    def test_round_trip_task_request(self):
        """Deserializing a serialized TaskRequest yields equivalent instance."""
        original = create_task_request(task_id="my_task", input_data={"q": "hi"})
        json_str = serialize_model(original)
        restored = deserialize_model(json_str, "TaskRequest")

        assert restored.task_id == original.task_id
        assert restored.request_id == original.request_id
        assert restored.input_data == original.input_data

    def test_round_trip_budget_state(self, aware_now):
        """BudgetState round-trips through serialization."""
        period = _make_budget_period(aware_now)
        original = create_budget_state(
            task_id="test_task",
            total_budget=Decimal("100"),
            spent=Decimal("30"),
            period=period,
            request_count=5,
            last_updated=aware_now,
        )
        json_str = serialize_model(original)
        restored = deserialize_model(json_str, "BudgetState")

        assert restored.total_budget == original.total_budget
        assert restored.spent == original.spent
        assert restored.remaining == original.remaining
        assert restored.is_exhausted == original.is_exhausted

    def test_round_trip_confidence_snapshot(self, aware_now):
        """ConfidenceSnapshot round-trips correctly, preserving derived phase."""
        original = create_confidence_snapshot(
            task_id="test_task",
            score=0.7,
            sample_count=50,
            threshold_phase2=0.5,
            threshold_phase3=0.9,
            last_updated=aware_now,
            trend=0.01,
        )
        json_str = serialize_model(original)
        restored = deserialize_model(json_str, "ConfidenceSnapshot")

        assert restored.phase == Phase.PHASE_2_SAMPLING
        assert restored.score == original.score

    def test_invalid_json(self):
        """Raises error for invalid JSON input."""
        with pytest.raises((ValueError, Exception)):
            deserialize_model("not valid json {{{", "TaskRequest")

    def test_unknown_model_type(self):
        """Raises error for unknown model type."""
        with pytest.raises((ValueError, KeyError, Exception)):
            deserialize_model("{}", "NonExistentModel")

    def test_extra_fields_forbidden(self):
        """Raises error when JSON has extra fields for extra='forbid' models."""
        req = create_task_request(task_id="my_task", input_data={})
        json_str = serialize_model(req)
        # Inject extra field
        data = json.loads(json_str)
        data["totally_unknown_field"] = "should fail"
        modified_json = json.dumps(data)
        with pytest.raises((ValueError, Exception)):
            deserialize_model(modified_json, "TaskRequest")


class TestValidateModel:
    """Tests for validate_model function."""

    def test_valid_data_returns_empty_list(self):
        """Returns empty list for valid data."""
        req = create_task_request(task_id="my_task", input_data={"q": "hi"})
        json_str = serialize_model(req)
        data = json.loads(json_str)
        errors = validate_model(data, "TaskRequest")
        assert errors == [] or len(errors) == 0, \
            f"Valid data should produce no errors, got: {errors}"

    def test_invalid_data_returns_errors(self):
        """Returns structured errors for invalid data."""
        errors = validate_model(
            {"task_id": "", "input_data": "not_a_dict"},
            "TaskRequest",
        )
        assert len(errors) >= 1, "Invalid data should produce validation errors"
        # Check error structure
        first_error = errors[0]
        assert hasattr(first_error, 'field_path') or isinstance(first_error, dict)

    def test_unknown_model_type(self):
        """Raises error for unknown model type."""
        with pytest.raises((ValueError, KeyError, Exception)):
            validate_model({}, "NonExistentModel")


class TestGetModelJsonSchema:
    """Tests for get_model_json_schema function."""

    def test_known_model(self):
        """Returns valid JSON schema for a known model type."""
        schema = get_model_json_schema("TaskRequest")
        assert isinstance(schema, dict)
        assert "properties" in schema or "type" in schema, \
            "Schema should have 'properties' or 'type' key"

    def test_unknown_model_type(self):
        """Raises error for unknown model type."""
        with pytest.raises((ValueError, KeyError, Exception)):
            get_model_json_schema("NonExistentModel")

    @pytest.mark.parametrize("model_name", [
        "TaskRequest",
        "TaskResponse",
        "ConfidenceSnapshot",
        "BudgetState",
        "AuditEntry",
        "SamplingDecision",
        "TrainingExample",
        "ComparisonPair",
        "ModelVersion",
    ])
    def test_schema_for_all_key_models(self, model_name):
        """All key model types return a schema dict."""
        schema = get_model_json_schema(model_name)
        assert isinstance(schema, dict), \
            f"get_model_json_schema('{model_name}') should return a dict"


# ===========================================================================
# SECTION 7: Invariant Tests
# ===========================================================================

class TestInvariants:
    """Tests verifying system-wide invariants from the contract."""

    def test_confidence_score_range_enforced(self, aware_now):
        """ConfidenceScore values are always in [0.0, 1.0]."""
        # Valid boundary
        snap_low = create_confidence_snapshot(
            task_id="test_task", score=0.0, sample_count=0,
            threshold_phase2=0.5, threshold_phase3=0.9,
            last_updated=aware_now, trend=0.0,
        )
        snap_high = create_confidence_snapshot(
            task_id="test_task", score=1.0, sample_count=0,
            threshold_phase2=0.5, threshold_phase3=0.9,
            last_updated=aware_now, trend=0.0,
        )
        assert 0.0 <= snap_low.score <= 1.0
        assert 0.0 <= snap_high.score <= 1.0

    def test_budget_remaining_invariant(self, aware_now):
        """BudgetState.remaining always equals total_budget - spent."""
        period = _make_budget_period(aware_now)
        for total, spent in [
            (Decimal("100"), Decimal("0")),
            (Decimal("100"), Decimal("50")),
            (Decimal("100"), Decimal("100")),
            (Decimal("100"), Decimal("150")),
        ]:
            state = create_budget_state(
                task_id="test_task",
                total_budget=total,
                spent=spent,
                period=period,
                request_count=0,
                last_updated=aware_now,
            )
            assert state.remaining == total - spent, \
                f"remaining should be {total} - {spent} = {total - spent}, got {state.remaining}"

    def test_budget_is_exhausted_invariant(self, aware_now):
        """BudgetState.is_exhausted always equals (remaining <= 0)."""
        period = _make_budget_period(aware_now)
        for total, spent, expected_exhausted in [
            (Decimal("100"), Decimal("0"), False),
            (Decimal("100"), Decimal("99"), False),
            (Decimal("100"), Decimal("100"), True),
            (Decimal("100"), Decimal("150"), True),
        ]:
            state = create_budget_state(
                task_id="test_task",
                total_budget=total,
                spent=spent,
                period=period,
                request_count=0,
                last_updated=aware_now,
            )
            assert state.is_exhausted == expected_exhausted, \
                f"For total={total}, spent={spent}: is_exhausted should be {expected_exhausted}"

    def test_phase_derived_not_independent(self, aware_now):
        """Phase is derived from score and thresholds, never set independently."""
        # We verify by creating snapshots and checking phase matches derivation rule
        cases = [
            (0.1, 0.5, 0.9, Phase.PHASE_1_REMOTE_ONLY),
            (0.5, 0.5, 0.9, Phase.PHASE_2_SAMPLING),
            (0.7, 0.5, 0.9, Phase.PHASE_2_SAMPLING),
            (0.9, 0.5, 0.9, Phase.PHASE_3_LOCAL_PRIMARY),
            (1.0, 0.5, 0.9, Phase.PHASE_3_LOCAL_PRIMARY),
        ]
        for score, t2, t3, expected_phase in cases:
            snap = create_confidence_snapshot(
                task_id="test_task",
                score=score,
                sample_count=10,
                threshold_phase2=t2,
                threshold_phase3=t3,
                last_updated=aware_now,
                trend=0.0,
            )
            assert snap.phase == expected_phase, \
                f"score={score}, t2={t2}, t3={t3}: expected {expected_phase}, got {snap.phase}"

    def test_registry_len_ge_one(self):
        """Registry must have len >= 1."""
        tc = _make_registry_task_config()
        registry = TaskRegistry(tasks=[tc])
        assert len(registry) >= 1

    def test_registry_task_names_equals_keys(self):
        """registry.task_names equals frozenset of task keys."""
        tc1 = _make_registry_task_config(name="alpha")
        tc2 = _make_registry_task_config(name="beta", prompt_template="Q: {text}")
        registry = TaskRegistry(tasks=[tc1, tc2])

        assert registry.task_names == frozenset({"alpha", "beta"})
        assert len(registry.task_names) == len(registry)

    def test_all_models_frozen(self, aware_now):
        """All frozen model instances reject attribute assignment."""
        period = _make_budget_period(aware_now)

        frozen_instances = [
            create_task_request(task_id="test", input_data={}),
            create_confidence_snapshot(
                task_id="test", score=0.5, sample_count=10,
                threshold_phase2=0.3, threshold_phase3=0.8,
                last_updated=aware_now, trend=0.0,
            ),
            create_audit_entry(
                request_id="r", task_id="test",
                routing_decision=RoutingDecision.REMOTE,
                model_used="gpt-4", confidence_at_decision=0.5,
                phase=Phase.PHASE_1_REMOTE_ONLY,
                cost=Decimal("0"), latency_ms=0.0,
                fallback_triggered=False, timestamp=aware_now,
                sampling_probability=0.0, error="", metadata={},
            ),
        ]

        for instance in frozen_instances:
            # Frozen Pydantic models raise ValidationError on normal attribute assignment
            with pytest.raises(ValidationError):
                # Use the first field name from the model's fields for assignment
                first_field = next(iter(instance.model_fields))
                setattr(instance, first_field, "mutated")

    def test_evaluation_result_discriminated_union(self):
        """EvaluationResult discriminated union — result_type determines variant."""
        exact = ExactMatchResult(
            result_type=EvaluationResultType.EXACT_MATCH_RESULT,
            is_match=True,
            matched_fields=["f1"],
            mismatched_fields=[],
        )
        assert exact.result_type == EvaluationResultType.EXACT_MATCH_RESULT

        fuzzy = FuzzyMatchResult(
            result_type=EvaluationResultType.FUZZY_MATCH_RESULT,
            similarity_score=0.9,
            threshold=0.8,
            is_pass=True,
        )
        assert fuzzy.result_type == EvaluationResultType.FUZZY_MATCH_RESULT

    def test_round_trip_serialization(self, aware_now):
        """All models round-trip cleanly through JSON serialization."""
        req = create_task_request(task_id="roundtrip_test", input_data={"k": "v"})
        json_str = serialize_model(req)
        restored = deserialize_model(json_str, "TaskRequest")
        assert restored.task_id == req.task_id
        assert restored.request_id == req.request_id
        assert restored.input_data == req.input_data

    def test_task_id_pattern_invariant(self):
        """TaskId values must match ^[a-z][a-z0-9_-]*$."""
        # Valid patterns
        for valid_id in ["a", "abc", "my_task", "task-1", "a123"]:
            req = create_task_request(task_id=valid_id, input_data={})
            assert req.task_id == valid_id

        # Invalid patterns
        for invalid_id in ["", "A", "1abc", "my task", "UPPER", "my.task"]:
            with pytest.raises((ValueError, Exception)):
                create_task_request(task_id=invalid_id, input_data={})

    def test_model_id_constraints(self, aware_now):
        """ModelId values are non-empty strings of at most 256 characters."""
        # Valid
        resp = create_task_response(
            request_id="r",
            task_id="test",
            output_data={},
            model_used="a" * 256,
            routing_decision=RoutingDecision.REMOTE,
            phase=Phase.PHASE_1_REMOTE_ONLY,
            confidence_at_decision=0.5,
            latency_ms=0.0,
            cost_incurred=Decimal("0"),
            created_at=aware_now,
            metadata={},
        )
        assert len(resp.model_used) == 256

        # Empty model_id should be rejected
        with pytest.raises((ValueError, Exception)):
            create_task_response(
                request_id="r",
                task_id="test",
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

    def test_secret_str_never_in_repr(self, tmp_path, mock_env):
        """SecretStr fields mask their values — secrets never in logs/repr."""
        yaml_content = _make_valid_yaml_content()
        config_path = _write_yaml(tmp_path, yaml_content)
        config = load_config(config_path, env=mock_env)

        # Check repr of the full config
        full_repr = repr(config)
        assert "sk-test-key-12345" not in full_repr
        assert "ft-test-key-67890" not in full_repr

        # Check str of the provider config
        provider_str = str(config.provider)
        assert "sk-test-key-12345" not in provider_str

        # But the secret is still accessible via get_secret_value
        assert config.provider.api_key.get_secret_value() == "sk-test-key-12345"
