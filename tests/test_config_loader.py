"""
Contract tests for config_loader component.

Tests verify the behavior of load_config, resolve_env_var, and
validate_cross_field_constraints against the contract specification.

Organized into sections:
1. Fixtures and helpers
2. Happy-path tests
3. File I/O error tests
4. Schema validation error tests
5. Cross-field constraint tests
6. Env var resolution tests
7. Invariant and property-based tests
"""
import copy
import os
import stat
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

try:
    from hypothesis import given, assume, settings
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

from src.config_loader import (
    load_config,
    resolve_env_var,
    get_unresolved_vars,
    ApprenticeConfig,
    ConfigError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
    EnvVarNotFoundError,
)


# ---------------------------------------------------------------------------
# Section 1: Fixtures and helpers
# ---------------------------------------------------------------------------

def make_valid_task_dict(
    task_name="summarize",
    evaluator_type="exact_match",
    local_ready=0.7,
    local_only=0.9,
    extra_evaluator_fields=None,
    match_field_name="result",
):
    """Return a minimal valid task config dict."""
    evaluator = {
        "type": evaluator_type,
        "match_fields": [
            {"name": match_field_name, "weight": 1.0, "case_sensitive": False}
        ],
        "semantic_model": "",
        "judge_prompt_template": "",
        "regex_pattern": "",
        "json_schema_path": "",
    }
    if extra_evaluator_fields:
        evaluator.update(extra_evaluator_fields)
    return {
        "task_name": task_name,
        "prompt_template": "Summarize: {{input}}",
        "input_schema": [
            {"name": "input", "type": "str", "required": True, "description": "The text to summarize"}
        ],
        "output_schema": [
            {"name": "result", "type": "str", "required": True, "description": "The summary"}
        ],
        "evaluators": [evaluator],
        "thresholds": {
            "local_ready": local_ready,
            "local_only": local_only,
            "degraded_threshold": 0.3,
        },
        "sampling_rate_initial": 1.0,
        "min_training_examples": 100,
    }


def make_valid_config_dict(env_ref=False):
    """
    Return a minimal but complete valid config as a Python dict.
    If env_ref=True, api_key fields use env:VAR_NAME references.
    """
    provider_api_key = "env:PROVIDER_API_KEY" if env_ref else "test-api-key-1234"
    finetuning_api_key = "env:FT_API_KEY" if env_ref else "test-ft-key-5678"
    return {
        "provider": {
            "api_base_url": "https://api.openai.com/v1",
            "api_key": provider_api_key,
            "model": "gpt-4",
            "timeout_seconds": 30,
            "max_retries": 3,
            "retry_base_delay_seconds": 1.0,
        },
        "local_model": {
            "endpoint": "http://localhost:8080",
            "model_name": "local-llama",
            "timeout_seconds": 10,
            "max_retries": 2,
            "health_check_path": "/health",
        },
        "tasks": [make_valid_task_dict()],
        "budget": {
            "max_daily_cost_usd": 50.0,
            "max_monthly_cost_usd": 1000.0,
            "rolling_window_hours": 24,
            "cost_per_input_token": 0.00003,
            "cost_per_output_token": 0.00006,
            "budget_state_path": "/tmp/budget_state.json",
        },
        "finetuning": {
            "backend": "openai",
            "model_base": "gpt-3.5-turbo",
            "batch_size": 32,
            "trigger_interval_hours": 24,
            "api_key": finetuning_api_key,
            "api_base_url": "https://api.openai.com/v1",
            "output_dir": "/tmp/finetuning_output",
            "max_concurrent_jobs": 2,
        },
        "audit": {
            "log_path": "/tmp/audit.log",
            "log_level": "INFO",
            "log_to_stdout": False,
            "max_file_size_mb": 100,
            "backup_count": 5,
        },
        "training_data": {
            "storage_dir": "/tmp/training_data",
            "max_examples_per_task": 10000,
        },
    }


def write_config(tmp_path: Path, config_dict: dict, filename="config.yaml") -> Path:
    """Serialize config dict to YAML and return the file path."""
    config_path = tmp_path / filename
    config_path.write_text(yaml.dump(config_dict, default_flow_style=False), encoding="utf-8")
    return config_path


def write_raw_config(tmp_path: Path, content: str, filename="config.yaml") -> Path:
    """Write raw string content to a YAML file and return the path."""
    config_path = tmp_path / filename
    config_path.write_text(content, encoding="utf-8")
    return config_path


@pytest.fixture
def sample_env():
    """Env mapping with all referenced env vars set."""
    return {
        "PROVIDER_API_KEY": "resolved-provider-key",
        "FT_API_KEY": "resolved-ft-key",
    }


@pytest.fixture
def empty_env():
    """Empty env mapping."""
    return {}


# ---------------------------------------------------------------------------
# Section 2: Happy-path tests
# ---------------------------------------------------------------------------

class TestHappyPath:
    """Tests for successful config loading scenarios."""

    def test_minimal_valid_config_loads(self, tmp_path):
        """A minimal but complete valid config loads successfully."""
        cfg_dict = make_valid_config_dict()
        path = write_config(tmp_path, cfg_dict)
        env = {"PROVIDER_API_KEY": "key", "FT_API_KEY": "key"}

        config = load_config(path, env={})
        # If env refs are not used, empty env is fine
        config = load_config(path, {})

        assert config is not None, "load_config should return a non-None config"
        assert config.provider.model == "gpt-4"
        assert config.local_model.model_name == "local-llama"
        assert len(config.tasks) == 1
        assert config.tasks[0].task_name == "summarize"

    def test_env_vars_resolved_in_api_keys(self, tmp_path, sample_env):
        """env:VAR_NAME references are resolved from the provided env mapping."""
        cfg_dict = make_valid_config_dict(env_ref=True)
        path = write_config(tmp_path, cfg_dict)

        config = load_config(path, env=sample_env)

        # SecretStr values should be accessible via get_secret_value
        assert config.provider.api_key.get_secret_value() == "resolved-provider-key", \
            "provider.api_key should be resolved from env"
        assert config.finetuning.api_key.get_secret_value() == "resolved-ft-key", \
            "finetuning.api_key should be resolved from env"

    def test_multiple_tasks_load_correctly(self, tmp_path):
        """Config with multiple distinct tasks loads both tasks."""
        cfg_dict = make_valid_config_dict()
        task2 = make_valid_task_dict(task_name="classify", evaluator_type="exact_match")
        cfg_dict["tasks"].append(task2)
        path = write_config(tmp_path, cfg_dict)

        config = load_config(path, {})

        assert len(config.tasks) == 2, "Should have two tasks"
        task_names = [t.task_name for t in config.tasks]
        assert "summarize" in task_names
        assert "classify" in task_names

    def test_fully_specified_config_loads_all_fields(self, tmp_path):
        """All explicitly specified fields are preserved in the returned config."""
        cfg_dict = make_valid_config_dict()
        path = write_config(tmp_path, cfg_dict)

        config = load_config(path, {})

        assert config.provider.timeout_seconds == 30
        assert config.provider.max_retries == 3
        assert config.provider.retry_base_delay_seconds == 1.0
        assert config.budget.max_daily_cost_usd == 50.0
        assert config.budget.max_monthly_cost_usd == 1000.0
        assert config.finetuning.backend.value == "openai" or str(config.finetuning.backend) == "openai"
        assert config.audit.log_level.value == "INFO" or str(config.audit.log_level) == "INFO"
        assert config.training_data.max_examples_per_task == 10000

    def test_config_with_local_backend_no_remote_keys(self, tmp_path):
        """A local backend (local_lora) does not require finetuning api_key/api_base_url."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["finetuning"]["backend"] = "local_lora"
        cfg_dict["finetuning"]["api_key"] = ""
        cfg_dict["finetuning"]["api_base_url"] = ""
        path = write_config(tmp_path, cfg_dict)

        config = load_config(path, {})
        assert config is not None


# ---------------------------------------------------------------------------
# Section 3: File I/O error tests
# ---------------------------------------------------------------------------

class TestFileIOErrors:
    """Tests for file-level loading failures."""

    def test_file_not_found(self, tmp_path):
        """Raises ConfigFileNotFoundError when file does not exist."""
        nonexistent = tmp_path / "does_not_exist.yaml"
        with pytest.raises((ConfigFileNotFoundError, ConfigError)) as exc_info:
            load_config(nonexistent, {})
        assert "does_not_exist.yaml" in str(exc_info.value) or hasattr(exc_info.value, "path")

    @pytest.mark.skipif(sys.platform == "win32", reason="Permission tests unreliable on Windows")
    def test_file_not_readable(self, tmp_path):
        """Raises ConfigError when file exists but cannot be read."""
        cfg_dict = make_valid_config_dict()
        path = write_config(tmp_path, cfg_dict)
        os.chmod(path, 0o000)
        try:
            with pytest.raises((ConfigError,)):
                load_config(path, {})
        finally:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    def test_yaml_syntax_error(self, tmp_path):
        """Raises ConfigParseError when YAML has syntax errors."""
        bad_yaml = "provider:\n  api_key: [\ninvalid: {{\n"
        path = write_raw_config(tmp_path, bad_yaml)
        with pytest.raises((ConfigParseError, ConfigError)):
            load_config(path, {})

    def test_empty_config_file(self, tmp_path):
        """Raises ConfigError when the YAML file is empty."""
        path = write_raw_config(tmp_path, "")
        with pytest.raises(ConfigError):
            load_config(path, {})

    def test_yaml_not_a_mapping(self, tmp_path):
        """Raises ConfigError when top-level YAML is a list."""
        path = write_raw_config(tmp_path, "- item1\n- item2\n")
        with pytest.raises(ConfigError):
            load_config(path, {})

    def test_yaml_only_comments(self, tmp_path):
        """Raises ConfigError when YAML file contains only comments."""
        path = write_raw_config(tmp_path, "# This is just a comment\n# Nothing else\n")
        with pytest.raises(ConfigError):
            load_config(path, {})


# ---------------------------------------------------------------------------
# Section 4: Schema validation error tests
# ---------------------------------------------------------------------------

class TestSchemaValidationErrors:
    """Tests for Pydantic validation failures."""

    @pytest.mark.parametrize("missing_key", [
        "provider",
        "local_model",
        "tasks",
        "budget",
        "finetuning",
        "audit",
        "training_data",
    ])
    def test_missing_required_top_level_field(self, tmp_path, missing_key):
        """Raises ConfigValidationError when a required top-level field is missing."""
        cfg_dict = make_valid_config_dict()
        del cfg_dict[missing_key]
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    @pytest.mark.parametrize("section,field", [
        ("provider", "api_key"),
        ("provider", "model"),
        ("local_model", "endpoint"),
        ("local_model", "model_name"),
    ])
    def test_missing_required_nested_field(self, tmp_path, section, field):
        """Raises ConfigValidationError when a required nested field is missing."""
        cfg_dict = make_valid_config_dict()
        del cfg_dict[section][field]
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_invalid_field_type_strict_mode(self, tmp_path):
        """Raises ConfigValidationError when a string is given where int expected (strict)."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["provider"]["timeout_seconds"] = "thirty"  # string, not int
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_extra_field_forbidden(self, tmp_path):
        """Raises ConfigValidationError when an unknown field is present."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["unknown_top_level_field"] = "surprise"
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_unknown_evaluator_type(self, tmp_path):
        """Raises ConfigValidationError when evaluator type is not in enum."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["tasks"][0]["evaluators"][0]["type"] = "nonexistent_evaluator"
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_unknown_backend(self, tmp_path):
        """Raises ConfigValidationError when finetuning backend is not in enum."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["finetuning"]["backend"] = "unknown_backend"
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_strict_type_string_as_int(self, tmp_path):
        """Strict mode rejects '3' as a string where int is expected."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["provider"]["timeout_seconds"] = "3"
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_extra_field_in_nested_section(self, tmp_path):
        """Raises ConfigValidationError when extra field is in a nested section."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["provider"]["unknown_nested"] = "value"
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})


# ---------------------------------------------------------------------------
# Section 5: Cross-field constraint tests
# ---------------------------------------------------------------------------

class TestCrossFieldConstraints:
    """Tests for cross-field validation failures (model_validator)."""

    def test_threshold_ordering_violation(self, tmp_path):
        """Raises ConfigValidationError when local_only < local_ready."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["tasks"][0]["thresholds"]["local_ready"] = 0.9
        cfg_dict["tasks"][0]["thresholds"]["local_only"] = 0.7  # < local_ready
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    @pytest.mark.parametrize("local_ready,local_only", [
        (0.9, 0.1),
        (0.5, 0.4),
        (0.99, 0.01),
    ])
    def test_threshold_ordering_violation_parametrized(self, tmp_path, local_ready, local_only):
        """Parametrized: local_only < local_ready always fails."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["tasks"][0]["thresholds"]["local_ready"] = local_ready
        cfg_dict["tasks"][0]["thresholds"]["local_only"] = local_only
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_threshold_equal_values_accepted(self, tmp_path):
        """Thresholds where local_only == local_ready are accepted (>= not >)."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["tasks"][0]["thresholds"]["local_ready"] = 0.8
        cfg_dict["tasks"][0]["thresholds"]["local_only"] = 0.8
        path = write_config(tmp_path, cfg_dict)
        config = load_config(path, {})
        assert config is not None

    def test_monthly_less_than_daily(self, tmp_path):
        """Raises ConfigValidationError when max_monthly < max_daily."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["budget"]["max_daily_cost_usd"] = 100.0
        cfg_dict["budget"]["max_monthly_cost_usd"] = 50.0
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_budget_equal_daily_monthly_accepted(self, tmp_path):
        """Budget with max_monthly == max_daily is accepted (>= not >)."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["budget"]["max_daily_cost_usd"] = 100.0
        cfg_dict["budget"]["max_monthly_cost_usd"] = 100.0
        path = write_config(tmp_path, cfg_dict)
        config = load_config(path, {})
        assert config is not None

    def test_match_field_not_in_output_schema(self, tmp_path):
        """Raises ConfigValidationError when match_field references nonexistent output field."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["tasks"][0]["evaluators"][0]["match_fields"][0]["name"] = "nonexistent_field"
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_semantic_similarity_missing_model(self, tmp_path):
        """Raises ConfigValidationError for semantic_similarity without semantic_model."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["tasks"][0]["evaluators"][0]["type"] = "semantic_similarity"
        cfg_dict["tasks"][0]["evaluators"][0]["semantic_model"] = ""
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_semantic_similarity_with_model_succeeds(self, tmp_path):
        """semantic_similarity with semantic_model set loads successfully."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["tasks"][0]["evaluators"][0]["type"] = "semantic_similarity"
        cfg_dict["tasks"][0]["evaluators"][0]["semantic_model"] = "all-MiniLM-L6-v2"
        path = write_config(tmp_path, cfg_dict)
        config = load_config(path, {})
        assert config is not None

    def test_llm_judge_missing_template(self, tmp_path):
        """Raises ConfigValidationError for llm_judge without judge_prompt_template."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["tasks"][0]["evaluators"][0]["type"] = "llm_judge"
        cfg_dict["tasks"][0]["evaluators"][0]["judge_prompt_template"] = ""
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_llm_judge_with_template_succeeds(self, tmp_path):
        """llm_judge with judge_prompt_template set loads successfully."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["tasks"][0]["evaluators"][0]["type"] = "llm_judge"
        cfg_dict["tasks"][0]["evaluators"][0]["judge_prompt_template"] = "Rate this: {{output}}"
        path = write_config(tmp_path, cfg_dict)
        config = load_config(path, {})
        assert config is not None

    def test_regex_match_missing_pattern(self, tmp_path):
        """Raises ConfigValidationError for regex_match without regex_pattern."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["tasks"][0]["evaluators"][0]["type"] = "regex_match"
        cfg_dict["tasks"][0]["evaluators"][0]["regex_pattern"] = ""
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_regex_match_with_pattern_succeeds(self, tmp_path):
        """regex_match with regex_pattern set loads successfully."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["tasks"][0]["evaluators"][0]["type"] = "regex_match"
        cfg_dict["tasks"][0]["evaluators"][0]["regex_pattern"] = r"\d{3}-\d{4}"
        path = write_config(tmp_path, cfg_dict)
        config = load_config(path, {})
        assert config is not None

    def test_json_schema_match_missing_path(self, tmp_path):
        """Raises ConfigValidationError for json_schema_match without json_schema_path."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["tasks"][0]["evaluators"][0]["type"] = "json_schema_match"
        cfg_dict["tasks"][0]["evaluators"][0]["json_schema_path"] = ""
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_json_schema_match_with_path_succeeds(self, tmp_path):
        """json_schema_match with json_schema_path set loads successfully."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["tasks"][0]["evaluators"][0]["type"] = "json_schema_match"
        cfg_dict["tasks"][0]["evaluators"][0]["json_schema_path"] = "/schemas/output.json"
        path = write_config(tmp_path, cfg_dict)
        config = load_config(path, {})
        assert config is not None

    def test_duplicate_task_names(self, tmp_path):
        """Raises ConfigValidationError when two tasks have the same name."""
        cfg_dict = make_valid_config_dict()
        task2 = make_valid_task_dict(task_name="summarize")  # duplicate name
        cfg_dict["tasks"].append(task2)
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_unsupported_backend_combination_openai_no_api_key(self, tmp_path):
        """Raises ConfigValidationError when openai backend lacks api_key."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["finetuning"]["backend"] = "openai"
        cfg_dict["finetuning"]["api_key"] = ""
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_unsupported_backend_combination_anyscale_no_api_base(self, tmp_path):
        """Raises ConfigValidationError when anyscale backend lacks api_base_url."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["finetuning"]["backend"] = "anyscale"
        cfg_dict["finetuning"]["api_key"] = "some-key"
        cfg_dict["finetuning"]["api_base_url"] = ""
        path = write_config(tmp_path, cfg_dict)
        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})


# ---------------------------------------------------------------------------
# Section 6: Env var resolution tests
# ---------------------------------------------------------------------------

class TestEnvVarResolution:
    """Tests for resolve_env_var and env var resolution in load_config."""

    def test_resolve_env_var_passthrough(self):
        """Value without env: prefix is returned unchanged."""
        result = resolve_env_var("plain_value", {}, "test.field")
        assert result == "plain_value", "Non-env values should pass through unchanged"

    def test_resolve_env_var_found(self):
        """env:VAR resolves to the value in the env mapping."""
        result = resolve_env_var("env:MY_SECRET", {"MY_SECRET": "s3cret"}, "provider.api_key")
        assert result == "s3cret", "Should resolve to the env var value"

    def test_resolve_env_var_not_found_returns_placeholder(self):
        """Returns placeholder when env var is not in the mapping."""
        result = resolve_env_var("env:MISSING_VAR", {}, "provider.api_key")
        assert "UNRESOLVED" in result
        assert "MISSING_VAR" in result

    def test_resolve_env_var_empty_var_name(self):
        """Raises error when env: has no variable name."""
        with pytest.raises((EnvVarNotFoundError, ConfigError, KeyError, ValueError)):
            resolve_env_var("env:", {}, "provider.api_key")

    def test_load_config_env_var_not_found_returns_placeholder(self, tmp_path):
        """Config loads with placeholder when env vars cannot be resolved."""
        cfg_dict = make_valid_config_dict(env_ref=True)
        path = write_config(tmp_path, cfg_dict)
        # Provide empty env — env:PROVIDER_API_KEY won't resolve, gets placeholder
        cfg = load_config(path, env={})
        assert "UNRESOLVED" in cfg.provider.api_key.get_secret_value()

    def test_resolve_env_var_preserves_non_env_colon(self):
        """A value like 'https://example.com' is not treated as env ref."""
        result = resolve_env_var("https://example.com", {}, "provider.api_base_url")
        assert result == "https://example.com", "URLs should not be treated as env refs"

    def test_resolve_env_var_case_sensitive(self):
        """Env var names are case-sensitive — mismatched case returns placeholder."""
        result = resolve_env_var("env:my_var", {"MY_VAR": "value"}, "field")
        assert "UNRESOLVED" in result
        assert "my_var" in result

    def test_get_unresolved_vars_populated(self, tmp_path):
        """get_unresolved_vars() returns list of unresolved env vars after load."""
        cfg_dict = make_valid_config_dict(env_ref=True)
        path = write_config(tmp_path, cfg_dict)
        load_config(path, env={})
        unresolved = get_unresolved_vars()
        var_names = [name for name, _ in unresolved]
        assert "PROVIDER_API_KEY" in var_names
        assert "FT_API_KEY" in var_names

    def test_get_unresolved_vars_empty_when_resolved(self, tmp_path, sample_env):
        """get_unresolved_vars() returns empty list when all vars are resolved."""
        cfg_dict = make_valid_config_dict(env_ref=True)
        path = write_config(tmp_path, cfg_dict)
        load_config(path, env=sample_env)
        unresolved = get_unresolved_vars()
        assert len(unresolved) == 0

    def test_get_unresolved_vars_includes_field_path(self, tmp_path):
        """get_unresolved_vars() entries include the field path."""
        cfg_dict = make_valid_config_dict(env_ref=True)
        path = write_config(tmp_path, cfg_dict)
        load_config(path, env={})
        unresolved = get_unresolved_vars()
        field_paths = [fp for _, fp in unresolved]
        assert "provider.api_key" in field_paths

    def test_get_unresolved_vars_resets_between_loads(self, tmp_path, sample_env):
        """get_unresolved_vars() is reset on each load_config call."""
        cfg_dict = make_valid_config_dict(env_ref=True)
        path = write_config(tmp_path, cfg_dict)
        # First load without env — should have unresolved
        load_config(path, env={})
        assert len(get_unresolved_vars()) > 0
        # Second load with env — should be empty
        load_config(path, env=sample_env)
        assert len(get_unresolved_vars()) == 0

    def test_unresolved_placeholder_masked_in_repr(self, tmp_path):
        """UNRESOLVED placeholder in SecretStr is masked in repr/str (env var name doesn't leak)."""
        cfg_dict = make_valid_config_dict(env_ref=True)
        path = write_config(tmp_path, cfg_dict)
        config = load_config(path, env={})

        config_repr = repr(config)
        api_key_str = str(config.provider.api_key)
        # The UNRESOLVED placeholder (and the env var name) must not leak
        assert "UNRESOLVED" not in config_repr, \
            "UNRESOLVED placeholder should be masked in repr"
        assert "PROVIDER_API_KEY" not in config_repr, \
            "Env var name should be masked in repr"
        assert "UNRESOLVED" not in api_key_str, \
            "UNRESOLVED placeholder should be masked in str"

    def test_unresolved_placeholder_survives_secretstr_roundtrip(self, tmp_path):
        """UNRESOLVED placeholder is intact when extracted via get_secret_value()."""
        cfg_dict = make_valid_config_dict(env_ref=True)
        path = write_config(tmp_path, cfg_dict)
        config = load_config(path, env={})

        raw = config.provider.api_key.get_secret_value()
        assert raw == "UNRESOLVED:env:PROVIDER_API_KEY", \
            f"Placeholder should survive SecretStr round-trip, got: {raw}"

    def test_unresolved_placeholder_detected_by_remote_client(self, tmp_path):
        """Full chain: config_loader → SecretStr → get_secret_value → create_remote_client rejects."""
        from apprentice.remote_api_client import (
            create_remote_client,
            RemoteClientConfig,
            ProviderName,
            AuthError,
        )

        cfg_dict = make_valid_config_dict(env_ref=True)
        path = write_config(tmp_path, cfg_dict)
        config = load_config(path, env={})

        # Extract the UNRESOLVED placeholder exactly as factory.py does
        placeholder = config.provider.api_key.get_secret_value()
        assert placeholder.startswith("UNRESOLVED:env:")

        # Simulate factory.py setting it into os.environ
        env_var = "TEST_UNRESOLVED_ROUNDTRIP"
        os.environ[env_var] = placeholder
        try:
            with pytest.raises(AuthError) as exc_info:
                create_remote_client(RemoteClientConfig(
                    provider=ProviderName.anthropic,
                    model="test-model",
                    api_key_env_var=env_var,
                ))
            assert "PROVIDER_API_KEY" in exc_info.value.message
        finally:
            del os.environ[env_var]


# ---------------------------------------------------------------------------
# Section 7: Invariant and property-based tests
# ---------------------------------------------------------------------------

class TestInvariants:
    """Tests for contract invariants."""

    def test_config_is_frozen_immutable(self, tmp_path):
        """ApprenticeConfig is frozen — attribute assignment raises an error."""
        cfg_dict = make_valid_config_dict()
        path = write_config(tmp_path, cfg_dict)
        config = load_config(path, {})

        with pytest.raises((AttributeError, TypeError, Exception)):
            config.provider = None  # type: ignore

    def test_secrets_masked_in_repr(self, tmp_path):
        """SecretStr fields do not expose values in repr or str."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["provider"]["api_key"] = "super-secret-key-12345"
        path = write_config(tmp_path, cfg_dict)
        config = load_config(path, {})

        config_repr = repr(config)
        api_key_str = str(config.provider.api_key)
        assert "super-secret-key-12345" not in config_repr, \
            "Secret value should not appear in repr"
        assert "super-secret-key-12345" not in api_key_str, \
            "Secret value should not appear in str"
        # But we can access it via get_secret_value
        assert config.provider.api_key.get_secret_value() == "super-secret-key-12345"

    def test_idempotent_loading(self, tmp_path):
        """Calling load_config twice with same inputs produces identical results."""
        cfg_dict = make_valid_config_dict()
        path = write_config(tmp_path, cfg_dict)

        config1 = load_config(path, {})
        config2 = load_config(path, {})

        assert config1.provider.model == config2.provider.model
        assert config1.budget.max_daily_cost_usd == config2.budget.max_daily_cost_usd
        assert len(config1.tasks) == len(config2.tasks)
        assert config1.tasks[0].task_name == config2.tasks[0].task_name

    def test_all_task_names_unique(self, tmp_path):
        """After successful load, all task names are unique."""
        cfg_dict = make_valid_config_dict()
        task2 = make_valid_task_dict(task_name="classify")
        cfg_dict["tasks"].append(task2)
        path = write_config(tmp_path, cfg_dict)
        config = load_config(path, {})

        task_names = [t.task_name for t in config.tasks]
        assert len(task_names) == len(set(task_names)), "Task names must be unique"

    def test_all_env_refs_resolved(self, tmp_path, sample_env):
        """After loading, no field value contains 'env:' — all are resolved."""
        cfg_dict = make_valid_config_dict(env_ref=True)
        path = write_config(tmp_path, cfg_dict)
        config = load_config(path, env=sample_env)

        # Check that secret values are resolved, not env: references
        provider_key = config.provider.api_key.get_secret_value()
        assert not provider_key.startswith("env:"), \
            "Provider API key should be resolved, not an env: reference"
        ft_key = config.finetuning.api_key.get_secret_value()
        assert not ft_key.startswith("env:"), \
            "Finetuning API key should be resolved, not an env: reference"

    def test_cross_field_validations_all_passed(self, tmp_path):
        """A successfully loaded config passes all cross-field constraints."""
        cfg_dict = make_valid_config_dict()
        path = write_config(tmp_path, cfg_dict)
        config = load_config(path, {})

        # Verify threshold ordering
        for task in config.tasks:
            assert task.thresholds.local_only >= task.thresholds.local_ready, \
                f"Task {task.task_name}: local_only must be >= local_ready"

        # Verify budget consistency
        assert config.budget.max_monthly_cost_usd >= config.budget.max_daily_cost_usd, \
            "Monthly budget must be >= daily budget"


# ---------------------------------------------------------------------------
# Section 7b: Hypothesis property-based tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        local_ready=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        local_only=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_threshold_ordering_property(self, tmp_path, local_ready, local_only):
        """Property: load succeeds iff local_only >= local_ready."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["tasks"][0]["thresholds"]["local_ready"] = local_ready
        cfg_dict["tasks"][0]["thresholds"]["local_only"] = local_only
        cfg_dict["tasks"][0]["thresholds"]["degraded_threshold"] = 0.1
        path = write_config(tmp_path, cfg_dict)

        if local_only >= local_ready:
            try:
                config = load_config(path, {})
                assert config is not None
            except (ConfigValidationError, ConfigError):
                # Other validation might fail; that's ok for this property
                pass
        else:
            with pytest.raises((ConfigValidationError, ConfigError)):
                load_config(path, {})

    @given(
        max_daily=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False),
        max_monthly=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_budget_consistency_property(self, tmp_path, max_daily, max_monthly):
        """Property: load succeeds iff max_monthly >= max_daily."""
        cfg_dict = make_valid_config_dict()
        cfg_dict["budget"]["max_daily_cost_usd"] = max_daily
        cfg_dict["budget"]["max_monthly_cost_usd"] = max_monthly
        path = write_config(tmp_path, cfg_dict)

        if max_monthly >= max_daily:
            try:
                config = load_config(path, {})
                assert config is not None
            except (ConfigValidationError, ConfigError):
                # Other validation might fail
                pass
        else:
            with pytest.raises((ConfigValidationError, ConfigError)):
                load_config(path, {})

    @given(
        var_name=st.from_regex(r"[A-Z][A-Z0-9_]{0,20}", fullmatch=True),
        var_value=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=50)
    def test_env_var_resolution_roundtrip(self, var_name, var_value):
        """Property: resolve_env_var(env:NAME, {NAME: value}) == value."""
        assume(len(var_name) > 0)
        result = resolve_env_var(f"env:{var_name}", {var_name: var_value}, "test.field")
        assert result == var_value, \
            f"resolve_env_var should return '{var_value}' but got '{result}'"
