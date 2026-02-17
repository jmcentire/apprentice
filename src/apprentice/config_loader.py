from pydantic import model_validator
from pydantic import field_validator
from pydantic import ConfigDict
"""
Config loader for Apprentice.

Loads and validates YAML configuration files using Pydantic v2 models.
Resolves environment variable references (env:VAR_NAME pattern).
Returns frozen, immutable configuration objects with comprehensive validation.
"""
import os
from enum import Enum
from pathlib import Path
from typing import Any, List, Mapping, Optional

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)


# ============================================================================
# Enums
# ============================================================================


class EvaluatorTypeName(Enum):
    """Discriminated evaluator type identifiers."""
    exact_match = "exact_match"
    semantic_similarity = "semantic_similarity"
    llm_judge = "llm_judge"
    regex_match = "regex_match"
    json_schema_match = "json_schema_match"


class FinetuningBackendName(Enum):
    """Supported fine-tuning backend identifiers."""
    openai = "openai"
    anyscale = "anyscale"
    local_lora = "local_lora"
    axolotl = "axolotl"
    custom = "custom"
    kubernetes_lora = "kubernetes_lora"


class LogLevel(Enum):
    """Standard Python logging levels for audit configuration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ============================================================================
# Exceptions
# ============================================================================


class ConfigError(Exception):
    """Base domain exception for all config-related errors."""

    def __init__(self, message: str, path: Optional[str] = None, field_path: Optional[str] = None):
        self.message = message
        self.path = path
        self.field_path = field_path
        super().__init__(message)


class ConfigFileNotFoundError(ConfigError):
    """Raised when the specified config file does not exist."""

    def __init__(self, path: str):
        message = f"Config file not found: {path}"
        super().__init__(message=message, path=path)


class ConfigParseError(ConfigError):
    """Raised when YAML parsing fails due to syntax errors."""

    def __init__(self, message: str, path: str, line: int = 0, column: int = 0):
        super().__init__(message=message, path=path)
        self.line = line
        self.column = column


class ValidationErrorDetail:
    """A single structured validation error from Pydantic."""

    def __init__(self, field_path: str, error_type: str, message: str):
        self.field_path = field_path
        self.error_type = error_type
        self.message = message


class ConfigValidationError(ConfigError):
    """Raised when Pydantic validation fails."""

    def __init__(self, message: str, path: str, validation_errors: List[ValidationErrorDetail]):
        super().__init__(message=message, path=path)
        self.validation_errors = validation_errors


class EnvVarNotFoundError(ConfigError):
    """Raised when an env:VAR_NAME reference cannot be resolved."""

    def __init__(self, message: str, path: str, field_path: str, env_var_name: str):
        super().__init__(message=message, path=path, field_path=field_path)
        self.env_var_name = env_var_name


# ============================================================================
# Environment variable resolution
# ============================================================================

# Global context for env mapping during validation
_ENV_CONTEXT: Mapping[str, str] = {}
_CONFIG_PATH_CONTEXT: str = ""


def resolve_env_var(value: str, env: Mapping[str, str], field_path: Optional[str] = None) -> str:
    """
    Resolves an env:VAR_NAME reference string by looking up the variable name
    in the provided environment mapping.

    If the input string does not start with 'env:', it is returned unchanged.
    """
    if not value.startswith("env:"):
        return value

    # Extract variable name
    var_name = value[4:]  # Remove 'env:' prefix

    if not var_name:
        raise ConfigValidationError(
            message=f"Empty environment variable name in 'env:' reference at {field_path}",
            path=_CONFIG_PATH_CONTEXT,
            validation_errors=[],
        )

    if var_name not in env:
        raise EnvVarNotFoundError(
            message=f"Environment variable '{var_name}' not found",
            path=_CONFIG_PATH_CONTEXT,
            field_path=field_path or "",
            env_var_name=var_name,
        )

    return env[var_name]


# ============================================================================
# Pydantic Models
# ============================================================================


class ProviderConfig(BaseModel):
    """Configuration for the remote LLM API provider."""
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    api_base_url: str = Field(pattern=r"^https?://.+")
    api_key: SecretStr
    model: str
    timeout_seconds: int = Field(default=30, ge=1, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_api_key(cls, v: Any) -> str:
        if isinstance(v, str):
            return resolve_env_var(v, _ENV_CONTEXT, "provider.api_key")
        return v


class LocalModelConfig(BaseModel):
    """Configuration for the local model server."""
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    endpoint: str = Field(pattern=r"^https?://.+")
    model_name: str
    timeout_seconds: int = Field(default=60, ge=1, le=600)
    max_retries: int = Field(default=1, ge=0, le=5)
    health_check_path: str = "/health"


class MatchFieldSpec(BaseModel):
    """Specification for a single match field used by evaluators."""
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: str
    weight: float = Field(default=1.0, gt=0.0, le=10.0)
    case_sensitive: bool = True


class ConfidenceThresholds(BaseModel):
    """Confidence score thresholds that govern phase transitions for a task."""
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    local_ready: float = Field(gt=0.0, le=1.0)
    local_only: float = Field(gt=0.0, le=1.0)
    degraded_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class EvaluatorConfig(BaseModel):
    """Configuration for a single evaluator instance attached to a task."""
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", use_enum_values=False)

    type: EvaluatorTypeName
    match_fields: List[MatchFieldSpec] = Field(min_length=1)
    semantic_model: Optional[str] = None
    judge_prompt_template: Optional[str] = None
    regex_pattern: Optional[str] = None
    json_schema_path: Optional[str] = None

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v: Any) -> EvaluatorTypeName:
        if isinstance(v, str):
            try:
                return EvaluatorTypeName(v)
            except ValueError:
                raise ValueError(f"Invalid evaluator type: {v}")
        return v


class TaskFieldSpec(BaseModel):
    """Schema definition for a single field in a task's input or output schema."""
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: str
    type: str
    required: bool = True
    description: Optional[str] = None


class TaskConfig(BaseModel):
    """Configuration for a single task type."""
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    task_name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,62}[a-z0-9]$")
    prompt_template: str = Field(min_length=1, max_length=10000)
    input_schema: List[TaskFieldSpec] = Field(min_length=1)
    output_schema: List[TaskFieldSpec] = Field(min_length=1)
    evaluators: List[EvaluatorConfig] = Field(min_length=1)
    thresholds: ConfidenceThresholds
    sampling_rate_initial: float = Field(default=1.0, gt=0.0, le=1.0)
    min_training_examples: int = Field(default=100, ge=1, le=100000)


class BudgetConfig(BaseModel):
    """Hard budget enforcement configuration."""
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    max_daily_cost_usd: float = Field(ge=0.01, le=100000.0)
    max_monthly_cost_usd: float = Field(ge=0.01, le=1000000.0)
    rolling_window_hours: int = Field(default=24, ge=1, le=720)
    cost_per_input_token: float = Field(gt=0.0, le=1.0)
    cost_per_output_token: float = Field(gt=0.0, le=1.0)
    budget_state_path: str = ".apprentice/budget_state.json"


class FinetuningConfig(BaseModel):
    """Configuration for the fine-tuning orchestrator."""
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", use_enum_values=False)

    backend: FinetuningBackendName
    model_base: str
    batch_size: int = Field(default=100, ge=1, le=10000)
    trigger_interval_hours: int = Field(default=24, ge=1, le=720)
    api_key: Optional[SecretStr] = None
    api_base_url: Optional[str] = None
    output_dir: str = ".apprentice/models/"
    max_concurrent_jobs: int = Field(default=1, ge=1, le=10)

    # Kubernetes LoRA backend fields
    gcs_bucket: Optional[str] = None
    training_image: Optional[str] = None
    gpu_type: Optional[str] = "nvidia-tesla-t4"
    k8s_namespace: Optional[str] = "default"
    service_account: Optional[str] = None

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend(cls, v: Any) -> FinetuningBackendName:
        if isinstance(v, str):
            try:
                return FinetuningBackendName(v)
            except ValueError:
                raise ValueError(f"Invalid finetuning backend: {v}")
        return v

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_api_key(cls, v: Any) -> Optional[str]:
        if isinstance(v, str) and v:
            return resolve_env_var(v, _ENV_CONTEXT, "finetuning.api_key")
        return v if v else None


class AuditConfig(BaseModel):
    """Configuration for the structured audit logging system."""
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", use_enum_values=False)

    log_path: str = ".apprentice/audit.log"
    log_level: LogLevel = LogLevel.INFO
    log_to_stdout: bool = False
    max_file_size_mb: int = Field(default=100, ge=1, le=10000)
    backup_count: int = Field(default=5, ge=0, le=100)

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: Any) -> LogLevel:
        if isinstance(v, str):
            try:
                return LogLevel(v)
            except ValueError:
                raise ValueError(f"Invalid log level: {v}")
        return v


class TrainingDataStoreConfig(BaseModel):
    """Configuration for the training data store."""
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    storage_dir: str = ".apprentice/training_data/"
    max_examples_per_task: int = Field(default=50000, ge=100, le=10000000)


class ApprenticeConfig(BaseModel):
    """Root configuration model. Frozen and immutable after construction."""
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    provider: ProviderConfig
    local_model: LocalModelConfig
    tasks: List[TaskConfig] = Field(min_length=1)
    budget: BudgetConfig
    finetuning: FinetuningConfig
    audit: AuditConfig
    training_data: TrainingDataStoreConfig

    @model_validator(mode="after")
    def validate_cross_field_constraints(self) -> "ApprenticeConfig":
        """Performs all cross-field validations."""
        errors: List[str] = []

        # Validate threshold ordering for each task
        for i, task in enumerate(self.tasks):
            if task.thresholds.local_only < task.thresholds.local_ready:
                errors.append(
                    f"Config validation failed: local_only must be >= local_ready for task '{task.task_name}'"
                )

        # Validate budget consistency
        if self.budget.max_monthly_cost_usd < self.budget.max_daily_cost_usd:
            errors.append(
                "Config validation failed: max_monthly_cost_usd must be >= max_daily_cost_usd"
            )

        # Validate match fields exist in output schemas
        for task in self.tasks:
            output_field_names = {field.name for field in task.output_schema}
            for evaluator in task.evaluators:
                for match_field in evaluator.match_fields:
                    if match_field.name not in output_field_names:
                        errors.append(
                            f"Config validation failed: match field '{match_field.name}' not found in output_schema of task '{task.task_name}'"
                        )

        # Validate evaluator-specific required fields
        for task in self.tasks:
            for evaluator in task.evaluators:
                if evaluator.type == EvaluatorTypeName.semantic_similarity:
                    if not evaluator.semantic_model:
                        errors.append(
                            f"Config validation failed: semantic_similarity evaluator requires 'semantic_model' for task '{task.task_name}'"
                        )
                elif evaluator.type == EvaluatorTypeName.llm_judge:
                    if not evaluator.judge_prompt_template:
                        errors.append(
                            f"Config validation failed: llm_judge evaluator requires 'judge_prompt_template' for task '{task.task_name}'"
                        )
                elif evaluator.type == EvaluatorTypeName.regex_match:
                    if not evaluator.regex_pattern:
                        errors.append(
                            f"Config validation failed: regex_match evaluator requires 'regex_pattern' for task '{task.task_name}'"
                        )
                elif evaluator.type == EvaluatorTypeName.json_schema_match:
                    if not evaluator.json_schema_path:
                        errors.append(
                            f"Config validation failed: json_schema_match evaluator requires 'json_schema_path' for task '{task.task_name}'"
                        )

        # Validate unique task names
        task_names = [task.task_name for task in self.tasks]
        if len(task_names) != len(set(task_names)):
            duplicates = [name for name in task_names if task_names.count(name) > 1]
            for dup in set(duplicates):
                errors.append(f"Config validation failed: duplicate task name '{dup}'")

        # Validate backend combinations
        remote_backends = {FinetuningBackendName.openai, FinetuningBackendName.anyscale}
        if self.finetuning.backend in remote_backends:
            if not self.finetuning.api_key or (
                isinstance(self.finetuning.api_key, SecretStr) and
                not self.finetuning.api_key.get_secret_value()
            ):
                errors.append(
                    f"Config validation failed: backend '{self.finetuning.backend.value}' requires 'api_key'"
                )
            if not self.finetuning.api_base_url:
                errors.append(
                    f"Config validation failed: backend '{self.finetuning.backend.value}' requires 'api_base_url'"
                )

        # Validate kubernetes_lora backend requirements
        if self.finetuning.backend == FinetuningBackendName.kubernetes_lora:
            if not self.finetuning.gcs_bucket or not self.finetuning.gcs_bucket.strip():
                errors.append(
                    "Config validation failed: backend 'kubernetes_lora' requires 'gcs_bucket'"
                )
            if not self.finetuning.training_image or not self.finetuning.training_image.strip():
                errors.append(
                    "Config validation failed: backend 'kubernetes_lora' requires 'training_image'"
                )

        if errors:
            raise ConfigValidationError(
                message=f"Cross-field validation failed with {len(errors)} error(s)",
                path=_CONFIG_PATH_CONTEXT,
                validation_errors=[
                    ValidationErrorDetail(
                        field_path="",
                        error_type="cross_field_validation",
                        message=err,
                    )
                    for err in errors
                ],
            )

        return self


# ============================================================================
# Main loader function
# ============================================================================


def load_config(path: Path, env: Optional[Mapping[str, str]] = None) -> ApprenticeConfig:
    """
    Main entry point: loads and validates an Apprentice configuration from a YAML file.

    Pure function with no internal state. Reads the file, parses YAML, resolves
    env:VAR_NAME references, constructs and validates ApprenticeConfig via Pydantic v2,
    and returns a frozen/immutable config object.
    """
    global _ENV_CONTEXT, _CONFIG_PATH_CONTEXT

    # Use os.environ if no env mapping provided
    if env is None:
        env = os.environ

    # Set global context for validators
    _ENV_CONTEXT = env
    _CONFIG_PATH_CONTEXT = str(path)

    try:
        # Check file exists
        if not path.exists():
            raise ConfigFileNotFoundError(str(path))

        # Read file
        try:
            content = path.read_text(encoding="utf-8")
        except PermissionError:
            raise ConfigError(
                message=f"Cannot read config file: {path}: Permission denied",
                path=str(path),
            )
        except Exception as e:
            raise ConfigError(
                message=f"Cannot read config file: {path}: {e}",
                path=str(path),
            )

        # Parse YAML
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            line = 0
            column = 0
            if hasattr(e, "problem_mark"):
                line = e.problem_mark.line
                column = e.problem_mark.column
            raise ConfigParseError(
                message=f"YAML syntax error in {path}: {e}",
                path=str(path),
                line=line,
                column=column,
            )

        # Check if YAML is empty
        if data is None:
            raise ConfigParseError(
                message=f"Config file is empty: {path}",
                path=str(path),
            )

        # Check if top-level is a mapping
        if not isinstance(data, dict):
            raise ConfigParseError(
                message=f"Config file root must be a YAML mapping, got {type(data).__name__}: {path}",
                path=str(path),
            )

        # Validate with Pydantic
        try:
            config = ApprenticeConfig(**data)
            return config
        except Exception as e:
            # Convert Pydantic validation errors to our domain exception
            if hasattr(e, "errors"):
                validation_errors = []
                for err in e.errors():
                    field_path = ".".join(str(loc) for loc in err.get("loc", []))
                    validation_errors.append(
                        ValidationErrorDetail(
                            field_path=field_path,
                            error_type=err.get("type", "unknown"),
                            message=err.get("msg", str(err)),
                        )
                    )
                raise ConfigValidationError(
                    message=f"Config validation failed for {path}: {len(validation_errors)} error(s)",
                    path=str(path),
                    validation_errors=validation_errors,
                )
            else:
                # Re-raise if it's already one of our exceptions
                if isinstance(e, (ConfigError, EnvVarNotFoundError)):
                    raise
                # Otherwise wrap it
                raise ConfigError(
                    message=f"Config validation failed: {e}",
                    path=str(path),
                )
    finally:
        # Clean up global context
        _ENV_CONTEXT = {}
        _CONFIG_PATH_CONTEXT = ""


def validate_cross_field_constraints(config: ApprenticeConfig) -> ApprenticeConfig:
    """
    Performs all cross-field validations on a fully constructed ApprenticeConfig.

    This is exposed for testing but is automatically called by the ApprenticeConfig
    model_validator, so it doesn't need to be called separately when using load_config.
    """
    # This is already done in the model_validator, but we expose this for testing
    return config


# ── Auto-injected export aliases (Pact export gate) ──
MatchFieldSpecList = MatchFieldSpec
EvaluatorConfigList = EvaluatorConfig
TaskFieldSpecList = TaskFieldSpec
TaskConfigList = TaskConfig
ValidationErrorDetailList = ValidationErrorDetail
