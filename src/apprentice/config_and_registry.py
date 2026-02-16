"""
Configuration, Task Registry & Data Models (config_and_registry) - Glue Module

This module wires together three child components:
  - config_loader: YAML config parsing and validation
  - task_registry: Task registry and task config management
  - data_models: Shared data models and factory functions

It re-exports all types and functions from children to satisfy the parent contract.
"""

# Import child modules via relative imports (intra-package)
from . import config_loader
from . import task_registry
from . import data_models

# ===========================================================================
# Re-export Config Loader types and functions
# ===========================================================================

# Main config loading function
load_config = config_loader.load_config
resolve_env_var = config_loader.resolve_env_var
validate_cross_field_constraints = config_loader.validate_cross_field_constraints

# Config models
ApprenticeConfig = config_loader.ApprenticeConfig
ProviderConfig = config_loader.ProviderConfig
LocalModelConfig = config_loader.LocalModelConfig
BudgetConfig = config_loader.BudgetConfig
FinetuningConfig = config_loader.FinetuningConfig
AuditConfig = config_loader.AuditConfig
TrainingDataStoreConfig = config_loader.TrainingDataStoreConfig

# Config task types (note: these are from config_loader, not task_registry)
ConfigConfidenceThresholds = config_loader.ConfidenceThresholds
ConfigTaskConfig = config_loader.TaskConfig
EvaluatorConfig = config_loader.EvaluatorConfig
MatchFieldSpec = config_loader.MatchFieldSpec
TaskFieldSpec = config_loader.TaskFieldSpec

# Config errors
ConfigError = config_loader.ConfigError
ConfigFileNotFoundError = config_loader.ConfigFileNotFoundError
ConfigParseError = config_loader.ConfigParseError
ConfigValidationError = config_loader.ConfigValidationError
EnvVarNotFoundError = config_loader.EnvVarNotFoundError
ValidationErrorDetail = config_loader.ValidationErrorDetail

# Config enums
EvaluatorTypeName = config_loader.EvaluatorTypeName
FinetuningBackendName = config_loader.FinetuningBackendName
LogLevel = config_loader.LogLevel

# ===========================================================================
# Re-export Task Registry types and functions
# ===========================================================================

# Task Registry class — wrapper to accept `tasks=` keyword (contract API)
# while delegating to the implementation's `task_list=` positional parameter.
class TaskRegistry(task_registry.TaskRegistry):
    """Thin adapter: accepts ``TaskRegistry(tasks=[...])`` as the contract requires,
    forwarding to the implementation which expects ``task_list=``."""

    def __init__(self, tasks=None, *, task_list=None, **data):
        # Callers may use either `tasks=` (contract) or `task_list=` (impl).
        resolved = task_list if task_list is not None else tasks
        if resolved is None:
            raise ValueError("Either 'tasks' or 'task_list' must be provided")
        super().__init__(task_list=resolved, **data)

# Registry task types (note: different from config task types)
RegistryTaskConfig = task_registry.TaskConfig
RegistryConfidenceThresholds = task_registry.ConfidenceThresholds

# Registry errors
TaskNotFoundError = task_registry.TaskNotFoundError

# Registry enums (note: EvaluatorType is different from config's EvaluatorTypeName)
EvaluatorType = task_registry.EvaluatorType

# ===========================================================================
# Re-export Data Models types and functions
# ===========================================================================

# Data model classes
TaskRequest = data_models.TaskRequest
TaskResponse = data_models.TaskResponse
TrainingExample = data_models.TrainingExample
ComparisonPair = data_models.ComparisonPair
ConfidenceSnapshot = data_models.ConfidenceSnapshot
BudgetPeriod = data_models.BudgetPeriod
BudgetState = data_models.BudgetState
AuditEntry = data_models.AuditEntry
SamplingDecision = data_models.SamplingDecision
ModelVersion = data_models.ModelVersion

# Evaluation result types
ExactMatchResult = data_models.ExactMatchResult
FuzzyMatchResult = data_models.FuzzyMatchResult
SemanticSimilarityResult = data_models.SemanticSimilarityResult
LlmJudgeResult = data_models.LlmJudgeResult
CustomEvalResult = data_models.CustomEvalResult
EvaluationResult = data_models.EvaluationResult

# Factory functions
create_task_request = data_models.create_task_request
create_task_response = data_models.create_task_response
create_confidence_snapshot = data_models.create_confidence_snapshot
create_budget_state = data_models.create_budget_state
record_cost = data_models.record_cost
create_audit_entry = data_models.create_audit_entry
create_sampling_decision = data_models.create_sampling_decision
create_training_example = data_models.create_training_example
create_comparison_pair = data_models.create_comparison_pair
create_model_version = data_models.create_model_version

# Serialization functions
serialize_model = data_models.serialize_model
deserialize_model = data_models.deserialize_model
get_model_json_schema = data_models.get_model_json_schema


def validate_model(data, model_type):
    """Wrapper that ensures returned ValidationErrorDetail objects have ``field_path``
    (the contract attribute name) in addition to the implementation's ``field``."""
    errors = data_models.validate_model(data, model_type)
    for err in errors:
        if hasattr(err, 'field') and not hasattr(err, 'field_path'):
            # Dynamically alias field_path -> field for contract compatibility
            try:
                object.__setattr__(err, 'field_path', err.field)
            except Exception:
                pass
    return errors

# Data model enums
Phase = data_models.Phase
RoutingDecision = data_models.RoutingDecision
BudgetPeriodUnit = data_models.BudgetPeriodUnit
EvaluationResultType = data_models.EvaluationResultType

# ===========================================================================
# Export list (all types and functions the parent contract requires)
# ===========================================================================

__all__ = [
    # Config loading
    "load_config",
    "resolve_env_var",
    "validate_cross_field_constraints",
    # Config models
    "ApprenticeConfig",
    "ProviderConfig",
    "LocalModelConfig",
    "BudgetConfig",
    "FinetuningConfig",
    "AuditConfig",
    "TrainingDataStoreConfig",
    # Config task types
    "ConfigConfidenceThresholds",
    "ConfigTaskConfig",
    "EvaluatorConfig",
    "MatchFieldSpec",
    "TaskFieldSpec",
    # Config errors
    "ConfigError",
    "ConfigFileNotFoundError",
    "ConfigParseError",
    "ConfigValidationError",
    "EnvVarNotFoundError",
    "ValidationErrorDetail",
    # Config enums
    "EvaluatorTypeName",
    "FinetuningBackendName",
    "LogLevel",
    # Task Registry
    "TaskRegistry",
    "RegistryTaskConfig",
    "RegistryConfidenceThresholds",
    "TaskNotFoundError",
    "EvaluatorType",
    # Data models
    "TaskRequest",
    "TaskResponse",
    "TrainingExample",
    "ComparisonPair",
    "ConfidenceSnapshot",
    "BudgetPeriod",
    "BudgetState",
    "AuditEntry",
    "SamplingDecision",
    "ModelVersion",
    # Evaluation results
    "ExactMatchResult",
    "FuzzyMatchResult",
    "SemanticSimilarityResult",
    "LlmJudgeResult",
    "CustomEvalResult",
    "EvaluationResult",
    # Factory functions
    "create_task_request",
    "create_task_response",
    "create_confidence_snapshot",
    "create_budget_state",
    "record_cost",
    "create_audit_entry",
    "create_sampling_decision",
    "create_training_example",
    "create_comparison_pair",
    "create_model_version",
    # Serialization
    "serialize_model",
    "deserialize_model",
    "validate_model",
    "get_model_json_schema",
    # Enums
    "Phase",
    "RoutingDecision",
    "BudgetPeriodUnit",
    "EvaluationResultType",
]
