"""
Fine-Tuning Orchestrator Component

Orchestrates the fine-tuning process with pluggable backends (Unsloth, OpenAI, HuggingFace).
Manages model versions, handles failures, and provides full provenance tracking.
"""

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============================================================================
# Enums
# ============================================================================

class FineTuneErrorCategory(str, Enum):
    """Categorization of fine-tuning errors to drive retry and fallback logic."""
    transient = "transient"
    data_error = "data_error"
    infrastructure = "infrastructure"
    permanent = "permanent"


class BackendType(str, Enum):
    """Supported fine-tuning backend types, used as discriminator in config and provenance."""
    unsloth = "unsloth"
    openai = "openai"
    huggingface = "huggingface"


class QuantizationType(str, Enum):
    """Supported GGUF quantization types for local backends (llama.cpp quantize)."""
    Q4_0 = "Q4_0"
    Q4_K_M = "Q4_K_M"
    Q4_K_S = "Q4_K_S"
    Q5_0 = "Q5_0"
    Q5_K_M = "Q5_K_M"
    Q5_K_S = "Q5_K_S"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    F16 = "F16"


# ============================================================================
# Data Models
# ============================================================================

class TrainingExample(BaseModel):
    """A single training example consisting of a user prompt and the desired assistant response, plus metadata for provenance tracking."""
    model_config = ConfigDict(frozen=True)

    example_id: str = Field(..., min_length=1)
    user_prompt: str = Field(..., min_length=1)
    assistant_response: str = Field(..., min_length=1)
    system_prompt: Optional[str] = None
    collected_at: str
    source: str

    @field_validator('collected_at')
    @classmethod
    def validate_iso8601(cls, v: str) -> str:
        """Validate ISO 8601 datetime string."""
        datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class HyperparameterOverrides(BaseModel):
    """Optional hyperparameter overrides for the fine-tuning run. All fields are optional; backends use their own defaults when not specified."""
    model_config = ConfigDict(frozen=True)

    learning_rate: Optional[float] = Field(None, gt=0.0, le=1.0)
    num_epochs: Optional[int] = Field(None, ge=1, le=100)
    batch_size: Optional[int] = Field(None, ge=1, le=512)
    lora_rank: Optional[int] = Field(None, ge=1, le=256)
    lora_alpha: Optional[int] = Field(None, ge=1, le=512)
    warmup_steps: Optional[int] = Field(None, ge=0, le=10000)


class FineTuneRequest(BaseModel):
    """Frozen Pydantic model representing a request to fine-tune a model."""
    model_config = ConfigDict(frozen=True)

    run_id: str
    base_model: str = Field(..., min_length=1)
    training_examples: List['TrainingExample'] = Field(..., min_length=1)
    hyperparameter_overrides: Optional[HyperparameterOverrides] = None


class TrainingMetrics(BaseModel):
    """Training metrics produced by a fine-tuning run, when available."""
    final_loss: Optional[float] = None
    num_steps: Optional[int] = None
    num_epochs_completed: Optional[float] = None
    training_duration_seconds: Optional[float] = None
    additional_metrics: dict = Field(default_factory=dict)


class FineTuneError(BaseModel):
    """Structured error result from a failed fine-tuning attempt."""
    status: str = Field("error", pattern="^error$")
    category: FineTuneErrorCategory
    message: str
    retry_recommended: bool
    run_id: str
    backend_type: str
    occurred_at: str
    details: dict = Field(default_factory=dict)

    @field_validator('occurred_at')
    @classmethod
    def validate_iso8601(cls, v: str) -> str:
        """Validate ISO 8601 datetime string."""
        datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class ModelVersion(BaseModel):
    """Frozen Pydantic model representing a successfully fine-tuned model version with full provenance."""
    model_config = ConfigDict(frozen=True)

    status: str = Field("success", pattern="^success$")
    version_id: str
    base_model: str
    backend_type: str
    training_example_ids: List[str]
    training_example_count: int = Field(..., ge=1)
    run_id: str
    started_at: str
    completed_at: str
    artifact_location: str
    ollama_model_name: Optional[str] = None
    metrics: Optional[TrainingMetrics] = None
    quantization_type: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    @field_validator('started_at', 'completed_at')
    @classmethod
    def validate_iso8601(cls, v: str) -> str:
        """Validate ISO 8601 datetime string."""
        datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

    @model_validator(mode='after')
    def validate_count_matches_ids(self) -> 'ModelVersion':
        """Ensure training_example_count matches len(training_example_ids)."""
        if self.training_example_count != len(self.training_example_ids):
            raise ValueError(
                f"training_example_count ({self.training_example_count}) must equal "
                f"len(training_example_ids) ({len(self.training_example_ids)})"
            )
        return self


# Type aliases
FineTuneResult = Union[ModelVersion, FineTuneError]
StringList = List[str]
ModelVersionList = List[ModelVersion]
TrainingExampleList = List[TrainingExample]


# ============================================================================
# Backend Config Models
# ============================================================================

class UnslothBackendConfig(BaseModel):
    """Configuration for the Unsloth local LoRA fine-tuning backend."""
    backend_type: Literal["unsloth"] = "unsloth"
    model_output_dir: str = Field(..., min_length=1)
    gguf_output_dir: str = Field(..., min_length=1)
    quantization_type: QuantizationType
    llama_cpp_path: str = Field(..., min_length=1)
    ollama_model_name_prefix: str = Field(..., min_length=1)
    ollama_base_url: str = "http://localhost:11434"
    modelfile_template: Optional[str] = None
    max_seq_length: int = Field(2048, ge=128, le=32768)
    default_lora_rank: int = Field(16, ge=1, le=256)
    default_learning_rate: float = 2e-4
    default_num_epochs: int = Field(3, ge=1, le=100)


class OpenAIBackendConfig(BaseModel):
    """Configuration for the OpenAI fine-tuning API backend."""
    backend_type: Literal["openai"] = "openai"
    api_base_url: str = "https://api.openai.com/v1"
    api_key_env_var: str = "OPENAI_API_KEY"
    default_num_epochs: int = Field(3, ge=1, le=50)
    poll_interval_seconds: int = Field(60, ge=5, le=600)
    poll_timeout_seconds: int = Field(7200, ge=60, le=86400)
    suffix: str = "apprentice"


class HuggingFaceBackendConfig(BaseModel):
    """Configuration for the Hugging Face Trainer fine-tuning backend."""
    backend_type: Literal["huggingface"] = "huggingface"
    model_output_dir: str = Field(..., min_length=1)
    gguf_output_dir: str = Field(..., min_length=1)
    quantization_type: QuantizationType
    llama_cpp_path: str = Field(..., min_length=1)
    ollama_model_name_prefix: str = Field(..., min_length=1)
    ollama_base_url: str = "http://localhost:11434"
    use_peft: bool = True
    default_learning_rate: float = 2e-4
    default_num_epochs: int = Field(3, ge=1, le=100)
    max_seq_length: int = Field(2048, ge=128, le=32768)


class KubernetesLoRAOrchestratorConfig(BaseModel):
    """Minimal orchestrator config stub for the Kubernetes LoRA backend.

    The full configuration lives in kubernetes_lora_backend.KubernetesLoRABackendConfig.
    This stub provides just the backend_type discriminator needed by OrchestratorConfig.
    """
    backend_type: Literal["kubernetes_lora"] = "kubernetes_lora"


class LocalNoOpOrchestratorConfig(BaseModel):
    """Orchestrator config for the LocalNoOpBackend (dev/testing)."""
    backend_type: Literal["noop"] = "noop"


BackendConfig = Union[UnslothBackendConfig, OpenAIBackendConfig, HuggingFaceBackendConfig, KubernetesLoRAOrchestratorConfig, LocalNoOpOrchestratorConfig]


class OrchestratorConfig(BaseModel):
    """Top-level configuration for the FineTuningOrchestrator."""
    backend: BackendConfig = Field(..., discriminator='backend_type')
    min_training_examples: int = Field(..., ge=1)
    model_version_store_path: str = Field(..., min_length=1)
    retry_on_next_batch: bool = True
    max_consecutive_failures: int = Field(3, ge=1, le=100)


# ============================================================================
# Internal Data Models
# ============================================================================

class CommandResult(BaseModel):
    """Result of executing a subprocess command via the CommandRunner protocol."""
    return_code: int
    stdout: str
    stderr: str
    command: str
    duration_seconds: float


class TrainArtifact(BaseModel):
    """Internal intermediate artifact produced by the training phase of local backends."""
    model_dir: str
    base_model: str
    adapter_dir: Optional[str] = None
    metrics: Optional[TrainingMetrics] = None


class GGUFArtifact(BaseModel):
    """Internal intermediate artifact produced by the GGUF quantization phase of local backends."""
    gguf_path: str
    quantization_type: QuantizationType
    file_size_bytes: int
    source_model_dir: str


class OllamaRegistration(BaseModel):
    """Result of registering a GGUF model with Ollama via 'ollama create' with a Modelfile."""
    model_name: str
    modelfile_path: str
    gguf_path: str


# ============================================================================
# Protocol / ABC Interfaces
# ============================================================================

class FineTuneBackend(ABC):
    """Protocol for fine-tuning backends."""

    @abstractmethod
    def fine_tune(self, request: FineTuneRequest) -> FineTuneResult:
        """Execute a fine-tuning run against the configured backend."""
        pass


class CommandRunner(ABC):
    """Protocol for running subprocess commands."""

    @abstractmethod
    def run_command(
        self,
        args: List[str],
        cwd: Optional[str] = None,
        timeout_seconds: int = 3600
    ) -> CommandResult:
        """Execute a subprocess command and capture output."""
        pass


class LocalNoOpBackend(FineTuneBackend):
    """A no-op fine-tuning backend for local development and testing.

    Simulates a successful fine-tuning run without doing any actual training.
    Writes artifact metadata to an output directory so the pipeline can
    execute end-to-end. Useful for integration testing, CI pipelines, and
    local development where real GPU training is unavailable.
    """

    def __init__(self, output_dir: str = "./models/noop"):
        self.output_dir = Path(output_dir)

    def fine_tune(self, request: FineTuneRequest) -> FineTuneResult:
        """Simulate a fine-tuning run and produce a ModelVersion artifact."""
        started_at = datetime.now(timezone.utc).isoformat()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = self.output_dir / request.run_id
        artifact_dir.mkdir(exist_ok=True)

        # Write training manifest
        manifest = {
            "run_id": request.run_id,
            "base_model": request.base_model,
            "backend_type": "noop",
            "training_example_count": len(request.training_examples),
            "training_example_ids": [e.example_id for e in request.training_examples],
            "started_at": started_at,
        }
        manifest_path = artifact_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        completed_at = datetime.now(timezone.utc).isoformat()

        return ModelVersion(
            status="success",
            version_id=f"noop-{request.run_id}",
            base_model=request.base_model,
            backend_type="noop",
            training_example_ids=[e.example_id for e in request.training_examples],
            training_example_count=len(request.training_examples),
            run_id=request.run_id,
            started_at=started_at,
            completed_at=completed_at,
            artifact_location=str(artifact_dir),
            metrics=TrainingMetrics(
                final_loss=0.0,
                num_steps=0,
                num_epochs_completed=0.0,
                training_duration_seconds=0.0,
            ),
        )


class ModelVersionStore:
    """Persistent storage for ModelVersion artifacts."""

    def __init__(self, store_path: str):
        """Initialize the model version store.

        Args:
            store_path: File path for persisting ModelVersion artifacts as JSON.
        """
        self.store_path = Path(store_path)
        self._versions: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load existing versions from disk if the file exists."""
        if self.store_path.exists():
            try:
                with open(self.store_path, 'r') as f:
                    data = json.load(f)
                    self._versions = data.get('versions', {})
            except (IOError, json.JSONDecodeError) as e:
                raise IOError(f"IOError reading model version store: {e}")

    def _save(self) -> None:
        """Persist versions to disk."""
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.store_path, 'w') as f:
                json.dump({'versions': self._versions}, f, indent=2)
        except IOError as e:
            raise IOError(f"IOError writing model version to disk: {e}")

    def save_model_version(self, model_version: ModelVersion) -> None:
        """Persist a ModelVersion artifact to durable storage.

        Args:
            model_version: The ModelVersion to save.

        Raises:
            ValueError: If a ModelVersion with the same version_id already exists.
            IOError: If there's a file I/O error when writing to disk.
        """
        if model_version.status != "success":
            raise ValueError("Can only save ModelVersions with status='success'")

        if model_version.version_id in self._versions:
            raise ValueError(f"Duplicate version_id: {model_version.version_id}")

        self._versions[model_version.version_id] = model_version.model_dump()
        self._save()

    def get_model_version(self, version_id: str) -> ModelVersion:
        """Retrieve a specific ModelVersion by its version_id.

        Args:
            version_id: The unique identifier of the version to retrieve.

        Returns:
            The ModelVersion matching the given version_id.

        Raises:
            ValueError: If version_id is empty or no ModelVersion exists with that id.
            IOError: If there's a file I/O error when reading from disk.
        """
        if not version_id:
            raise ValueError("version_id must be non-empty")

        if version_id not in self._versions:
            raise ValueError(f"ModelVersion not found: {version_id}")

        return ModelVersion(**self._versions[version_id])

    def get_latest_model_version(self) -> Optional[ModelVersion]:
        """Retrieve the most recently completed ModelVersion.

        Returns:
            The ModelVersion with the latest completed_at timestamp, or None if no versions exist.

        Raises:
            IOError: If there's a file I/O error when reading from disk.
        """
        if not self._versions:
            return None

        # Sort by completed_at descending
        sorted_versions = sorted(
            self._versions.values(),
            key=lambda v: v['completed_at'],
            reverse=True
        )

        return ModelVersion(**sorted_versions[0])

    def list_model_versions(self, limit: int = 100) -> List[ModelVersion]:
        """List all persisted ModelVersions, ordered by completed_at descending.

        Args:
            limit: Maximum number of versions to return (1-10000).

        Returns:
            List of ModelVersions ordered by completed_at descending.

        Raises:
            IOError: If there's a file I/O error when reading from disk.
        """
        if not self._versions:
            return []

        # Sort by completed_at descending
        sorted_versions = sorted(
            self._versions.values(),
            key=lambda v: v['completed_at'],
            reverse=True
        )

        # Apply limit
        limited = sorted_versions[:limit]

        return [ModelVersion(**v) for v in limited]


# ============================================================================
# Core Orchestrator
# ============================================================================

class FineTuningOrchestrator:
    """Orchestrates the fine-tuning process with full error handling and provenance tracking."""

    def __init__(
        self,
        backend: FineTuneBackend,
        model_version_store: ModelVersionStore,
        config: OrchestratorConfig
    ):
        """Initialize the orchestrator.

        Args:
            backend: The fine-tuning backend to use.
            model_version_store: Storage for model versions.
            config: Orchestrator configuration.
        """
        self.backend = backend
        self.model_version_store = model_version_store
        self.config = config
        self._consecutive_failures = 0

    def trigger_fine_tuning(
        self,
        training_examples: List[TrainingExample],
        base_model: str,
        hyperparameter_overrides: Optional[HyperparameterOverrides] = None
    ) -> FineTuneResult:
        """Main entry point for fine-tuning orchestration.

        Args:
            training_examples: List of training examples for this batch.
            base_model: Identifier of the base model to fine-tune.
            hyperparameter_overrides: Optional hyperparameter overrides.

        Returns:
            ModelVersion on success, FineTuneError on failure.
        """
        # Generate unique run_id
        run_id = str(uuid.uuid4())

        # Validate batch size
        if not training_examples or len(training_examples) < self.config.min_training_examples:
            return FineTuneError(
                status="error",
                category=FineTuneErrorCategory.data_error,
                message=f"Batch size ({len(training_examples)}) below minimum threshold ({self.config.min_training_examples})",
                retry_recommended=False,
                run_id=run_id,
                backend_type=self.config.backend.backend_type,
                occurred_at=datetime.now(timezone.utc).isoformat(),
                details={"batch_size": len(training_examples), "min_required": self.config.min_training_examples}
            )

        # Construct fine-tune request
        request = FineTuneRequest(
            run_id=run_id,
            base_model=base_model,
            training_examples=training_examples,
            hyperparameter_overrides=hyperparameter_overrides
        )

        # Call backend
        try:
            result = self.backend.fine_tune(request)
        except Exception as e:
            # Backend raised an exception despite contract - wrap it
            self._consecutive_failures += 1
            return FineTuneError(
                status="error",
                category=FineTuneErrorCategory.infrastructure,
                message=f"Unexpected backend exception: {str(e)}",
                retry_recommended=True,
                run_id=run_id,
                backend_type=self.config.backend.backend_type,
                occurred_at=datetime.now(timezone.utc).isoformat(),
                details={"exception_type": type(e).__name__, "exception_message": str(e)}
            )

        # Handle result
        if isinstance(result, FineTuneError):
            self._consecutive_failures += 1

            # Check if we've exceeded max failures
            if self._consecutive_failures > self.config.max_consecutive_failures:
                # Log at ERROR level but still return the error
                pass

            return result

        # Success - persist the model version
        try:
            self.model_version_store.save_model_version(result)
            self._consecutive_failures = 0  # Reset on success
            return result
        except Exception as e:
            # Failed to persist
            self._consecutive_failures += 1
            return FineTuneError(
                status="error",
                category=FineTuneErrorCategory.infrastructure,
                message=f"Fine-tuning succeeded but failed to persist ModelVersion: {str(e)}",
                retry_recommended=True,
                run_id=run_id,
                backend_type=self.config.backend.backend_type,
                occurred_at=datetime.now(timezone.utc).isoformat(),
                details={"persist_error": str(e)}
            )


# ============================================================================
# Config Parser
# ============================================================================

def parse_orchestrator_config(raw_config: dict) -> OrchestratorConfig:
    """Parse and validate the OrchestratorConfig from a YAML configuration dictionary.

    Args:
        raw_config: Dictionary containing the configuration.

    Returns:
        Validated OrchestratorConfig instance.

    Raises:
        ValueError: If config is invalid or missing required fields.
        TypeError: If field types are incorrect.
    """
    if raw_config is None:
        raise ValueError("Configuration cannot be None")

    if not isinstance(raw_config, dict):
        raise TypeError(f"Configuration must be a dict, got {type(raw_config).__name__}")

    # Validate backend exists
    if 'backend' not in raw_config:
        raise ValueError("Missing required configuration field: backend")

    backend_config = raw_config.get('backend')
    if not isinstance(backend_config, dict):
        raise TypeError(f"backend must be a dict, got {type(backend_config).__name__}")

    # Validate backend_type
    backend_type = backend_config.get('backend_type')
    if not backend_type:
        raise ValueError("Missing required field: backend_type")

    if backend_type not in ['unsloth', 'openai', 'huggingface']:
        raise ValueError(
            f"Unknown backend_type: {backend_type}. Must be one of: unsloth, openai, huggingface"
        )

    try:
        # Use Pydantic's validation
        config = OrchestratorConfig(**raw_config)
        return config
    except Exception as e:
        # Re-raise with better context
        error_str = str(e).lower()
        if 'validation' in error_str or 'field' in error_str:
            raise ValueError(f"Configuration validation error: {e}")
        elif 'type' in error_str:
            raise TypeError(f"Type error in configuration: {e}")
        else:
            raise
