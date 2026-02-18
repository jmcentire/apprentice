"""
Tests for the KubernetesLoRABackend and its integration points.

Covers:
  1. KubernetesLoRABackendConfig validation
  2. Training data upload (GCS)
  3. K8s Job creation and spec correctness
  4. Job polling (success, failure, timeout)
  5. GGUF download from GCS
  6. Metrics download from GCS
  7. Ollama registration
  8. Error classification
  9. End-to-end fine_tune() with all mocks
  10. Config loader cross-field validation for kubernetes_lora
  11. Factory wiring
  12. GCS cleanup
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest

from apprentice.kubernetes_lora_backend import (
    KubernetesLoRABackend,
    KubernetesLoRABackendConfig,
)
from apprentice.fine_tuning_orchestrator import (
    FineTuneBackend,
    FineTuneError,
    FineTuneErrorCategory,
    FineTuneRequest,
    ModelVersion,
    TrainingExample,
    TrainingMetrics,
    KubernetesLoRAOrchestratorConfig,
    LocalNoOpOrchestratorConfig,
    OrchestratorConfig,
    FineTuningOrchestrator,
    ModelVersionStore,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config():
    return KubernetesLoRABackendConfig(
        gcs_bucket="test-bucket",
        training_image="gcr.io/test/trainer:latest",
        namespace="test-ns",
        gpu_type="nvidia-tesla-t4",
        service_account_name="test-sa",
        base_model="unsloth/llama-3.1-8b-bnb-4bit",
        ollama_base_url="http://localhost:11434",
        local_gguf_dir="/tmp/test-gguf",
        poll_interval_seconds=5,
        job_timeout_seconds=60,
    )


@pytest.fixture
def backend(config):
    return KubernetesLoRABackend(config)


@pytest.fixture
def sample_request():
    examples = [
        TrainingExample(
            example_id=str(uuid.uuid4()),
            user_prompt=f"Question {i}",
            assistant_response=f"Answer {i}",
            collected_at=datetime.now(timezone.utc).isoformat(),
            source="test",
        )
        for i in range(5)
    ]
    return FineTuneRequest(
        run_id=str(uuid.uuid4()),
        base_model="llama3.1:8b",
        training_examples=examples,
    )


@pytest.fixture
def sample_request_with_system():
    """Request with system prompts on examples."""
    examples = [
        TrainingExample(
            example_id=str(uuid.uuid4()),
            user_prompt="What is 2+2?",
            assistant_response="4",
            system_prompt="You are a math tutor.",
            collected_at=datetime.now(timezone.utc).isoformat(),
            source="test",
        )
    ]
    return FineTuneRequest(
        run_id="test-run-abc",
        base_model="llama3.1:8b",
        training_examples=examples,
    )


def make_mock_gcs_client():
    """Create a mock GCS client with bucket/blob interface."""
    mock_blob = MagicMock()
    mock_blob.upload_from_string = MagicMock()
    mock_blob.download_to_filename = MagicMock()
    mock_blob.download_as_text = MagicMock(return_value=json.dumps({
        "final_loss": 0.42,
        "num_steps": 100,
        "num_epochs_completed": 3.0,
        "training_duration_seconds": 1200.0,
        "additional_metrics": {"lr": 0.0002},
    }))
    # For _download_gguf: blob.exists(), blob.reload(), blob.size
    mock_blob.exists = MagicMock(return_value=True)
    mock_blob.reload = MagicMock()
    mock_blob.size = 1024 * 1024  # 1 MB

    mock_bucket = MagicMock()
    mock_bucket.blob = MagicMock(return_value=mock_blob)

    mock_client = MagicMock()
    mock_client.bucket = MagicMock(return_value=mock_bucket)

    return mock_client, mock_bucket, mock_blob


def make_mock_k8s_clients(succeed=False, fail=False):
    """Create mock K8s batch and core API clients.

    Args:
        succeed: Job completed successfully (succeeded=1).
        fail: Job failed (failed=1).
        Both False: Job still pending.
    """
    mock_batch = MagicMock()
    mock_core = MagicMock()

    # Job status — use SimpleNamespace so .succeeded/.failed are real None/int,
    # not MagicMock objects (which are always truthy).
    if succeed:
        mock_status = SimpleNamespace(succeeded=1, failed=None)
    elif fail:
        mock_status = SimpleNamespace(succeeded=None, failed=1)
    else:
        mock_status = SimpleNamespace(succeeded=None, failed=None, active=1)

    mock_job = SimpleNamespace(status=mock_status)

    mock_batch.create_namespaced_job = MagicMock()
    mock_batch.read_namespaced_job_status = MagicMock(return_value=mock_job)
    mock_batch.delete_namespaced_job = MagicMock()

    # Pod logs
    mock_pod = SimpleNamespace(metadata=SimpleNamespace(name="test-pod-xyz"))
    mock_pods = SimpleNamespace(items=[mock_pod])
    mock_core.list_namespaced_pod = MagicMock(return_value=mock_pods)
    mock_core.read_namespaced_pod_log = MagicMock(return_value="some log output")
    mock_core.list_namespaced_event = MagicMock(return_value=SimpleNamespace(items=[]))

    return mock_batch, mock_core


# ============================================================================
# 1. Config validation
# ============================================================================


class TestKubernetesLoRABackendConfig:

    def test_minimal_valid_config(self):
        cfg = KubernetesLoRABackendConfig(
            gcs_bucket="my-bucket",
            training_image="gcr.io/proj/trainer:v1",
        )
        assert cfg.backend_type == "kubernetes_lora"
        assert cfg.gcs_bucket == "my-bucket"
        assert cfg.namespace == "default"
        assert cfg.gpu_type == "nvidia-tesla-t4"
        assert cfg.spot_tolerations is True
        assert cfg.lora_rank == 16

    def test_missing_gcs_bucket_raises(self):
        with pytest.raises(Exception):
            KubernetesLoRABackendConfig(training_image="gcr.io/proj/trainer:v1")

    def test_missing_training_image_raises(self):
        with pytest.raises(Exception):
            KubernetesLoRABackendConfig(gcs_bucket="my-bucket")

    def test_gpu_count_bounds(self):
        cfg = KubernetesLoRABackendConfig(
            gcs_bucket="b", training_image="img:v1", gpu_count=4,
        )
        assert cfg.gpu_count == 4

        with pytest.raises(Exception):
            KubernetesLoRABackendConfig(
                gcs_bucket="b", training_image="img:v1", gpu_count=0,
            )

    def test_all_fields_customized(self):
        cfg = KubernetesLoRABackendConfig(
            gcs_bucket="custom-bucket",
            gcs_prefix="custom-prefix",
            training_image="us-docker.pkg.dev/proj/repo/trainer:v2",
            namespace="ml-training",
            gpu_type="nvidia-l4",
            gpu_count=2,
            gpu_resource_key="nvidia.com/gpu",
            node_pool_label="gpu-pool",
            spot_tolerations=False,
            job_timeout_seconds=3600,
            poll_interval_seconds=15,
            service_account_name="ml-sa",
            backoff_limit=3,
            image_pull_policy="IfNotPresent",
            cpu_request="4",
            cpu_limit="16",
            memory_request="16Gi",
            memory_limit="64Gi",
            base_model="unsloth/mistral-7b-bnb-4bit",
            quantization_type="Q5_K_M",
            max_seq_length=4096,
            lora_rank=32,
            learning_rate=1e-4,
            num_epochs=5,
            ollama_base_url="http://ollama:11434",
            ollama_model_name_prefix="custom",
            ollama_create_timeout_seconds=300,
            ollama_create_retries=5,
            local_gguf_dir="/models/gguf",
            min_disk_free_bytes=10_000_000_000,
        )
        assert cfg.namespace == "ml-training"
        assert cfg.gpu_count == 2
        assert cfg.spot_tolerations is False
        assert cfg.lora_rank == 32
        assert cfg.backoff_limit == 3
        assert cfg.image_pull_policy == "IfNotPresent"
        assert cfg.cpu_request == "4"
        assert cfg.memory_limit == "64Gi"
        assert cfg.ollama_create_retries == 5


# ============================================================================
# 2. Implements FineTuneBackend ABC
# ============================================================================


class TestBackendABC:

    def test_implements_fine_tune_backend(self, backend):
        assert isinstance(backend, FineTuneBackend)

    def test_has_fine_tune_method(self, backend):
        assert hasattr(backend, 'fine_tune')
        assert callable(backend.fine_tune)


# ============================================================================
# 3. Training data upload
# ============================================================================


class TestUploadTrainingData:

    def test_upload_formats_chat_jsonl(self, backend, sample_request):
        mock_client, mock_bucket, mock_blob = make_mock_gcs_client()
        backend._gcs_client = mock_client

        result = backend._upload_training_data(sample_request)

        assert result.startswith("gs://test-bucket/")
        assert sample_request.run_id in result
        assert "data.jsonl" in result

        # Verify blob.upload_from_string was called
        mock_blob.upload_from_string.assert_called_once()
        call_args = mock_blob.upload_from_string.call_args
        uploaded_data = call_args[0][0]

        # Verify JSONL format
        lines = uploaded_data.strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            parsed = json.loads(line)
            assert "messages" in parsed
            messages = parsed["messages"]
            assert messages[-2]["role"] == "user"
            assert messages[-1]["role"] == "assistant"

    def test_upload_includes_system_prompt(self, backend, sample_request_with_system):
        mock_client, mock_bucket, mock_blob = make_mock_gcs_client()
        backend._gcs_client = mock_client

        backend._upload_training_data(sample_request_with_system)

        call_args = mock_blob.upload_from_string.call_args
        uploaded_data = call_args[0][0]
        parsed = json.loads(uploaded_data.strip())
        messages = parsed["messages"]

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a math tutor."

    def test_upload_gcs_path_structure(self, backend, sample_request):
        mock_client, mock_bucket, mock_blob = make_mock_gcs_client()
        backend._gcs_client = mock_client

        result = backend._upload_training_data(sample_request)

        expected_prefix = f"gs://test-bucket/training-runs/{sample_request.run_id}/data.jsonl"
        assert result == expected_prefix


# ============================================================================
# 4. K8s Job creation
# ============================================================================


class TestCreateTrainingJob:

    def _patch_kubernetes_module(self):
        """Create a mock kubernetes.client module and install it in sys.modules."""
        mock_k8s_client = MagicMock()
        # Each V1* class should be callable and return a plain MagicMock
        for cls_name in [
            'V1EnvVar', 'V1ResourceRequirements', 'V1Container', 'V1Toleration',
            'V1PodSpec', 'V1ObjectMeta', 'V1Job', 'V1JobSpec', 'V1PodTemplateSpec',
            'V1DeleteOptions',
        ]:
            setattr(mock_k8s_client, cls_name, MagicMock())
        return mock_k8s_client

    def test_job_name_format(self, backend, sample_request):
        mock_batch, mock_core = make_mock_k8s_clients()
        backend._k8s_batch_client = mock_batch
        backend._k8s_core_client = mock_core

        mock_k8s_client = self._patch_kubernetes_module()
        mock_k8s_mod = MagicMock(client=mock_k8s_client)
        with patch.dict("sys.modules", {
            "kubernetes": mock_k8s_mod,
            "kubernetes.client": mock_k8s_client,
            "kubernetes.config": MagicMock(),
        }):
            job_name = backend._create_training_job(sample_request, "gs://test/data.jsonl")

        assert job_name.startswith("apprentice-ft-")
        assert job_name == f"apprentice-ft-{sample_request.run_id[:8]}"

    def test_job_submitted_to_k8s(self, backend, sample_request):
        mock_batch, mock_core = make_mock_k8s_clients()
        backend._k8s_batch_client = mock_batch
        backend._k8s_core_client = mock_core

        mock_k8s_client = self._patch_kubernetes_module()
        mock_k8s_mod = MagicMock(client=mock_k8s_client)
        with patch.dict("sys.modules", {
            "kubernetes": mock_k8s_mod,
            "kubernetes.client": mock_k8s_client,
            "kubernetes.config": MagicMock(),
        }):
            backend._create_training_job(sample_request, "gs://test/data.jsonl")

        mock_batch.create_namespaced_job.assert_called_once()
        call_kwargs = mock_batch.create_namespaced_job.call_args
        assert call_kwargs.kwargs.get("namespace") == "test-ns" or call_kwargs[1].get("namespace") == "test-ns"


# ============================================================================
# 5. Job polling
# ============================================================================


class TestWaitForJob:

    def test_returns_on_success(self, backend):
        mock_batch, mock_core = make_mock_k8s_clients(succeed=True)
        backend._k8s_batch_client = mock_batch
        backend._k8s_core_client = mock_core

        # Should not raise
        backend._wait_for_job("test-job", "test-run-id")
        mock_batch.read_namespaced_job_status.assert_called_once()

    def test_raises_on_failure(self, backend):
        mock_batch, mock_core = make_mock_k8s_clients(fail=True)
        backend._k8s_batch_client = mock_batch
        backend._k8s_core_client = mock_core

        with pytest.raises(RuntimeError, match="failed"):
            backend._wait_for_job("test-job", "test-run-id")

    def test_timeout_raises(self, backend):
        """Job that never completes should raise TimeoutError."""
        mock_batch, mock_core = make_mock_k8s_clients(succeed=False, fail=False)
        backend._k8s_batch_client = mock_batch
        backend._k8s_core_client = mock_core

        # Set very short timeout
        backend.config = backend.config.model_copy(update={
            "job_timeout_seconds": 1,
            "poll_interval_seconds": 5,
        })

        with patch("apprentice.kubernetes_lora_backend.time") as mock_time:
            # Simulate time progression past deadline
            mock_time.monotonic = MagicMock(side_effect=[0, 0, 100])
            mock_time.sleep = MagicMock()

            with pytest.raises(TimeoutError, match="timed out"):
                backend._wait_for_job("test-job", "test-run-id")

    def test_polls_until_success(self, backend):
        """Job transitions from pending to succeeded."""
        mock_batch, _ = make_mock_k8s_clients()
        mock_core = MagicMock()
        backend._k8s_batch_client = mock_batch
        backend._k8s_core_client = mock_core

        # First call: pending, second call: succeeded
        pending_job = SimpleNamespace(status=SimpleNamespace(succeeded=None, failed=None, active=1))
        success_job = SimpleNamespace(status=SimpleNamespace(succeeded=1, failed=None))

        mock_batch.read_namespaced_job_status = MagicMock(
            side_effect=[pending_job, success_job]
        )

        with patch("apprentice.kubernetes_lora_backend.time") as mock_time:
            # Need enough monotonic values:
            # 1. deadline calc (start), 2. while check, 3. poll_count%log check,
            # 4. sleep, 5. while check again, 6. elapsed calc on success
            mock_time.monotonic = MagicMock(side_effect=[0, 10, 20, 30, 40, 50])
            mock_time.sleep = MagicMock()

            backend._wait_for_job("test-job", "test-run-id")

        assert mock_batch.read_namespaced_job_status.call_count == 2


# ============================================================================
# 6. GGUF download
# ============================================================================


class TestDownloadGGUF:

    def test_downloads_to_local_path(self, backend, tmp_path):
        mock_client, mock_bucket, mock_blob = make_mock_gcs_client()
        backend._gcs_client = mock_client

        # Use a real temp directory that exists (for shutil.disk_usage)
        backend.config = backend.config.model_copy(update={
            "local_gguf_dir": str(tmp_path),
            "min_disk_free_bytes": 0,
        })

        # Mock stat() only on the downloaded file, not on all Path objects
        # (global Path.stat patch breaks mkdir's internal is_dir() check on Python 3.13+)
        _original_stat = Path.stat
        def _selective_stat(self, *args, **kwargs):
            if self.name.endswith(".gguf"):
                return SimpleNamespace(st_size=1024)
            return _original_stat(self, *args, **kwargs)

        with patch.object(Path, 'stat', _selective_stat):
            result = backend._download_gguf("test-run-123")

        assert "test-run-123.gguf" in result
        mock_blob.download_to_filename.assert_called_once()
        mock_blob.exists.assert_called_once()

    def test_raises_on_missing_blob(self, backend, tmp_path):
        mock_client, mock_bucket, mock_blob = make_mock_gcs_client()
        mock_blob.exists = MagicMock(return_value=False)
        backend._gcs_client = mock_client

        backend.config = backend.config.model_copy(update={
            "local_gguf_dir": str(tmp_path),
        })

        with pytest.raises(FileNotFoundError, match="not found"):
            backend._download_gguf("test-run-missing")

    def test_raises_on_insufficient_disk(self, backend, tmp_path):
        mock_client, mock_bucket, mock_blob = make_mock_gcs_client()
        mock_blob.size = 100_000_000_000  # 100 GB — bigger than any /tmp
        backend._gcs_client = mock_client

        backend.config = backend.config.model_copy(update={
            "local_gguf_dir": str(tmp_path),
            "min_disk_free_bytes": 100_000_000_000_000,  # require 100 TB free
        })

        with pytest.raises(OSError, match="[Ii]nsufficient disk"):
            backend._download_gguf("test-run-big")


# ============================================================================
# 7. Metrics download
# ============================================================================


class TestDownloadMetrics:

    def test_parses_metrics_json(self, backend):
        mock_client, mock_bucket, mock_blob = make_mock_gcs_client()
        backend._gcs_client = mock_client

        metrics = backend._download_metrics("test-run-123")

        assert metrics is not None
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.final_loss == 0.42
        assert metrics.num_steps == 100
        assert metrics.num_epochs_completed == 3.0
        assert metrics.training_duration_seconds == 1200.0

    def test_returns_none_on_missing_metrics(self, backend):
        mock_client, mock_bucket, mock_blob = make_mock_gcs_client()
        mock_blob.download_as_text = MagicMock(side_effect=Exception("Not found"))
        backend._gcs_client = mock_client

        metrics = backend._download_metrics("test-run-missing")
        assert metrics is None


# ============================================================================
# 8. Ollama registration
# ============================================================================


class TestRegisterOllama:

    def test_calls_ollama_create_api(self, backend):
        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post = MagicMock(return_value=mock_response)
        mock_httpx.Timeout = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            model_name = backend._register_ollama("abc123def456", "/tmp/model.gguf")

        assert model_name.startswith("apprentice:")
        assert "abc123def456" in model_name
        mock_httpx.post.assert_called_once()

        call_args = mock_httpx.post.call_args
        assert "/api/create" in call_args[0][0]
        body = call_args[1]["json"]
        assert body["name"] == model_name
        assert "/tmp/model.gguf" in body["modelfile"]

    def test_propagates_ollama_errors(self, backend):
        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(side_effect=Exception("Connection refused"))
        mock_httpx.post = MagicMock(return_value=mock_response)
        mock_httpx.Timeout = MagicMock(return_value=MagicMock())

        # Set retries to 1 to avoid long test
        backend.config = backend.config.model_copy(update={
            "ollama_create_retries": 1,
        })

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with pytest.raises(Exception, match="Connection refused"):
                backend._register_ollama("abc123", "/tmp/model.gguf")


# ============================================================================
# 9. Error classification
# ============================================================================


class TestClassifyError:

    def test_timeout_is_transient(self, backend):
        assert backend._classify_error(TimeoutError("timed out")) == FineTuneErrorCategory.transient

    def test_preemption_is_transient(self, backend):
        assert backend._classify_error(RuntimeError("node was preempted")) == FineTuneErrorCategory.transient

    def test_spot_is_transient(self, backend):
        assert backend._classify_error(RuntimeError("spot instance terminated")) == FineTuneErrorCategory.transient

    def test_oom_is_infrastructure(self, backend):
        assert backend._classify_error(RuntimeError("CUDA out of memory")) == FineTuneErrorCategory.infrastructure

    def test_cuda_is_infrastructure(self, backend):
        assert backend._classify_error(RuntimeError("CUDA error: device-side assert")) == FineTuneErrorCategory.infrastructure

    def test_disk_is_infrastructure(self, backend):
        assert backend._classify_error(OSError("Insufficient disk space")) == FineTuneErrorCategory.infrastructure

    def test_json_error_is_data_error(self, backend):
        assert backend._classify_error(ValueError("JSON decode error")) == FineTuneErrorCategory.data_error

    def test_format_error_is_data_error(self, backend):
        assert backend._classify_error(ValueError("Invalid data format")) == FineTuneErrorCategory.data_error

    def test_unknown_error_is_infrastructure(self, backend):
        assert backend._classify_error(RuntimeError("something unknown happened")) == FineTuneErrorCategory.infrastructure


# ============================================================================
# 10. End-to-end fine_tune()
# ============================================================================


class TestFineTuneEndToEnd:
    """End-to-end tests mock individual steps to avoid needing kubernetes/gcloud packages."""

    def _run_e2e(self, backend, sample_request, *, create_job_side_effect=None, wait_side_effect=None):
        """Helper: run fine_tune() with all internal methods mocked."""
        job_name = f"apprentice-ft-{sample_request.run_id[:8]}"
        model_name = f"apprentice:{sample_request.run_id[:12]}"
        gguf_path = f"/tmp/test-gguf/{sample_request.run_id}.gguf"

        mock_metrics = TrainingMetrics(
            final_loss=0.42, num_steps=100,
            num_epochs_completed=3.0, training_duration_seconds=1200.0,
        )

        with patch.object(backend, '_upload_training_data', return_value=f"gs://test-bucket/data.jsonl") as mock_upload, \
             patch.object(backend, '_create_training_job', return_value=job_name) as mock_create, \
             patch.object(backend, '_wait_for_job', side_effect=wait_side_effect) as mock_wait, \
             patch.object(backend, '_download_gguf', return_value=gguf_path) as mock_download_gguf, \
             patch.object(backend, '_download_metrics', return_value=mock_metrics) as mock_download_metrics, \
             patch.object(backend, '_register_ollama', return_value=model_name) as mock_register, \
             patch.object(backend, '_cleanup_job') as mock_cleanup:

            if create_job_side_effect:
                mock_create.side_effect = create_job_side_effect

            result = backend.fine_tune(sample_request)
            return result, mock_create, mock_wait, mock_cleanup

    def test_success_returns_model_version(self, backend, sample_request):
        result, _, _, _ = self._run_e2e(backend, sample_request)

        assert isinstance(result, ModelVersion)
        assert result.status == "success"
        assert result.backend_type == "kubernetes_lora"
        assert result.run_id == sample_request.run_id
        assert result.version_id == f"k8s-lora-{sample_request.run_id}"
        assert result.training_example_count == 5
        assert len(result.training_example_ids) == 5
        assert result.quantization_type == "Q4_K_M"
        assert result.ollama_model_name is not None
        assert result.metrics is not None
        assert result.metrics.final_loss == 0.42

    def test_gcs_upload_failure_returns_error(self, backend, sample_request):
        """Upload failure should return FineTuneError."""
        with patch.object(backend, '_upload_training_data', side_effect=Exception("GCS permission denied")):
            result = backend.fine_tune(sample_request)

        assert isinstance(result, FineTuneError)
        assert result.status == "error"
        assert result.run_id == sample_request.run_id
        assert result.backend_type == "kubernetes_lora"
        assert "GCS permission denied" in result.message

    def test_k8s_job_failure_returns_error(self, backend, sample_request):
        result, _, _, _ = self._run_e2e(
            backend, sample_request,
            wait_side_effect=RuntimeError("Training job 'test' failed. Logs: OOM"),
        )

        assert isinstance(result, FineTuneError)
        assert result.status == "error"
        assert "failed" in result.message.lower()

    def test_timeout_returns_transient_error(self, backend, sample_request):
        result, _, _, _ = self._run_e2e(
            backend, sample_request,
            wait_side_effect=TimeoutError("Training job timed out after 60s"),
        )

        assert isinstance(result, FineTuneError)
        assert result.category == FineTuneErrorCategory.transient
        assert result.retry_recommended is True

    def test_metadata_includes_provenance(self, backend, sample_request):
        result, _, _, _ = self._run_e2e(backend, sample_request)

        assert result.metadata["gcs_bucket"] == "test-bucket"
        assert result.metadata["training_image"] == "gcr.io/test/trainer:latest"
        assert result.metadata["gpu_type"] == "nvidia-tesla-t4"

    def test_cleanup_called_on_success(self, backend, sample_request):
        _, _, _, mock_cleanup = self._run_e2e(backend, sample_request)
        mock_cleanup.assert_called_once()

    def test_all_steps_called_in_order(self, backend, sample_request):
        """Verify the full pipeline: upload -> create -> wait -> download -> register -> cleanup."""
        call_order = []

        job_name = f"apprentice-ft-{sample_request.run_id[:8]}"
        model_name = f"apprentice:{sample_request.run_id[:12]}"
        gguf_path = f"/tmp/test-gguf/{sample_request.run_id}.gguf"

        mock_metrics = TrainingMetrics(
            final_loss=0.42, num_steps=100,
            num_epochs_completed=3.0, training_duration_seconds=1200.0,
        )

        with patch.object(backend, '_upload_training_data', side_effect=lambda *a: (call_order.append("upload"), "gs://test/data.jsonl")[1]), \
             patch.object(backend, '_create_training_job', side_effect=lambda *a: (call_order.append("create_job"), job_name)[1]), \
             patch.object(backend, '_wait_for_job', side_effect=lambda *a, **kw: call_order.append("wait")), \
             patch.object(backend, '_download_gguf', side_effect=lambda *a: (call_order.append("download_gguf"), gguf_path)[1]), \
             patch.object(backend, '_download_metrics', side_effect=lambda *a: (call_order.append("download_metrics"), mock_metrics)[1]), \
             patch.object(backend, '_register_ollama', side_effect=lambda *a: (call_order.append("register"), model_name)[1]), \
             patch.object(backend, '_cleanup_job', side_effect=lambda *a, **kw: call_order.append("cleanup")):

            result = backend.fine_tune(sample_request)

        assert isinstance(result, ModelVersion)
        assert call_order == ["upload", "create_job", "wait", "download_gguf", "download_metrics", "register", "cleanup"]


# ============================================================================
# 11. Config loader cross-field validation
# ============================================================================


class TestConfigLoaderKubernetesLora:

    def _make_valid_k8s_config(self):
        """Full valid config dict with kubernetes_lora backend."""
        return {
            "provider": {
                "api_base_url": "https://api.openai.com/v1",
                "api_key": "test-key",
                "model": "gpt-4",
                "timeout_seconds": 30,
                "max_retries": 3,
                "retry_base_delay_seconds": 1.0,
            },
            "local_model": {
                "endpoint": "http://localhost:11434",
                "model_name": "llama3.1:8b",
                "timeout_seconds": 60,
                "max_retries": 1,
                "health_check_path": "/health",
            },
            "tasks": [{
                "task_name": "test_task",
                "prompt_template": "Do something with {text}",
                "input_schema": [{"name": "text", "type": "string", "required": True}],
                "output_schema": [{"name": "result", "type": "string", "required": True}],
                "evaluators": [{
                    "type": "exact_match",
                    "match_fields": [{"name": "result", "weight": 1.0, "case_sensitive": True}],
                }],
                "thresholds": {"local_ready": 0.7, "local_only": 0.85, "degraded_threshold": 0.3},
                "sampling_rate_initial": 1.0,
                "min_training_examples": 100,
            }],
            "budget": {
                "max_daily_cost_usd": 10.0,
                "max_monthly_cost_usd": 150.0,
                "rolling_window_hours": 24,
                "cost_per_input_token": 0.000003,
                "cost_per_output_token": 0.000015,
                "budget_state_path": ".apprentice/budget_state.json",
            },
            "finetuning": {
                "backend": "kubernetes_lora",
                "model_base": "llama3.1:8b",
                "batch_size": 100,
                "trigger_interval_hours": 24,
                "output_dir": ".apprentice/models/",
                "max_concurrent_jobs": 1,
                "gcs_bucket": "my-training-bucket",
                "training_image": "gcr.io/proj/trainer:latest",
            },
            "audit": {
                "log_path": ".apprentice/audit.log",
                "log_level": "INFO",
                "log_to_stdout": False,
                "max_file_size_mb": 100,
                "backup_count": 5,
            },
            "training_data": {
                "storage_dir": ".apprentice/training_data/",
                "max_examples_per_task": 50000,
            },
        }

    def test_valid_kubernetes_lora_config_loads(self, tmp_path):
        import yaml
        from apprentice.config_loader import load_config

        cfg_dict = self._make_valid_k8s_config()
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg_dict))

        config = load_config(path, {})
        assert config.finetuning.backend.value == "kubernetes_lora"
        assert config.finetuning.gcs_bucket == "my-training-bucket"
        assert config.finetuning.training_image == "gcr.io/proj/trainer:latest"

    def test_kubernetes_lora_missing_gcs_bucket_raises(self, tmp_path):
        import yaml
        from apprentice.config_loader import load_config, ConfigValidationError, ConfigError

        cfg_dict = self._make_valid_k8s_config()
        cfg_dict["finetuning"]["gcs_bucket"] = None
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg_dict))

        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_kubernetes_lora_missing_training_image_raises(self, tmp_path):
        import yaml
        from apprentice.config_loader import load_config, ConfigValidationError, ConfigError

        cfg_dict = self._make_valid_k8s_config()
        cfg_dict["finetuning"]["training_image"] = None
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg_dict))

        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_kubernetes_lora_missing_both_raises(self, tmp_path):
        import yaml
        from apprentice.config_loader import load_config, ConfigValidationError, ConfigError

        cfg_dict = self._make_valid_k8s_config()
        cfg_dict["finetuning"]["gcs_bucket"] = None
        cfg_dict["finetuning"]["training_image"] = None
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg_dict))

        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_kubernetes_lora_empty_string_gcs_bucket_raises(self, tmp_path):
        """Empty string should also fail validation."""
        import yaml
        from apprentice.config_loader import load_config, ConfigValidationError, ConfigError

        cfg_dict = self._make_valid_k8s_config()
        cfg_dict["finetuning"]["gcs_bucket"] = ""
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg_dict))

        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_kubernetes_lora_whitespace_gcs_bucket_raises(self, tmp_path):
        """Whitespace-only string should also fail validation."""
        import yaml
        from apprentice.config_loader import load_config, ConfigValidationError, ConfigError

        cfg_dict = self._make_valid_k8s_config()
        cfg_dict["finetuning"]["gcs_bucket"] = "   "
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg_dict))

        with pytest.raises((ConfigValidationError, ConfigError)):
            load_config(path, {})

    def test_non_k8s_backend_ignores_gcs_fields(self, tmp_path):
        """local_lora backend should not require gcs_bucket."""
        import yaml
        from apprentice.config_loader import load_config

        cfg_dict = self._make_valid_k8s_config()
        cfg_dict["finetuning"]["backend"] = "local_lora"
        cfg_dict["finetuning"].pop("gcs_bucket")
        cfg_dict["finetuning"].pop("training_image")
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg_dict))

        config = load_config(path, {})
        assert config.finetuning.backend.value == "local_lora"
        assert config.finetuning.gcs_bucket is None

    def test_kubernetes_lora_with_optional_fields(self, tmp_path):
        import yaml
        from apprentice.config_loader import load_config

        cfg_dict = self._make_valid_k8s_config()
        cfg_dict["finetuning"]["gpu_type"] = "nvidia-l4"
        cfg_dict["finetuning"]["k8s_namespace"] = "ml-training"
        cfg_dict["finetuning"]["service_account"] = "ml-sa"
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg_dict))

        config = load_config(path, {})
        assert config.finetuning.gpu_type == "nvidia-l4"
        assert config.finetuning.k8s_namespace == "ml-training"
        assert config.finetuning.service_account == "ml-sa"


# ============================================================================
# 12. Orchestrator config stub
# ============================================================================


class TestOrchestratorConfigStub:

    def test_kubernetes_lora_orchestrator_config_creates(self):
        cfg = KubernetesLoRAOrchestratorConfig()
        assert cfg.backend_type == "kubernetes_lora"

    def test_noop_orchestrator_config_creates(self):
        cfg = LocalNoOpOrchestratorConfig()
        assert cfg.backend_type == "noop"

    def test_orchestrator_config_accepts_kubernetes_lora(self):
        orch_cfg = OrchestratorConfig(
            backend=KubernetesLoRAOrchestratorConfig(),
            min_training_examples=10,
            model_version_store_path="/tmp/versions.json",
        )
        assert orch_cfg.backend.backend_type == "kubernetes_lora"

    def test_orchestrator_config_accepts_noop(self):
        orch_cfg = OrchestratorConfig(
            backend=LocalNoOpOrchestratorConfig(),
            min_training_examples=10,
            model_version_store_path="/tmp/versions.json",
        )
        assert orch_cfg.backend.backend_type == "noop"

    def test_orchestrator_wraps_backend(self, backend, tmp_path):
        store = ModelVersionStore(store_path=str(tmp_path / "versions.json"))
        orch = FineTuningOrchestrator(
            backend=backend,
            model_version_store=store,
            config=OrchestratorConfig(
                backend=KubernetesLoRAOrchestratorConfig(),
                min_training_examples=5,
                model_version_store_path=str(tmp_path / "versions.json"),
            ),
        )
        assert orch.backend is backend
        assert orch._consecutive_failures == 0

    def test_orchestrator_rejects_small_batch(self, backend, tmp_path):
        store = ModelVersionStore(store_path=str(tmp_path / "versions.json"))
        orch = FineTuningOrchestrator(
            backend=backend,
            model_version_store=store,
            config=OrchestratorConfig(
                backend=KubernetesLoRAOrchestratorConfig(),
                min_training_examples=100,
                model_version_store_path=str(tmp_path / "versions.json"),
            ),
        )
        examples = [
            TrainingExample(
                example_id=str(uuid.uuid4()),
                user_prompt="q",
                assistant_response="a",
                collected_at=datetime.now(timezone.utc).isoformat(),
                source="test",
            )
            for _ in range(5)
        ]
        result = orch.trigger_fine_tuning(examples, "llama3.1:8b")
        assert isinstance(result, FineTuneError)
        assert result.category == FineTuneErrorCategory.data_error


# ============================================================================
# 13. Cleanup
# ============================================================================


class TestCleanupJob:

    def test_cleanup_deletes_job(self, backend):
        mock_batch, mock_core = make_mock_k8s_clients()
        backend._k8s_batch_client = mock_batch
        backend._k8s_core_client = mock_core

        mock_k8s_client = MagicMock()
        mock_k8s_client.V1DeleteOptions = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"kubernetes": MagicMock(client=mock_k8s_client), "kubernetes.client": mock_k8s_client}):
            backend._cleanup_job("test-job", "test-run-id")

        mock_batch.delete_namespaced_job.assert_called_once()

    def test_cleanup_swallows_errors(self, backend):
        mock_batch, mock_core = make_mock_k8s_clients()
        mock_batch.delete_namespaced_job = MagicMock(side_effect=Exception("API error"))
        backend._k8s_batch_client = mock_batch
        backend._k8s_core_client = mock_core

        # Should not raise
        backend._cleanup_job("test-job", "test-run-id")


# ============================================================================
# 14. GCS cleanup
# ============================================================================


class TestCleanupGCSArtifacts:

    def test_cleanup_deletes_blobs(self, backend):
        mock_client, mock_bucket, mock_blob = make_mock_gcs_client()
        backend._gcs_client = mock_client

        mock_blob1 = MagicMock()
        mock_blob2 = MagicMock()
        mock_bucket.list_blobs = MagicMock(return_value=[mock_blob1, mock_blob2])

        backend.cleanup_gcs_artifacts("test-run-123")

        mock_blob1.delete.assert_called_once()
        mock_blob2.delete.assert_called_once()

    def test_cleanup_swallows_errors(self, backend):
        mock_client, mock_bucket, mock_blob = make_mock_gcs_client()
        mock_bucket.list_blobs = MagicMock(side_effect=Exception("GCS error"))
        backend._gcs_client = mock_client

        # Should not raise
        backend.cleanup_gcs_artifacts("test-run-123")
