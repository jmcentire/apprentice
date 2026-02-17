"""Kubernetes LoRA Fine-Tuning Backend.

Submits K8s Jobs with GPU nodes to run QLoRA fine-tuning via Unsloth.
Stores training data and GGUF artifacts in GCS. Downloads the GGUF,
registers it with Ollama via /api/create, and returns a real ModelVersion
so the existing validate -> promote pipeline works.

Requires optional dependencies: ``pip install apprentice[gke]``
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from apprentice.fine_tuning_orchestrator import (
    FineTuneBackend,
    FineTuneError,
    FineTuneErrorCategory,
    FineTuneRequest,
    FineTuneResult,
    ModelVersion,
    TrainingMetrics,
)

logger = logging.getLogger("apprentice.kubernetes_lora")


# ============================================================================
# Configuration
# ============================================================================


class KubernetesLoRABackendConfig(BaseModel):
    """Configuration for the Kubernetes LoRA fine-tuning backend."""

    backend_type: Literal["kubernetes_lora"] = "kubernetes_lora"

    # GCS
    gcs_bucket: str = Field(..., min_length=1)
    gcs_prefix: str = "training-runs"

    # K8s Job
    training_image: str = Field(..., min_length=1)
    namespace: str = "default"
    gpu_type: str = "nvidia-tesla-t4"
    gpu_count: int = Field(default=1, ge=1, le=8)
    gpu_resource_key: str = "nvidia.com/gpu"
    node_pool_label: str = ""
    spot_tolerations: bool = True
    job_timeout_seconds: int = Field(default=7200, ge=60, le=86400)
    poll_interval_seconds: int = Field(default=30, ge=5, le=600)
    service_account_name: str = ""
    backoff_limit: int = Field(default=2, ge=0, le=10)
    image_pull_policy: str = "Always"

    # K8s resource limits (CPU/memory for the training container)
    cpu_request: str = "2"
    cpu_limit: str = "8"
    memory_request: str = "8Gi"
    memory_limit: str = "32Gi"

    # Training hyperparameters
    base_model: str = "unsloth/llama-3.1-8b-bnb-4bit"
    quantization_type: str = "Q4_K_M"
    max_seq_length: int = Field(default=2048, ge=128, le=32768)
    lora_rank: int = Field(default=16, ge=1, le=256)
    learning_rate: float = 2e-4
    num_epochs: int = Field(default=3, ge=1, le=100)

    # Ollama registration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model_name_prefix: str = "apprentice"
    ollama_create_timeout_seconds: int = Field(default=600, ge=30, le=3600)
    ollama_create_retries: int = Field(default=3, ge=0, le=10)
    local_gguf_dir: str = ".apprentice/models/gguf"

    # Safety
    min_disk_free_bytes: int = Field(default=5_000_000_000, ge=0)  # 5 GB


# ============================================================================
# Retry helper
# ============================================================================


def _retry(fn, retries: int = 3, base_delay: float = 2.0, operation: str = "operation"):
    """Retry a callable with exponential backoff. Returns the result or re-raises."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "[run:?] %s failed (attempt %d/%d): %s. Retrying in %.1fs",
                    operation, attempt, retries, e, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "[run:?] %s failed after %d attempts: %s",
                    operation, retries, e,
                )
    raise last_exc


# ============================================================================
# Backend Implementation
# ============================================================================


class KubernetesLoRABackend(FineTuneBackend):
    """Fine-tuning backend that runs QLoRA training as K8s Jobs on GPU nodes.

    The workflow:
      1. Upload training JSONL to GCS
      2. Create a K8s Job targeting a GPU node
      3. Poll until the Job completes
      4. Download the GGUF model artifact from GCS
      5. Download training metrics from GCS
      6. Register the GGUF with Ollama via /api/create
      7. Return a ModelVersion for the validate -> promote pipeline
    """

    def __init__(self, config: KubernetesLoRABackendConfig):
        self.config = config
        self._gcs_client: Any = None
        self._k8s_batch_client: Any = None
        self._k8s_core_client: Any = None

    # ── Public interface ─────────────────────────────────────────────────

    def fine_tune(self, request: FineTuneRequest) -> FineTuneResult:
        """Execute a fine-tuning run via K8s Job. Synchronous — intended to be
        called via ``asyncio.to_thread()`` from the pipeline loop."""
        run_id = request.run_id
        started_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "[run:%s] Starting fine-tune: %d examples, base_model=%s",
            run_id, len(request.training_examples), request.base_model,
        )

        try:
            # 1. Upload training data to GCS
            gcs_data_path = self._upload_training_data(request)

            # 2. Create K8s Job
            job_name = self._create_training_job(request, gcs_data_path)

            # 3. Poll until complete
            self._wait_for_job(job_name, run_id)

            # 4. Download GGUF from GCS
            gguf_local_path = self._download_gguf(run_id)

            # 5. Download metrics from GCS
            metrics = self._download_metrics(run_id)

            # 6. Register with Ollama
            model_name = self._register_ollama(run_id, gguf_local_path)

            # 7. Clean up K8s Job
            self._cleanup_job(job_name, run_id)

            completed_at = datetime.now(timezone.utc).isoformat()

            logger.info(
                "[run:%s] Fine-tune complete: model=%s, artifact=%s",
                run_id, model_name, gguf_local_path,
            )

            return ModelVersion(
                status="success",
                version_id=f"k8s-lora-{run_id}",
                base_model=request.base_model,
                backend_type="kubernetes_lora",
                training_example_ids=[e.example_id for e in request.training_examples],
                training_example_count=len(request.training_examples),
                run_id=run_id,
                started_at=started_at,
                completed_at=completed_at,
                artifact_location=gguf_local_path,
                ollama_model_name=model_name,
                metrics=metrics,
                quantization_type=self.config.quantization_type,
                metadata={
                    "gcs_bucket": self.config.gcs_bucket,
                    "gcs_prefix": self.config.gcs_prefix,
                    "training_image": self.config.training_image,
                    "gpu_type": self.config.gpu_type,
                    "base_model_id": self.config.base_model,
                },
            )

        except Exception as e:
            category = self._classify_error(e)
            logger.error(
                "[run:%s] Fine-tune failed (%s): %s",
                run_id, category.value, e,
            )
            return FineTuneError(
                status="error",
                category=category,
                message=str(e),
                retry_recommended=category == FineTuneErrorCategory.transient,
                run_id=run_id,
                backend_type="kubernetes_lora",
                occurred_at=datetime.now(timezone.utc).isoformat(),
                details={
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                },
            )

    # ── GCS helpers ──────────────────────────────────────────────────────

    def _get_gcs_client(self) -> Any:
        if self._gcs_client is None:
            try:
                from google.cloud import storage  # lazy import
                self._gcs_client = storage.Client()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize GCS client for bucket "
                    f"'{self.config.gcs_bucket}'. Check credentials and "
                    f"google-cloud-storage installation: {e}"
                ) from e
        return self._gcs_client

    def _gcs_path(self, run_id: str, filename: str) -> str:
        return f"{self.config.gcs_prefix}/{run_id}/{filename}"

    def _upload_training_data(self, request: FineTuneRequest) -> str:
        """Format training examples as chat-template JSONL and upload to GCS."""
        run_id = request.run_id
        client = self._get_gcs_client()
        bucket = client.bucket(self.config.gcs_bucket)

        lines = []
        for ex in request.training_examples:
            messages = []
            if ex.system_prompt:
                messages.append({"role": "system", "content": ex.system_prompt})
            messages.append({"role": "user", "content": ex.user_prompt})
            messages.append({"role": "assistant", "content": ex.assistant_response})
            lines.append(json.dumps({"messages": messages}))

        data = "\n".join(lines)
        blob_path = self._gcs_path(run_id, "data.jsonl")
        blob = bucket.blob(blob_path)

        logger.info(
            "[run:%s] Uploading %d examples (%.1f KB) to gs://%s/%s",
            run_id, len(lines), len(data) / 1024,
            self.config.gcs_bucket, blob_path,
        )

        def _upload():
            blob.upload_from_string(data, content_type="application/jsonl")

        _retry(_upload, retries=3, operation=f"GCS upload gs://{self.config.gcs_bucket}/{blob_path}")

        return f"gs://{self.config.gcs_bucket}/{blob_path}"

    def _download_gguf(self, run_id: str) -> str:
        """Download the GGUF model artifact from GCS to local disk."""
        client = self._get_gcs_client()
        bucket = client.bucket(self.config.gcs_bucket)

        blob_path = self._gcs_path(run_id, "model.gguf")
        blob = bucket.blob(blob_path)

        # Verify the blob exists before downloading
        if not blob.exists():
            raise FileNotFoundError(
                f"GGUF artifact not found at gs://{self.config.gcs_bucket}/{blob_path}. "
                f"Training job may have failed to upload."
            )

        blob.reload()  # fetch metadata including size
        blob_size = blob.size or 0

        logger.info(
            "[run:%s] Downloading GGUF (%.1f MB) from gs://%s/%s",
            run_id, blob_size / (1024 * 1024),
            self.config.gcs_bucket, blob_path,
        )

        # Check available disk space
        local_dir = Path(self.config.local_gguf_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        disk_usage = shutil.disk_usage(str(local_dir))
        if blob_size > 0 and disk_usage.free < blob_size + self.config.min_disk_free_bytes:
            raise OSError(
                f"Insufficient disk space: GGUF is {blob_size / (1024**3):.1f} GB, "
                f"available is {disk_usage.free / (1024**3):.1f} GB "
                f"(min free threshold: {self.config.min_disk_free_bytes / (1024**3):.1f} GB)"
            )

        local_path = local_dir / f"{run_id}.gguf"

        def _download():
            blob.download_to_filename(str(local_path))

        _retry(_download, retries=3, operation=f"GGUF download {blob_path}")

        actual_size = local_path.stat().st_size
        logger.info("[run:%s] GGUF downloaded: %s (%.1f MB)", run_id, local_path, actual_size / (1024 * 1024))
        return str(local_path)

    def _download_metrics(self, run_id: str) -> Optional[TrainingMetrics]:
        """Download metrics.json from GCS and parse into TrainingMetrics."""
        client = self._get_gcs_client()
        bucket = client.bucket(self.config.gcs_bucket)

        blob_path = self._gcs_path(run_id, "metrics.json")
        blob = bucket.blob(blob_path)

        try:
            raw = json.loads(blob.download_as_text())
            metrics = TrainingMetrics(
                final_loss=raw.get("final_loss"),
                num_steps=raw.get("num_steps"),
                num_epochs_completed=raw.get("num_epochs_completed"),
                training_duration_seconds=raw.get("training_duration_seconds"),
                additional_metrics=raw.get("additional_metrics", {}),
            )
            logger.info(
                "[run:%s] Training metrics: loss=%.4f, steps=%s, duration=%ss",
                run_id, metrics.final_loss or 0, metrics.num_steps, metrics.training_duration_seconds,
            )
            return metrics
        except Exception as e:
            logger.warning("[run:%s] Could not download metrics: %s", run_id, e)
            return None

    def cleanup_gcs_artifacts(self, run_id: str) -> None:
        """Delete training artifacts from GCS after successful promotion.

        Call this after a model has been validated and promoted to avoid
        accumulating GCS storage costs.
        """
        try:
            client = self._get_gcs_client()
            bucket = client.bucket(self.config.gcs_bucket)
            prefix = f"{self.config.gcs_prefix}/{run_id}/"
            blobs = list(bucket.list_blobs(prefix=prefix))
            for blob in blobs:
                blob.delete()
            logger.info("[run:%s] Cleaned up %d GCS artifacts at %s", run_id, len(blobs), prefix)
        except Exception as e:
            logger.warning("[run:%s] GCS artifact cleanup failed: %s", run_id, e)

    # ── K8s helpers ──────────────────────────────────────────────────────

    def _get_k8s_clients(self) -> tuple[Any, Any]:
        if self._k8s_batch_client is None:
            try:
                from kubernetes import client, config  # lazy import
                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    config.load_kube_config()
                self._k8s_batch_client = client.BatchV1Api()
                self._k8s_core_client = client.CoreV1Api()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize K8s client. Check kubeconfig or "
                    f"in-cluster service account: {e}"
                ) from e
        return self._k8s_batch_client, self._k8s_core_client

    def _create_training_job(self, request: FineTuneRequest, gcs_data_path: str) -> str:
        """Create a K8s Job spec and submit it."""
        from kubernetes import client as k8s  # lazy import

        batch_api, _ = self._get_k8s_clients()
        cfg = self.config

        job_name = f"apprentice-ft-{request.run_id[:8]}"

        # Environment variables for the training container
        env_vars = [
            k8s.V1EnvVar(name="GCS_BUCKET", value=cfg.gcs_bucket),
            k8s.V1EnvVar(name="GCS_PREFIX", value=cfg.gcs_prefix),
            k8s.V1EnvVar(name="RUN_ID", value=request.run_id),
            k8s.V1EnvVar(name="BASE_MODEL", value=cfg.base_model),
            k8s.V1EnvVar(name="QUANTIZATION_TYPE", value=cfg.quantization_type),
            k8s.V1EnvVar(name="MAX_SEQ_LENGTH", value=str(cfg.max_seq_length)),
            k8s.V1EnvVar(name="LORA_RANK", value=str(cfg.lora_rank)),
            k8s.V1EnvVar(name="LEARNING_RATE", value=str(cfg.learning_rate)),
            k8s.V1EnvVar(name="NUM_EPOCHS", value=str(cfg.num_epochs)),
        ]

        # Resource requirements — GPU, CPU, and memory
        resources = k8s.V1ResourceRequirements(
            requests={
                cfg.gpu_resource_key: str(cfg.gpu_count),
                "cpu": cfg.cpu_request,
                "memory": cfg.memory_request,
            },
            limits={
                cfg.gpu_resource_key: str(cfg.gpu_count),
                "cpu": cfg.cpu_limit,
                "memory": cfg.memory_limit,
            },
        )

        # Container
        container = k8s.V1Container(
            name="trainer",
            image=cfg.training_image,
            image_pull_policy=cfg.image_pull_policy,
            env=env_vars,
            resources=resources,
        )

        # Node selector
        node_selector = {
            "cloud.google.com/gke-accelerator": cfg.gpu_type,
        }
        if cfg.node_pool_label:
            node_selector["cloud.google.com/gke-nodepool"] = cfg.node_pool_label

        # Tolerations for spot/preemptible nodes
        tolerations = []
        if cfg.spot_tolerations:
            tolerations.append(k8s.V1Toleration(
                key="cloud.google.com/gke-spot",
                operator="Equal",
                value="true",
                effect="NoSchedule",
            ))

        # Pod spec
        pod_spec = k8s.V1PodSpec(
            containers=[container],
            restart_policy="Never",
            node_selector=node_selector,
            tolerations=tolerations if tolerations else None,
        )
        if cfg.service_account_name:
            pod_spec.service_account_name = cfg.service_account_name

        # Job spec
        job = k8s.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=k8s.V1ObjectMeta(
                name=job_name,
                namespace=cfg.namespace,
                labels={
                    "app": "apprentice-trainer",
                    "run-id": request.run_id,
                },
            ),
            spec=k8s.V1JobSpec(
                template=k8s.V1PodTemplateSpec(spec=pod_spec),
                backoff_limit=cfg.backoff_limit,
                active_deadline_seconds=cfg.job_timeout_seconds,
            ),
        )

        batch_api.create_namespaced_job(namespace=cfg.namespace, body=job)
        logger.info(
            "[run:%s] Created K8s Job '%s' (image=%s, gpu=%s×%d, backoff=%d)",
            request.run_id, job_name, cfg.training_image,
            cfg.gpu_type, cfg.gpu_count, cfg.backoff_limit,
        )
        return job_name

    def _wait_for_job(self, job_name: str, run_id: str = "") -> None:
        """Poll K8s Job status until complete, failed, or timed out."""
        batch_api, _ = self._get_k8s_clients()
        cfg = self.config

        deadline = time.monotonic() + cfg.job_timeout_seconds
        poll_count = 0
        log_every = max(1, 120 // cfg.poll_interval_seconds)  # log every ~2 min

        while time.monotonic() < deadline:
            job = batch_api.read_namespaced_job_status(
                name=job_name, namespace=cfg.namespace,
            )
            status = job.status
            poll_count += 1

            if status.succeeded and status.succeeded >= 1:
                elapsed = cfg.job_timeout_seconds - (deadline - time.monotonic())
                logger.info(
                    "[run:%s] Job '%s' succeeded after %d polls (%.0fs)",
                    run_id, job_name, poll_count, elapsed,
                )
                return

            if status.failed and status.failed >= 1:
                log_snippet = self._get_job_logs(job_name)
                events = self._get_pod_events(job_name)
                raise RuntimeError(
                    f"Training job '{job_name}' failed. "
                    f"Events: {events}. Logs: {log_snippet}"
                )

            if poll_count % log_every == 0:
                elapsed = cfg.job_timeout_seconds - (deadline - time.monotonic())
                active = getattr(status, 'active', None) or 0
                logger.info(
                    "[run:%s] Job '%s' still running: active=%s, elapsed=%.0fs, polls=%d",
                    run_id, job_name, active, elapsed, poll_count,
                )

            time.sleep(cfg.poll_interval_seconds)

        raise TimeoutError(
            f"Training job '{job_name}' timed out after {cfg.job_timeout_seconds}s "
            f"({poll_count} polls)"
        )

    def _get_job_logs(self, job_name: str) -> str:
        """Best-effort retrieval of pod logs from a failed Job."""
        _, core_api = self._get_k8s_clients()
        cfg = self.config

        try:
            pods = core_api.list_namespaced_pod(
                namespace=cfg.namespace,
                label_selector=f"job-name={job_name}",
            )
            if pods.items:
                log = core_api.read_namespaced_pod_log(
                    name=pods.items[0].metadata.name,
                    namespace=cfg.namespace,
                    tail_lines=50,
                )
                return log[:2000]
        except Exception:
            pass
        return "(logs unavailable)"

    def _get_pod_events(self, job_name: str) -> str:
        """Best-effort retrieval of pod events (OOMKilled, ImagePullBackOff, etc.)."""
        _, core_api = self._get_k8s_clients()
        cfg = self.config

        try:
            pods = core_api.list_namespaced_pod(
                namespace=cfg.namespace,
                label_selector=f"job-name={job_name}",
            )
            if not pods.items:
                return "(no pods found)"

            pod_name = pods.items[0].metadata.name
            events = core_api.list_namespaced_event(
                namespace=cfg.namespace,
                field_selector=f"involvedObject.name={pod_name}",
            )
            if events.items:
                summaries = []
                for ev in events.items[-10:]:
                    summaries.append(f"{ev.reason}: {ev.message}")
                return "; ".join(summaries)[:1000]
        except Exception:
            pass
        return "(events unavailable)"

    def _cleanup_job(self, job_name: str, run_id: str = "") -> None:
        """Delete completed K8s Job and its pods."""
        batch_api, _ = self._get_k8s_clients()
        cfg = self.config

        try:
            from kubernetes import client as k8s
            batch_api.delete_namespaced_job(
                name=job_name,
                namespace=cfg.namespace,
                body=k8s.V1DeleteOptions(propagation_policy="Background"),
            )
            logger.info("[run:%s] Cleaned up K8s Job '%s'", run_id, job_name)
        except Exception as e:
            logger.warning("[run:%s] K8s Job cleanup failed for '%s': %s", run_id, job_name, e)

    # ── Ollama registration ──────────────────────────────────────────────

    def _register_ollama(self, run_id: str, gguf_path: str) -> str:
        """Register GGUF model with Ollama via /api/create with retry."""
        import httpx

        model_name = f"{self.config.ollama_model_name_prefix}:{run_id[:12]}"
        modelfile = f"FROM {gguf_path}\n"
        url = f"{self.config.ollama_base_url}/api/create"
        timeout = float(self.config.ollama_create_timeout_seconds)

        logger.info("[run:%s] Registering model '%s' with Ollama at %s", run_id, model_name, url)

        def _create():
            response = httpx.post(
                url,
                json={"name": model_name, "modelfile": modelfile},
                timeout=httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=10.0),
            )
            response.raise_for_status()

        _retry(
            _create,
            retries=self.config.ollama_create_retries,
            base_delay=5.0,
            operation=f"Ollama create '{model_name}'",
        )

        logger.info("[run:%s] Model '%s' registered with Ollama", run_id, model_name)
        return model_name

    # ── Error classification ─────────────────────────────────────────────

    def _classify_error(self, exc: Exception) -> FineTuneErrorCategory:
        """Map exceptions to FineTuneErrorCategory for orchestrator retry logic."""
        exc_type = type(exc).__name__
        msg = str(exc).lower()

        # Timeouts and spot preemptions are transient
        if isinstance(exc, TimeoutError):
            return FineTuneErrorCategory.transient
        if "preempt" in msg or "spot" in msg or "timeout" in msg:
            return FineTuneErrorCategory.transient

        # K8s API errors (quota, scheduling) are usually transient
        if "apiexception" in exc_type.lower() or "409" in msg or "429" in msg:
            return FineTuneErrorCategory.transient

        # OOM / CUDA errors are infrastructure
        if "oom" in msg or "cuda" in msg or "out of memory" in msg or "gpu" in msg:
            return FineTuneErrorCategory.infrastructure

        # Disk space
        if "disk" in msg or "no space" in msg or "insufficient" in msg:
            return FineTuneErrorCategory.infrastructure

        # Data format errors
        if "json" in msg or "format" in msg or "parse" in msg:
            return FineTuneErrorCategory.data_error

        # Default to infrastructure (retriable)
        return FineTuneErrorCategory.infrastructure
