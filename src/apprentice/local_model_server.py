"""
Local Model Server Client - Async abstraction over local inference servers.

Implements a unified interface for Ollama, vLLM, and llama.cpp backends.
All operational failures are returned as values, never raised as exceptions.
"""
import asyncio
import logging
import time
import uuid
from enum import Enum
from typing import Optional, Union

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ── ENUMS ──────────────────────────────────────────────


class UnavailableReason(str, Enum):
    """Discriminated reason why the local model server cannot fulfill a request."""
    server_not_running = "server_not_running"
    no_model_loaded = "no_model_loaded"
    model_loading = "model_loading"
    inference_timeout = "inference_timeout"
    inference_error = "inference_error"


class PromotionStatus(str, Enum):
    """Outcome status of a model promotion attempt."""
    success = "success"
    smoke_test_failed = "smoke_test_failed"
    load_failed = "load_failed"
    server_unavailable = "server_unavailable"
    already_active = "already_active"
    lock_timeout = "lock_timeout"


class LocalBackendType(str, Enum):
    """Supported local inference server backend types."""
    ollama = "ollama"
    vllm = "vllm"
    llamacpp = "llamacpp"


# ── DATA MODELS ────────────────────────────────────────


class InferenceRequest(BaseModel):
    """Structured request for local model inference."""
    prompt: str = Field(..., min_length=1)
    model_id: Optional[str] = None
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stop_sequences: list = Field(default_factory=list)
    request_id: Optional[str] = None


class InferenceSuccess(BaseModel):
    """Successful inference result from any local backend."""
    result_type: str = Field(default="success")
    text: str
    model_id: str
    tokens_generated: int = Field(ge=0)
    tokens_prompt: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    backend: LocalBackendType
    request_id: str

    @field_validator('result_type')
    @classmethod
    def validate_result_type(cls, v):
        if v != 'success':
            raise ValueError("result_type must be 'success'")
        return v


class LocalModelUnavailable(BaseModel):
    """Structured unavailability result - NOT an exception."""
    result_type: str = Field(default="unavailable")
    reason: UnavailableReason
    detail: Optional[str] = None
    backend: LocalBackendType
    request_id: str
    latency_ms: float = Field(ge=0.0)

    @field_validator('result_type')
    @classmethod
    def validate_result_type(cls, v):
        if v != 'unavailable':
            raise ValueError("result_type must be 'unavailable'")
        return v


InferenceResult = Union[InferenceSuccess, LocalModelUnavailable]


class HealthStatus(BaseModel):
    """Structured health check result for a local inference server."""
    is_alive: bool
    is_ready: bool
    loaded_model: Optional[str] = None
    backend: LocalBackendType
    reason: Optional[str] = None
    latency_ms: float = Field(ge=0.0)


class PromotionResult(BaseModel):
    """Result of a model promotion attempt."""
    status: PromotionStatus
    previous_model_id: Optional[str] = None
    new_model_id: str
    is_active: bool
    smoke_test_passed: bool = False
    smoke_test_response: Optional[str] = None
    detail: Optional[str] = None
    duration_ms: float = Field(ge=0.0)


# ── CONFIG MODELS ──────────────────────────────────────


class BaseClientConfig(BaseModel):
    """Common configuration fields shared by all local model client backends."""
    base_url: str = Field(..., pattern=r'^https?://.+')
    connect_timeout_s: float = Field(default=5.0, ge=0.1, le=60.0)
    inference_timeout_s: float = Field(default=120.0, ge=1.0, le=600.0)
    health_check_timeout_s: float = Field(default=3.0, ge=0.1, le=30.0)
    promotion_lock_timeout_s: float = Field(default=300.0, ge=0.01, le=900.0)


class OllamaClientConfig(BaseClientConfig):
    """Configuration for the Ollama backend client."""
    base_url: str = Field(default="http://localhost:11434", pattern=r'^https?://.+')
    keep_alive: str = Field(default="5m")
    num_gpu_layers: int = Field(default=-1, ge=-1)


class VLLMClientConfig(BaseClientConfig):
    """Configuration for the vLLM backend client."""
    base_url: str = Field(default="http://localhost:8000", pattern=r'^https?://.+')
    api_key: Optional[str] = None
    model_name_override: Optional[str] = None


class LlamaCppClientConfig(BaseClientConfig):
    """Configuration for the llama.cpp server backend client."""
    base_url: str = Field(default="http://localhost:8080", pattern=r'^https?://.+')
    n_predict: int = Field(default=512, ge=1, le=32768)
    slot_id: int = Field(default=-1)


# ── CLIENT IMPLEMENTATIONS ─────────────────────────────


class OllamaClient:
    """Ollama backend client implementation."""

    def __init__(self, config: OllamaClientConfig, http_client: Optional[httpx.AsyncClient] = None):
        self.config = config
        self._external_client = http_client
        self._client: Optional[httpx.AsyncClient] = http_client
        self._active_model: str = ""
        self._promotion_lock = asyncio.Lock()

    async def __aenter__(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout_s,
                    read=self.config.inference_timeout_s,
                    write=10.0,
                    pool=5.0
                )
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client is not None and self._external_client is None:
            await self._client.aclose()

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Send an inference request to the Ollama server."""
        start_time = time.time()

        # Validate request type (programming error)
        if not isinstance(request, InferenceRequest):
            raise TypeError(f"request must be InferenceRequest, got {type(request).__name__}")

        # Generate request_id if not provided
        req_id = request.request_id if request.request_id else str(uuid.uuid4())

        try:
            # Prepare the request payload
            payload = {
                "model": request.model_id or self._active_model,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "num_predict": request.max_tokens,
                    "temperature": request.temperature,
                }
            }

            if request.stop_sequences:
                payload["options"]["stop"] = request.stop_sequences

            response = await self._client.post(
                "/api/generate",
                json=payload,
                timeout=self.config.inference_timeout_s
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                return InferenceSuccess(
                    result_type="success",
                    text=data.get("response", ""),
                    model_id=data.get("model", request.model_id or self._active_model),
                    tokens_generated=data.get("eval_count", 0),
                    tokens_prompt=data.get("prompt_eval_count", 0),
                    latency_ms=latency_ms,
                    backend=LocalBackendType.ollama,
                    request_id=req_id
                )
            else:
                # HTTP error status
                return LocalModelUnavailable(
                    result_type="unavailable",
                    reason=UnavailableReason.inference_error,
                    detail=f"HTTP {response.status_code}: {response.text}",
                    backend=LocalBackendType.ollama,
                    request_id=req_id,
                    latency_ms=latency_ms
                )

        except httpx.ReadTimeout:
            latency_ms = (time.time() - start_time) * 1000
            return LocalModelUnavailable(
                result_type="unavailable",
                reason=UnavailableReason.inference_timeout,
                detail="Read timeout during inference",
                backend=LocalBackendType.ollama,
                request_id=req_id,
                latency_ms=latency_ms
            )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            latency_ms = (time.time() - start_time) * 1000
            return LocalModelUnavailable(
                result_type="unavailable",
                reason=UnavailableReason.server_not_running,
                detail="Cannot connect to Ollama server",
                backend=LocalBackendType.ollama,
                request_id=req_id,
                latency_ms=latency_ms
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.exception("Unexpected error during inference")
            return LocalModelUnavailable(
                result_type="unavailable",
                reason=UnavailableReason.inference_error,
                detail=f"Unexpected error: {str(e)}",
                backend=LocalBackendType.ollama,
                request_id=req_id,
                latency_ms=latency_ms
            )

    async def health(self) -> HealthStatus:
        """Perform a health check against the Ollama server."""
        start_time = time.time()

        try:
            response = await self._client.get(
                "/api/tags",
                timeout=self.config.health_check_timeout_s
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])

                is_alive = True
                is_ready = len(models) > 0
                loaded_model = models[0]["name"] if models else ""

                return HealthStatus(
                    is_alive=is_alive,
                    is_ready=is_ready,
                    loaded_model=loaded_model,
                    backend=LocalBackendType.ollama,
                    reason=None if is_ready else "No models loaded",
                    latency_ms=latency_ms
                )
            else:
                latency_ms = (time.time() - start_time) * 1000
                return HealthStatus(
                    is_alive=False,
                    is_ready=False,
                    loaded_model="",
                    backend=LocalBackendType.ollama,
                    reason=f"HTTP {response.status_code}",
                    latency_ms=latency_ms
                )

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
            latency_ms = (time.time() - start_time) * 1000
            return HealthStatus(
                is_alive=False,
                is_ready=False,
                loaded_model="",
                backend=LocalBackendType.ollama,
                reason="Server unreachable",
                latency_ms=latency_ms
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.exception("Unexpected error during health check")
            return HealthStatus(
                is_alive=False,
                is_ready=False,
                loaded_model="",
                backend=LocalBackendType.ollama,
                reason=f"Error: {str(e)}",
                latency_ms=latency_ms
            )

    async def promote_model(
        self,
        new_model_id: str,
        smoke_test_prompt: str,
        expected_substring: Optional[str] = None
    ) -> PromotionResult:
        """Perform a blue-green model swap with smoke testing."""
        start_time = time.time()

        # Validate inputs (programming errors)
        if not isinstance(new_model_id, str):
            raise TypeError("new_model_id must be a string")
        if not new_model_id or not new_model_id.strip():
            raise ValueError("new_model_id must be a non-empty string")
        if not isinstance(smoke_test_prompt, str) or not smoke_test_prompt.strip():
            raise ValueError("smoke_test_prompt must be a non-empty string")

        new_model_id = new_model_id.strip()
        smoke_test_prompt = smoke_test_prompt.strip()

        # Try to acquire the promotion lock
        try:
            acquired = await asyncio.wait_for(
                self._promotion_lock.acquire(),
                timeout=self.config.promotion_lock_timeout_s
            )
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return PromotionResult(
                status=PromotionStatus.lock_timeout,
                previous_model_id=self._active_model,
                new_model_id=new_model_id,
                is_active=False,
                smoke_test_passed=False,
                detail="Another promotion is in progress",
                duration_ms=duration_ms
            )

        try:
            previous_model = self._active_model

            # Check if already active
            if new_model_id == self._active_model:
                duration_ms = (time.time() - start_time) * 1000
                return PromotionResult(
                    status=PromotionStatus.already_active,
                    previous_model_id=previous_model,
                    new_model_id=new_model_id,
                    is_active=True,
                    smoke_test_passed=False,
                    detail="Model is already active",
                    duration_ms=duration_ms
                )

            # Step 1: Load the new model (pull)
            try:
                response = await self._client.post(
                    "/api/pull",
                    json={"name": new_model_id},
                    timeout=self.config.inference_timeout_s
                )

                if response.status_code != 200:
                    duration_ms = (time.time() - start_time) * 1000
                    return PromotionResult(
                        status=PromotionStatus.load_failed,
                        previous_model_id=previous_model,
                        new_model_id=new_model_id,
                        is_active=False,
                        smoke_test_passed=False,
                        detail=f"Failed to pull model: HTTP {response.status_code}",
                        duration_ms=duration_ms
                    )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
                duration_ms = (time.time() - start_time) * 1000
                return PromotionResult(
                    status=PromotionStatus.server_unavailable,
                    previous_model_id=previous_model,
                    new_model_id=new_model_id,
                    is_active=False,
                    smoke_test_passed=False,
                    detail="Server unreachable during model load",
                    duration_ms=duration_ms
                )
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                return PromotionResult(
                    status=PromotionStatus.load_failed,
                    previous_model_id=previous_model,
                    new_model_id=new_model_id,
                    is_active=False,
                    smoke_test_passed=False,
                    detail=f"Error loading model: {str(e)}",
                    duration_ms=duration_ms
                )

            # Step 2: Run smoke test
            smoke_test_request = InferenceRequest(
                prompt=smoke_test_prompt,
                model_id=new_model_id,
                max_tokens=100,
                temperature=0.7,
                request_id=f"smoke-test-{uuid.uuid4()}"
            )

            smoke_result = await self.infer(smoke_test_request)

            if isinstance(smoke_result, LocalModelUnavailable):
                duration_ms = (time.time() - start_time) * 1000
                return PromotionResult(
                    status=PromotionStatus.load_failed,
                    previous_model_id=previous_model,
                    new_model_id=new_model_id,
                    is_active=False,
                    smoke_test_passed=False,
                    smoke_test_response="",
                    detail=f"Smoke test failed: {smoke_result.reason}",
                    duration_ms=duration_ms
                )

            # Check if expected substring is in the response
            smoke_test_response = smoke_result.text
            if expected_substring and expected_substring not in smoke_test_response:
                duration_ms = (time.time() - start_time) * 1000
                return PromotionResult(
                    status=PromotionStatus.smoke_test_failed,
                    previous_model_id=previous_model,
                    new_model_id=new_model_id,
                    is_active=False,
                    smoke_test_passed=False,
                    smoke_test_response=smoke_test_response,
                    detail=f"Expected substring '{expected_substring}' not found in response",
                    duration_ms=duration_ms
                )

            # Step 3: Swap active model
            self._active_model = new_model_id

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Successfully promoted model from '{previous_model}' to '{new_model_id}'")

            return PromotionResult(
                status=PromotionStatus.success,
                previous_model_id=previous_model,
                new_model_id=new_model_id,
                is_active=True,
                smoke_test_passed=True,
                smoke_test_response=smoke_test_response,
                detail=None,
                duration_ms=duration_ms
            )

        finally:
            self._promotion_lock.release()

    async def get_active_model(self) -> str:
        """Return the identifier of the currently active model."""
        return self._active_model


# ── ADDITIONAL CLIENT IMPLEMENTATIONS ──────────────────


class VLLMClient:
    """vLLM backend client — uses OpenAI-compatible /v1/completions API."""

    def __init__(self, config: VLLMClientConfig, http_client: Optional[httpx.AsyncClient] = None):
        self.config = config
        self._external_client = http_client
        self._active_model = config.model_name_override or ""
        self._promotion_lock = asyncio.Lock()

    def _get_client(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        return httpx.AsyncClient(timeout=httpx.Timeout(
            connect=self.config.connect_timeout_s,
            read=self.config.inference_timeout_s,
            pool=None, write=None,
        ))

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Send prompt to vLLM via OpenAI-compatible chat/completions endpoint."""
        client = self._get_client()
        try:
            model = self._active_model or request.model
            if not model:
                return LocalModelUnavailable(reason=UnavailableReason.no_model_loaded)

            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }

            start = time.perf_counter()
            resp = await client.post(f"{self.config.base_url}/v1/chat/completions", json=payload, headers=headers)
            elapsed = (time.perf_counter() - start) * 1000

            if resp.status_code != 200:
                return LocalModelUnavailable(reason=UnavailableReason.inference_error)

            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})

            return InferenceSuccess(
                content=content,
                model=data.get("model", model),
                tokens_generated=usage.get("completion_tokens", 0),
                duration_ms=elapsed,
            )
        except httpx.ConnectError:
            return LocalModelUnavailable(reason=UnavailableReason.server_not_running)
        except httpx.ReadTimeout:
            return LocalModelUnavailable(reason=UnavailableReason.inference_timeout)
        finally:
            if self._external_client is None:
                await client.aclose()

    async def health(self) -> HealthStatus:
        """Check vLLM health via /health endpoint."""
        client = self._get_client()
        try:
            resp = await client.get(f"{self.config.base_url}/health",
                                    timeout=self.config.health_check_timeout_s)
            return HealthStatus(
                is_ready=resp.status_code == 200,
                active_model=self._active_model,
                server_version="vllm",
            )
        except Exception:
            return HealthStatus(is_ready=False, active_model="", server_version="vllm")
        finally:
            if self._external_client is None:
                await client.aclose()

    async def promote_model(self, model_tag: str, smoke_test_prompt: str = "Hello") -> PromotionResult:
        """vLLM model promotion — update active model name (vLLM loads models at startup)."""
        async with self._promotion_lock:
            old = self._active_model
            self._active_model = model_tag
            # Smoke test
            result = await self.infer(InferenceRequest(prompt=smoke_test_prompt, model=model_tag))
            if isinstance(result, InferenceSuccess):
                return PromotionResult(
                    status=PromotionStatus.success,
                    previous_model=old,
                    new_model=model_tag,
                    smoke_test_response=result.content[:200],
                )
            else:
                self._active_model = old
                return PromotionResult(
                    status=PromotionStatus.smoke_test_failed,
                    previous_model=old,
                    new_model=model_tag,
                    smoke_test_response="",
                )

    async def get_active_model(self) -> str:
        return self._active_model


class LlamaCppClient:
    """llama.cpp server backend client — uses /completion endpoint."""

    def __init__(self, config: LlamaCppClientConfig, http_client: Optional[httpx.AsyncClient] = None):
        self.config = config
        self._external_client = http_client
        self._active_model = ""
        self._promotion_lock = asyncio.Lock()

    def _get_client(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        return httpx.AsyncClient(timeout=httpx.Timeout(
            connect=self.config.connect_timeout_s,
            read=self.config.inference_timeout_s,
            pool=None, write=None,
        ))

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Send prompt to llama.cpp server via /completion endpoint."""
        client = self._get_client()
        try:
            payload = {
                "prompt": request.prompt,
                "n_predict": self.config.n_predict,
                "temperature": request.temperature,
            }
            if self.config.slot_id >= 0:
                payload["slot_id"] = self.config.slot_id

            start = time.perf_counter()
            resp = await client.post(f"{self.config.base_url}/completion", json=payload)
            elapsed = (time.perf_counter() - start) * 1000

            if resp.status_code != 200:
                return LocalModelUnavailable(reason=UnavailableReason.inference_error)

            data = resp.json()
            content = data.get("content", "")
            tokens = data.get("tokens_predicted", 0)

            return InferenceSuccess(
                content=content,
                model=self._active_model or "llama.cpp",
                tokens_generated=tokens,
                duration_ms=elapsed,
            )
        except httpx.ConnectError:
            return LocalModelUnavailable(reason=UnavailableReason.server_not_running)
        except httpx.ReadTimeout:
            return LocalModelUnavailable(reason=UnavailableReason.inference_timeout)
        finally:
            if self._external_client is None:
                await client.aclose()

    async def health(self) -> HealthStatus:
        """Check llama.cpp health via /health endpoint."""
        client = self._get_client()
        try:
            resp = await client.get(f"{self.config.base_url}/health",
                                    timeout=self.config.health_check_timeout_s)
            data = resp.json() if resp.status_code == 200 else {}
            return HealthStatus(
                is_ready=data.get("status") == "ok" if data else resp.status_code == 200,
                active_model=self._active_model,
                server_version="llama.cpp",
            )
        except Exception:
            return HealthStatus(is_ready=False, active_model="", server_version="llama.cpp")
        finally:
            if self._external_client is None:
                await client.aclose()

    async def promote_model(self, model_tag: str, smoke_test_prompt: str = "Hello") -> PromotionResult:
        """llama.cpp promotion — the server loads one model at startup, so this just updates the label."""
        async with self._promotion_lock:
            old = self._active_model
            self._active_model = model_tag
            result = await self.infer(InferenceRequest(prompt=smoke_test_prompt, model=model_tag))
            if isinstance(result, InferenceSuccess):
                return PromotionResult(
                    status=PromotionStatus.success,
                    previous_model=old,
                    new_model=model_tag,
                    smoke_test_response=result.content[:200],
                )
            else:
                self._active_model = old
                return PromotionResult(
                    status=PromotionStatus.smoke_test_failed,
                    previous_model=old,
                    new_model=model_tag,
                    smoke_test_response="",
                )

    async def get_active_model(self) -> str:
        return self._active_model


# ── Auto-injected export aliases (Pact export gate) ──
health = HealthStatus
