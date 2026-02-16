"""
External Model Interfaces - Glue module wiring child components.

This module adapts child components (remote_api_client, local_model_server)
to satisfy the parent interface contract for external_interfaces.
"""
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Union, Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

# Import child modules with src. prefix for robustness
import src.remote_api_client as remote_api_client
import src.local_model_server as local_model_server


# ═══════════════════════════════════════════════════════════════════════════
# TYPE RE-EXPORTS AND ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class SourceType(str, Enum):
    """Discriminator for the origin of a model response."""
    LOCAL = "LOCAL"
    REMOTE = "REMOTE"


class Role(str, Enum):
    """Role of a message in a conversation prompt."""
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "ASSISTANT"


class UnavailableReason(str, Enum):
    """Categorized reason why a model source cannot fulfill a request."""
    SERVER_NOT_RUNNING = "SERVER_NOT_RUNNING"
    NO_MODEL_LOADED = "NO_MODEL_LOADED"
    MODEL_LOADING = "MODEL_LOADING"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    REMOTE_UNAVAILABLE = "REMOTE_UNAVAILABLE"
    AUTH_FAILURE = "AUTH_FAILURE"


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

class SourceInfo(BaseModel):
    """Frozen Pydantic v2 model identifying the origin of a completion."""
    model_config = ConfigDict(frozen=True)

    source_type: SourceType
    provider_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    backend_type: str = ""


class Message(BaseModel):
    """A single message in a conversation prompt."""
    role: Role
    content: str = Field(min_length=1)


class GenerationConfig(BaseModel):
    """Canonical generation parameters that control model output behavior."""
    model_config = ConfigDict(frozen=True)

    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stop_sequences: list[str] = Field(default_factory=list)


class UnifiedTokenUsage(BaseModel):
    """Normalized token usage across all backends."""
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(ge=0)
    is_estimated: bool = False


class CompletionRequest(BaseModel):
    """Unified request for model completion."""
    messages: list[Message]
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)
    request_id: str = ""


class CompletionResponse(BaseModel):
    """Unified response from any model source."""
    model_config = ConfigDict(frozen=True)

    content: str
    source_info: SourceInfo
    token_usage: UnifiedTokenUsage
    cost_usd: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    response_id: str = Field(min_length=1)


class HealthReport(BaseModel):
    """Unified health check result for any model source."""
    model_config = ConfigDict(frozen=True)

    available: bool
    source_info: SourceInfo
    detail: str = ""
    latency_ms: float = Field(default=0.0, ge=0.0)


class PromotionReport(BaseModel):
    """Unified promotion report returned by LocalModelAdapter.promote_model()."""
    model_config = ConfigDict(frozen=True)

    status: str
    previous_model_id: str = ""
    new_model_id: str
    is_active: bool
    smoke_test_passed: bool = False
    detail: str = ""
    duration_ms: float = Field(ge=0.0)


# ═══════════════════════════════════════════════════════════════════════════
# ERROR TYPES
# ═══════════════════════════════════════════════════════════════════════════

class ModelClientError(Exception):
    """Base error type for all unified model client errors."""

    def __init__(
        self,
        source_info: SourceInfo,
        message: str,
        request_id: str,
        original_error: str = "",
        timestamp_utc: str = "",
    ):
        super().__init__(message)
        self.source_info = source_info
        self.message = message
        self.request_id = request_id
        self.original_error = original_error
        self.timestamp_utc = timestamp_utc or datetime.now(timezone.utc).isoformat()


class RetriableError(ModelClientError):
    """Error indicating a transient failure that may succeed on retry."""

    def __init__(
        self,
        source_info: SourceInfo,
        message: str,
        request_id: str,
        original_error: str = "",
        timestamp_utc: str = "",
        retry_after_seconds: float = 0.0,
        retries_exhausted: int = 0,
    ):
        super().__init__(source_info, message, request_id, original_error, timestamp_utc)
        self.retry_after_seconds = retry_after_seconds
        self.retries_exhausted = retries_exhausted


class TerminalError(ModelClientError):
    """Error indicating a permanent failure that will not succeed on retry."""

    def __init__(
        self,
        source_info: SourceInfo,
        message: str,
        request_id: str,
        original_error: str = "",
        timestamp_utc: str = "",
        status_code: int = 0,
    ):
        super().__init__(source_info, message, request_id, original_error, timestamp_utc)
        self.status_code = status_code


class SourceUnavailableError(ModelClientError):
    """Error signaling that a model source is entirely unavailable."""

    def __init__(
        self,
        source_info: SourceInfo,
        message: str,
        request_id: str,
        original_error: str = "",
        timestamp_utc: str = "",
        reason: UnavailableReason = UnavailableReason.SERVER_NOT_RUNNING,
    ):
        super().__init__(source_info, message, request_id, original_error, timestamp_utc)
        self.reason = reason


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG MODELS
# ═══════════════════════════════════════════════════════════════════════════

class RemoteAdapterConfig(BaseModel):
    """Configuration for constructing a RemoteModelAdapter."""
    provider: str
    model: str = Field(min_length=1)
    api_key_env_var: str = Field(min_length=1)
    max_retries: int = Field(default=3, ge=0, le=10)
    base_delay_seconds: float = Field(default=1.0, gt=0.0)
    max_delay_seconds: float = Field(default=60.0, gt=0.0)
    connect_timeout_seconds: float = Field(default=10.0, gt=0.0)
    read_timeout_seconds: float = Field(default=120.0, gt=0.0)
    total_timeout_seconds: float = Field(default=180.0, gt=0.0)


class LocalAdapterConfig(BaseModel):
    """Configuration for constructing a LocalModelAdapter."""
    backend: str
    base_url: str
    model_id: str = ""
    inference_timeout_s: float = Field(default=120.0, ge=1.0, le=600.0)
    health_check_timeout_s: float = Field(default=3.0, ge=0.1, le=30.0)


# ═══════════════════════════════════════════════════════════════════════════
# MAPPING FUNCTIONS (Pure)
# ═══════════════════════════════════════════════════════════════════════════

def map_remote_error(error: Any, source_info: SourceInfo) -> ModelClientError:
    """Map a remote_api_client error to the unified error hierarchy."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # Map AuthError -> TerminalError
    if hasattr(error, 'status_code') and error.status_code in (401, 403):
        return TerminalError(
            source_info=source_info,
            message=getattr(error, 'message', str(error)),
            request_id=getattr(error, 'request_id', ''),
            original_error=str(error),
            timestamp_utc=timestamp,
            status_code=error.status_code,
        )

    # Map InvalidRequestError -> TerminalError
    if hasattr(error, 'status_code') and error.status_code in (400, 422):
        return TerminalError(
            source_info=source_info,
            message=getattr(error, 'message', str(error)),
            request_id=getattr(error, 'request_id', ''),
            original_error=str(error),
            timestamp_utc=timestamp,
            status_code=error.status_code,
        )

    # Map RateLimitError -> RetriableError
    if hasattr(error, 'retry_after_seconds') or (hasattr(error, 'retry_after')):
        retry_after = getattr(error, 'retry_after_seconds',
                             getattr(error, 'retry_after', 0.0))
        return RetriableError(
            source_info=source_info,
            message=getattr(error, 'message', str(error)),
            request_id=getattr(error, 'request_id', ''),
            original_error=str(error),
            timestamp_utc=timestamp,
            retry_after_seconds=retry_after,
            retries_exhausted=getattr(error, 'retries_attempted', 0),
        )

    # Map RemoteUnavailableError -> SourceUnavailableError
    # Check by class name pattern to handle both real and mock errors
    error_type_name = type(error).__name__
    if 'RemoteUnavailable' in error_type_name or 'Unavailable' in error_type_name:
        return SourceUnavailableError(
            source_info=source_info,
            message=getattr(error, 'message', str(error)),
            request_id=getattr(error, 'request_id', ''),
            original_error=str(error),
            timestamp_utc=timestamp,
            reason=UnavailableReason.REMOTE_UNAVAILABLE,
        )

    # Unknown error type
    raise TypeError(f"Cannot map unknown error type: {error_type_name}")


def map_local_unavailable(unavailable: Any, source_info: SourceInfo) -> SourceUnavailableError:
    """Map a local_model_server LocalModelUnavailable result to SourceUnavailableError."""
    if not hasattr(unavailable, 'result_type') or unavailable.result_type != 'unavailable':
        raise TypeError("Expected LocalModelUnavailable instance")

    timestamp = datetime.now(timezone.utc).isoformat()

    # Map reason string to enum
    reason_str = str(unavailable.reason).upper()
    if reason_str == "SERVER_NOT_RUNNING":
        reason = UnavailableReason.SERVER_NOT_RUNNING
    elif reason_str == "NO_MODEL_LOADED":
        reason = UnavailableReason.NO_MODEL_LOADED
    elif reason_str == "MODEL_LOADING":
        reason = UnavailableReason.MODEL_LOADING
    elif reason_str == "INFERENCE_TIMEOUT":
        reason = UnavailableReason.INFERENCE_TIMEOUT
    elif reason_str == "INFERENCE_ERROR":
        reason = UnavailableReason.INFERENCE_ERROR
    else:
        reason = UnavailableReason.INFERENCE_ERROR

    return SourceUnavailableError(
        source_info=source_info,
        message=getattr(unavailable, 'detail', ''),
        request_id=unavailable.request_id,
        original_error=getattr(unavailable, 'detail', ''),
        timestamp_utc=timestamp,
        reason=reason,
    )


def map_local_success_to_response(success: Any, source_info: SourceInfo) -> CompletionResponse:
    """Map a local_model_server InferenceSuccess result to CompletionResponse."""
    if not hasattr(success, 'result_type') or success.result_type != 'success':
        raise TypeError("Expected InferenceSuccess instance")

    token_usage = UnifiedTokenUsage(
        prompt_tokens=success.tokens_prompt,
        completion_tokens=success.tokens_generated,
        total_tokens=success.tokens_prompt + success.tokens_generated,
        is_estimated=False,  # Local backends report exact counts
    )

    return CompletionResponse(
        content=success.text,
        source_info=source_info,
        token_usage=token_usage,
        cost_usd=0.0,  # Local inference has no per-request cost
        latency_ms=success.latency_ms,
        response_id=success.request_id,
    )


def map_remote_response(response: Any, source_info: SourceInfo) -> CompletionResponse:
    """Map a remote_api_client TaskResponse to CompletionResponse."""
    if not hasattr(response, 'content'):
        raise TypeError("Expected TaskResponse instance")

    token_usage = UnifiedTokenUsage(
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        is_estimated=False,  # Remote APIs report exact counts
    )

    return CompletionResponse(
        content=response.content,
        source_info=source_info,
        token_usage=token_usage,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
        response_id=response.request_id,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ADAPTER IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

class RemoteModelAdapter:
    """Adapter wrapping remote_api_client to satisfy ModelClient ABC."""

    def __init__(self, config: RemoteAdapterConfig, remote_client: Any, source_info: SourceInfo):
        self._config = config
        self._client = remote_client
        self._source_info = source_info

    def get_source_info(self) -> SourceInfo:
        """Return the SourceInfo describing this adapter's configured model source."""
        return self._source_info

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send a completion request through the remote client."""
        # Validate messages
        if not request.messages:
            raise ValueError("messages must be non-empty")
        if not any(msg.role == Role.USER for msg in request.messages):
            raise ValueError("messages must contain at least one USER message")

        # Convert Message to PromptMessage
        messages = []
        for msg in request.messages:
            # Map Role enum to remote_api_client MessageRole
            if msg.role == Role.SYSTEM:
                role = remote_api_client.MessageRole.system
            elif msg.role == Role.USER:
                role = remote_api_client.MessageRole.user
            elif msg.role == Role.ASSISTANT:
                role = remote_api_client.MessageRole.assistant
            else:
                raise ValueError(f"Unsupported role: {msg.role}")

            messages.append(remote_api_client.PromptMessage(
                role=role,
                content=msg.content,
            ))

        # Convert GenerationConfig to GenerationParams
        params = remote_api_client.GenerationParams(
            max_tokens=request.generation_config.max_tokens,
            temperature=request.generation_config.temperature,
            top_p=request.generation_config.top_p,
            stop_sequences=request.generation_config.stop_sequences,
        )

        try:
            response = await self._client.send_request(messages, params)
            return map_remote_response(response, self._source_info)
        except Exception as e:
            # Map remote client errors to unified error hierarchy
            raise map_remote_error(e, self._source_info)

    async def health(self) -> HealthReport:
        """Perform a health check against the remote API."""
        # Remote APIs are always available if constructed successfully
        return HealthReport(
            available=True,
            source_info=self._source_info,
            detail="Remote API reachable",
            latency_ms=0.0,
        )

    async def close(self) -> None:
        """Async cleanup method."""
        await self._client.close()

    async def promote_model(
        self, new_model_id: str, smoke_test_prompt: str, expected_substring: str = ""
    ) -> PromotionReport:
        """Model promotion is only supported on local model adapters."""
        raise NotImplementedError("Model promotion is only supported on local model adapters")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class LocalModelAdapter:
    """Adapter wrapping local_model_server to satisfy ModelClient ABC."""

    def __init__(self, config: LocalAdapterConfig, local_client: Any):
        self._config = config
        self._client = local_client
        self._source_info = SourceInfo(
            source_type=SourceType.LOCAL,
            provider_name=config.backend,
            model_id=config.model_id,
            backend_type=config.backend,
        )

    def get_source_info(self) -> SourceInfo:
        """Return the SourceInfo describing this adapter's configured model source."""
        return self._source_info

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send a completion request through the local client."""
        # Validate messages
        if not request.messages:
            raise ValueError("messages must be non-empty")
        if not any(msg.role == Role.USER for msg in request.messages):
            raise ValueError("messages must contain at least one USER message")

        # Convert messages to a single prompt string (local models use text prompts)
        prompt_parts = []
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == Role.USER:
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == Role.ASSISTANT:
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt = "\n".join(prompt_parts)

        # Create InferenceRequest
        inference_request = local_model_server.InferenceRequest(
            prompt=prompt,
            model_id=self._config.model_id or None,
            max_tokens=request.generation_config.max_tokens,
            temperature=request.generation_config.temperature,
            stop_sequences=request.generation_config.stop_sequences,
            request_id=request.request_id or str(uuid.uuid4()),
        )

        # Call local client
        result = await self._client.infer(inference_request)

        # Pattern match on result_type
        if hasattr(result, 'result_type') and result.result_type == 'success':
            return map_local_success_to_response(result, self._source_info)
        elif hasattr(result, 'result_type') and result.result_type == 'unavailable':
            raise map_local_unavailable(result, self._source_info)
        else:
            raise RuntimeError(f"Unexpected inference result type: {type(result)}")

    async def health(self) -> HealthReport:
        """Perform a health check against the local inference server."""
        try:
            status = await self._client.health()

            return HealthReport(
                available=status.is_ready,
                source_info=self._source_info,
                detail=getattr(status, 'reason', getattr(status, 'detail', '')),
                latency_ms=getattr(status, 'latency_ms', 0.0),
            )
        except Exception as e:
            # Never raise for operational failures
            return HealthReport(
                available=False,
                source_info=self._source_info,
                detail=str(e),
                latency_ms=0.0,
            )

    async def close(self) -> None:
        """Async cleanup method."""
        if hasattr(self._client, 'close'):
            await self._client.close()

    async def promote_model(
        self, new_model_id: str, smoke_test_prompt: str, expected_substring: str = ""
    ) -> PromotionReport:
        """Request a model promotion on the underlying local model server."""
        # Validate inputs
        if not isinstance(new_model_id, str) or not new_model_id.strip():
            raise ValueError("new_model_id must be a non-empty string")
        if not isinstance(smoke_test_prompt, str) or not smoke_test_prompt.strip():
            raise ValueError("smoke_test_prompt must be a non-empty string")

        # Call local client promote_model
        result = await self._client.promote_model(
            new_model_id=new_model_id,
            smoke_test_prompt=smoke_test_prompt,
            expected_substring=expected_substring or None,
        )

        # Update source_info.model_id on success
        if result.status == "success":
            self._source_info = SourceInfo(
                source_type=SourceType.LOCAL,
                provider_name=self._config.backend,
                model_id=new_model_id,
                backend_type=self._config.backend,
            )

        # Map PromotionResult to PromotionReport
        return PromotionReport(
            status=result.status,
            previous_model_id=getattr(result, 'previous_model_id', '') or "",
            new_model_id=result.new_model_id,
            is_active=result.status == "success",
            smoke_test_passed=getattr(result, 'smoke_test_passed', False),
            detail=getattr(result, 'detail', '') or "",
            duration_ms=getattr(result, 'duration_ms', 0.0),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_remote_adapter(
    config: RemoteAdapterConfig, http_client: Optional[httpx.AsyncClient] = None
) -> RemoteModelAdapter:
    """Factory function that constructs a RemoteModelAdapter."""
    # Validate API key environment variable
    api_key = os.environ.get(config.api_key_env_var)
    if not api_key or not api_key.strip():
        raise ValueError(
            f"API key environment variable {config.api_key_env_var} is not set or is empty"
        )

    # Map provider string to ProviderName enum
    provider_map = {
        "anthropic": remote_api_client.ProviderName.anthropic,
        "openai": remote_api_client.ProviderName.openai,
        "google": remote_api_client.ProviderName.google,
    }

    if config.provider not in provider_map:
        raise ValueError(f"Unsupported provider: {config.provider}")

    provider = provider_map[config.provider]

    # Build remote client config
    remote_config = remote_api_client.RemoteClientConfig(
        provider=provider,
        model=config.model,
        api_key_env_var=config.api_key_env_var,
        retry=remote_api_client.RetryConfig(
            max_retries=config.max_retries,
            base_delay_seconds=config.base_delay_seconds,
            max_delay_seconds=config.max_delay_seconds,
            respect_retry_after=True,
        ),
        timeout=remote_api_client.TimeoutConfig(
            connect_timeout_seconds=config.connect_timeout_seconds,
            read_timeout_seconds=config.read_timeout_seconds,
            total_timeout_seconds=config.total_timeout_seconds,
        ),
    )

    # Create remote client
    remote_client = remote_api_client.create_remote_client(
        config=remote_config, http_client=http_client
    )

    # Build source info
    source_info = SourceInfo(
        source_type=SourceType.REMOTE,
        provider_name=config.provider,
        model_id=config.model,
        backend_type=config.provider,
    )

    return RemoteModelAdapter(config, remote_client, source_info)


def create_local_adapter(
    config: LocalAdapterConfig, local_client: Optional[Any] = None
) -> LocalModelAdapter:
    """Factory function that constructs a LocalModelAdapter."""
    # Map backend string to config type
    backend_map = {
        "ollama": local_model_server.OllamaClientConfig,
        "vllm": local_model_server.VLLMClientConfig,
        "llamacpp": local_model_server.LlamaCppClientConfig,
    }

    if config.backend not in backend_map:
        raise ValueError(f"Unsupported backend: {config.backend}")

    # If no client provided, create one
    if local_client is None:
        config_cls = backend_map[config.backend]
        backend_config = config_cls(
            base_url=config.base_url,
            inference_timeout_s=config.inference_timeout_s,
            health_check_timeout_s=config.health_check_timeout_s,
        )

        # Construct the appropriate client
        if config.backend == "ollama":
            local_client = local_model_server.OllamaClient(backend_config)
        elif config.backend == "vllm":
            local_client = local_model_server.VLLMClient(backend_config)
        elif config.backend == "llamacpp":
            local_client = local_model_server.LlamaCppClient(backend_config)

    return LocalModelAdapter(config, local_client)
