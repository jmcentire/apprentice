"""
Remote API Client - Main module and public interface.

Async abstraction over commercial AI APIs (Anthropic, OpenAI, Google).
Implements retry with exponential backoff, cost tracking, and diversity sampling.
"""

import asyncio
import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class ProviderName(Enum):
    """Supported remote AI API providers."""
    anthropic = "anthropic"
    openai = "openai"
    google = "google"


class MessageRole(Enum):
    """Role of a message in a conversation prompt."""
    system = "system"
    user = "user"
    assistant = "assistant"


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────

class TokenUsage(BaseModel):
    """Frozen Pydantic model representing token counts for a single API request/response pair."""
    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)


class TokenPricing(BaseModel):
    """Frozen Pydantic model representing per-token pricing for a specific model."""
    model_config = ConfigDict(frozen=True)

    input_cost_per_token: float = Field(..., ge=0.0)
    output_cost_per_token: float = Field(..., ge=0.0)


class PricingTable(BaseModel):
    """Mapping of model identifiers to their token pricing."""
    entries: dict[str, TokenPricing] = Field(default_factory=dict)


class RetryConfig(BaseModel):
    """Configuration for exponential backoff retry behavior."""
    max_retries: int = Field(..., ge=0, le=10)
    base_delay_seconds: float = Field(..., gt=0.0)
    max_delay_seconds: float = Field(..., gt=0.0)
    respect_retry_after: bool = True


class TimeoutConfig(BaseModel):
    """Timeout settings for httpx async requests to the provider API."""
    connect_timeout_seconds: float = Field(..., gt=0.0)
    read_timeout_seconds: float = Field(..., gt=0.0)
    total_timeout_seconds: float = Field(..., gt=0.0)


class RemoteClientConfig(BaseModel):
    """Configuration for a single remote API client instance."""
    provider: ProviderName
    model: str = Field(..., min_length=1)
    api_key_env_var: str = Field(..., min_length=1)
    retry: RetryConfig = Field(default_factory=lambda: RetryConfig(
        max_retries=3, base_delay_seconds=1.0, max_delay_seconds=10.0, respect_retry_after=True
    ))
    timeout: TimeoutConfig = Field(default_factory=lambda: TimeoutConfig(
        connect_timeout_seconds=5.0, read_timeout_seconds=30.0, total_timeout_seconds=60.0
    ))
    pricing_overrides: PricingTable = Field(default_factory=PricingTable)


class TaskResponse(BaseModel):
    """Frozen Pydantic model representing a successful structured response."""
    model_config = ConfigDict(frozen=True)

    content: str
    model: str
    provider: ProviderName
    usage: TokenUsage
    cost_usd: float = Field(..., ge=0.0)
    latency_ms: float = Field(..., ge=0.0)
    request_id: str = Field(..., min_length=1)


class ErrorInfo(BaseModel):
    """Structured error information for secondary provider failures in diversity requests."""
    provider: ProviderName
    error_type: str
    message: str
    retries_attempted: int = Field(..., ge=0)


SecondaryResult = Union[TaskResponse, ErrorInfo]


class DiversityResult(BaseModel):
    """Result of a multi-provider diversity request."""
    primary: TaskResponse
    secondary: list[SecondaryResult]
    total_cost_usd: float = Field(..., ge=0.0)


class PromptMessage(BaseModel):
    """A single message in a prompt conversation."""
    role: MessageRole
    content: str = Field(..., min_length=1)


class GenerationParams(BaseModel):
    """Optional generation parameters that control the model's output behavior."""
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stop_sequences: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Error Types
# ──────────────────────────────────────────────────────────────────────────────

class RemoteClientError(Exception):
    """Base error type for all remote API client errors."""

    def __init__(
        self,
        provider: ProviderName,
        message: str,
        request_id: str,
        retries_attempted: int,
        status_code: int = 0,
    ):
        super().__init__(message)
        self.provider = provider
        self.message = message
        self.request_id = request_id
        self.retries_attempted = retries_attempted
        self.status_code = status_code


class AuthError(RemoteClientError):
    """Raised when authentication fails."""
    pass


class RateLimitError(RemoteClientError):
    """Raised when the provider returns HTTP 429 and all retries are exhausted."""

    def __init__(
        self,
        provider: ProviderName,
        message: str,
        request_id: str,
        retries_attempted: int,
        status_code: int,
        retry_after_seconds: float = 0.0,
    ):
        super().__init__(provider, message, request_id, retries_attempted, status_code)
        self.retry_after_seconds = retry_after_seconds


class RemoteUnavailableError(RemoteClientError):
    """Raised when the remote API is unreachable after all retries."""

    def __init__(
        self,
        provider: ProviderName,
        message: str,
        request_id: str,
        retries_attempted: int,
        status_code: int,
        last_error_type: Optional[str] = None,
    ):
        super().__init__(provider, message, request_id, retries_attempted, status_code)
        self.last_error_type = last_error_type


class InvalidRequestError(RemoteClientError):
    """Raised when the provider rejects the request due to client-side issues."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Pricing Defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PRICING_TABLES = {
    ProviderName.anthropic: {
        "claude-3-opus": TokenPricing(input_cost_per_token=0.000015, output_cost_per_token=0.000075),
        "claude-3-sonnet": TokenPricing(input_cost_per_token=0.000003, output_cost_per_token=0.000015),
        "claude-3-haiku": TokenPricing(input_cost_per_token=0.00000025, output_cost_per_token=0.00000125),
    },
    ProviderName.openai: {
        "gpt-4": TokenPricing(input_cost_per_token=0.00003, output_cost_per_token=0.00006),
        "gpt-3.5-turbo": TokenPricing(input_cost_per_token=0.0000015, output_cost_per_token=0.000002),
    },
    ProviderName.google: {
        "gemini-pro": TokenPricing(input_cost_per_token=0.000000125, output_cost_per_token=0.000000375),
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Pure Functions
# ──────────────────────────────────────────────────────────────────────────────

def compute_cost(usage: TokenUsage, pricing: TokenPricing) -> float:
    """Compute the USD cost of a single API request given token usage and pricing."""
    return (usage.input_tokens * pricing.input_cost_per_token +
            usage.output_tokens * pricing.output_cost_per_token)


def get_pricing(
    provider: ProviderName,
    model: str,
    pricing_overrides: PricingTable = PricingTable(entries={}),
) -> Optional[TokenPricing]:
    """Look up the TokenPricing for a given provider and model."""
    # Check overrides first
    if model in pricing_overrides.entries:
        return pricing_overrides.entries[model]

    # Fall back to defaults
    defaults = DEFAULT_PRICING_TABLES.get(provider, {})
    return defaults.get(model)


# ──────────────────────────────────────────────────────────────────────────────
# Base Client
# ──────────────────────────────────────────────────────────────────────────────

class BaseRemoteClient(ABC):
    """Base ABC for remote clients with shared retry/cost-tracking logic."""

    def __init__(
        self,
        config: RemoteClientConfig,
        api_key: SecretStr,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.config = config
        self._api_key = api_key
        self._owns_http_client = http_client is None

        if http_client is None:
            timeout = httpx.Timeout(
                connect=config.timeout.connect_timeout_seconds,
                read=config.timeout.read_timeout_seconds,
                pool=None,
                write=None,
            )
            self._http_client = httpx.AsyncClient(timeout=timeout)
        else:
            self._http_client = http_client

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.config.provider.value}, model={self.config.model})"

    def __repr__(self) -> str:
        return str(self)

    @abstractmethod
    async def _make_api_request(
        self,
        messages: list[PromptMessage],
        params: GenerationParams,
        request_id: str,
    ) -> TaskResponse:
        """Provider-specific API request implementation."""
        pass

    async def send_request(
        self,
        messages: list[PromptMessage],
        params: GenerationParams = GenerationParams(),
    ) -> TaskResponse:
        """Send a prompt to the configured remote AI provider."""
        # Validate messages
        if not messages:
            raise ValueError("messages must be non-empty")
        if not any(msg.role == MessageRole.user for msg in messages):
            raise ValueError("messages must contain at least one user message")

        request_id = str(uuid.uuid4())
        last_exception = None
        retry_count = 0

        for attempt in range(self.config.retry.max_retries + 1):
            try:
                start_time = time.perf_counter()
                response = await self._make_api_request(messages, params, request_id)
                return response

            except (AuthError, InvalidRequestError):
                # Non-retryable errors
                raise

            except RateLimitError as e:
                last_exception = e
                if attempt < self.config.retry.max_retries:
                    retry_count += 1
                    delay = self._calculate_retry_delay(attempt, e.retry_after_seconds)
                    await asyncio.sleep(delay)
                else:
                    e.retries_attempted = retry_count
                    raise

            except RemoteUnavailableError as e:
                last_exception = e
                if attempt < self.config.retry.max_retries:
                    retry_count += 1
                    delay = self._calculate_retry_delay(attempt)
                    await asyncio.sleep(delay)
                else:
                    e.retries_attempted = retry_count
                    raise

            except Exception as e:
                # Wrap unexpected errors as RemoteUnavailableError
                last_exception = RemoteUnavailableError(
                    provider=self.config.provider,
                    message=f"Unexpected error: {str(e)}",
                    request_id=request_id,
                    retries_attempted=retry_count,
                    status_code=0,
                    last_error_type=type(e).__name__,
                )
                if attempt < self.config.retry.max_retries:
                    retry_count += 1
                    delay = self._calculate_retry_delay(attempt)
                    await asyncio.sleep(delay)
                else:
                    raise last_exception

        # Should not reach here, but if we do, raise the last exception
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry logic error")

    def _calculate_retry_delay(self, attempt: int, retry_after: float = 0.0) -> float:
        """Calculate retry delay with full-jitter exponential backoff."""
        if retry_after > 0.0 and self.config.retry.respect_retry_after:
            return min(retry_after, self.config.retry.max_delay_seconds)

        max_delay = min(
            self.config.retry.max_delay_seconds,
            self.config.retry.base_delay_seconds * (2 ** attempt),
        )
        return random.uniform(0, max_delay)

    async def close(self) -> None:
        """Async cleanup method."""
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()


# ──────────────────────────────────────────────────────────────────────────────
# Anthropic Client
# ──────────────────────────────────────────────────────────────────────────────

class AnthropicClient(BaseRemoteClient):
    """Concrete implementation for Anthropic API."""

    async def _make_api_request(
        self,
        messages: list[PromptMessage],
        params: GenerationParams,
        request_id: str,
    ) -> TaskResponse:
        """Make a request to Anthropic API."""
        start_time = time.perf_counter()

        # Build request payload
        system_messages = [msg.content for msg in messages if msg.role == MessageRole.system]
        conversation_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
            if msg.role in (MessageRole.user, MessageRole.assistant)
        ]

        payload = {
            "model": self.config.model,
            "messages": conversation_messages,
            "max_tokens": params.max_tokens,
        }

        if system_messages:
            payload["system"] = " ".join(system_messages)

        if params.temperature != 0.0:
            payload["temperature"] = params.temperature

        if params.top_p != 1.0:
            payload["top_p"] = params.top_p

        if params.stop_sequences:
            payload["stop_sequences"] = params.stop_sequences

        headers = {
            "x-api-key": self._api_key.get_secret_value(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        try:
            response = await self._http_client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
            )

            elapsed = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                usage = TokenUsage(
                    input_tokens=data["usage"]["input_tokens"],
                    output_tokens=data["usage"]["output_tokens"],
                )

                # Extract content from response
                content_blocks = data.get("content", [])
                if isinstance(content_blocks, list):
                    # Filter for text blocks (may or may not have "type" field)
                    content = "".join([
                        block.get("text", "")
                        for block in content_blocks
                        if "text" in block and (block.get("type") == "text" or "type" not in block)
                    ])
                else:
                    content = str(content_blocks)

                pricing = get_pricing(self.config.provider, data["model"], self.config.pricing_overrides)
                cost = compute_cost(usage, pricing) if pricing else 0.0

                return TaskResponse(
                    content=content,
                    model=data["model"],
                    provider=self.config.provider,
                    usage=usage,
                    cost_usd=cost,
                    latency_ms=elapsed,
                    request_id=request_id,
                )

            elif response.status_code in (401, 403):
                raise AuthError(
                    provider=self.config.provider,
                    message=f"Authentication failed: {response.text}",
                    request_id=request_id,
                    retries_attempted=0,
                    status_code=response.status_code,
                )

            elif response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 0))
                raise RateLimitError(
                    provider=self.config.provider,
                    message=f"Rate limited: {response.text}",
                    request_id=request_id,
                    retries_attempted=0,
                    status_code=429,
                    retry_after_seconds=retry_after,
                )

            elif response.status_code in (400, 422):
                raise InvalidRequestError(
                    provider=self.config.provider,
                    message=f"Invalid request: {response.text}",
                    request_id=request_id,
                    retries_attempted=0,
                    status_code=response.status_code,
                )

            else:
                # 5xx errors
                raise RemoteUnavailableError(
                    provider=self.config.provider,
                    message=f"Remote unavailable: {response.text}",
                    request_id=request_id,
                    retries_attempted=0,
                    status_code=response.status_code,
                )

        except httpx.HTTPError as e:
            raise RemoteUnavailableError(
                provider=self.config.provider,
                message=f"HTTP error: {str(e)}",
                request_id=request_id,
                retries_attempted=0,
                status_code=0,
                last_error_type=type(e).__name__,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Factory Function
# ──────────────────────────────────────────────────────────────────────────────

def create_remote_client(
    config: RemoteClientConfig,
    http_client: Optional[httpx.AsyncClient] = None,
) -> BaseRemoteClient:
    """Factory function that constructs a concrete RemoteClient implementation."""
    # Read API key from environment
    api_key_value = os.environ.get(config.api_key_env_var)
    if not api_key_value:
        raise AuthError(
            provider=config.provider,
            message=f"API key environment variable {config.api_key_env_var} is not set or is empty",
            request_id="",
            retries_attempted=0,
            status_code=0,
        )

    if not api_key_value.strip():
        raise AuthError(
            provider=config.provider,
            message=f"API key environment variable {config.api_key_env_var} is empty",
            request_id="",
            retries_attempted=0,
            status_code=0,
        )

    api_key = SecretStr(api_key_value)

    if config.provider == ProviderName.anthropic:
        return AnthropicClient(config, api_key, http_client)
    elif config.provider == ProviderName.openai:
        raise ValueError(f"Provider {config.provider.value} not yet implemented")
    elif config.provider == ProviderName.google:
        raise ValueError(f"Provider {config.provider.value} not yet implemented")
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


# ──────────────────────────────────────────────────────────────────────────────
# Diversity Request
# ──────────────────────────────────────────────────────────────────────────────

async def send_diversity_request(
    primary_client: BaseRemoteClient,
    secondary_clients: list[BaseRemoteClient],
    messages: list[PromptMessage],
    params: GenerationParams = GenerationParams(),
) -> DiversityResult:
    """Send the same prompt to a primary provider and zero or more secondary providers."""
    # Validate messages
    if not messages:
        raise ValueError("messages must be non-empty")
    if not any(msg.role == MessageRole.user for msg in messages):
        raise ValueError("messages must contain at least one user message")

    # Fan out requests concurrently
    async def call_primary():
        return await primary_client.send_request(messages, params)

    async def call_secondary(client):
        try:
            return await client.send_request(messages, params)
        except RemoteClientError as e:
            return ErrorInfo(
                provider=e.provider,
                error_type=type(e).__name__,
                message=e.message,
                retries_attempted=e.retries_attempted,
            )
        except Exception as e:
            return ErrorInfo(
                provider=client.config.provider,
                error_type=type(e).__name__,
                message=str(e),
                retries_attempted=0,
            )

    # Execute all requests concurrently
    tasks = [call_primary()] + [call_secondary(c) for c in secondary_clients]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Extract primary result
    primary_result = results[0]
    if isinstance(primary_result, Exception):
        raise primary_result

    # Extract secondary results
    secondary_results = results[1:]

    # Compute total cost
    total_cost = primary_result.cost_usd
    for result in secondary_results:
        if isinstance(result, TaskResponse):
            total_cost += result.cost_usd

    return DiversityResult(
        primary=primary_result,
        secondary=secondary_results,
        total_cost_usd=total_cost,
    )
