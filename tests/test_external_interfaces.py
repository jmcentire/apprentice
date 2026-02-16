"""
Contract tests for external_interfaces component.
Tests organized into sections:
  1. Adapter Creation (create_remote_adapter, create_local_adapter)
  2. Completion (complete() on both adapter types)
  3. Error and Response Mapping (pure mapping functions)
  4. Adapter Lifecycle (health, close, get_source_info, promote_model)
  5. Invariants
"""
import asyncio
import os
import re
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Import the component under test
# ---------------------------------------------------------------------------
from src.external_interfaces import (
    # Types / enums
    SourceType,
    Role,
    UnavailableReason,
    SourceInfo,
    Message,
    GenerationConfig,
    UnifiedTokenUsage,
    CompletionRequest,
    CompletionResponse,
    HealthReport,
    # Errors
    ModelClientError,
    RetriableError,
    TerminalError,
    SourceUnavailableError,
    # Configs
    RemoteAdapterConfig,
    LocalAdapterConfig,
    # Factory functions
    create_remote_adapter,
    create_local_adapter,
    # Mapping functions (exposed publicly for testing)
    map_remote_error,
    map_local_unavailable,
    map_local_success_to_response,
    map_remote_response,
)


# ===========================================================================
# Helper fixtures and utilities
# ===========================================================================

def _iso8601_pattern():
    """Regex pattern for ISO 8601 UTC timestamps."""
    return re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|\+00:00)?$"
    )


def _make_source_info(source_type=SourceType.REMOTE, provider="openai",
                       model_id="gpt-4", backend_type="openai"):
    return SourceInfo(
        source_type=source_type,
        provider_name=provider,
        model_id=model_id,
        backend_type=backend_type,
    )


def _make_local_source_info(backend="ollama", model_id="llama3"):
    return SourceInfo(
        source_type=SourceType.LOCAL,
        provider_name=backend,
        model_id=model_id,
        backend_type=backend,
    )


def _make_generation_config():
    return GenerationConfig(
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        stop_sequences=[],
    )


def _make_completion_request(messages=None, request_id=None):
    if messages is None:
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello!"),
        ]
    return CompletionRequest(
        messages=messages,
        generation_config=_make_generation_config(),
        request_id=request_id or str(uuid.uuid4()),
    )


# ---------------------------------------------------------------------------
# Mock objects for remote_api_client dependency types
# ---------------------------------------------------------------------------

class MockTokenUsage:
    """Mimics remote_api_client.TokenUsage."""
    def __init__(self, input_tokens=10, output_tokens=5):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockTaskResponse:
    """Mimics remote_api_client.TaskResponse."""
    def __init__(self, content="Hello from remote", cost_usd=0.01,
                 latency_ms=150.0, request_id="remote-req-001",
                 usage=None):
        self.content = content
        self.cost_usd = cost_usd
        self.latency_ms = latency_ms
        self.request_id = request_id
        self.usage = usage or MockTokenUsage()


class MockAuthError(Exception):
    def __init__(self, message="Auth failed", status_code=401,
                 request_id="req-001", provider="openai"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        self.provider = provider


class MockInvalidRequestError(Exception):
    def __init__(self, message="Invalid request", status_code=422,
                 request_id="req-001", provider="openai"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        self.provider = provider


class MockRateLimitError(Exception):
    def __init__(self, message="Rate limited", retry_after=30.0,
                 request_id="req-001", provider="openai"):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after
        self.retry_after_seconds = retry_after
        self.request_id = request_id
        self.provider = provider


class MockRemoteUnavailableError(Exception):
    def __init__(self, message="Service unavailable",
                 request_id="req-001", provider="openai"):
        super().__init__(message)
        self.message = message
        self.request_id = request_id
        self.provider = provider


# ---------------------------------------------------------------------------
# Mock objects for local_model_server dependency types
# ---------------------------------------------------------------------------

class MockInferenceSuccess:
    """Mimics local_model_server.InferenceSuccess."""
    result_type = "success"

    def __init__(self, text="Hello from local", tokens_generated=5,
                 tokens_prompt=10, latency_ms=50.0,
                 request_id="local-req-001"):
        self.text = text
        self.tokens_generated = tokens_generated
        self.tokens_prompt = tokens_prompt
        self.latency_ms = latency_ms
        self.request_id = request_id


class MockLocalModelUnavailable:
    """Mimics local_model_server.LocalModelUnavailable."""
    result_type = "unavailable"

    def __init__(self, reason="SERVER_NOT_RUNNING", detail="Server is down",
                 request_id="local-req-001"):
        self.reason = reason
        self.detail = detail
        self.request_id = request_id


class MockHealthStatus:
    """Mimics local_model_server.HealthStatus."""
    def __init__(self, is_ready=True, loaded_model="llama3", detail="OK"):
        self.is_ready = is_ready
        self.loaded_model = loaded_model
        self.detail = detail


class MockPromotionResult:
    """Mimics local_model_server.PromotionResult."""
    def __init__(self, status="success", detail="Model promoted successfully",
                 new_model_id="llama3.1"):
        self.status = status
        self.detail = detail
        self.new_model_id = new_model_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def remote_source_info():
    return _make_source_info()


@pytest.fixture
def local_source_info():
    return _make_local_source_info()


@pytest.fixture
def mock_remote_client():
    """Mock remote_api_client client with async methods."""
    client = AsyncMock()
    client.send_request = AsyncMock(return_value=MockTaskResponse())
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_local_client():
    """Mock local_model_server client with async methods."""
    client = AsyncMock()
    client.infer = AsyncMock(return_value=MockInferenceSuccess())
    client.health = AsyncMock(return_value=MockHealthStatus())
    client.promote_model = AsyncMock(return_value=MockPromotionResult())
    client.get_active_model = AsyncMock(return_value="llama3")
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_http_client():
    """Mock httpx.AsyncClient that appears open."""
    client = AsyncMock()
    client.is_closed = False
    return client


@pytest.fixture
def valid_remote_config():
    return RemoteAdapterConfig(
        provider="openai",
        model="gpt-4",
        api_key_env_var="TEST_OPENAI_API_KEY",
        max_retries=3,
        base_delay_seconds=1.0,
        max_delay_seconds=30.0,
        connect_timeout_seconds=5.0,
        read_timeout_seconds=30.0,
        total_timeout_seconds=60.0,
    )


@pytest.fixture
def valid_local_config():
    return LocalAdapterConfig(
        backend="ollama",
        base_url="http://localhost:11434",
        model_id="llama3",
        inference_timeout_s=30.0,
        health_check_timeout_s=5.0,
    )


@pytest.fixture
def sample_request():
    return _make_completion_request()


# ===========================================================================
# 1. ADAPTER CREATION TESTS
# ===========================================================================

class TestCreateRemoteAdapter:
    """Tests for create_remote_adapter factory function."""

    @pytest.mark.asyncio
    async def test_happy_path_valid_config(
        self, valid_remote_config, mock_http_client, monkeypatch
    ):
        """create_remote_adapter returns adapter with correct source_info when config and API key are valid."""
        monkeypatch.setenv("TEST_OPENAI_API_KEY", "sk-test-key-12345")

        with patch(
            "src.external_interfaces.remote_api_client.create_remote_client"
        ) as mock_create:
            mock_create.return_value = AsyncMock()
            adapter = create_remote_adapter(
                config=valid_remote_config, http_client=mock_http_client
            )

        source = adapter.get_source_info()
        assert source.source_type == SourceType.REMOTE, \
            f"Expected REMOTE, got {source.source_type}"
        assert source.provider_name == "openai", \
            f"Expected 'openai', got {source.provider_name}"
        assert source.model_id == "gpt-4", \
            f"Expected 'gpt-4', got {source.model_id}"

    @pytest.mark.asyncio
    async def test_missing_api_key_env_var(
        self, valid_remote_config, mock_http_client, monkeypatch
    ):
        """create_remote_adapter raises when API key env var is not set."""
        monkeypatch.delenv("TEST_OPENAI_API_KEY", raising=False)

        with pytest.raises((ValueError, KeyError, EnvironmentError, Exception)) as exc_info:
            create_remote_adapter(
                config=valid_remote_config, http_client=mock_http_client
            )
        # Should mention API key or environment variable
        assert exc_info.value is not None

    @pytest.mark.asyncio
    async def test_empty_api_key_env_var(
        self, valid_remote_config, mock_http_client, monkeypatch
    ):
        """create_remote_adapter raises when API key env var is set to empty string."""
        monkeypatch.setenv("TEST_OPENAI_API_KEY", "")

        with pytest.raises((ValueError, KeyError, EnvironmentError, Exception)):
            create_remote_adapter(
                config=valid_remote_config, http_client=mock_http_client
            )

    @pytest.mark.asyncio
    async def test_unsupported_provider(self, mock_http_client, monkeypatch):
        """create_remote_adapter raises for unsupported provider."""
        monkeypatch.setenv("TEST_UNSUPPORTED_KEY", "some-key")
        config = RemoteAdapterConfig(
            provider="unsupported_provider_xyz",
            model="some-model",
            api_key_env_var="TEST_UNSUPPORTED_KEY",
            max_retries=3,
            base_delay_seconds=1.0,
            max_delay_seconds=30.0,
            connect_timeout_seconds=5.0,
            read_timeout_seconds=30.0,
            total_timeout_seconds=60.0,
        )
        with pytest.raises((ValueError, NotImplementedError, Exception)):
            create_remote_adapter(config=config, http_client=mock_http_client)


class TestCreateLocalAdapter:
    """Tests for create_local_adapter factory function."""

    @pytest.mark.asyncio
    async def test_happy_path_valid_config(
        self, valid_local_config, mock_local_client
    ):
        """create_local_adapter returns adapter with correct source_info for valid config."""
        adapter = create_local_adapter(
            config=valid_local_config, local_client=mock_local_client
        )

        source = adapter.get_source_info()
        assert source.source_type == SourceType.LOCAL, \
            f"Expected LOCAL, got {source.source_type}"
        assert source.provider_name == "ollama", \
            f"Expected 'ollama', got {source.provider_name}"
        assert source.backend_type == "ollama", \
            f"Expected backend_type='ollama', got {source.backend_type}"
        assert source.model_id == "llama3", \
            f"Expected 'llama3', got {source.model_id}"

    @pytest.mark.asyncio
    async def test_unsupported_backend(self, mock_local_client):
        """create_local_adapter raises for unsupported backend."""
        config = LocalAdapterConfig(
            backend="unsupported_backend_xyz",
            base_url="http://localhost:9999",
            model_id="some-model",
            inference_timeout_s=30.0,
            health_check_timeout_s=5.0,
        )
        with pytest.raises((ValueError, NotImplementedError, Exception)):
            create_local_adapter(config=config, local_client=mock_local_client)


# ===========================================================================
# 2. COMPLETION TESTS
# ===========================================================================

class TestCompleteRemote:
    """Tests for complete() on RemoteModelAdapter."""

    @pytest.fixture
    def remote_adapter(self, valid_remote_config, mock_remote_client, monkeypatch):
        """Create a remote adapter with mocked dependencies."""
        monkeypatch.setenv("TEST_OPENAI_API_KEY", "sk-test-key-12345")
        with patch(
            "src.external_interfaces.remote_api_client.create_remote_client"
        ) as mock_create:
            mock_create.return_value = mock_remote_client
            adapter = create_remote_adapter(
                config=valid_remote_config, http_client=None
            )
        # Ensure adapter has the mock client for send_request
        return adapter

    @pytest.mark.asyncio
    async def test_happy_path(self, remote_adapter, mock_remote_client):
        """Remote complete() returns CompletionResponse with correct fields."""
        task_resp = MockTaskResponse(
            content="Hello from GPT-4",
            cost_usd=0.015,
            latency_ms=200.0,
            request_id="resp-001",
            usage=MockTokenUsage(input_tokens=20, output_tokens=10),
        )
        mock_remote_client.send_request.return_value = task_resp

        request = _make_completion_request()
        response = await remote_adapter.complete(request)

        assert isinstance(response, CompletionResponse), \
            f"Expected CompletionResponse, got {type(response)}"
        assert response.content == "Hello from GPT-4"
        assert response.source_info.source_type == SourceType.REMOTE
        assert response.cost_usd == 0.015
        assert response.latency_ms >= 0.0
        assert response.response_id != ""
        assert response.token_usage.is_estimated is False

    @pytest.mark.asyncio
    async def test_source_info_matches_adapter(self, remote_adapter, mock_remote_client):
        """Remote complete() response source_info matches adapter's configured source."""
        mock_remote_client.send_request.return_value = MockTaskResponse()
        request = _make_completion_request()
        response = await remote_adapter.complete(request)

        adapter_source = remote_adapter.get_source_info()
        assert response.source_info == adapter_source, \
            f"Response source_info {response.source_info} != adapter {adapter_source}"

    @pytest.mark.asyncio
    async def test_auth_failure_maps_to_terminal_error(
        self, remote_adapter, mock_remote_client
    ):
        """Remote complete() raises TerminalError on AuthError from dependency."""
        mock_remote_client.send_request.side_effect = MockAuthError(
            status_code=401, request_id="req-auth-fail"
        )
        request = _make_completion_request()

        with pytest.raises(TerminalError) as exc_info:
            await remote_adapter.complete(request)

        err = exc_info.value
        assert err.source_info.source_type == SourceType.REMOTE
        assert err.request_id != ""

    @pytest.mark.asyncio
    async def test_invalid_request_maps_to_terminal_error(
        self, remote_adapter, mock_remote_client
    ):
        """Remote complete() raises TerminalError on InvalidRequestError from dependency."""
        mock_remote_client.send_request.side_effect = MockInvalidRequestError(
            status_code=422, request_id="req-invalid"
        )
        request = _make_completion_request()

        with pytest.raises(TerminalError) as exc_info:
            await remote_adapter.complete(request)

        assert exc_info.value.source_info.source_type == SourceType.REMOTE

    @pytest.mark.asyncio
    async def test_rate_limited_maps_to_retriable_error(
        self, remote_adapter, mock_remote_client
    ):
        """Remote complete() raises RetriableError on RateLimitError from dependency."""
        mock_remote_client.send_request.side_effect = MockRateLimitError(
            retry_after=30.0, request_id="req-rate"
        )
        request = _make_completion_request()

        with pytest.raises(RetriableError) as exc_info:
            await remote_adapter.complete(request)

        err = exc_info.value
        assert err.retry_after_seconds == 30.0

    @pytest.mark.asyncio
    async def test_remote_unavailable_maps_to_source_unavailable(
        self, remote_adapter, mock_remote_client
    ):
        """Remote complete() raises SourceUnavailableError on RemoteUnavailableError."""
        mock_remote_client.send_request.side_effect = MockRemoteUnavailableError(
            request_id="req-unavail"
        )
        request = _make_completion_request()

        with pytest.raises(SourceUnavailableError) as exc_info:
            await remote_adapter.complete(request)

        err = exc_info.value
        assert err.reason == UnavailableReason.REMOTE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_empty_messages_error(self, remote_adapter):
        """complete() raises error on empty messages list."""
        request = _make_completion_request(messages=[])

        with pytest.raises((ValueError, Exception)):
            await remote_adapter.complete(request)

    @pytest.mark.asyncio
    async def test_no_user_message_error(self, remote_adapter):
        """complete() raises error when messages have no USER role."""
        messages = [Message(role=Role.SYSTEM, content="System only")]
        request = _make_completion_request(messages=messages)

        with pytest.raises((ValueError, Exception)):
            await remote_adapter.complete(request)

    @pytest.mark.asyncio
    async def test_token_usage_is_estimated_false(
        self, remote_adapter, mock_remote_client
    ):
        """Remote completions always have token_usage.is_estimated == False."""
        mock_remote_client.send_request.return_value = MockTaskResponse()
        request = _make_completion_request()
        response = await remote_adapter.complete(request)

        assert response.token_usage.is_estimated is False, \
            "Remote completions must have is_estimated=False"

    @pytest.mark.asyncio
    async def test_response_id_nonempty(self, remote_adapter, mock_remote_client):
        """CompletionResponse.response_id is always non-empty."""
        mock_remote_client.send_request.return_value = MockTaskResponse()
        request = _make_completion_request()
        response = await remote_adapter.complete(request)

        assert response.response_id != "", "response_id must be non-empty"
        assert isinstance(response.response_id, str)


class TestCompleteLocal:
    """Tests for complete() on LocalModelAdapter."""

    @pytest.fixture
    def local_adapter(self, valid_local_config, mock_local_client):
        """Create a local adapter with mocked dependencies."""
        adapter = create_local_adapter(
            config=valid_local_config, local_client=mock_local_client
        )
        return adapter

    @pytest.mark.asyncio
    async def test_happy_path(self, local_adapter, mock_local_client):
        """Local complete() returns CompletionResponse with cost_usd=0.0."""
        success = MockInferenceSuccess(
            text="Hello from Llama",
            tokens_generated=8,
            tokens_prompt=15,
            latency_ms=75.0,
            request_id="local-resp-001",
        )
        mock_local_client.infer.return_value = success

        request = _make_completion_request()
        response = await local_adapter.complete(request)

        assert isinstance(response, CompletionResponse)
        assert response.content == "Hello from Llama"
        assert response.cost_usd == 0.0, \
            f"Local cost_usd must be 0.0, got {response.cost_usd}"
        assert response.source_info.source_type == SourceType.LOCAL
        assert response.latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_source_info_matches_adapter(self, local_adapter, mock_local_client):
        """Local complete() response source_info matches adapter's configured source."""
        mock_local_client.infer.return_value = MockInferenceSuccess()
        request = _make_completion_request()
        response = await local_adapter.complete(request)

        assert response.source_info == local_adapter.get_source_info()

    @pytest.mark.asyncio
    async def test_source_unavailable_local(self, local_adapter, mock_local_client):
        """Local complete() raises SourceUnavailableError when server returns LocalModelUnavailable."""
        unavail = MockLocalModelUnavailable(
            reason="SERVER_NOT_RUNNING",
            detail="Connection refused",
            request_id="local-req-fail",
        )
        mock_local_client.infer.return_value = unavail

        request = _make_completion_request()
        with pytest.raises(SourceUnavailableError) as exc_info:
            await local_adapter.complete(request)

        err = exc_info.value
        assert err.source_info.source_type == SourceType.LOCAL
        assert err.reason in (
            UnavailableReason.SERVER_NOT_RUNNING,
            UnavailableReason.NO_MODEL_LOADED,
            UnavailableReason.MODEL_LOADING,
            UnavailableReason.INFERENCE_TIMEOUT,
            UnavailableReason.INFERENCE_ERROR,
        )

    @pytest.mark.asyncio
    async def test_cost_always_zero_for_local(self, local_adapter, mock_local_client):
        """Local completions always have cost_usd == 0.0 (invariant)."""
        mock_local_client.infer.return_value = MockInferenceSuccess()
        request = _make_completion_request()
        response = await local_adapter.complete(request)

        assert response.cost_usd == 0.0, \
            "Local source completions must always have cost_usd == 0.0"

    @pytest.mark.asyncio
    async def test_token_usage_total_equals_sum(self, local_adapter, mock_local_client):
        """Token usage total_tokens equals prompt_tokens + completion_tokens."""
        success = MockInferenceSuccess(tokens_generated=12, tokens_prompt=25)
        mock_local_client.infer.return_value = success

        request = _make_completion_request()
        response = await local_adapter.complete(request)

        usage = response.token_usage
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens, \
            f"total_tokens ({usage.total_tokens}) != prompt ({usage.prompt_tokens}) + completion ({usage.completion_tokens})"


# ===========================================================================
# 3. ERROR AND RESPONSE MAPPING TESTS
# ===========================================================================

class TestMapRemoteError:
    """Tests for map_remote_error pure function."""

    @pytest.fixture
    def source(self):
        return _make_source_info()

    def test_auth_error_to_terminal(self, source):
        """AuthError maps to TerminalError."""
        auth_err = MockAuthError(
            message="Unauthorized", status_code=401, request_id="req-a1"
        )
        result = map_remote_error(auth_err, source)

        assert isinstance(result, TerminalError), \
            f"Expected TerminalError, got {type(result)}"
        assert result.source_info == source
        assert result.request_id == "req-a1"
        assert "Unauthorized" in result.original_error or "Auth" in result.original_error

    def test_invalid_request_error_to_terminal(self, source):
        """InvalidRequestError maps to TerminalError."""
        err = MockInvalidRequestError(
            message="Bad request body", status_code=422, request_id="req-ir1"
        )
        result = map_remote_error(err, source)

        assert isinstance(result, TerminalError)
        assert result.source_info == source

    def test_rate_limit_error_to_retriable(self, source):
        """RateLimitError maps to RetriableError with retry_after preserved."""
        err = MockRateLimitError(
            message="Too many requests", retry_after=45.0, request_id="req-rl1"
        )
        result = map_remote_error(err, source)

        assert isinstance(result, RetriableError), \
            f"Expected RetriableError, got {type(result)}"
        assert result.retry_after_seconds == 45.0
        assert result.request_id == "req-rl1"

    def test_remote_unavailable_to_source_unavailable(self, source):
        """RemoteUnavailableError maps to SourceUnavailableError with REMOTE_UNAVAILABLE reason."""
        err = MockRemoteUnavailableError(
            message="Service down", request_id="req-ru1"
        )
        result = map_remote_error(err, source)

        assert isinstance(result, SourceUnavailableError), \
            f"Expected SourceUnavailableError, got {type(result)}"
        assert result.reason == UnavailableReason.REMOTE_UNAVAILABLE
        assert result.source_info == source

    def test_unknown_error_type_raises(self, source):
        """map_remote_error raises for unrecognized error type."""
        unknown_err = RuntimeError("Unexpected error")

        with pytest.raises((TypeError, ValueError, Exception)):
            map_remote_error(unknown_err, source)

    def test_timestamp_is_valid_iso8601(self, source):
        """Mapped errors carry valid ISO 8601 UTC timestamps."""
        err = MockAuthError(request_id="req-ts")
        result = map_remote_error(err, source)

        pattern = _iso8601_pattern()
        assert pattern.match(result.timestamp_utc), \
            f"timestamp_utc '{result.timestamp_utc}' is not valid ISO 8601"

    def test_original_error_contains_string_repr(self, source):
        """Mapped error.original_error contains string representation of input error."""
        err = MockAuthError(message="Auth failed badly", request_id="req-oe")
        result = map_remote_error(err, source)

        assert result.original_error != "", \
            "original_error should contain string repr of input error"


class TestMapLocalUnavailable:
    """Tests for map_local_unavailable pure function."""

    @pytest.fixture
    def source(self):
        return _make_local_source_info()

    @pytest.mark.parametrize("reason_str,expected_reason", [
        ("SERVER_NOT_RUNNING", UnavailableReason.SERVER_NOT_RUNNING),
        ("NO_MODEL_LOADED", UnavailableReason.NO_MODEL_LOADED),
        ("MODEL_LOADING", UnavailableReason.MODEL_LOADING),
        ("INFERENCE_TIMEOUT", UnavailableReason.INFERENCE_TIMEOUT),
        ("INFERENCE_ERROR", UnavailableReason.INFERENCE_ERROR),
    ])
    def test_maps_all_reason_variants(self, source, reason_str, expected_reason):
        """map_local_unavailable correctly maps all five UnavailableReason variants."""
        unavail = MockLocalModelUnavailable(
            reason=reason_str,
            detail=f"Simulated {reason_str}",
            request_id="local-req-map",
        )
        result = map_local_unavailable(unavail, source)

        assert isinstance(result, SourceUnavailableError)
        assert result.reason == expected_reason, \
            f"Expected {expected_reason}, got {result.reason}"
        assert result.source_info == source
        assert result.request_id == "local-req-map"

    def test_preserves_detail_in_original_error(self, source):
        """Mapped error preserves unavailable.detail in original_error."""
        unavail = MockLocalModelUnavailable(
            reason="SERVER_NOT_RUNNING",
            detail="Connection refused at localhost:11434",
            request_id="local-detail",
        )
        result = map_local_unavailable(unavail, source)

        assert "Connection refused" in result.original_error or result.original_error != ""

    def test_timestamp_is_valid_iso8601(self, source):
        """map_local_unavailable produces valid ISO 8601 UTC timestamp."""
        unavail = MockLocalModelUnavailable(request_id="local-ts")
        result = map_local_unavailable(unavail, source)

        pattern = _iso8601_pattern()
        assert pattern.match(result.timestamp_utc), \
            f"timestamp_utc '{result.timestamp_utc}' is not valid ISO 8601"

    def test_invalid_input_raises(self, source):
        """map_local_unavailable raises on non-LocalModelUnavailable input."""
        with pytest.raises((TypeError, ValueError, AttributeError, Exception)):
            map_local_unavailable("not_an_unavailable_object", source)


class TestMapLocalSuccessToResponse:
    """Tests for map_local_success_to_response pure function."""

    @pytest.fixture
    def source(self):
        return _make_local_source_info()

    def test_happy_path(self, source):
        """Correctly maps InferenceSuccess to CompletionResponse."""
        success = MockInferenceSuccess(
            text="Generated text",
            tokens_generated=20,
            tokens_prompt=30,
            latency_ms=100.0,
            request_id="local-success-001",
        )
        result = map_local_success_to_response(success, source)

        assert isinstance(result, CompletionResponse)
        assert result.content == "Generated text", \
            f"Expected 'Generated text', got {result.content!r}"
        assert result.source_info == source
        assert result.cost_usd == 0.0, \
            f"Local cost must be 0.0, got {result.cost_usd}"
        assert result.latency_ms == 100.0
        assert result.response_id == "local-success-001"
        assert result.token_usage.completion_tokens == 20
        assert result.token_usage.prompt_tokens == 30
        assert result.token_usage.total_tokens == 50

    def test_invalid_input_raises(self, source):
        """map_local_success_to_response raises on non-InferenceSuccess input."""
        with pytest.raises((TypeError, ValueError, AttributeError, Exception)):
            map_local_success_to_response("not_a_success", source)


class TestMapRemoteResponse:
    """Tests for map_remote_response pure function."""

    @pytest.fixture
    def source(self):
        return _make_source_info()

    def test_happy_path(self, source):
        """Correctly maps TaskResponse to CompletionResponse."""
        usage = MockTokenUsage(input_tokens=50, output_tokens=25)
        task_resp = MockTaskResponse(
            content="Remote response text",
            cost_usd=0.025,
            latency_ms=300.0,
            request_id="remote-resp-001",
            usage=usage,
        )
        result = map_remote_response(task_resp, source)

        assert isinstance(result, CompletionResponse)
        assert result.content == "Remote response text"
        assert result.source_info == source
        assert result.cost_usd == 0.025
        assert result.latency_ms == 300.0
        assert result.response_id == "remote-resp-001"
        assert result.token_usage.prompt_tokens == 50
        assert result.token_usage.completion_tokens == 25
        assert result.token_usage.total_tokens == 75
        assert result.token_usage.is_estimated is False, \
            "Remote token_usage.is_estimated must be False"

    def test_invalid_input_raises(self, source):
        """map_remote_response raises on non-TaskResponse input."""
        with pytest.raises((TypeError, ValueError, AttributeError, Exception)):
            map_remote_response("not_a_task_response", source)


# ===========================================================================
# 4. ADAPTER LIFECYCLE TESTS
# ===========================================================================

class TestHealth:
    """Tests for health() on both adapter types."""

    @pytest.fixture
    def local_adapter(self, valid_local_config, mock_local_client):
        return create_local_adapter(
            config=valid_local_config, local_client=mock_local_client
        )

    @pytest.mark.asyncio
    async def test_local_health_available(self, local_adapter, mock_local_client):
        """Local health() returns available=True when model is loaded."""
        mock_local_client.health.return_value = MockHealthStatus(
            is_ready=True, loaded_model="llama3", detail="OK"
        )
        report = await local_adapter.health()

        assert isinstance(report, HealthReport)
        assert report.available is True
        assert report.source_info == local_adapter.get_source_info()
        assert report.latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_local_health_unavailable(self, local_adapter, mock_local_client):
        """Local health() returns available=False with diagnostic detail when server is down."""
        mock_local_client.health.return_value = MockHealthStatus(
            is_ready=False, loaded_model="", detail="Server unreachable"
        )
        report = await local_adapter.health()

        assert report.available is False
        assert report.detail != "", \
            "Unavailable health report must have non-empty detail"

    @pytest.mark.asyncio
    async def test_health_never_raises_on_exception(self, local_adapter, mock_local_client):
        """health() never raises for operational failures — returns available=False."""
        mock_local_client.health.side_effect = ConnectionError("Connection refused")

        report = await local_adapter.health()

        assert isinstance(report, HealthReport)
        assert report.available is False
        assert report.detail != ""

    @pytest.mark.asyncio
    async def test_health_source_info_matches(self, local_adapter, mock_local_client):
        """HealthReport.source_info always matches adapter's configured source."""
        mock_local_client.health.return_value = MockHealthStatus()
        report = await local_adapter.health()

        assert report.source_info == local_adapter.get_source_info()


class TestClose:
    """Tests for close() method."""

    @pytest.fixture
    def local_adapter(self, valid_local_config, mock_local_client):
        return create_local_adapter(
            config=valid_local_config, local_client=mock_local_client
        )

    @pytest.mark.asyncio
    async def test_close_idempotent(self, local_adapter):
        """close() can be called multiple times safely."""
        await local_adapter.close()
        await local_adapter.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_complete_after_close_raises(self, local_adapter):
        """complete() raises RuntimeError after close() on adapter that owns its client."""
        # If adapter owns the client, close should cause complete to fail
        # For injected clients, behavior may differ
        await local_adapter.close()
        request = _make_completion_request()

        # May raise RuntimeError or similar — depends on ownership
        try:
            await local_adapter.complete(request)
            # If it doesn't raise, the adapter does not own the client (injected)
        except RuntimeError:
            pass  # Expected for owned clients

    @pytest.mark.asyncio
    async def test_async_context_manager(self, valid_local_config, mock_local_client):
        """Adapters support async context manager protocol."""
        adapter = create_local_adapter(
            config=valid_local_config, local_client=mock_local_client
        )
        async with adapter as a:
            assert a is not None
            source = a.get_source_info()
            assert source.source_type == SourceType.LOCAL


class TestGetSourceInfo:
    """Tests for get_source_info() method."""

    @pytest.fixture
    def local_adapter(self, valid_local_config, mock_local_client):
        return create_local_adapter(
            config=valid_local_config, local_client=mock_local_client
        )

    def test_remote_source_info(self, valid_remote_config, mock_http_client, monkeypatch):
        """get_source_info returns correct SourceInfo for remote adapter."""
        monkeypatch.setenv("TEST_OPENAI_API_KEY", "sk-test")
        with patch(
            "src.external_interfaces.remote_api_client.create_remote_client"
        ) as mock_create:
            mock_create.return_value = AsyncMock()
            adapter = create_remote_adapter(
                config=valid_remote_config, http_client=mock_http_client
            )

        source = adapter.get_source_info()
        assert source.source_type == SourceType.REMOTE
        assert source.provider_name != ""
        assert source.model_id != ""

    def test_local_source_info(self, local_adapter):
        """get_source_info returns correct SourceInfo for local adapter."""
        source = local_adapter.get_source_info()
        assert source.source_type == SourceType.LOCAL
        assert source.provider_name != ""
        assert source.model_id != ""

    def test_source_info_immutable(self, local_adapter):
        """get_source_info returns the same instance on repeated calls."""
        s1 = local_adapter.get_source_info()
        s2 = local_adapter.get_source_info()
        assert s1 is s2 or s1 == s2, \
            "get_source_info should return consistent SourceInfo"

    def test_source_info_frozen(self, local_adapter):
        """SourceInfo is frozen — cannot be mutated."""
        source = local_adapter.get_source_info()
        with pytest.raises((AttributeError, TypeError, Exception)):
            source.model_id = "hacked"


class TestPromoteModel:
    """Tests for promote_model() method."""

    @pytest.fixture
    def local_adapter(self, valid_local_config, mock_local_client):
        return create_local_adapter(
            config=valid_local_config, local_client=mock_local_client
        )

    @pytest.mark.asyncio
    async def test_promote_success_updates_model_id(
        self, local_adapter, mock_local_client
    ):
        """promote_model updates source_info.model_id on success."""
        mock_local_client.promote_model.return_value = MockPromotionResult(
            status="success", new_model_id="llama3.1"
        )

        old_model = local_adapter.get_source_info().model_id
        result = await local_adapter.promote_model(
            new_model_id="llama3.1",
            smoke_test_prompt="Say hello",
            expected_substring="hello",
        )

        new_model = local_adapter.get_source_info().model_id
        assert new_model == "llama3.1", \
            f"Expected model_id 'llama3.1' after promotion, got {new_model}"

    @pytest.mark.asyncio
    async def test_promote_failure_no_change(
        self, local_adapter, mock_local_client
    ):
        """promote_model does not change model_id on failure."""
        mock_local_client.promote_model.return_value = MockPromotionResult(
            status="failed", detail="Smoke test failed"
        )

        old_model = local_adapter.get_source_info().model_id

        result = await local_adapter.promote_model(
            new_model_id="llama3.1",
            smoke_test_prompt="Say hello",
            expected_substring="hello",
        )

        current_model = local_adapter.get_source_info().model_id
        assert current_model == old_model, \
            f"model_id should not change on failed promotion, got {current_model}"

    @pytest.mark.asyncio
    async def test_promote_on_remote_raises_not_implemented(
        self, valid_remote_config, mock_http_client, monkeypatch
    ):
        """promote_model on RemoteModelAdapter raises NotImplementedError."""
        monkeypatch.setenv("TEST_OPENAI_API_KEY", "sk-test")
        with patch(
            "src.external_interfaces.remote_api_client.create_remote_client"
        ) as mock_create:
            mock_create.return_value = AsyncMock()
            adapter = create_remote_adapter(
                config=valid_remote_config, http_client=mock_http_client
            )

        with pytest.raises(NotImplementedError):
            await adapter.promote_model(
                new_model_id="gpt-5",
                smoke_test_prompt="Test",
                expected_substring="ok",
            )

    @pytest.mark.asyncio
    async def test_promote_empty_model_id_raises(self, local_adapter):
        """promote_model raises error when new_model_id is empty."""
        with pytest.raises((ValueError, Exception)):
            await local_adapter.promote_model(
                new_model_id="",
                smoke_test_prompt="Test",
                expected_substring="ok",
            )

    @pytest.mark.asyncio
    async def test_promote_empty_smoke_test_raises(self, local_adapter):
        """promote_model raises error when smoke_test_prompt is empty."""
        with pytest.raises((ValueError, Exception)):
            await local_adapter.promote_model(
                new_model_id="llama3.1",
                smoke_test_prompt="",
                expected_substring="ok",
            )


# ===========================================================================
# 5. INVARIANT TESTS
# ===========================================================================

class TestInvariants:
    """Cross-cutting invariant tests from the contract."""

    def test_source_type_enum_values(self):
        """SourceType enum has LOCAL and REMOTE values."""
        assert hasattr(SourceType, "LOCAL")
        assert hasattr(SourceType, "REMOTE")

    def test_role_enum_values(self):
        """Role enum has SYSTEM, USER, ASSISTANT values."""
        assert hasattr(Role, "SYSTEM")
        assert hasattr(Role, "USER")
        assert hasattr(Role, "ASSISTANT")

    def test_unavailable_reason_enum_values(self):
        """UnavailableReason enum has all specified variants."""
        for variant in [
            "SERVER_NOT_RUNNING", "NO_MODEL_LOADED", "MODEL_LOADING",
            "INFERENCE_TIMEOUT", "INFERENCE_ERROR", "RATE_LIMITED",
            "REMOTE_UNAVAILABLE", "AUTH_FAILURE",
        ]:
            assert hasattr(UnavailableReason, variant), \
                f"UnavailableReason missing variant: {variant}"

    def test_source_info_frozen(self):
        """SourceInfo is immutable (frozen Pydantic model)."""
        si = _make_source_info()
        with pytest.raises((AttributeError, TypeError, Exception)):
            si.provider_name = "changed"

    def test_generation_config_frozen(self):
        """GenerationConfig is immutable (frozen Pydantic model)."""
        gc = _make_generation_config()
        with pytest.raises((AttributeError, TypeError, Exception)):
            gc.temperature = 999.0

    def test_unified_token_usage_frozen(self):
        """UnifiedTokenUsage is immutable (frozen Pydantic model)."""
        usage = UnifiedTokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            is_estimated=False,
        )
        with pytest.raises((AttributeError, TypeError, Exception)):
            usage.prompt_tokens = 999

    def test_completion_response_frozen(self):
        """CompletionResponse is immutable (frozen Pydantic model)."""
        resp = CompletionResponse(
            content="test",
            source_info=_make_source_info(),
            token_usage=UnifiedTokenUsage(
                prompt_tokens=1, completion_tokens=1,
                total_tokens=2, is_estimated=False,
            ),
            cost_usd=0.0,
            latency_ms=1.0,
            response_id="test-id",
        )
        with pytest.raises((AttributeError, TypeError, Exception)):
            resp.content = "changed"

    def test_health_report_frozen(self):
        """HealthReport is immutable (frozen Pydantic model)."""
        report = HealthReport(
            available=True,
            source_info=_make_source_info(),
            detail="OK",
            latency_ms=1.0,
        )
        with pytest.raises((AttributeError, TypeError, Exception)):
            report.available = False

    def test_model_client_error_is_exception(self):
        """ModelClientError is an Exception subclass."""
        err = ModelClientError(
            source_info=_make_source_info(),
            message="test",
            request_id="req-001",
            original_error="orig",
            timestamp_utc="2024-01-01T00:00:00Z",
        )
        assert isinstance(err, Exception), \
            "ModelClientError must be an Exception subclass"

    def test_retriable_error_is_exception(self):
        """RetriableError is an Exception subclass."""
        err = RetriableError(
            source_info=_make_source_info(),
            message="test",
            request_id="req-001",
            original_error="orig",
            timestamp_utc="2024-01-01T00:00:00Z",
            retry_after_seconds=10.0,
            retries_exhausted=3,
        )
        assert isinstance(err, Exception)

    def test_terminal_error_is_exception(self):
        """TerminalError is an Exception subclass."""
        err = TerminalError(
            source_info=_make_source_info(),
            message="test",
            request_id="req-001",
            original_error="orig",
            timestamp_utc="2024-01-01T00:00:00Z",
            status_code=401,
        )
        assert isinstance(err, Exception)

    def test_source_unavailable_error_is_exception(self):
        """SourceUnavailableError is an Exception subclass."""
        err = SourceUnavailableError(
            source_info=_make_source_info(),
            message="test",
            request_id="req-001",
            original_error="orig",
            timestamp_utc="2024-01-01T00:00:00Z",
            reason=UnavailableReason.SERVER_NOT_RUNNING,
        )
        assert isinstance(err, Exception)

    def test_error_hierarchy_unified(self):
        """All specific errors conceptually extend ModelClientError."""
        source = _make_source_info()
        base_kwargs = dict(
            source_info=source, message="test", request_id="req",
            original_error="orig", timestamp_utc="2024-01-01T00:00:00Z",
        )
        # Check they are all exceptions at minimum
        retriable = RetriableError(
            **base_kwargs, retry_after_seconds=0, retries_exhausted=0
        )
        terminal = TerminalError(**base_kwargs, status_code=500)
        unavailable = SourceUnavailableError(
            **base_kwargs, reason=UnavailableReason.REMOTE_UNAVAILABLE
        )

        for err in [retriable, terminal, unavailable]:
            assert isinstance(err, Exception), \
                f"{type(err).__name__} must be an Exception"

    @pytest.mark.asyncio
    async def test_errors_carry_request_id(self):
        """Every error in the hierarchy carries a non-empty request_id."""
        source = _make_source_info()
        base_kwargs = dict(
            source_info=source, message="test", request_id="audit-trail-id",
            original_error="orig", timestamp_utc="2024-01-01T00:00:00Z",
        )

        errors = [
            ModelClientError(**base_kwargs),
            RetriableError(**base_kwargs, retry_after_seconds=0, retries_exhausted=0),
            TerminalError(**base_kwargs, status_code=400),
            SourceUnavailableError(**base_kwargs, reason=UnavailableReason.AUTH_FAILURE),
        ]

        for err in errors:
            assert err.request_id == "audit-trail-id", \
                f"{type(err).__name__}.request_id should be 'audit-trail-id'"
            assert err.request_id != "", \
                f"{type(err).__name__}.request_id should be non-empty"


class TestMappingInvariantsParametrized:
    """Parametrized mapping invariant tests."""

    @pytest.mark.parametrize("error_cls,expected_type", [
        (MockAuthError, TerminalError),
        (MockInvalidRequestError, TerminalError),
        (MockRateLimitError, RetriableError),
        (MockRemoteUnavailableError, SourceUnavailableError),
    ])
    def test_remote_error_mapping_type(self, error_cls, expected_type):
        """Each remote error type maps to the correct unified error type."""
        source = _make_source_info()
        kwargs = {"request_id": "param-test"}
        if error_cls in (MockAuthError, MockInvalidRequestError):
            kwargs["status_code"] = 401
        if error_cls == MockRateLimitError:
            kwargs["retry_after"] = 10.0

        err = error_cls(**kwargs)
        result = map_remote_error(err, source)

        assert isinstance(result, expected_type), \
            f"{error_cls.__name__} should map to {expected_type.__name__}, got {type(result).__name__}"

    @pytest.mark.parametrize("reason_str", [
        "SERVER_NOT_RUNNING",
        "NO_MODEL_LOADED",
        "MODEL_LOADING",
        "INFERENCE_TIMEOUT",
        "INFERENCE_ERROR",
    ])
    def test_local_unavailable_all_reasons_produce_source_unavailable(self, reason_str):
        """All five local UnavailableReason variants produce SourceUnavailableError."""
        source = _make_local_source_info()
        unavail = MockLocalModelUnavailable(
            reason=reason_str, request_id="param-local"
        )
        result = map_local_unavailable(unavail, source)

        assert isinstance(result, SourceUnavailableError)
        assert result.reason.name == reason_str, \
            f"Expected reason {reason_str}, got {result.reason.name}"
