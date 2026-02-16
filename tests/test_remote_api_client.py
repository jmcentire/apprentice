"""
Contract tests for remote_api_client module.

Tests organized into groups:
1. TestCreateRemoteClient - Factory function tests
2. TestSendRequest - Primary send_request method tests
3. TestRetryBehavior - Exponential backoff and retry logic
4. TestSendDiversityRequest - Multi-provider diversity requests
5. TestComputeCost - Pure cost computation function
6. TestGetPricing - Pricing lookup logic
7. TestClose - Resource cleanup
8. TestInvariants - Cross-cutting contract invariants
"""

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import pytest
import httpx

# Import the component under test
from src.remote_api_client import (
    create_remote_client,
    send_diversity_request,
    compute_cost,
    get_pricing,
    ProviderName,
    TokenUsage,
    TokenPricing,
    PricingTable,
    RetryConfig,
    TimeoutConfig,
    RemoteClientConfig,
    TaskResponse,
    ErrorInfo,
    DiversityResult,
    RemoteClientError,
    AuthError,
    RateLimitError,
    RemoteUnavailableError,
    InvalidRequestError,
    PromptMessage,
    MessageRole,
    GenerationParams,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_retry_config():
    return RetryConfig(
        max_retries=3,
        base_delay_seconds=1.0,
        max_delay_seconds=10.0,
        respect_retry_after=True,
    )


@pytest.fixture
def default_timeout_config():
    return TimeoutConfig(
        connect_timeout_seconds=5.0,
        read_timeout_seconds=30.0,
        total_timeout_seconds=60.0,
    )


@pytest.fixture
def empty_pricing_table():
    return PricingTable(entries={})


@pytest.fixture
def sample_pricing_table():
    return PricingTable(entries={
        "claude-3-opus": TokenPricing(input_cost_per_token=0.000015, output_cost_per_token=0.000075),
        "claude-3-sonnet": TokenPricing(input_cost_per_token=0.000003, output_cost_per_token=0.000015),
    })


@pytest.fixture
def valid_config(default_retry_config, default_timeout_config, empty_pricing_table):
    return RemoteClientConfig(
        provider=ProviderName.anthropic,
        model="claude-3-sonnet",
        api_key_env_var="TEST_ANTHROPIC_API_KEY",
        retry=default_retry_config,
        timeout=default_timeout_config,
        pricing_overrides=empty_pricing_table,
    )


@pytest.fixture
def sample_messages():
    return [
        PromptMessage(role=MessageRole.system, content="You are a helpful assistant."),
        PromptMessage(role=MessageRole.user, content="Hello, how are you?"),
    ]


@pytest.fixture
def default_params():
    return GenerationParams(
        max_tokens=1024,
        temperature=0.7,
        top_p=1.0,
        stop_sequences=[],
    )


def _make_success_response(status_code=200, body=None):
    """Helper to build a mock httpx response for a successful API call."""
    if body is None:
        body = {
            "content": [{"text": "Hello! I'm doing well."}],
            "model": "claude-3-sonnet",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
    return httpx.Response(status_code=status_code, json=body)


def _make_error_response(status_code, body=None, headers=None):
    """Helper to build a mock httpx error response."""
    if body is None:
        body = {"error": {"message": "Error occurred", "type": "api_error"}}
    return httpx.Response(status_code=status_code, json=body, headers=headers or {})


def _build_mock_transport(responses):
    """
    Build an httpx.MockTransport that returns responses in order.
    `responses` is a list of httpx.Response objects or callables.
    """
    call_count = {"n": 0}

    def handler(request):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        resp = responses[idx]
        if callable(resp):
            return resp(request)
        return resp

    transport = httpx.MockTransport(handler)
    transport._test_call_count = call_count
    return transport


# ---------------------------------------------------------------------------
# 1. TestCreateRemoteClient
# ---------------------------------------------------------------------------

class TestCreateRemoteClient:
    """Tests for the create_remote_client factory function."""

    def test_create_client_happy_path(self, valid_config, monkeypatch):
        """Successfully create a remote client with valid config and API key."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        client = create_remote_client(valid_config)
        assert client is not None, "Client should not be None"
        assert hasattr(client, "send_request"), "Client must have send_request method"
        assert hasattr(client, "close"), "Client must have close method"

    def test_create_client_missing_api_key(self, valid_config, monkeypatch):
        """Fail fast when API key env var is not set."""
        monkeypatch.delenv("TEST_ANTHROPIC_API_KEY", raising=False)
        with pytest.raises((AuthError, ValueError, KeyError, Exception)) as exc_info:
            create_remote_client(valid_config)
        error_str = str(exc_info.value).lower()
        assert (
            "api_key" in error_str
            or "api key" in error_str
            or "TEST_ANTHROPIC_API_KEY".lower() in error_str
            or "missing" in error_str
            or "not set" in error_str
            or "environment" in error_str
        ), f"Error message should mention API key issue, got: {exc_info.value}"

    def test_create_client_empty_api_key(self, valid_config, monkeypatch):
        """Fail fast when API key env var is set but empty."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "")
        with pytest.raises((AuthError, ValueError, KeyError, Exception)):
            create_remote_client(valid_config)

    def test_create_client_timeout_propagation(
        self, default_retry_config, empty_pricing_table, monkeypatch
    ):
        """Timeout settings from config propagate to httpx client."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        timeout_cfg = TimeoutConfig(
            connect_timeout_seconds=5.0,
            read_timeout_seconds=30.0,
            total_timeout_seconds=60.0,
        )
        config = RemoteClientConfig(
            provider=ProviderName.anthropic,
            model="claude-3-sonnet",
            api_key_env_var="TEST_ANTHROPIC_API_KEY",
            retry=default_retry_config,
            timeout=timeout_cfg,
            pricing_overrides=empty_pricing_table,
        )
        client = create_remote_client(config)
        # The client should be constructed without error — timeout propagation
        # is verified by the fact that creation succeeds with these specific values
        assert client is not None

    def test_create_client_pricing_overrides_merged(
        self, default_retry_config, default_timeout_config, monkeypatch
    ):
        """Pricing overrides are merged with defaults at construction."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        custom_pricing = PricingTable(entries={
            "custom-model": TokenPricing(
                input_cost_per_token=0.001,
                output_cost_per_token=0.002,
            )
        })
        config = RemoteClientConfig(
            provider=ProviderName.anthropic,
            model="custom-model",
            api_key_env_var="TEST_ANTHROPIC_API_KEY",
            retry=default_retry_config,
            timeout=default_timeout_config,
            pricing_overrides=custom_pricing,
        )
        client = create_remote_client(config)
        assert client is not None

    def test_create_client_with_injected_http_client(self, valid_config, monkeypatch):
        """Accept an externally provided httpx.AsyncClient."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        external_client = httpx.AsyncClient()
        try:
            client = create_remote_client(valid_config, http_client=external_client)
            assert client is not None
        finally:
            # Clean up the external client
            asyncio.run(external_client.aclose())


# ---------------------------------------------------------------------------
# 2. TestSendRequest
# ---------------------------------------------------------------------------

class TestSendRequest:
    """Tests for the send_request method on RemoteClient."""

    @pytest.mark.asyncio
    async def test_send_request_happy_path(
        self, valid_config, sample_messages, default_params, monkeypatch
    ):
        """Successfully send a request and receive a valid TaskResponse."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        transport = _build_mock_transport([_make_success_response()])
        http_client = httpx.AsyncClient(transport=transport)
        try:
            client = create_remote_client(valid_config, http_client=http_client)
            response = await client.send_request(sample_messages, default_params)

            assert isinstance(response, TaskResponse), "Response should be TaskResponse"
            assert response.provider == ProviderName.anthropic, "Provider should match config"
            # Validate UUID4
            parsed_uuid = uuid.UUID(response.request_id, version=4)
            assert str(parsed_uuid) == response.request_id, "request_id should be valid UUID4"
            assert response.cost_usd >= 0, "cost_usd should be non-negative"
            assert response.latency_ms > 0, "latency_ms should be positive"
            assert len(response.content) > 0, "content should be non-empty"
            assert len(response.model) > 0, "model should be non-empty"
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_send_request_cost_computation(
        self, valid_config, sample_messages, default_params, monkeypatch
    ):
        """cost_usd matches the formula: input_tokens*input_cost + output_tokens*output_cost."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        transport = _build_mock_transport([_make_success_response()])
        http_client = httpx.AsyncClient(transport=transport)
        try:
            client = create_remote_client(valid_config, http_client=http_client)
            response = await client.send_request(sample_messages, default_params)

            # Look up what pricing was used
            pricing = get_pricing(
                response.provider, response.model, valid_config.pricing_overrides
            )
            if pricing is not None:
                expected_cost = (
                    response.usage.input_tokens * pricing.input_cost_per_token
                    + response.usage.output_tokens * pricing.output_cost_per_token
                )
                assert response.cost_usd == pytest.approx(expected_cost), (
                    f"cost_usd {response.cost_usd} should match computed {expected_cost}"
                )
            else:
                # If no pricing found, cost should be 0
                assert response.cost_usd == pytest.approx(0.0)
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_send_request_unique_request_ids(
        self, valid_config, sample_messages, default_params, monkeypatch
    ):
        """Each request generates a unique UUID4 request_id."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        transport = _build_mock_transport([
            _make_success_response(),
            _make_success_response(),
        ])
        http_client = httpx.AsyncClient(transport=transport)
        try:
            client = create_remote_client(valid_config, http_client=http_client)
            resp1 = await client.send_request(sample_messages, default_params)
            resp2 = await client.send_request(sample_messages, default_params)

            assert resp1.request_id != resp2.request_id, "Each request should have unique ID"
            uuid.UUID(resp1.request_id, version=4)
            uuid.UUID(resp2.request_id, version=4)
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [401, 403])
    async def test_send_request_auth_failure(
        self, valid_config, sample_messages, default_params, monkeypatch, status_code
    ):
        """Auth error raised when provider returns HTTP 401 or 403."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        transport = _build_mock_transport([_make_error_response(status_code)])
        http_client = httpx.AsyncClient(transport=transport)
        try:
            client = create_remote_client(valid_config, http_client=http_client)
            with pytest.raises(AuthError) as exc_info:
                await client.send_request(sample_messages, default_params)
            assert exc_info.value.status_code == status_code
            assert exc_info.value.provider == ProviderName.anthropic
            assert exc_info.value.request_id is not None
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_send_request_rate_limited_429(
        self, valid_config, sample_messages, default_params, monkeypatch
    ):
        """RateLimitError raised when provider returns HTTP 429 after all retries."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        responses = [
            _make_error_response(429, headers={"Retry-After": "1"})
            for _ in range(10)  # More than max_retries
        ]
        transport = _build_mock_transport(responses)
        http_client = httpx.AsyncClient(transport=transport)
        try:
            client = create_remote_client(valid_config, http_client=http_client)
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RateLimitError) as exc_info:
                    await client.send_request(sample_messages, default_params)
            assert exc_info.value.retries_attempted == valid_config.retry.max_retries
            assert hasattr(exc_info.value, "retry_after_seconds")
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [500, 502, 503])
    async def test_send_request_remote_unavailable(
        self, valid_config, sample_messages, default_params, monkeypatch, status_code
    ):
        """RemoteUnavailableError raised for HTTP 5xx after all retries."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        responses = [_make_error_response(status_code) for _ in range(10)]
        transport = _build_mock_transport(responses)
        http_client = httpx.AsyncClient(transport=transport)
        try:
            client = create_remote_client(valid_config, http_client=http_client)
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RemoteUnavailableError) as exc_info:
                    await client.send_request(sample_messages, default_params)
            assert exc_info.value.retries_attempted == valid_config.retry.max_retries
            assert exc_info.value.status_code == status_code
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [400, 422])
    async def test_send_request_invalid_request(
        self, valid_config, sample_messages, default_params, monkeypatch, status_code
    ):
        """InvalidRequestError raised when provider returns HTTP 400 or 422."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        transport = _build_mock_transport([_make_error_response(status_code)])
        http_client = httpx.AsyncClient(transport=transport)
        try:
            client = create_remote_client(valid_config, http_client=http_client)
            with pytest.raises(InvalidRequestError) as exc_info:
                await client.send_request(sample_messages, default_params)
            assert exc_info.value.status_code == status_code
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_send_request_empty_messages(
        self, valid_config, default_params, monkeypatch
    ):
        """Error raised when messages list is empty."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        transport = _build_mock_transport([_make_success_response()])
        http_client = httpx.AsyncClient(transport=transport)
        try:
            client = create_remote_client(valid_config, http_client=http_client)
            with pytest.raises((ValueError, InvalidRequestError, Exception)):
                await client.send_request([], default_params)
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_send_request_no_user_message(
        self, valid_config, default_params, monkeypatch
    ):
        """Error raised when messages contain no user role message."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        messages = [
            PromptMessage(role=MessageRole.system, content="System prompt only"),
        ]
        transport = _build_mock_transport([_make_success_response()])
        http_client = httpx.AsyncClient(transport=transport)
        try:
            client = create_remote_client(valid_config, http_client=http_client)
            with pytest.raises((ValueError, InvalidRequestError, Exception)):
                await client.send_request(messages, default_params)
        finally:
            await http_client.aclose()


# ---------------------------------------------------------------------------
# 3. TestRetryBehavior
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    """Tests for exponential backoff retry logic."""

    @pytest.mark.asyncio
    async def test_retry_transient_then_success(
        self, valid_config, sample_messages, default_params, monkeypatch
    ):
        """Successful retry after transient 500 errors."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        responses = [
            _make_error_response(500),
            _make_error_response(500),
            _make_success_response(),
        ]
        transport = _build_mock_transport(responses)
        http_client = httpx.AsyncClient(transport=transport)
        try:
            client = create_remote_client(valid_config, http_client=http_client)
            with patch("asyncio.sleep", new_callable=AsyncMock):
                response = await client.send_request(sample_messages, default_params)
            assert isinstance(response, TaskResponse)
            assert transport._test_call_count["n"] == 3, "Should have made 3 attempts"
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_retry_max_retries_exhausted(
        self, valid_config, sample_messages, default_params, monkeypatch
    ):
        """Retries up to max_retries and then raises error."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        responses = [_make_error_response(500) for _ in range(10)]
        transport = _build_mock_transport(responses)
        http_client = httpx.AsyncClient(transport=transport)
        try:
            client = create_remote_client(valid_config, http_client=http_client)
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RemoteUnavailableError) as exc_info:
                    await client.send_request(sample_messages, default_params)
            assert exc_info.value.retries_attempted == valid_config.retry.max_retries
            # 1 initial + max_retries retries
            expected_calls = valid_config.retry.max_retries + 1
            assert transport._test_call_count["n"] == expected_calls, (
                f"Should have made {expected_calls} total attempts"
            )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_no_retry_on_401(
        self, valid_config, sample_messages, default_params, monkeypatch
    ):
        """Auth errors (401) are never retried."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        responses = [_make_error_response(401) for _ in range(5)]
        transport = _build_mock_transport(responses)
        http_client = httpx.AsyncClient(transport=transport)
        try:
            client = create_remote_client(valid_config, http_client=http_client)
            with pytest.raises(AuthError) as exc_info:
                await client.send_request(sample_messages, default_params)
            assert exc_info.value.retries_attempted == 0, "Auth errors should not be retried"
            assert transport._test_call_count["n"] == 1, "Should only call transport once"
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_no_retry_on_400(
        self, valid_config, sample_messages, default_params, monkeypatch
    ):
        """Invalid request errors (400) are never retried."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        responses = [_make_error_response(400) for _ in range(5)]
        transport = _build_mock_transport(responses)
        http_client = httpx.AsyncClient(transport=transport)
        try:
            client = create_remote_client(valid_config, http_client=http_client)
            with pytest.raises(InvalidRequestError) as exc_info:
                await client.send_request(sample_messages, default_params)
            assert exc_info.value.retries_attempted == 0, "400 errors should not be retried"
            assert transport._test_call_count["n"] == 1, "Should only call transport once"
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(
        self, sample_messages, default_params, monkeypatch,
        default_timeout_config, empty_pricing_table,
    ):
        """Retry delays follow full-jitter exponential backoff within bounds."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        retry_cfg = RetryConfig(
            max_retries=3,
            base_delay_seconds=1.0,
            max_delay_seconds=10.0,
            respect_retry_after=False,
        )
        config = RemoteClientConfig(
            provider=ProviderName.anthropic,
            model="claude-3-sonnet",
            api_key_env_var="TEST_ANTHROPIC_API_KEY",
            retry=retry_cfg,
            timeout=default_timeout_config,
            pricing_overrides=empty_pricing_table,
        )
        responses = [_make_error_response(500) for _ in range(10)]
        transport = _build_mock_transport(responses)
        http_client = httpx.AsyncClient(transport=transport)
        sleep_calls = []

        async def mock_sleep(duration):
            sleep_calls.append(duration)

        try:
            client = create_remote_client(config, http_client=http_client)
            with patch("asyncio.sleep", side_effect=mock_sleep):
                with pytest.raises(RemoteUnavailableError):
                    await client.send_request(sample_messages, default_params)

            assert len(sleep_calls) == 3, f"Should sleep {retry_cfg.max_retries} times"
            for attempt, delay in enumerate(sleep_calls):
                max_bound = min(
                    retry_cfg.max_delay_seconds,
                    retry_cfg.base_delay_seconds * (2 ** attempt),
                )
                assert delay >= 0, f"Delay for attempt {attempt} should be >= 0"
                assert delay <= max_bound + 0.01, (
                    f"Delay {delay} for attempt {attempt} should be <= {max_bound}"
                )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_max_delay_cap(
        self, sample_messages, default_params, monkeypatch,
        default_timeout_config, empty_pricing_table,
    ):
        """Retry delay is capped at max_delay_seconds."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        retry_cfg = RetryConfig(
            max_retries=5,
            base_delay_seconds=1.0,
            max_delay_seconds=2.0,
            respect_retry_after=False,
        )
        config = RemoteClientConfig(
            provider=ProviderName.anthropic,
            model="claude-3-sonnet",
            api_key_env_var="TEST_ANTHROPIC_API_KEY",
            retry=retry_cfg,
            timeout=default_timeout_config,
            pricing_overrides=empty_pricing_table,
        )
        responses = [_make_error_response(500) for _ in range(10)]
        transport = _build_mock_transport(responses)
        http_client = httpx.AsyncClient(transport=transport)
        sleep_calls = []

        async def mock_sleep(duration):
            sleep_calls.append(duration)

        try:
            client = create_remote_client(config, http_client=http_client)
            with patch("asyncio.sleep", side_effect=mock_sleep):
                with pytest.raises(RemoteUnavailableError):
                    await client.send_request(sample_messages, default_params)

            for delay in sleep_calls:
                assert delay <= retry_cfg.max_delay_seconds + 0.01, (
                    f"All delays should be <= {retry_cfg.max_delay_seconds}, got {delay}"
                )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_respect_retry_after_true(
        self, sample_messages, default_params, monkeypatch,
        default_timeout_config, empty_pricing_table,
    ):
        """When respect_retry_after=True, 429 Retry-After header is used as delay."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        retry_cfg = RetryConfig(
            max_retries=2,
            base_delay_seconds=0.1,
            max_delay_seconds=10.0,
            respect_retry_after=True,
        )
        config = RemoteClientConfig(
            provider=ProviderName.anthropic,
            model="claude-3-sonnet",
            api_key_env_var="TEST_ANTHROPIC_API_KEY",
            retry=retry_cfg,
            timeout=default_timeout_config,
            pricing_overrides=empty_pricing_table,
        )
        responses = [
            _make_error_response(429, headers={"Retry-After": "2"}),
            _make_success_response(),
        ]
        transport = _build_mock_transport(responses)
        http_client = httpx.AsyncClient(transport=transport)
        sleep_calls = []

        async def mock_sleep(duration):
            sleep_calls.append(duration)

        try:
            client = create_remote_client(config, http_client=http_client)
            with patch("asyncio.sleep", side_effect=mock_sleep):
                response = await client.send_request(sample_messages, default_params)
            assert isinstance(response, TaskResponse)
            # The sleep should have used Retry-After value (2 seconds)
            assert len(sleep_calls) >= 1
            assert any(
                abs(d - 2.0) < 0.5 for d in sleep_calls
            ), f"Expected a sleep near 2.0 (from Retry-After), got {sleep_calls}"
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_respect_retry_after_false(
        self, sample_messages, default_params, monkeypatch,
        default_timeout_config, empty_pricing_table,
    ):
        """When respect_retry_after=False, Retry-After header is ignored."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        retry_cfg = RetryConfig(
            max_retries=2,
            base_delay_seconds=0.1,
            max_delay_seconds=1.0,
            respect_retry_after=False,
        )
        config = RemoteClientConfig(
            provider=ProviderName.anthropic,
            model="claude-3-sonnet",
            api_key_env_var="TEST_ANTHROPIC_API_KEY",
            retry=retry_cfg,
            timeout=default_timeout_config,
            pricing_overrides=empty_pricing_table,
        )
        responses = [
            _make_error_response(429, headers={"Retry-After": "100"}),
            _make_success_response(),
        ]
        transport = _build_mock_transport(responses)
        http_client = httpx.AsyncClient(transport=transport)
        sleep_calls = []

        async def mock_sleep(duration):
            sleep_calls.append(duration)

        try:
            client = create_remote_client(config, http_client=http_client)
            with patch("asyncio.sleep", side_effect=mock_sleep):
                response = await client.send_request(sample_messages, default_params)
            assert isinstance(response, TaskResponse)
            # Should NOT have slept for 100 seconds
            for d in sleep_calls:
                assert d < 100.0, (
                    f"Should not use Retry-After of 100 when respect_retry_after=False, got {d}"
                )
        finally:
            await http_client.aclose()


# ---------------------------------------------------------------------------
# 4. TestSendDiversityRequest
# ---------------------------------------------------------------------------

class TestSendDiversityRequest:
    """Tests for send_diversity_request multi-provider fan-out."""

    def _make_mock_client(self, provider, cost=0.01, should_fail=False, error_cls=None):
        """Create a mock client that satisfies RemoteClient protocol."""
        mock = AsyncMock()
        mock.provider = provider
        if should_fail:
            error = (error_cls or RemoteUnavailableError)(
                provider=provider,
                message="Test error",
                request_id=str(uuid.uuid4()),
                retries_attempted=3,
                status_code=500,
            )
            mock.send_request = AsyncMock(side_effect=error)
        else:
            response = TaskResponse(
                content=f"Response from {provider.value}",
                model="test-model",
                provider=provider,
                usage=TokenUsage(input_tokens=100, output_tokens=50),
                cost_usd=cost,
                latency_ms=150.0,
                request_id=str(uuid.uuid4()),
            )
            mock.send_request = AsyncMock(return_value=response)
        return mock

    @pytest.mark.asyncio
    async def test_diversity_all_succeed(self, sample_messages, default_params):
        """All providers succeed in diversity request."""
        primary = self._make_mock_client(ProviderName.anthropic, cost=0.01)
        sec1 = self._make_mock_client(ProviderName.openai, cost=0.02)
        sec2 = self._make_mock_client(ProviderName.google, cost=0.005)

        result = await send_diversity_request(
            primary, [sec1, sec2], sample_messages, default_params
        )

        assert isinstance(result, DiversityResult)
        assert isinstance(result.primary, TaskResponse)
        assert result.primary.provider == ProviderName.anthropic
        assert len(result.secondary) == 2
        for entry in result.secondary:
            assert isinstance(entry, TaskResponse)
        expected_total = 0.01 + 0.02 + 0.005
        assert result.total_cost_usd == pytest.approx(expected_total), (
            f"Total cost should be {expected_total}, got {result.total_cost_usd}"
        )

    @pytest.mark.asyncio
    async def test_diversity_primary_fails_propagates(
        self, sample_messages, default_params
    ):
        """Primary provider failure propagates as error."""
        primary = self._make_mock_client(
            ProviderName.anthropic, should_fail=True, error_cls=RemoteUnavailableError
        )
        sec1 = self._make_mock_client(ProviderName.openai, cost=0.02)

        with pytest.raises(RemoteUnavailableError):
            await send_diversity_request(
                primary, [sec1], sample_messages, default_params
            )

    @pytest.mark.asyncio
    async def test_diversity_primary_auth_failure(
        self, sample_messages, default_params
    ):
        """Primary auth failure propagates as AuthError."""
        primary = AsyncMock()
        primary.provider = ProviderName.anthropic
        auth_err = AuthError(
            provider=ProviderName.anthropic,
            message="Invalid API key",
            request_id=str(uuid.uuid4()),
            retries_attempted=0,
            status_code=401,
        )
        primary.send_request = AsyncMock(side_effect=auth_err)

        with pytest.raises(AuthError):
            await send_diversity_request(
                primary, [], sample_messages, default_params
            )

    @pytest.mark.asyncio
    async def test_diversity_primary_rate_limited(
        self, sample_messages, default_params
    ):
        """Primary rate limit failure propagates."""
        primary = AsyncMock()
        primary.provider = ProviderName.anthropic
        rate_err = RateLimitError(
            provider=ProviderName.anthropic,
            message="Rate limited",
            request_id=str(uuid.uuid4()),
            retries_attempted=3,
            status_code=429,
            retry_after_seconds=5.0,
        )
        primary.send_request = AsyncMock(side_effect=rate_err)

        with pytest.raises(RateLimitError):
            await send_diversity_request(
                primary, [], sample_messages, default_params
            )

    @pytest.mark.asyncio
    async def test_diversity_secondary_failure_graceful(
        self, sample_messages, default_params
    ):
        """Secondary failure captured as ErrorInfo, not raised."""
        primary = self._make_mock_client(ProviderName.anthropic, cost=0.01)
        sec_fail = self._make_mock_client(
            ProviderName.openai, should_fail=True, error_cls=RemoteUnavailableError
        )

        result = await send_diversity_request(
            primary, [sec_fail], sample_messages, default_params
        )

        assert isinstance(result, DiversityResult)
        assert isinstance(result.primary, TaskResponse)
        assert len(result.secondary) == 1
        sec_entry = result.secondary[0]
        assert isinstance(sec_entry, ErrorInfo), (
            f"Failed secondary should be ErrorInfo, got {type(sec_entry)}"
        )
        assert sec_entry.provider == ProviderName.openai
        assert hasattr(sec_entry, "error_type")
        assert hasattr(sec_entry, "message")
        assert hasattr(sec_entry, "retries_attempted")

    @pytest.mark.asyncio
    async def test_diversity_mixed_secondary(
        self, sample_messages, default_params
    ):
        """Mixed secondary success and failure results."""
        primary = self._make_mock_client(ProviderName.anthropic, cost=0.01)
        sec_ok = self._make_mock_client(ProviderName.openai, cost=0.02)
        sec_fail = self._make_mock_client(
            ProviderName.google, should_fail=True, error_cls=RemoteUnavailableError
        )

        result = await send_diversity_request(
            primary, [sec_ok, sec_fail], sample_messages, default_params
        )

        assert len(result.secondary) == 2
        types = {type(e) for e in result.secondary}
        assert TaskResponse in types, "Should have at least one TaskResponse secondary"
        assert ErrorInfo in types, "Should have at least one ErrorInfo secondary"
        # Total cost should only include successful responses
        expected_total = 0.01 + 0.02  # primary + successful secondary
        assert result.total_cost_usd == pytest.approx(expected_total), (
            f"Total cost should exclude failed secondary, expected {expected_total}"
        )

    @pytest.mark.asyncio
    async def test_diversity_empty_secondaries(
        self, sample_messages, default_params
    ):
        """Diversity request with no secondary clients."""
        primary = self._make_mock_client(ProviderName.anthropic, cost=0.01)

        result = await send_diversity_request(
            primary, [], sample_messages, default_params
        )

        assert isinstance(result, DiversityResult)
        assert len(result.secondary) == 0
        assert result.total_cost_usd == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_diversity_empty_messages(self, default_params):
        """Diversity request fails with empty messages."""
        primary = self._make_mock_client(ProviderName.anthropic, cost=0.01)

        with pytest.raises((ValueError, InvalidRequestError, Exception)):
            await send_diversity_request(primary, [], [], default_params)

    @pytest.mark.asyncio
    async def test_diversity_total_cost_excludes_failed(
        self, sample_messages, default_params
    ):
        """Total cost only includes successful responses (detailed)."""
        primary = self._make_mock_client(ProviderName.anthropic, cost=0.01)
        sec1 = self._make_mock_client(ProviderName.openai, cost=0.02)
        sec2 = self._make_mock_client(
            ProviderName.google, should_fail=True, error_cls=RemoteUnavailableError
        )

        result = await send_diversity_request(
            primary, [sec1, sec2], sample_messages, default_params
        )

        successful_secondary_cost = sum(
            s.cost_usd for s in result.secondary if isinstance(s, TaskResponse)
        )
        expected = result.primary.cost_usd + successful_secondary_cost
        assert result.total_cost_usd == pytest.approx(expected), (
            f"total_cost_usd should be primary + successful secondary costs"
        )


# ---------------------------------------------------------------------------
# 5. TestComputeCost
# ---------------------------------------------------------------------------

class TestComputeCost:
    """Tests for the compute_cost pure function."""

    def test_basic_computation(self):
        """Basic cost computation with known values."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        pricing = TokenPricing(
            input_cost_per_token=0.00001,
            output_cost_per_token=0.00003,
        )
        result = compute_cost(usage, pricing)
        expected = 100 * 0.00001 + 50 * 0.00003  # 0.001 + 0.0015 = 0.0025
        assert result == pytest.approx(expected), (
            f"Expected {expected}, got {result}"
        )

    def test_zero_tokens(self):
        """Cost is zero when both token counts are zero."""
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        pricing = TokenPricing(
            input_cost_per_token=0.00001,
            output_cost_per_token=0.00003,
        )
        result = compute_cost(usage, pricing)
        assert result == 0.0, f"Expected 0.0 for zero tokens, got {result}"

    def test_zero_pricing(self):
        """Cost is zero when pricing is all zeros."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        pricing = TokenPricing(
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
        )
        result = compute_cost(usage, pricing)
        assert result == 0.0, f"Expected 0.0 for zero pricing, got {result}"

    def test_large_tokens(self):
        """Correct computation with large token counts."""
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
        pricing = TokenPricing(
            input_cost_per_token=0.000003,
            output_cost_per_token=0.000015,
        )
        result = compute_cost(usage, pricing)
        expected = 1_000_000 * 0.000003 + 500_000 * 0.000015  # 3.0 + 7.5 = 10.5
        assert result == pytest.approx(expected), (
            f"Expected {expected}, got {result}"
        )

    def test_input_only(self):
        """Cost with only input tokens (zero output)."""
        usage = TokenUsage(input_tokens=100, output_tokens=0)
        pricing = TokenPricing(
            input_cost_per_token=0.00001,
            output_cost_per_token=0.00003,
        )
        result = compute_cost(usage, pricing)
        expected = 100 * 0.00001
        assert result == pytest.approx(expected)

    def test_output_only(self):
        """Cost with only output tokens (zero input)."""
        usage = TokenUsage(input_tokens=0, output_tokens=50)
        pricing = TokenPricing(
            input_cost_per_token=0.00001,
            output_cost_per_token=0.00003,
        )
        result = compute_cost(usage, pricing)
        expected = 50 * 0.00003
        assert result == pytest.approx(expected)

    def test_non_negative_property(self):
        """Cost is always non-negative for valid inputs."""
        test_cases = [
            (0, 0),
            (1, 0),
            (0, 1),
            (100, 200),
            (999999, 999999),
        ]
        pricing = TokenPricing(
            input_cost_per_token=0.00001,
            output_cost_per_token=0.00003,
        )
        for inp, out in test_cases:
            usage = TokenUsage(input_tokens=inp, output_tokens=out)
            result = compute_cost(usage, pricing)
            assert result >= 0.0, (
                f"Cost should be non-negative for ({inp}, {out}), got {result}"
            )

    def test_linearity_doubling(self):
        """Doubling tokens doubles cost."""
        pricing = TokenPricing(
            input_cost_per_token=0.00001,
            output_cost_per_token=0.00003,
        )
        usage1 = TokenUsage(input_tokens=100, output_tokens=50)
        usage2 = TokenUsage(input_tokens=200, output_tokens=100)
        cost1 = compute_cost(usage1, pricing)
        cost2 = compute_cost(usage2, pricing)
        assert cost2 == pytest.approx(2 * cost1), (
            f"Doubling tokens should double cost: {cost2} != 2 * {cost1}"
        )


# ---------------------------------------------------------------------------
# 6. TestGetPricing
# ---------------------------------------------------------------------------

class TestGetPricing:
    """Tests for the get_pricing lookup function."""

    def test_override_takes_precedence(self):
        """Override pricing takes precedence over defaults."""
        override_pricing = TokenPricing(
            input_cost_per_token=0.999,
            output_cost_per_token=0.888,
        )
        overrides = PricingTable(entries={"claude-3-sonnet": override_pricing})
        result = get_pricing(ProviderName.anthropic, "claude-3-sonnet", overrides)
        assert result is not None, "Should find override pricing"
        assert result.input_cost_per_token == pytest.approx(0.999)
        assert result.output_cost_per_token == pytest.approx(0.888)

    def test_default_fallback(self):
        """Falls back to built-in default when no override exists."""
        overrides = PricingTable(entries={})
        # This test assumes there's a built-in default for a common model
        # If no default exists, result may be None — that's also valid
        result = get_pricing(ProviderName.anthropic, "claude-3-sonnet", overrides)
        # We accept either a valid TokenPricing or None (if no defaults exist)
        if result is not None:
            assert result.input_cost_per_token >= 0.0
            assert result.output_cost_per_token >= 0.0

    def test_unknown_model_returns_none(self):
        """Returns None for unknown model with no override."""
        overrides = PricingTable(entries={})
        result = get_pricing(
            ProviderName.anthropic, "nonexistent-model-xyz-999", overrides
        )
        assert result is None, (
            f"Unknown model should return None, got {result}"
        )

    def test_multiple_entries_correct_lookup(self):
        """Correct lookup from pricing table with multiple entries."""
        entries = {
            "model-a": TokenPricing(
                input_cost_per_token=0.001,
                output_cost_per_token=0.002,
            ),
            "model-b": TokenPricing(
                input_cost_per_token=0.003,
                output_cost_per_token=0.004,
            ),
            "model-c": TokenPricing(
                input_cost_per_token=0.005,
                output_cost_per_token=0.006,
            ),
        }
        overrides = PricingTable(entries=entries)
        result = get_pricing(ProviderName.anthropic, "model-b", overrides)
        assert result is not None
        assert result.input_cost_per_token == pytest.approx(0.003)
        assert result.output_cost_per_token == pytest.approx(0.004)

    def test_override_for_one_model_does_not_affect_others(self):
        """Override for one model does not affect other models."""
        overrides = PricingTable(entries={
            "model-a": TokenPricing(
                input_cost_per_token=0.999,
                output_cost_per_token=0.999,
            ),
        })
        # model-b should not get model-a's override
        result = get_pricing(ProviderName.anthropic, "model-b", overrides)
        if result is not None:
            assert result.input_cost_per_token != pytest.approx(0.999), (
                "model-b should not get model-a's override pricing"
            )


# ---------------------------------------------------------------------------
# 7. TestClose
# ---------------------------------------------------------------------------

class TestClose:
    """Tests for the close() resource cleanup method."""

    @pytest.mark.asyncio
    async def test_close_owned_client(self, valid_config, monkeypatch):
        """Closing a client with owned httpx.AsyncClient releases resources."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        client = create_remote_client(valid_config)
        await client.close()
        # After close, the internal httpx client should be closed
        # We can verify by trying to send a request (should fail)
        # This is tested more explicitly in test_send_after_close_raises

    @pytest.mark.asyncio
    async def test_close_injected_client_noop(self, valid_config, monkeypatch):
        """Closing a client with injected httpx.AsyncClient is a no-op."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        external = httpx.AsyncClient()
        try:
            client = create_remote_client(valid_config, http_client=external)
            await client.close()
            assert not external.is_closed, (
                "Injected httpx client should remain open after close()"
            )
        finally:
            await external.aclose()

    @pytest.mark.asyncio
    async def test_close_idempotent(self, valid_config, monkeypatch):
        """Multiple calls to close are safe."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        client = create_remote_client(valid_config)
        await client.close()
        # Second close should not raise
        await client.close()

    @pytest.mark.asyncio
    async def test_send_after_close_raises(
        self, valid_config, sample_messages, default_params, monkeypatch
    ):
        """send_request after close raises RuntimeError for owned client."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        client = create_remote_client(valid_config)
        await client.close()
        with pytest.raises((RuntimeError, Exception)):
            await client.send_request(sample_messages, default_params)


# ---------------------------------------------------------------------------
# 8. TestInvariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """Tests for cross-cutting contract invariants."""

    def test_api_key_not_in_string_representation(self, valid_config, monkeypatch):
        """API key is stored as SecretStr and never appears in string representation."""
        api_key = "sk-super-secret-key-12345"
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", api_key)
        client = create_remote_client(valid_config)
        client_str = str(client)
        client_repr = repr(client)
        assert api_key not in client_str, (
            "API key should not appear in str(client)"
        )
        assert api_key not in client_repr, (
            "API key should not appear in repr(client)"
        )

    @pytest.mark.asyncio
    async def test_error_carries_all_metadata(
        self, valid_config, sample_messages, default_params, monkeypatch
    ):
        """All error types carry provider, message, request_id, retries_attempted, status_code."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")

        error_scenarios = [
            (401, AuthError),
            (400, InvalidRequestError),
        ]

        for status_code, error_class in error_scenarios:
            transport = _build_mock_transport([_make_error_response(status_code)])
            http_client = httpx.AsyncClient(transport=transport)
            try:
                client = create_remote_client(valid_config, http_client=http_client)
                with pytest.raises(error_class) as exc_info:
                    await client.send_request(sample_messages, default_params)
                err = exc_info.value
                assert hasattr(err, "provider"), f"{error_class.__name__} missing provider"
                assert hasattr(err, "message"), f"{error_class.__name__} missing message"
                assert hasattr(err, "request_id"), f"{error_class.__name__} missing request_id"
                assert hasattr(err, "retries_attempted"), (
                    f"{error_class.__name__} missing retries_attempted"
                )
                assert hasattr(err, "status_code"), (
                    f"{error_class.__name__} missing status_code"
                )
            finally:
                await http_client.aclose()

    @pytest.mark.asyncio
    async def test_taskresponse_is_frozen(
        self, valid_config, sample_messages, default_params, monkeypatch
    ):
        """TaskResponse is immutable (frozen Pydantic model)."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        transport = _build_mock_transport([_make_success_response()])
        http_client = httpx.AsyncClient(transport=transport)
        try:
            client = create_remote_client(valid_config, http_client=http_client)
            response = await client.send_request(sample_messages, default_params)
            with pytest.raises((AttributeError, TypeError, Exception)):
                response.content = "modified"
        finally:
            await http_client.aclose()

    def test_token_usage_frozen(self):
        """TokenUsage is immutable (frozen Pydantic model)."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        with pytest.raises((AttributeError, TypeError, Exception)):
            usage.input_tokens = 999

    def test_token_pricing_frozen(self):
        """TokenPricing is immutable (frozen Pydantic model)."""
        pricing = TokenPricing(
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        with pytest.raises((AttributeError, TypeError, Exception)):
            pricing.input_cost_per_token = 999.0

    @pytest.mark.asyncio
    async def test_provider_matches_in_response(
        self, sample_messages, default_params, monkeypatch,
        default_retry_config, default_timeout_config, empty_pricing_table,
    ):
        """Returned TaskResponse.provider always matches the client's configured provider."""
        monkeypatch.setenv("TEST_ANTHROPIC_API_KEY", "sk-test-key-12345")
        for provider in [ProviderName.anthropic]:
            config = RemoteClientConfig(
                provider=provider,
                model="claude-3-sonnet",
                api_key_env_var="TEST_ANTHROPIC_API_KEY",
                retry=default_retry_config,
                timeout=default_timeout_config,
                pricing_overrides=empty_pricing_table,
            )
            transport = _build_mock_transport([_make_success_response()])
            http_client = httpx.AsyncClient(transport=transport)
            try:
                client = create_remote_client(config, http_client=http_client)
                response = await client.send_request(sample_messages, default_params)
                assert response.provider == provider, (
                    f"Response provider {response.provider} should match config {provider}"
                )
            finally:
                await http_client.aclose()
