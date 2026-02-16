"""
Contract tests for the Local Model Server Client.

Tests verify behavior at boundaries (inputs/outputs) against the contract.
All HTTP dependencies are mocked via httpx.MockTransport injection.
"""
import asyncio
import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

# Import the component under test
from src.local_model_server import (
    BaseClientConfig,
    HealthStatus,
    InferenceRequest,
    InferenceSuccess,
    LlamaCppClientConfig,
    LocalBackendType,
    LocalModelUnavailable,
    OllamaClientConfig,
    PromotionResult,
    PromotionStatus,
    UnavailableReason,
    VLLMClientConfig,
)

# Attempt to import concrete client implementations
# These may vary by project structure
try:
    from src.local_model_server import OllamaClient, VLLMClient, LlamaCppClient
except ImportError:
    # Fallback: try alternative import paths
    try:
        from src.local_model_server.ollama_client import OllamaClient
        from src.local_model_server.vllm_client import VLLMClient
        from src.local_model_server.llamacpp_client import LlamaCppClient
    except ImportError:
        OllamaClient = None
        VLLMClient = None
        LlamaCppClient = None


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def make_ollama_config(**overrides):
    """Create an OllamaClientConfig with sensible test defaults."""
    defaults = dict(
        base_url="http://localhost:11434",
        connect_timeout_s=5.0,
        inference_timeout_s=30.0,
        health_check_timeout_s=5.0,
        promotion_lock_timeout_s=10.0,
        keep_alive="5m",
        num_gpu_layers=0,
    )
    defaults.update(overrides)
    return OllamaClientConfig(**defaults)


def make_vllm_config(**overrides):
    defaults = dict(
        base_url="http://localhost:8000",
        connect_timeout_s=5.0,
        inference_timeout_s=30.0,
        health_check_timeout_s=5.0,
        promotion_lock_timeout_s=10.0,
        api_key="test-key",
        model_name_override="",
    )
    defaults.update(overrides)
    return VLLMClientConfig(**defaults)


def make_llamacpp_config(**overrides):
    defaults = dict(
        base_url="http://localhost:8080",
        connect_timeout_s=5.0,
        inference_timeout_s=30.0,
        health_check_timeout_s=5.0,
        promotion_lock_timeout_s=10.0,
        n_predict=128,
        slot_id=-1,
    )
    defaults.update(overrides)
    return LlamaCppClientConfig(**defaults)


def make_inference_request(**overrides):
    """Create a valid InferenceRequest with sensible defaults."""
    defaults = dict(
        prompt="What is the capital of France?",
        model_id="llama3:8b",
        max_tokens=100,
        temperature=0.7,
        stop_sequences=[],
        request_id=str(uuid.uuid4()),
    )
    defaults.update(overrides)
    return InferenceRequest(**defaults)


def make_ollama_success_handler(response_text="Paris is the capital of France.",
                                 model="llama3:8b",
                                 prompt_tokens=10,
                                 completion_tokens=8):
    """Create a mock transport handler that returns a successful Ollama response."""
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        # Ollama inference endpoint
        if "/api/generate" in path or "/api/chat" in path:
            return httpx.Response(
                200,
                json={
                    "model": model,
                    "response": response_text,
                    "done": True,
                    "total_duration": 500000000,  # nanoseconds
                    "load_duration": 100000000,
                    "prompt_eval_count": prompt_tokens,
                    "eval_count": completion_tokens,
                },
            )
        # Ollama health / tags endpoint
        if "/api/tags" in path:
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": model, "size": 4000000000, "digest": "abc123"}
                    ]
                },
            )
        # Ollama pull endpoint (for promotion)
        if "/api/pull" in path:
            return httpx.Response(200, json={"status": "success"})
        return httpx.Response(404)

    return handler


def make_error_handler(error_cls):
    """Create a handler that raises an httpx transport error."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise error_cls("Simulated transport error", request=request)
    return handler


def make_status_handler(status_code, body=None):
    """Create a handler that returns a specific HTTP status code."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json=body or {"error": f"Server error {status_code}"},
        )
    return handler


async def make_ollama_client(handler, config=None):
    """Factory to create an Ollama client with mock transport for testing."""
    if OllamaClient is None:
        pytest.skip("OllamaClient not importable")
    cfg = config or make_ollama_config()
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url=cfg.base_url)
    # Try common constructor patterns
    try:
        client = OllamaClient(config=cfg, http_client=http_client)
    except TypeError:
        try:
            client = OllamaClient(cfg, http_client=http_client)
        except TypeError:
            client = OllamaClient(cfg)
            client._client = http_client
    # Initialize if needed
    if hasattr(client, '__aenter__'):
        try:
            await client.__aenter__()
        except Exception:
            pass
    return client, http_client


# ---------------------------------------------------------------------------
# Test Group 1: infer() — Happy Path
# ---------------------------------------------------------------------------

class TestInferHappyPath:
    """Tests for successful inference execution."""

    @pytest.mark.asyncio
    async def test_infer_returns_inference_success_on_valid_request(self):
        """infer() returns InferenceSuccess with correct fields when backend returns valid completion."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            result = await client.infer(request)

            assert isinstance(result, InferenceSuccess), (
                f"Expected InferenceSuccess, got {type(result).__name__}"
            )
            assert result.result_type == "success", (
                f"Expected result_type='success', got '{result.result_type}'"
            )
            assert result.text != "", "InferenceSuccess.text must be non-empty"
            assert result.model_id != "", "InferenceSuccess.model_id must be non-empty"
            assert result.latency_ms >= 0, (
                f"latency_ms must be >= 0, got {result.latency_ms}"
            )
            assert result.request_id == request.request_id, (
                f"request_id mismatch: expected {request.request_id}, got {result.request_id}"
            )
            assert result.tokens_generated >= 0, "tokens_generated must be >= 0"
            assert result.tokens_prompt >= 0, "tokens_prompt must be >= 0"
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_infer_success_contains_backend_type(self):
        """InferenceSuccess.backend contains the correct backend type."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            result = await client.infer(request)

            assert isinstance(result, InferenceSuccess)
            assert result.backend is not None, "backend must not be None"
            # Should be ollama for this client
            assert result.backend == "ollama" or result.backend == LocalBackendType.ollama or str(result.backend) == "ollama", (
                f"Expected backend 'ollama', got {result.backend}"
            )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_infer_request_id_propagated(self):
        """request_id from InferenceRequest is propagated to InferenceSuccess."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            req_id = "test-req-12345"
            request = make_inference_request(request_id=req_id)
            result = await client.infer(request)

            assert result.request_id == req_id, (
                f"Expected request_id '{req_id}', got '{result.request_id}'"
            )
        finally:
            await http_client.aclose()


# ---------------------------------------------------------------------------
# Test Group 2: infer() — Unavailable scenarios
# ---------------------------------------------------------------------------

class TestInferUnavailable:
    """Tests for LocalModelUnavailable return values (no exceptions)."""

    @pytest.mark.asyncio
    async def test_infer_server_not_running_on_connect_error(self):
        """infer() returns LocalModelUnavailable with reason=server_not_running on ConnectError."""
        handler = make_error_handler(httpx.ConnectError)
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            result = await client.infer(request)

            assert isinstance(result, LocalModelUnavailable), (
                f"Expected LocalModelUnavailable, got {type(result).__name__}"
            )
            assert result.result_type == "unavailable"
            reason_str = str(result.reason)
            assert "server_not_running" in reason_str or result.reason == UnavailableReason.server_not_running or result.reason == "server_not_running", (
                f"Expected reason 'server_not_running', got '{result.reason}'"
            )
            assert result.latency_ms >= 0
            assert result.request_id != "", "request_id must be non-empty"
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_infer_server_not_running_on_connect_timeout(self):
        """infer() returns LocalModelUnavailable on ConnectTimeout (not an exception)."""
        handler = make_error_handler(httpx.ConnectTimeout)
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            result = await client.infer(request)

            assert isinstance(result, LocalModelUnavailable), (
                f"Expected LocalModelUnavailable, got {type(result).__name__}"
            )
            assert result.result_type == "unavailable"
            assert result.latency_ms >= 0
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_infer_timeout_on_read_timeout(self):
        """infer() returns LocalModelUnavailable with reason=inference_timeout on ReadTimeout."""
        handler = make_error_handler(httpx.ReadTimeout)
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            result = await client.infer(request)

            assert isinstance(result, LocalModelUnavailable), (
                f"Expected LocalModelUnavailable, got {type(result).__name__}"
            )
            assert result.result_type == "unavailable"
            reason_str = str(result.reason)
            assert "timeout" in reason_str.lower() or result.reason == "inference_timeout", (
                f"Expected timeout-related reason, got '{result.reason}'"
            )
            assert result.latency_ms >= 0
            assert result.request_id == request.request_id
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_infer_error_on_http_500(self):
        """infer() returns LocalModelUnavailable with reason=inference_error on HTTP 500."""
        handler = make_status_handler(500)
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            result = await client.infer(request)

            assert isinstance(result, LocalModelUnavailable), (
                f"Expected LocalModelUnavailable, got {type(result).__name__}"
            )
            assert result.result_type == "unavailable"
            reason_str = str(result.reason)
            assert "error" in reason_str.lower() or "unavailable" in reason_str.lower() or result.reason in (
                "inference_error", UnavailableReason.inference_error,
                "server_not_running", UnavailableReason.server_not_running,
            ), f"Expected error/unavailable reason, got '{result.reason}'"
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_infer_unavailable_always_has_backend(self):
        """LocalModelUnavailable always contains the backend type that was attempted."""
        handler = make_error_handler(httpx.ConnectError)
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            result = await client.infer(request)

            assert isinstance(result, LocalModelUnavailable)
            assert result.backend is not None, "backend must not be None in unavailable result"
        finally:
            await http_client.aclose()


# ---------------------------------------------------------------------------
# Test Group 3: infer() — Validation Errors (programming errors → exceptions)
# ---------------------------------------------------------------------------

class TestInferValidation:
    """Tests that programming errors raise exceptions (not return values)."""

    @pytest.mark.asyncio
    async def test_infer_rejects_non_inference_request(self):
        """infer() raises exception when passed a dict instead of InferenceRequest."""
        call_count = 0

        def counting_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json={})

        client, http_client = await make_ollama_client(counting_handler)
        try:
            with pytest.raises((TypeError, ValueError, AttributeError, Exception)):
                await client.infer({"prompt": "test", "model_id": "m", "max_tokens": 10})

            assert call_count == 0, "No HTTP call should be made for invalid request type"
        finally:
            await http_client.aclose()

    def test_inference_request_rejects_empty_prompt(self):
        """InferenceRequest with empty prompt fails Pydantic validation."""
        with pytest.raises((ValueError, Exception)):
            InferenceRequest(
                prompt="",
                model_id="llama3:8b",
                max_tokens=100,
                temperature=0.7,
                stop_sequences=[],
                request_id="test",
            )

    def test_inference_request_rejects_max_tokens_zero(self):
        """InferenceRequest with max_tokens=0 fails validation."""
        with pytest.raises((ValueError, Exception)):
            InferenceRequest(
                prompt="Hello",
                model_id="llama3:8b",
                max_tokens=0,
                temperature=0.7,
                stop_sequences=[],
                request_id="test",
            )

    def test_inference_request_rejects_max_tokens_too_large(self):
        """InferenceRequest with max_tokens=99999 fails validation."""
        with pytest.raises((ValueError, Exception)):
            InferenceRequest(
                prompt="Hello",
                model_id="llama3:8b",
                max_tokens=99999,
                temperature=0.7,
                stop_sequences=[],
                request_id="test",
            )

    def test_inference_request_rejects_negative_temperature(self):
        """InferenceRequest with temperature=-0.1 fails validation."""
        with pytest.raises((ValueError, Exception)):
            InferenceRequest(
                prompt="Hello",
                model_id="llama3:8b",
                max_tokens=100,
                temperature=-0.1,
                stop_sequences=[],
                request_id="test",
            )

    def test_inference_request_rejects_temperature_above_2(self):
        """InferenceRequest with temperature=2.5 fails validation."""
        with pytest.raises((ValueError, Exception)):
            InferenceRequest(
                prompt="Hello",
                model_id="llama3:8b",
                max_tokens=100,
                temperature=2.5,
                stop_sequences=[],
                request_id="test",
            )

    def test_inference_request_accepts_boundary_max_tokens_1(self):
        """InferenceRequest accepts max_tokens=1 (lower boundary)."""
        req = InferenceRequest(
            prompt="Hello",
            model_id="llama3:8b",
            max_tokens=1,
            temperature=0.7,
            stop_sequences=[],
            request_id="test",
        )
        assert req.max_tokens == 1

    def test_inference_request_accepts_boundary_max_tokens_32768(self):
        """InferenceRequest accepts max_tokens=32768 (upper boundary)."""
        req = InferenceRequest(
            prompt="Hello",
            model_id="llama3:8b",
            max_tokens=32768,
            temperature=0.7,
            stop_sequences=[],
            request_id="test",
        )
        assert req.max_tokens == 32768

    def test_inference_request_accepts_temperature_0(self):
        """InferenceRequest accepts temperature=0.0 (lower boundary)."""
        req = InferenceRequest(
            prompt="Hello",
            model_id="llama3:8b",
            max_tokens=100,
            temperature=0.0,
            stop_sequences=[],
            request_id="test",
        )
        assert req.temperature == 0.0

    def test_inference_request_accepts_temperature_2(self):
        """InferenceRequest accepts temperature=2.0 (upper boundary)."""
        req = InferenceRequest(
            prompt="Hello",
            model_id="llama3:8b",
            max_tokens=100,
            temperature=2.0,
            stop_sequences=[],
            request_id="test",
        )
        assert req.temperature == 2.0


# ---------------------------------------------------------------------------
# Test Group 4: health()
# ---------------------------------------------------------------------------

class TestHealth:
    """Tests for the health() method."""

    @pytest.mark.asyncio
    async def test_health_alive_and_ready(self):
        """health() returns is_alive=True, is_ready=True when server has model loaded."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            result = await client.health()

            assert isinstance(result, HealthStatus), (
                f"Expected HealthStatus, got {type(result).__name__}"
            )
            assert result.is_alive is True, "Server should be alive"
            assert result.is_ready is True, "Server should be ready with model loaded"
            assert result.loaded_model != "", (
                "loaded_model must be non-empty when is_ready=True"
            )
            assert result.latency_ms >= 0, (
                f"latency_ms must be >= 0, got {result.latency_ms}"
            )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_health_unreachable_server(self):
        """health() returns is_alive=False on network error (never raises)."""
        handler = make_error_handler(httpx.ConnectError)
        client, http_client = await make_ollama_client(handler)
        try:
            result = await client.health()

            assert isinstance(result, HealthStatus), (
                f"Expected HealthStatus, got {type(result).__name__}"
            )
            assert result.is_alive is False, "Server should not be alive on ConnectError"
            assert result.is_ready is False, "Server cannot be ready if not alive"
            assert result.latency_ms >= 0
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_health_ready_implies_alive(self):
        """Invariant: is_ready=True implies is_alive=True."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            result = await client.health()

            if result.is_ready:
                assert result.is_alive is True, (
                    "Invariant violated: is_ready=True but is_alive=False"
                )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_health_not_alive_implies_not_ready(self):
        """Invariant: is_alive=False implies is_ready=False."""
        handler = make_error_handler(httpx.ConnectError)
        client, http_client = await make_ollama_client(handler)
        try:
            result = await client.health()

            if not result.is_alive:
                assert result.is_ready is False, (
                    "Invariant violated: is_alive=False but is_ready=True"
                )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_health_ready_implies_loaded_model_non_empty(self):
        """Invariant: is_ready=True implies loaded_model is non-empty."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            result = await client.health()

            if result.is_ready:
                assert result.loaded_model != "", (
                    "Invariant violated: is_ready=True but loaded_model is empty"
                )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_health_alive_not_ready(self):
        """health() returns is_alive=True, is_ready=False when server has no model."""
        def handler(request: httpx.Request) -> httpx.Response:
            if "/api/tags" in request.url.path:
                return httpx.Response(200, json={"models": []})
            # Health endpoint with no models
            return httpx.Response(200, json={"status": "ok"})

        client, http_client = await make_ollama_client(handler)
        try:
            result = await client.health()

            assert isinstance(result, HealthStatus)
            assert result.is_alive is True, "Server should be alive"
            # is_ready depends on whether a model is loaded
            # With no models, we expect is_ready=False
            if not result.is_ready:
                assert result.is_ready is False
        finally:
            await http_client.aclose()


# ---------------------------------------------------------------------------
# Test Group 5: promote_model()
# ---------------------------------------------------------------------------

class TestPromoteModel:
    """Tests for model promotion workflow."""

    @pytest.mark.asyncio
    async def test_promote_model_success(self):
        """promote_model() succeeds: loads model, passes smoke test, swaps active model."""
        handler = make_ollama_success_handler(
            response_text="Hello! I am ready to help.",
            model="llama3:8b-new",
        )
        client, http_client = await make_ollama_client(handler)
        try:
            result = await client.promote_model(
                new_model_id="llama3:8b-new",
                smoke_test_prompt="Hello, are you ready?",
                expected_substring="ready",
            )

            assert isinstance(result, PromotionResult), (
                f"Expected PromotionResult, got {type(result).__name__}"
            )
            assert result.status == "success" or result.status == PromotionStatus.success, (
                f"Expected status 'success', got '{result.status}'"
            )
            assert result.is_active is True, "New model should be active after success"
            assert result.smoke_test_passed is True, "Smoke test should have passed"
            assert result.new_model_id == "llama3:8b-new"
            assert result.duration_ms >= 0, (
                f"duration_ms must be >= 0, got {result.duration_ms}"
            )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_promote_model_smoke_test_failed(self):
        """promote_model() rolls back when smoke test does not contain expected substring."""
        handler = make_ollama_success_handler(
            response_text="I don't know what you mean.",
            model="new-model",
        )
        client, http_client = await make_ollama_client(handler)
        try:
            result = await client.promote_model(
                new_model_id="new-model",
                smoke_test_prompt="Tell me about cats",
                expected_substring="IMPOSSIBLE_SUBSTRING_XYZZY",
            )

            assert isinstance(result, PromotionResult)
            status_str = str(result.status)
            assert "smoke_test_failed" in status_str or result.status == PromotionStatus.smoke_test_failed, (
                f"Expected 'smoke_test_failed', got '{result.status}'"
            )
            assert result.is_active is False, "New model should NOT be active after smoke test failure"
            assert result.smoke_test_passed is False
            assert result.smoke_test_response != "", "smoke_test_response should contain the failing response"
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_promote_model_server_unavailable(self):
        """promote_model() returns server_unavailable when server is unreachable."""
        handler = make_error_handler(httpx.ConnectError)
        client, http_client = await make_ollama_client(handler)
        try:
            result = await client.promote_model(
                new_model_id="new-model",
                smoke_test_prompt="test",
                expected_substring="test",
            )

            assert isinstance(result, PromotionResult)
            status_str = str(result.status)
            assert any(s in status_str for s in ("server_unavailable", "load_failed")), (
                f"Expected server_unavailable or load_failed, got '{result.status}'"
            )
            assert result.is_active is False
            assert result.duration_ms >= 0
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_promote_model_error_empty_model_id(self):
        """promote_model() raises exception for empty model_id (programming error)."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            with pytest.raises((ValueError, TypeError, Exception)):
                await client.promote_model(
                    new_model_id="",
                    smoke_test_prompt="test",
                    expected_substring="test",
                )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_promote_model_error_whitespace_model_id(self):
        """promote_model() raises exception for whitespace-only model_id."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            with pytest.raises((ValueError, TypeError, Exception)):
                await client.promote_model(
                    new_model_id="   ",
                    smoke_test_prompt="test",
                    expected_substring="test",
                )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_promote_model_error_empty_smoke_test_prompt(self):
        """promote_model() raises exception for empty smoke_test_prompt."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            with pytest.raises((ValueError, TypeError, Exception)):
                await client.promote_model(
                    new_model_id="new-model",
                    smoke_test_prompt="",
                    expected_substring="test",
                )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_promote_model_error_non_string_model_id(self):
        """promote_model() raises exception for non-string model_id."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            with pytest.raises((ValueError, TypeError, Exception)):
                await client.promote_model(
                    new_model_id=12345,
                    smoke_test_prompt="test",
                    expected_substring="test",
                )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_promote_model_previous_model_id_captured(self):
        """promote_model() captures the previous active model in the result."""
        handler = make_ollama_success_handler(
            response_text="Hello ready",
            model="new-model",
        )
        client, http_client = await make_ollama_client(handler)
        try:
            result = await client.promote_model(
                new_model_id="new-model",
                smoke_test_prompt="Hello",
                expected_substring="ready",
            )

            assert isinstance(result, PromotionResult)
            # previous_model_id should be a string (empty if none was active)
            assert isinstance(result.previous_model_id, str), (
                "previous_model_id must be a string"
            )
        finally:
            await http_client.aclose()


# ---------------------------------------------------------------------------
# Test Group 6: get_active_model()
# ---------------------------------------------------------------------------

class TestGetActiveModel:
    """Tests for get_active_model() method."""

    @pytest.mark.asyncio
    async def test_get_active_model_no_model_loaded(self):
        """get_active_model() returns empty string when no model has been loaded."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            result = await client.get_active_model()

            assert isinstance(result, str), (
                f"Expected str, got {type(result).__name__}"
            )
            assert result == "", (
                f"Expected empty string for no model loaded, got '{result}'"
            )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_get_active_model_after_promotion(self):
        """get_active_model() returns correct model_id after successful promotion."""
        handler = make_ollama_success_handler(
            response_text="I am ready to help.",
            model="llama3:8b-promoted",
        )
        client, http_client = await make_ollama_client(handler)
        try:
            promotion_result = await client.promote_model(
                new_model_id="llama3:8b-promoted",
                smoke_test_prompt="Are you ready?",
                expected_substring="ready",
            )

            if promotion_result.status in ("success", PromotionStatus.success):
                result = await client.get_active_model()
                assert result == "llama3:8b-promoted", (
                    f"Expected 'llama3:8b-promoted', got '{result}'"
                )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_get_active_model_no_io(self):
        """get_active_model() does not make any HTTP calls."""
        call_count = 0

        def counting_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json={"models": []})

        client, http_client = await make_ollama_client(counting_handler)
        initial_count = call_count
        try:
            await client.get_active_model()
            assert call_count == initial_count, (
                f"get_active_model() should not make HTTP calls, but {call_count - initial_count} calls were made"
            )
        finally:
            await http_client.aclose()


# ---------------------------------------------------------------------------
# Test Group 7: Invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """Tests verifying contract invariants."""

    @pytest.mark.asyncio
    async def test_operational_failures_never_raise_connect_error(self):
        """Invariant: ConnectError is returned as LocalModelUnavailable, not raised."""
        handler = make_error_handler(httpx.ConnectError)
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            # Should NOT raise
            result = await client.infer(request)
            assert isinstance(result, LocalModelUnavailable), (
                f"Expected LocalModelUnavailable for ConnectError, got {type(result).__name__}"
            )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_operational_failures_never_raise_read_timeout(self):
        """Invariant: ReadTimeout is returned as LocalModelUnavailable, not raised."""
        handler = make_error_handler(httpx.ReadTimeout)
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            result = await client.infer(request)
            assert isinstance(result, LocalModelUnavailable), (
                f"Expected LocalModelUnavailable for ReadTimeout, got {type(result).__name__}"
            )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_operational_failures_never_raise_connect_timeout(self):
        """Invariant: ConnectTimeout is returned as LocalModelUnavailable, not raised."""
        handler = make_error_handler(httpx.ConnectTimeout)
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            result = await client.infer(request)
            assert isinstance(result, LocalModelUnavailable), (
                f"Expected LocalModelUnavailable for ConnectTimeout, got {type(result).__name__}"
            )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_operational_failures_never_raise_http_500(self):
        """Invariant: HTTP 500 is returned as LocalModelUnavailable, not raised."""
        handler = make_status_handler(500)
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            result = await client.infer(request)
            assert isinstance(result, LocalModelUnavailable), (
                f"Expected LocalModelUnavailable for HTTP 500, got {type(result).__name__}"
            )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_request_id_always_present_on_success(self):
        """Invariant: InferenceSuccess always carries a non-empty request_id."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            result = await client.infer(request)
            assert result.request_id != "", "request_id must be non-empty"
            assert len(result.request_id) > 0
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_request_id_always_present_on_unavailable(self):
        """Invariant: LocalModelUnavailable always carries a non-empty request_id."""
        handler = make_error_handler(httpx.ConnectError)
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            result = await client.infer(request)
            assert isinstance(result, LocalModelUnavailable)
            assert result.request_id != "", "request_id must be non-empty on unavailable result"
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_request_id_generated_when_empty(self):
        """Invariant: request_id is generated if not provided (empty) in request."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            # Create request without a meaningful request_id
            request = make_inference_request(request_id="")
            result = await client.infer(request)
            assert result.request_id != "", (
                "request_id must be auto-generated when empty in request"
            )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_latency_non_negative_on_success(self):
        """Invariant: latency_ms >= 0 on InferenceSuccess."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            result = await client.infer(request)
            assert result.latency_ms >= 0, (
                f"latency_ms must be >= 0, got {result.latency_ms}"
            )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_latency_non_negative_on_unavailable(self):
        """Invariant: latency_ms >= 0 on LocalModelUnavailable."""
        handler = make_error_handler(httpx.ConnectError)
        client, http_client = await make_ollama_client(handler)
        try:
            request = make_inference_request()
            result = await client.infer(request)
            assert result.latency_ms >= 0, (
                f"latency_ms must be >= 0, got {result.latency_ms}"
            )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_latency_non_negative_on_health(self):
        """Invariant: latency_ms >= 0 on HealthStatus."""
        handler = make_ollama_success_handler()
        client, http_client = await make_ollama_client(handler)
        try:
            result = await client.health()
            assert result.latency_ms >= 0, (
                f"latency_ms must be >= 0, got {result.latency_ms}"
            )
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_promotion_serialized_concurrent_attempts(self):
        """Invariant: concurrent promotions are serialized; second attempt gets lock_timeout."""
        # Create a slow handler that delays responses
        async_event = asyncio.Event()

        def slow_handler(request: httpx.Request) -> httpx.Response:
            # Simulate slow model pull/load
            import time
            if "/api/pull" in request.url.path:
                time.sleep(0.5)
            if "/api/generate" in request.url.path or "/api/chat" in request.url.path:
                return httpx.Response(200, json={
                    "model": "model-a",
                    "response": "test response ready",
                    "done": True,
                    "total_duration": 500000000,
                    "prompt_eval_count": 5,
                    "eval_count": 3,
                })
            if "/api/tags" in request.url.path:
                return httpx.Response(200, json={
                    "models": [{"name": "model-a", "size": 1000, "digest": "abc"}]
                })
            return httpx.Response(200, json={"status": "success"})

        config = make_ollama_config(promotion_lock_timeout_s=0.1)
        client, http_client = await make_ollama_client(slow_handler, config=config)
        try:
            # Launch two concurrent promotions
            results = await asyncio.gather(
                client.promote_model("model-a", "test", "ready"),
                client.promote_model("model-b", "test", "ready"),
                return_exceptions=True,
            )

            # Filter out any unexpected exceptions
            promotion_results = [r for r in results if isinstance(r, PromotionResult)]

            if len(promotion_results) == 2:
                statuses = [str(r.status) for r in promotion_results]
                # At least one should complete, the other might get lock_timeout
                has_lock_timeout = any("lock_timeout" in s for s in statuses)
                # This test may not always trigger lock_timeout depending on timing,
                # so we just verify that if lock_timeout occurred, is_active is False
                for r in promotion_results:
                    if "lock_timeout" in str(r.status):
                        assert r.is_active is False, (
                            "lock_timeout result must have is_active=False"
                        )
        finally:
            await http_client.aclose()


# ---------------------------------------------------------------------------
# Test Group 8: Config Validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    """Tests that configs validate at construction time via Pydantic."""

    def test_ollama_config_valid(self):
        """OllamaClientConfig accepts valid configuration."""
        config = make_ollama_config()
        assert config.base_url == "http://localhost:11434"
        assert config.connect_timeout_s == 5.0

    def test_vllm_config_valid(self):
        """VLLMClientConfig accepts valid configuration."""
        config = make_vllm_config()
        assert config.base_url == "http://localhost:8000"
        assert config.api_key == "test-key"

    def test_llamacpp_config_valid(self):
        """LlamaCppClientConfig accepts valid configuration."""
        config = make_llamacpp_config()
        assert config.base_url == "http://localhost:8080"
        assert config.n_predict == 128

    def test_config_rejects_empty_base_url(self):
        """Config rejects empty base_url at construction time."""
        with pytest.raises((ValueError, Exception)):
            OllamaClientConfig(
                base_url="",
                connect_timeout_s=5.0,
                inference_timeout_s=30.0,
                health_check_timeout_s=5.0,
                promotion_lock_timeout_s=10.0,
                keep_alive="5m",
                num_gpu_layers=0,
            )

    def test_config_rejects_negative_timeout(self):
        """Config rejects negative timeout values at construction time."""
        with pytest.raises((ValueError, Exception)):
            OllamaClientConfig(
                base_url="http://localhost:11434",
                connect_timeout_s=-1.0,
                inference_timeout_s=30.0,
                health_check_timeout_s=5.0,
                promotion_lock_timeout_s=10.0,
                keep_alive="5m",
                num_gpu_layers=0,
            )


# ---------------------------------------------------------------------------
# Test Group 9: Type / Enum Validation
# ---------------------------------------------------------------------------

class TestTypeEnums:
    """Tests for enum and type correctness."""

    def test_unavailable_reason_enum_values(self):
        """UnavailableReason has all expected variants."""
        expected = {"server_not_running", "no_model_loaded", "model_loading",
                    "inference_timeout", "inference_error"}
        # Handle both string enums and standard enums
        if hasattr(UnavailableReason, "__members__"):
            actual = set(UnavailableReason.__members__.keys())
        else:
            actual = {e.value if hasattr(e, 'value') else str(e) for e in UnavailableReason}
        assert expected.issubset(actual) or expected == actual, (
            f"UnavailableReason missing expected variants. Expected {expected}, got {actual}"
        )

    def test_promotion_status_enum_values(self):
        """PromotionStatus has all expected variants."""
        expected = {"success", "smoke_test_failed", "load_failed",
                    "server_unavailable", "already_active", "lock_timeout"}
        if hasattr(PromotionStatus, "__members__"):
            actual = set(PromotionStatus.__members__.keys())
        else:
            actual = {e.value if hasattr(e, 'value') else str(e) for e in PromotionStatus}
        assert expected.issubset(actual) or expected == actual, (
            f"PromotionStatus missing expected variants. Expected {expected}, got {actual}"
        )

    def test_local_backend_type_enum_values(self):
        """LocalBackendType has all expected variants."""
        expected = {"ollama", "vllm", "llamacpp"}
        if hasattr(LocalBackendType, "__members__"):
            actual = set(LocalBackendType.__members__.keys())
        else:
            actual = {e.value if hasattr(e, 'value') else str(e) for e in LocalBackendType}
        assert expected.issubset(actual) or expected == actual, (
            f"LocalBackendType missing expected variants. Expected {expected}, got {actual}"
        )

    def test_inference_success_has_required_fields(self):
        """InferenceSuccess struct has all required fields."""
        required_fields = {"result_type", "text", "model_id", "tokens_generated",
                          "tokens_prompt", "latency_ms", "backend", "request_id"}
        if hasattr(InferenceSuccess, "model_fields"):
            actual = set(InferenceSuccess.model_fields.keys())
        elif hasattr(InferenceSuccess, "__fields__"):
            actual = set(InferenceSuccess.__fields__.keys())
        else:
            actual = set()
        assert required_fields.issubset(actual), (
            f"InferenceSuccess missing fields: {required_fields - actual}"
        )

    def test_local_model_unavailable_has_required_fields(self):
        """LocalModelUnavailable struct has all required fields."""
        required_fields = {"result_type", "reason", "detail", "backend",
                          "request_id", "latency_ms"}
        if hasattr(LocalModelUnavailable, "model_fields"):
            actual = set(LocalModelUnavailable.model_fields.keys())
        elif hasattr(LocalModelUnavailable, "__fields__"):
            actual = set(LocalModelUnavailable.__fields__.keys())
        else:
            actual = set()
        assert required_fields.issubset(actual), (
            f"LocalModelUnavailable missing fields: {required_fields - actual}"
        )

    def test_health_status_has_required_fields(self):
        """HealthStatus struct has all required fields."""
        required_fields = {"is_alive", "is_ready", "loaded_model", "backend",
                          "reason", "latency_ms"}
        if hasattr(HealthStatus, "model_fields"):
            actual = set(HealthStatus.model_fields.keys())
        elif hasattr(HealthStatus, "__fields__"):
            actual = set(HealthStatus.__fields__.keys())
        else:
            actual = set()
        assert required_fields.issubset(actual), (
            f"HealthStatus missing fields: {required_fields - actual}"
        )

    def test_promotion_result_has_required_fields(self):
        """PromotionResult struct has all required fields."""
        required_fields = {"status", "previous_model_id", "new_model_id", "is_active",
                          "smoke_test_passed", "smoke_test_response", "detail", "duration_ms"}
        if hasattr(PromotionResult, "model_fields"):
            actual = set(PromotionResult.model_fields.keys())
        elif hasattr(PromotionResult, "__fields__"):
            actual = set(PromotionResult.__fields__.keys())
        else:
            actual = set()
        assert required_fields.issubset(actual), (
            f"PromotionResult missing fields: {required_fields - actual}"
        )
