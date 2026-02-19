"""Integration tests for middleware pipeline invocation in Apprentice.run()."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from apprentice.apprentice_class import (
    Apprentice,
    ApprenticeConfig,
    BudgetEnforcementConfig,
    ConfidenceThresholds,
    ModelSource,
    PhaseLabel,
    RunStatus,
    TaskConfig,
)
from apprentice.middleware import MiddlewareContext, MiddlewarePipeline, MiddlewareResponse
from apprentice.pii_tokenizer import PIITokenizer, PIITokenizerConfig


def _make_config() -> ApprenticeConfig:
    return ApprenticeConfig(
        tasks=[
            TaskConfig(
                task_name="test_task",
                remote_model_id="test-model",
                local_model_id="local-model",
                budget_limit_usd=100.0,
                budget_period_days=30,
                confidence_thresholds=ConfidenceThresholds(
                    bootstrapping_upper=0.2,
                    active_learning_upper=0.5,
                    supervised_upper=0.8,
                    autonomous_lower=0.9,
                ),
            )
        ],
        remote_api_base_url="https://api.example.com",
        local_model_endpoint="http://localhost:11434",
        budget_enforcement=BudgetEnforcementConfig(
            hard_limit=True, warning_threshold_pct=80.0, log_every_spend=True,
        ),
        audit_log_path=".apprentice/audit.log",
        retry_max_attempts=3,
        retry_backoff_base_seconds=1.0,
    )


def _make_mocks():
    """Create mock components for Apprentice constructor."""
    task_registry = MagicMock()
    task_config = MagicMock()
    task_config.budget_limit_usd = 100.0
    task_config.local_model_id = "local-model"
    task_config.remote_model_id = "test-model"
    task_registry.get.return_value = task_config
    task_registry.list_tasks.return_value = ["test_task"]

    confidence_engine = MagicMock()
    confidence_engine.get_score.return_value = 0.5
    confidence_engine.get_phase.return_value = PhaseLabel.active_learning

    sampling_scheduler = MagicMock()
    sampling_scheduler.get_rate.return_value = 0.5

    budget_manager = MagicMock()
    budget_manager.can_spend.return_value = True
    budget_manager.get_remaining.return_value = 50.0
    budget_manager.get_used.return_value = 50.0

    local_client = MagicMock()
    local_client.is_available = AsyncMock(return_value=True)
    local_client.predict = AsyncMock(return_value={"reply": "Hello"})

    remote_client = MagicMock()
    remote_client.predict = AsyncMock(return_value={"reply": "Hello"})
    remote_client.get_cost = MagicMock(return_value=0.01)

    router = MagicMock()
    router.route.return_value = ModelSource.remote

    audit_log = MagicMock()
    audit_log.log = AsyncMock()

    training_data_store = MagicMock()
    training_data_store.store = AsyncMock()

    return {
        "task_registry": task_registry,
        "confidence_engine": confidence_engine,
        "sampling_scheduler": sampling_scheduler,
        "budget_manager": budget_manager,
        "local_client": local_client,
        "remote_client": remote_client,
        "router": router,
        "audit_log": audit_log,
        "training_data_store": training_data_store,
    }


class TestMiddlewarePipelineInRun:
    """Tests that the middleware pipeline is actually invoked in run()."""

    @pytest.mark.asyncio
    async def test_pii_tokenized_before_model_call(self):
        """Verify that PII is tokenized before the model sees the input."""
        mocks = _make_mocks()

        # Track what input_data the model actually receives
        captured_inputs = []

        async def capture_predict(task_name, input_data):
            captured_inputs.append(input_data)
            return {"reply": "I see the data"}

        mocks["remote_client"].predict = AsyncMock(side_effect=capture_predict)

        apprentice = Apprentice(config=_make_config(), **mocks)

        # Attach PII middleware pipeline
        pii_config = PIITokenizerConfig(
            enabled_entity_types=["email"],
            sensitive_fields=[],
        )
        tokenizer = PIITokenizer(config=pii_config)
        apprentice._middleware_pipeline = MiddlewarePipeline([tokenizer])

        # Enter async context
        apprentice._running = True

        result = await apprentice.run("test_task", {"text": "Contact alice@example.com"})

        # The model should have seen tokenized data, not raw PII
        assert len(captured_inputs) == 1
        assert "alice@example.com" not in captured_inputs[0]["text"]
        assert "<PII:email:" in captured_inputs[0]["text"]

        # The returned output should be detokenized (if model echoed tokens)
        assert result.status == RunStatus.success

    @pytest.mark.asyncio
    async def test_output_detokenized_for_user(self):
        """Verify that output is detokenized before returning to user."""
        mocks = _make_mocks()

        # Model echoes back the tokenized input
        async def echo_predict(task_name, input_data):
            return {"reply": input_data.get("text", "")}

        mocks["remote_client"].predict = AsyncMock(side_effect=echo_predict)

        apprentice = Apprentice(config=_make_config(), **mocks)

        pii_config = PIITokenizerConfig(
            enabled_entity_types=["email"],
            sensitive_fields=[],
        )
        tokenizer = PIITokenizer(config=pii_config)
        apprentice._middleware_pipeline = MiddlewarePipeline([tokenizer])
        apprentice._running = True

        result = await apprentice.run("test_task", {"text": "Email alice@example.com"})

        # Output should have the original PII restored
        assert "alice@example.com" in result.output["reply"]

    @pytest.mark.asyncio
    async def test_training_data_receives_tokenized(self):
        """Verify that training data store receives tokenized data, not raw PII."""
        mocks = _make_mocks()

        # Track what gets stored
        stored_inputs = []
        stored_outputs = []

        async def capture_store(task_name, input_data, output):
            stored_inputs.append(input_data)
            stored_outputs.append(output)

        mocks["training_data_store"].store = AsyncMock(side_effect=capture_store)
        mocks["remote_client"].predict = AsyncMock(return_value={"reply": "OK"})

        apprentice = Apprentice(config=_make_config(), **mocks)

        pii_config = PIITokenizerConfig(
            enabled_entity_types=["email"],
            sensitive_fields=[],
        )
        tokenizer = PIITokenizer(config=pii_config)
        apprentice._middleware_pipeline = MiddlewarePipeline([tokenizer])
        apprentice._running = True

        await apprentice.run("test_task", {"text": "Email alice@example.com"})

        # Training data store should have received tokenized input
        assert len(stored_inputs) == 1
        assert "alice@example.com" not in stored_inputs[0]["text"]
        assert "<PII:email:" in stored_inputs[0]["text"]

    @pytest.mark.asyncio
    async def test_no_middleware_pipeline_passthrough(self):
        """Verify run() works normally when no middleware pipeline is set."""
        mocks = _make_mocks()
        mocks["remote_client"].predict = AsyncMock(return_value={"reply": "Hello"})

        apprentice = Apprentice(config=_make_config(), **mocks)
        apprentice._middleware_pipeline = None
        apprentice._running = True

        result = await apprentice.run("test_task", {"text": "alice@example.com"})
        assert result.status == RunStatus.success

    @pytest.mark.asyncio
    async def test_sensitive_field_tokenized_in_run(self):
        """Verify sensitive fields are tokenized in the run pipeline."""
        mocks = _make_mocks()

        captured_inputs = []

        async def capture_predict(task_name, input_data):
            captured_inputs.append(input_data)
            return {"reply": "OK"}

        mocks["remote_client"].predict = AsyncMock(side_effect=capture_predict)

        apprentice = Apprentice(config=_make_config(), **mocks)

        pii_config = PIITokenizerConfig(
            enabled_entity_types=[],
            sensitive_fields=["password"],
        )
        tokenizer = PIITokenizer(config=pii_config)
        apprentice._middleware_pipeline = MiddlewarePipeline([tokenizer])
        apprentice._running = True

        await apprentice.run("test_task", {"password": "super_secret", "name": "Alice"})

        assert len(captured_inputs) == 1
        assert "super_secret" not in captured_inputs[0]["password"]
        assert captured_inputs[0]["name"] == "Alice"
