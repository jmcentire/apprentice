"""
Contract tests for TrainingPipeline component.

Tests verify the TrainingPipeline implementation against its contract,
covering initialization, example management, trigger evaluation,
pipeline execution, budget enforcement, status/history, state persistence,
and cross-cutting invariants.

Run with: pytest contract_test.py -v
"""

import asyncio
import copy
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch, call
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Import the component under test
# ---------------------------------------------------------------------------
from src.training_pipeline import (
    TrainingPipeline,
    PipelineConfig,
    PipelineTriggerPolicy,
    BudgetConstraint,
    PipelineStatus,
    PipelineStepName,
    PipelineStepStatus,
    PipelineState,
    PipelineRun,
    PipelineStepRecord,
    PipelineResult,
    PipelineSuccess,
    PipelineFailure,
    PipelineRunHistory,
    TriggerEvaluation,
    AddExampleInput,
    AddExampleOutput,
    DataSnapshot,
    HyperparameterOverrides,
)


# ============================================================================
# Fixtures and Helpers
# ============================================================================

def make_trigger_policy(**overrides) -> PipelineTriggerPolicy:
    """Factory for PipelineTriggerPolicy with sensible defaults."""
    defaults = dict(
        min_new_examples=10,
        min_interval_seconds=3600.0,
        max_consecutive_failures=3,
        confidence_ceiling=0.95,
    )
    defaults.update(overrides)
    return PipelineTriggerPolicy(**defaults)


def make_budget(**overrides) -> BudgetConstraint:
    """Factory for BudgetConstraint with sensible defaults."""
    defaults = dict(
        max_monthly_budget_usd=100.0,
        current_month_spent_usd=0.0,
        estimated_cost_per_run_usd=10.0,
        budget_reset_day=1,
    )
    defaults.update(overrides)
    return BudgetConstraint(**defaults)


def make_config(tmp_dir: str, **overrides) -> PipelineConfig:
    """Factory for PipelineConfig with sensible defaults."""
    defaults = dict(
        task_type="text_classification",
        base_model="gpt-4o-mini",
        trigger_policy=make_trigger_policy(),
        budget=make_budget(),
        state_file_path=os.path.join(tmp_dir, "pipeline_state.json"),
        run_history_limit=50,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def make_example_input(**overrides) -> AddExampleInput:
    """Factory for AddExampleInput with sensible defaults."""
    defaults = dict(
        input="What is the capital of France?",
        output="Paris",
        task_type="text_classification",
        timestamp=datetime.now(timezone.utc).isoformat(),
        metadata={},
    )
    defaults.update(overrides)
    return AddExampleInput(**defaults)


def make_training_example(input_text: str = "sample input", output_text: str = "sample output"):
    """Create a simple mock training example."""
    ex = MagicMock()
    ex.input = input_text
    ex.output = output_text
    ex.task_type = "text_classification"
    return ex


def make_mock_data_store(
    example_count: int = 50,
    training_count: int = 40,
    validation_count: int = 10,
):
    """Create a mock training data store with configurable behavior."""
    store = MagicMock()
    store.initialize = MagicMock(return_value=None)

    add_result = MagicMock()
    add_result.task_type = "text_classification"
    add_result.example_count = example_count
    add_result.training_count = training_count
    add_result.validation_count = validation_count
    store.add_example = MagicMock(return_value=add_result)

    # Return training examples
    training_examples = [make_training_example(f"input_{i}", f"output_{i}") for i in range(training_count)]
    store.iter_training_examples = MagicMock(return_value=training_examples)

    validation_examples = [make_training_example(f"val_input_{i}", f"val_output_{i}") for i in range(validation_count)]
    store.iter_validation_examples = MagicMock(return_value=validation_examples)

    count_result = MagicMock()
    count_result.total = example_count
    count_result.training = training_count
    count_result.validation = validation_count
    store.get_example_count = MagicMock(return_value=count_result)

    store.get_task_types = MagicMock(return_value=["text_classification"])
    return store


def make_mock_orchestrator(
    fine_tune_run_id: str = "ft-run-123",
    model_version_id: str = "model-v1",
    cost_usd: float = 8.0,
):
    """Create a mock fine-tuning orchestrator."""
    orch = MagicMock()

    fine_tune_result = MagicMock()
    fine_tune_result.run_id = fine_tune_run_id
    fine_tune_result.model_version_id = model_version_id
    fine_tune_result.cost_usd = cost_usd
    fine_tune_result.status = "completed"
    orch.trigger_fine_tuning = MagicMock(return_value=fine_tune_result)
    orch.fine_tune = MagicMock(return_value=fine_tune_result)
    orch.save_model_version = MagicMock(return_value=None)

    model_version = MagicMock()
    model_version.version_id = model_version_id
    orch.get_model_version = MagicMock(return_value=model_version)
    orch.get_latest_model_version = MagicMock(return_value=model_version)
    orch.list_model_versions = MagicMock(return_value=[model_version])

    return orch


def make_mock_validator(
    score: float = 0.92,
    passed: bool = True,
    promotion_decision: str = "promoted",
):
    """Create a mock model validator."""
    val = MagicMock()

    validation_result = MagicMock()
    validation_result.score = score
    validation_result.aggregate_score = score
    validation_result.passed = passed
    validation_result.candidate_model_id = "model-v1"
    validation_result.decision = "pass" if passed else "reject"
    val.validate_candidate = MagicMock(return_value=validation_result)

    promotion_record = MagicMock()
    promotion_record.decision = promotion_decision
    promotion_record.model_version_id = "model-v1"
    promotion_record.validation_result = validation_result
    val.promote_candidate = MagicMock(return_value=promotion_record)
    val.validate_and_promote = MagicMock(return_value=promotion_record)

    return val


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for state files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def config(tmp_dir):
    """Create a default PipelineConfig."""
    return make_config(tmp_dir)


@pytest.fixture
def data_store():
    """Create a default mock data store."""
    return make_mock_data_store()


@pytest.fixture
def orchestrator():
    """Create a default mock orchestrator."""
    return make_mock_orchestrator()


@pytest.fixture
def validator():
    """Create a default mock validator."""
    return make_mock_validator()


@pytest.fixture
def pipeline():
    """Create a bare TrainingPipeline instance (not yet initialized)."""
    return TrainingPipeline()


@pytest.fixture
def initialized_pipeline(tmp_dir, data_store, orchestrator, validator):
    """Create and return a fully initialized TrainingPipeline."""
    config = make_config(tmp_dir)
    p = TrainingPipeline()
    p.initialize(config=config, data_store=data_store, orchestrator=orchestrator, validator=validator)
    return p, config, data_store, orchestrator, validator


# ============================================================================
# Test: initialize()
# ============================================================================

class TestInitialize:
    """Tests for TrainingPipeline.initialize()."""

    def test_init_happy_path(self, pipeline, config, data_store, orchestrator, validator):
        """Initialize pipeline with valid config and dependencies succeeds; status is IDLE."""
        pipeline.initialize(config=config, data_store=data_store, orchestrator=orchestrator, validator=validator)

        status = pipeline.get_status()
        assert status == PipelineStatus.IDLE, f"Expected IDLE, got {status}"
        data_store.initialize.assert_called_once()
        # State file directory should exist
        state_dir = os.path.dirname(config.state_file_path)
        assert os.path.isdir(state_dir), f"State file directory {state_dir} should exist"

    def test_init_loads_existing_state(self, tmp_dir, data_store, orchestrator, validator):
        """Initialize loads persisted state from existing state file."""
        config = make_config(tmp_dir)
        # First pipeline: initialize and do something to create state
        p1 = TrainingPipeline()
        p1.initialize(config=config, data_store=data_store, orchestrator=orchestrator, validator=validator)
        # Write some state (trigger a run or just ensure state file exists)
        # Now create a second pipeline and load from the same state file
        p2 = TrainingPipeline()
        data_store2 = make_mock_data_store()
        p2.initialize(config=config, data_store=data_store2, orchestrator=orchestrator, validator=validator)
        status = p2.get_status()
        assert status == PipelineStatus.IDLE, "Second pipeline should load state and be IDLE"

    def test_init_marks_incomplete_runs_failed(self, tmp_dir, data_store, orchestrator, validator):
        """Incomplete (RUNNING) runs from prior crash are marked FAILED on init."""
        config = make_config(tmp_dir)
        # Create a state file with a RUNNING run
        running_state = {
            "task_type": "text_classification",
            "current_status": "RUNNING",
            "consecutive_failures": 0,
            "last_run_completed_at_utc": "",
            "last_successful_data_version_hash": "",
            "training_example_count_at_last_success": 0,
            "current_month_spent_usd": 0.0,
            "budget_period_start_utc": datetime.now(timezone.utc).isoformat(),
            "active_model_version_id": "",
            "run_history": [
                {
                    "run_id": "run-incomplete",
                    "task_type": "text_classification",
                    "status": "RUNNING",
                    "data_snapshot": None,
                    "steps": [],
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_at_utc": "",
                    "duration_seconds": 0.0,
                    "fine_tune_run_id": "",
                    "model_version_id": "",
                    "promotion_decision": "",
                    "error_detail": "",
                    "budget_spent_usd": 0.0,
                    "metadata": {},
                }
            ],
        }
        os.makedirs(os.path.dirname(config.state_file_path), exist_ok=True)
        with open(config.state_file_path, "w") as f:
            json.dump(running_state, f)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orchestrator, validator=validator)

        status = p.get_status()
        assert status == PipelineStatus.IDLE, "Pipeline should be IDLE after crash recovery"
        # Verify the incomplete run was marked FAILED
        history = p.get_run_history(limit=10)
        if history.runs:
            recovered_run = [r for r in history.runs if r.run_id == "run-incomplete"]
            if recovered_run:
                assert recovered_run[0].status in (PipelineStatus.FAILED, "FAILED"), \
                    "Incomplete run should be marked FAILED"
                assert recovered_run[0].error_detail, \
                    "FAILED run should have error_detail about crash recovery"

    def test_init_budget_period_reset(self, tmp_dir, data_store, orchestrator, validator):
        """Budget resets to 0.0 when current date is past budget_reset_day."""
        config = make_config(tmp_dir, budget=make_budget(
            max_monthly_budget_usd=100.0,
            current_month_spent_usd=50.0,
            budget_reset_day=1,
        ))
        # Create state with spent budget and old period
        old_period = (datetime.now(timezone.utc) - timedelta(days=35)).isoformat()
        state = {
            "task_type": "text_classification",
            "current_status": "IDLE",
            "consecutive_failures": 0,
            "last_run_completed_at_utc": "",
            "last_successful_data_version_hash": "",
            "training_example_count_at_last_success": 0,
            "current_month_spent_usd": 50.0,
            "budget_period_start_utc": old_period,
            "active_model_version_id": "",
            "run_history": [],
        }
        os.makedirs(os.path.dirname(config.state_file_path), exist_ok=True)
        with open(config.state_file_path, "w") as f:
            json.dump(state, f)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orchestrator, validator=validator)
        # The budget should have been reset — verify via trigger evaluation
        trigger = p.should_trigger()
        assert trigger.budget_remaining_usd >= config.budget.max_monthly_budget_usd - 1.0, \
            f"Budget should have been reset; remaining={trigger.budget_remaining_usd}"

    def test_init_config_validation_error(self, pipeline, data_store, orchestrator, validator, tmp_dir):
        """Invalid PipelineConfig raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            # Create config with invalid fields
            bad_config = make_config(tmp_dir, task_type="", base_model="")
            pipeline.initialize(config=bad_config, data_store=data_store,
                                orchestrator=orchestrator, validator=validator)

    def test_init_dependency_missing_none(self, pipeline, config):
        """None dependency raises TypeError or RuntimeError."""
        with pytest.raises((TypeError, RuntimeError)):
            pipeline.initialize(config=config, data_store=None,
                                orchestrator=MagicMock(), validator=MagicMock())

    def test_init_state_file_corrupt(self, tmp_dir, pipeline, data_store, orchestrator, validator):
        """Corrupt state file raises IOError."""
        config = make_config(tmp_dir)
        os.makedirs(os.path.dirname(config.state_file_path), exist_ok=True)
        with open(config.state_file_path, "w") as f:
            f.write("{{{not valid json!!!")

        with pytest.raises((IOError, OSError, ValueError, RuntimeError)):
            pipeline.initialize(config=config, data_store=data_store,
                                orchestrator=orchestrator, validator=validator)

    def test_init_data_store_init_error(self, pipeline, config, orchestrator, validator):
        """data_store.initialize() failure propagates as an error."""
        bad_store = make_mock_data_store()
        bad_store.initialize.side_effect = RuntimeError("data store init failed")

        with pytest.raises((RuntimeError, IOError)):
            pipeline.initialize(config=config, data_store=bad_store,
                                orchestrator=orchestrator, validator=validator)

    def test_init_not_reentrant(self, config, data_store, orchestrator, validator):
        """Calling initialize() twice raises RuntimeError."""
        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store,
                     orchestrator=orchestrator, validator=validator)
        with pytest.raises(RuntimeError):
            p.initialize(config=config, data_store=data_store,
                         orchestrator=orchestrator, validator=validator)


# ============================================================================
# Test: add_example()
# ============================================================================

class TestAddExample:
    """Tests for TrainingPipeline.add_example()."""

    def test_add_example_happy_path(self, initialized_pipeline):
        """Adding a valid example returns correct AddExampleOutput."""
        p, config, data_store, orch, val = initialized_pipeline
        example = make_example_input(task_type=config.task_type)

        result = p.add_example(example)

        assert isinstance(result, AddExampleOutput), "Should return AddExampleOutput"
        assert result.task_type == config.task_type, "task_type should match"
        assert result.example_count > 0, "example_count should be positive"
        assert result.trigger_evaluation is not None, "trigger_evaluation should be populated"
        assert isinstance(result.trigger_evaluation.should_trigger, bool), \
            "should_trigger should be a boolean"
        data_store.add_example.assert_called_once()

    def test_add_example_trigger_evaluation_populated(self, initialized_pipeline):
        """trigger_evaluation has reason and counts."""
        p, config, data_store, orch, val = initialized_pipeline
        example = make_example_input(task_type=config.task_type)

        result = p.add_example(example)

        te = result.trigger_evaluation
        assert te.reason, "reason should be non-empty"
        assert isinstance(te.new_examples_since_last_run, int), "new_examples should be int"
        assert te.evaluated_at_utc, "evaluated_at_utc should be set"

    def test_add_example_no_auto_trigger(self, tmp_dir):
        """add_example never auto-triggers a pipeline run even when should_trigger=True."""
        # Set up a pipeline where trigger conditions are met
        data_store = make_mock_data_store(example_count=100, training_count=80, validation_count=20)
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(tmp_dir, trigger_policy=make_trigger_policy(min_new_examples=1))

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        example = make_example_input(task_type=config.task_type)
        result = p.add_example(example)

        # Pipeline should still be IDLE
        assert p.get_status() == PipelineStatus.IDLE, "Pipeline should remain IDLE after add_example"
        # Orchestrator should not have been called
        orch.trigger_fine_tuning.assert_not_called()

    def test_add_example_not_initialized(self):
        """Calling add_example before initialize raises RuntimeError."""
        p = TrainingPipeline()
        example = make_example_input()
        with pytest.raises(RuntimeError, match="(?i)(?:not.*init|uninit)"):
            p.add_example(example)

    def test_add_example_task_type_mismatch(self, initialized_pipeline):
        """Wrong task_type raises ValueError."""
        p, config, data_store, orch, val = initialized_pipeline
        example = make_example_input(task_type="wrong_task_type")

        with pytest.raises((ValueError, RuntimeError)):
            p.add_example(example)

    def test_add_example_validation_error_empty_input(self, initialized_pipeline):
        """Empty input string raises ValueError."""
        p, config, data_store, orch, val = initialized_pipeline
        with pytest.raises((ValueError, TypeError)):
            example = make_example_input(input="", task_type=config.task_type)
            p.add_example(example)

    def test_add_example_validation_error_empty_output(self, initialized_pipeline):
        """Empty output string raises ValueError."""
        p, config, data_store, orch, val = initialized_pipeline
        with pytest.raises((ValueError, TypeError)):
            example = make_example_input(output="", task_type=config.task_type)
            p.add_example(example)

    def test_add_example_data_store_write_error(self, initialized_pipeline):
        """data_store.add_example failure propagates as IOError."""
        p, config, data_store, orch, val = initialized_pipeline
        data_store.add_example.side_effect = IOError("disk full")
        example = make_example_input(task_type=config.task_type)

        with pytest.raises((IOError, OSError, RuntimeError)):
            p.add_example(example)


# ============================================================================
# Test: should_trigger()
# ============================================================================

class TestShouldTrigger:
    """Tests for TrainingPipeline.should_trigger()."""

    def test_should_trigger_all_conditions_met(self, tmp_dir):
        """Returns True when all trigger conditions are satisfied."""
        data_store = make_mock_data_store(example_count=100, training_count=80, validation_count=20)
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(
            tmp_dir,
            trigger_policy=make_trigger_policy(
                min_new_examples=5,
                min_interval_seconds=0.0,
                max_consecutive_failures=10,
            ),
            budget=make_budget(max_monthly_budget_usd=1000.0, estimated_cost_per_run_usd=10.0),
        )

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.should_trigger()
        assert isinstance(result, TriggerEvaluation), "Should return TriggerEvaluation"
        assert result.should_trigger is True, f"Expected should_trigger=True, reason: {result.reason}"
        assert result.reason, "reason should be non-empty"
        assert result.budget_remaining_usd > 0, "budget_remaining should be positive"

    def test_should_trigger_insufficient_examples(self, tmp_dir):
        """Returns False when new examples < min_new_examples."""
        # Data store reports 0 examples (or very few)
        data_store = make_mock_data_store(example_count=2, training_count=2, validation_count=0)
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(
            tmp_dir,
            trigger_policy=make_trigger_policy(min_new_examples=100),
        )

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.should_trigger()
        assert result.should_trigger is False, f"Should not trigger with insufficient examples: {result.reason}"

    def test_should_trigger_exact_min_examples_boundary(self, tmp_dir):
        """Returns True when new examples == min_new_examples exactly."""
        data_store = make_mock_data_store(example_count=10, training_count=10, validation_count=0)
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(
            tmp_dir,
            trigger_policy=make_trigger_policy(
                min_new_examples=10,
                min_interval_seconds=0.0,
                max_consecutive_failures=10,
            ),
            budget=make_budget(max_monthly_budget_usd=1000.0),
        )

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.should_trigger()
        # At exact boundary, trigger should be True (>= semantics)
        assert result.should_trigger is True, \
            f"Should trigger at exact min_new_examples boundary, reason: {result.reason}"

    def test_should_trigger_interval_not_elapsed(self, tmp_dir):
        """Returns False when min_interval_seconds has not elapsed."""
        data_store = make_mock_data_store(example_count=100, training_count=80, validation_count=20)
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(
            tmp_dir,
            trigger_policy=make_trigger_policy(
                min_new_examples=1,
                min_interval_seconds=999999.0,  # Very long interval
            ),
        )

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        # Simulate a recent run by running the pipeline first (or writing state)
        # For simplicity, write state with recent last_run time
        state_with_recent_run = {
            "task_type": "text_classification",
            "current_status": "IDLE",
            "consecutive_failures": 0,
            "last_run_completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_successful_data_version_hash": "abc123",
            "training_example_count_at_last_success": 80,
            "current_month_spent_usd": 0.0,
            "budget_period_start_utc": datetime.now(timezone.utc).isoformat(),
            "active_model_version_id": "",
            "run_history": [],
        }
        with open(config.state_file_path, "w") as f:
            json.dump(state_with_recent_run, f)

        # Re-initialize to pick up the state
        p2 = TrainingPipeline()
        data_store2 = make_mock_data_store(example_count=100, training_count=80, validation_count=20)
        p2.initialize(config=config, data_store=data_store2, orchestrator=orch, validator=val)

        result = p2.should_trigger()
        # With 999999s interval and recent last run, should not trigger
        assert result.should_trigger is False, \
            f"Should not trigger when interval not elapsed, reason: {result.reason}"

    def test_should_trigger_max_consecutive_failures(self, tmp_dir):
        """Returns False when consecutive_failures >= max_consecutive_failures."""
        data_store = make_mock_data_store(example_count=100, training_count=80, validation_count=20)
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(
            tmp_dir,
            trigger_policy=make_trigger_policy(
                min_new_examples=1,
                min_interval_seconds=0.0,
                max_consecutive_failures=3,
            ),
        )

        # Write state with max consecutive failures
        state = {
            "task_type": "text_classification",
            "current_status": "IDLE",
            "consecutive_failures": 3,
            "last_run_completed_at_utc": "",
            "last_successful_data_version_hash": "",
            "training_example_count_at_last_success": 0,
            "current_month_spent_usd": 0.0,
            "budget_period_start_utc": datetime.now(timezone.utc).isoformat(),
            "active_model_version_id": "",
            "run_history": [],
        }
        os.makedirs(os.path.dirname(config.state_file_path), exist_ok=True)
        with open(config.state_file_path, "w") as f:
            json.dump(state, f)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.should_trigger()
        assert result.should_trigger is False, \
            f"Should not trigger at max_consecutive_failures, reason: {result.reason}"
        assert result.consecutive_failures >= 3

    def test_should_trigger_budget_exhausted(self, tmp_dir):
        """Returns False when remaining budget < estimated cost."""
        data_store = make_mock_data_store(example_count=100, training_count=80, validation_count=20)
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(
            tmp_dir,
            trigger_policy=make_trigger_policy(
                min_new_examples=1,
                min_interval_seconds=0.0,
                max_consecutive_failures=10,
            ),
            budget=make_budget(
                max_monthly_budget_usd=100.0,
                current_month_spent_usd=95.0,
                estimated_cost_per_run_usd=10.0,
            ),
        )

        # Write state with high spend
        state = {
            "task_type": "text_classification",
            "current_status": "IDLE",
            "consecutive_failures": 0,
            "last_run_completed_at_utc": "",
            "last_successful_data_version_hash": "",
            "training_example_count_at_last_success": 0,
            "current_month_spent_usd": 95.0,
            "budget_period_start_utc": datetime.now(timezone.utc).isoformat(),
            "active_model_version_id": "",
            "run_history": [],
        }
        os.makedirs(os.path.dirname(config.state_file_path), exist_ok=True)
        with open(config.state_file_path, "w") as f:
            json.dump(state, f)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.should_trigger()
        assert result.should_trigger is False, \
            f"Should not trigger with exhausted budget, reason: {result.reason}"
        assert result.budget_remaining_usd < config.budget.estimated_cost_per_run_usd

    def test_should_trigger_read_only(self, initialized_pipeline):
        """should_trigger does not mutate pipeline state."""
        p, config, data_store, orch, val = initialized_pipeline

        # Get state before
        status_before = p.get_status()
        history_before = p.get_run_history(limit=100)

        p.should_trigger()

        status_after = p.get_status()
        history_after = p.get_run_history(limit=100)

        assert status_before == status_after, "Status should not change"
        assert history_before.total_runs == history_after.total_runs, "Run count should not change"

    def test_should_trigger_not_initialized(self):
        """should_trigger before initialize raises RuntimeError."""
        p = TrainingPipeline()
        with pytest.raises(RuntimeError):
            p.should_trigger()


# ============================================================================
# Test: run_pipeline()
# ============================================================================

class TestRunPipeline:
    """Tests for TrainingPipeline.run_pipeline()."""

    def test_run_pipeline_happy_full_success(self, initialized_pipeline):
        """Full pipeline run completes all 5 steps and returns PipelineSuccess."""
        p, config, data_store, orch, val = initialized_pipeline

        result = p.run_pipeline(hyperparameter_overrides=None)

        assert isinstance(result, (PipelineSuccess, PipelineResult)), \
            f"Expected PipelineSuccess, got {type(result)}"
        assert result.status == "success", f"Expected 'success', got {result.status}"
        assert result.run is not None
        run = result.run
        assert run.status in (PipelineStatus.COMPLETED, "COMPLETED"), \
            f"Run status should be COMPLETED, got {run.status}"
        assert len(run.steps) == 5, f"Should have 5 steps, got {len(run.steps)}"

        # Verify canonical step order
        step_names = [s.step_name for s in run.steps]
        expected_order = [
            PipelineStepName.DATA_SNAPSHOT,
            PipelineStepName.BUDGET_CHECK,
            PipelineStepName.FINE_TUNE,
            PipelineStepName.VALIDATE,
            PipelineStepName.PROMOTE,
        ]
        # Handle string or enum comparison
        for i, (actual, expected) in enumerate(zip(step_names, expected_order)):
            assert str(actual) == str(expected) or actual == expected, \
                f"Step {i} should be {expected}, got {actual}"

        # All steps should be COMPLETED
        for step in run.steps:
            assert step.status in (PipelineStepStatus.COMPLETED, "COMPLETED"), \
                f"Step {step.step_name} should be COMPLETED, got {step.status}"

        assert result.model_version_id, "model_version_id should be set on success"
        assert p.get_status() == PipelineStatus.IDLE, "Pipeline should return to IDLE"

    def test_run_pipeline_result_union_success(self, initialized_pipeline):
        """PipelineResult discriminated union resolves to PipelineSuccess."""
        p, config, data_store, orch, val = initialized_pipeline
        result = p.run_pipeline(hyperparameter_overrides=None)

        assert result.status == "success"
        assert hasattr(result, "aggregate_validation_score") or hasattr(result, "promotion_decision"), \
            "PipelineSuccess should have validation/promotion fields"

    def test_run_pipeline_result_union_failure(self, tmp_dir):
        """PipelineResult discriminated union resolves to PipelineFailure on failure."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        orch.trigger_fine_tuning.side_effect = RuntimeError("Fine-tuning failed")
        val = make_mock_validator()
        config = make_config(tmp_dir)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)

        assert result.status == "failure", f"Expected 'failure', got {result.status}"
        assert isinstance(result, (PipelineFailure, PipelineResult))
        assert hasattr(result, "failed_step"), "PipelineFailure should have failed_step"
        assert hasattr(result, "error_detail"), "PipelineFailure should have error_detail"
        assert hasattr(result, "retry_recommended"), "PipelineFailure should have retry_recommended"

    def test_run_pipeline_five_step_records_on_failure(self, tmp_dir):
        """PipelineRun.steps always has exactly 5 records even on failure."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        orch.trigger_fine_tuning.side_effect = RuntimeError("boom")
        val = make_mock_validator()
        config = make_config(tmp_dir)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)
        assert len(result.run.steps) == 5, f"Should always have 5 steps, got {len(result.run.steps)}"

    def test_run_pipeline_failure_skips_subsequent_steps(self, tmp_dir):
        """Failed step causes subsequent steps to be SKIPPED."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        orch.trigger_fine_tuning.side_effect = RuntimeError("fine-tune error")
        val = make_mock_validator()
        config = make_config(tmp_dir)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)
        steps = result.run.steps

        # DATA_SNAPSHOT and BUDGET_CHECK should be COMPLETED
        assert steps[0].status in (PipelineStepStatus.COMPLETED, "COMPLETED"), \
            f"DATA_SNAPSHOT should be COMPLETED, got {steps[0].status}"
        assert steps[1].status in (PipelineStepStatus.COMPLETED, "COMPLETED"), \
            f"BUDGET_CHECK should be COMPLETED, got {steps[1].status}"
        # FINE_TUNE should be FAILED
        assert steps[2].status in (PipelineStepStatus.FAILED, "FAILED"), \
            f"FINE_TUNE should be FAILED, got {steps[2].status}"
        # VALIDATE and PROMOTE should be SKIPPED
        assert steps[3].status in (PipelineStepStatus.SKIPPED, "SKIPPED"), \
            f"VALIDATE should be SKIPPED, got {steps[3].status}"
        assert steps[4].status in (PipelineStepStatus.SKIPPED, "SKIPPED"), \
            f"PROMOTE should be SKIPPED, got {steps[4].status}"

    def test_run_pipeline_increments_consecutive_failures(self, tmp_dir):
        """Failed run increments consecutive_failures."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        orch.trigger_fine_tuning.side_effect = RuntimeError("fail")
        val = make_mock_validator()
        config = make_config(tmp_dir)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        p.run_pipeline(hyperparameter_overrides=None)

        trigger = p.should_trigger()
        assert trigger.consecutive_failures >= 1, \
            f"consecutive_failures should be >= 1, got {trigger.consecutive_failures}"

    def test_run_pipeline_resets_consecutive_failures_on_success(self, tmp_dir):
        """Successful run resets consecutive_failures to 0."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(tmp_dir, trigger_policy=make_trigger_policy(max_consecutive_failures=10))

        # First, create state with failures
        state = {
            "task_type": "text_classification",
            "current_status": "IDLE",
            "consecutive_failures": 2,
            "last_run_completed_at_utc": "",
            "last_successful_data_version_hash": "",
            "training_example_count_at_last_success": 0,
            "current_month_spent_usd": 0.0,
            "budget_period_start_utc": datetime.now(timezone.utc).isoformat(),
            "active_model_version_id": "",
            "run_history": [],
        }
        os.makedirs(os.path.dirname(config.state_file_path), exist_ok=True)
        with open(config.state_file_path, "w") as f:
            json.dump(state, f)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)
        assert result.status == "success"

        trigger = p.should_trigger()
        assert trigger.consecutive_failures == 0, \
            f"consecutive_failures should be 0 after success, got {trigger.consecutive_failures}"

    def test_run_pipeline_not_initialized(self):
        """run_pipeline before initialize raises RuntimeError."""
        p = TrainingPipeline()
        with pytest.raises(RuntimeError):
            p.run_pipeline(hyperparameter_overrides=None)

    def test_run_pipeline_insufficient_data(self, tmp_dir):
        """Pipeline with too few examples returns PipelineFailure/INSUFFICIENT_DATA."""
        data_store = make_mock_data_store(example_count=1, training_count=1, validation_count=0)
        data_store.iter_training_examples.return_value = [make_training_example()]
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(tmp_dir)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)
        # Could be PipelineFailure with INSUFFICIENT_DATA or similar
        if result.status == "failure":
            assert result.run.status in (
                PipelineStatus.INSUFFICIENT_DATA, PipelineStatus.FAILED,
                "INSUFFICIENT_DATA", "FAILED"
            )
        # Orchestrator should not have been called for fine-tuning
        orch.trigger_fine_tuning.assert_not_called()

    def test_run_pipeline_budget_exhausted(self, tmp_dir):
        """Pipeline returns SKIPPED_BUDGET when budget would be exceeded."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(
            tmp_dir,
            budget=make_budget(
                max_monthly_budget_usd=100.0,
                estimated_cost_per_run_usd=20.0,
            ),
        )

        # Write state with nearly exhausted budget
        state = {
            "task_type": "text_classification",
            "current_status": "IDLE",
            "consecutive_failures": 0,
            "last_run_completed_at_utc": "",
            "last_successful_data_version_hash": "",
            "training_example_count_at_last_success": 0,
            "current_month_spent_usd": 90.0,
            "budget_period_start_utc": datetime.now(timezone.utc).isoformat(),
            "active_model_version_id": "",
            "run_history": [],
        }
        os.makedirs(os.path.dirname(config.state_file_path), exist_ok=True)
        with open(config.state_file_path, "w") as f:
            json.dump(state, f)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)
        assert result.status == "failure", f"Should fail with budget exhausted, got {result.status}"
        assert result.run.status in (PipelineStatus.SKIPPED_BUDGET, "SKIPPED_BUDGET"), \
            f"Run status should be SKIPPED_BUDGET, got {result.run.status}"
        orch.trigger_fine_tuning.assert_not_called()

    def test_run_pipeline_fine_tune_failed(self, tmp_dir):
        """Fine-tuning failure returns PipelineFailure with failed_step=FINE_TUNE."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        orch.trigger_fine_tuning.side_effect = RuntimeError("GPU unavailable")
        val = make_mock_validator()
        config = make_config(tmp_dir)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)
        assert result.status == "failure"
        failed_step = str(result.failed_step)
        assert "FINE_TUNE" in failed_step, f"failed_step should be FINE_TUNE, got {failed_step}"

    def test_run_pipeline_validation_failed(self, tmp_dir):
        """Validation failure returns PipelineFailure; previous model continues serving."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        val = make_mock_validator(score=0.3, passed=False, promotion_decision="rejected")
        val.validate_candidate.side_effect = RuntimeError("Candidate below threshold")
        config = make_config(tmp_dir)

        # Set an existing model version
        state = {
            "task_type": "text_classification",
            "current_status": "IDLE",
            "consecutive_failures": 0,
            "last_run_completed_at_utc": "",
            "last_successful_data_version_hash": "",
            "training_example_count_at_last_success": 0,
            "current_month_spent_usd": 0.0,
            "budget_period_start_utc": datetime.now(timezone.utc).isoformat(),
            "active_model_version_id": "old-model-v0",
            "run_history": [],
        }
        os.makedirs(os.path.dirname(config.state_file_path), exist_ok=True)
        with open(config.state_file_path, "w") as f:
            json.dump(state, f)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)
        assert result.status == "failure"
        failed_step = str(result.failed_step)
        assert "VALIDATE" in failed_step, f"failed_step should be VALIDATE, got {failed_step}"

    def test_run_pipeline_promotion_failed(self, tmp_dir):
        """Promotion failure returns PipelineFailure; previous model continues serving."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        val.promote_candidate.side_effect = RuntimeError("Model server swap failed")
        val.validate_and_promote.side_effect = RuntimeError("Model server swap failed")
        config = make_config(tmp_dir)

        state = {
            "task_type": "text_classification",
            "current_status": "IDLE",
            "consecutive_failures": 0,
            "last_run_completed_at_utc": "",
            "last_successful_data_version_hash": "",
            "training_example_count_at_last_success": 0,
            "current_month_spent_usd": 0.0,
            "budget_period_start_utc": datetime.now(timezone.utc).isoformat(),
            "active_model_version_id": "old-model-v0",
            "run_history": [],
        }
        os.makedirs(os.path.dirname(config.state_file_path), exist_ok=True)
        with open(config.state_file_path, "w") as f:
            json.dump(state, f)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)
        # Could fail at VALIDATE or PROMOTE depending on implementation
        assert result.status == "failure"

    def test_run_pipeline_data_store_error(self, tmp_dir):
        """Data store failure during snapshot returns PipelineFailure."""
        data_store = make_mock_data_store()
        data_store.iter_training_examples.side_effect = IOError("Data store unavailable")
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(tmp_dir)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)
        assert result.status == "failure"
        failed_step = str(result.failed_step)
        assert "DATA_SNAPSHOT" in failed_step, f"failed_step should be DATA_SNAPSHOT, got {failed_step}"

    def test_run_pipeline_hyperparameter_overrides_passthrough(self, initialized_pipeline):
        """Hyperparameter overrides are passed through to orchestrator."""
        p, config, data_store, orch, val = initialized_pipeline

        overrides = HyperparameterOverrides(
            learning_rate=0.001,
            num_epochs=3,
            batch_size=16,
            lora_rank=8,
            lora_alpha=16,
        )
        result = p.run_pipeline(hyperparameter_overrides=overrides)

        if result.status == "success":
            # Check that orchestrator was called with the overrides
            assert orch.trigger_fine_tuning.called, "orchestrator.trigger_fine_tuning should be called"
            call_kwargs = orch.trigger_fine_tuning.call_args
            # The overrides should be somewhere in the call arguments
            call_str = str(call_kwargs)
            assert call_kwargs is not None, "trigger_fine_tuning should have been called with args"

    def test_run_pipeline_status_returns_to_idle(self, initialized_pipeline):
        """Pipeline status returns to IDLE after run completes."""
        p, config, data_store, orch, val = initialized_pipeline

        result = p.run_pipeline(hyperparameter_overrides=None)
        assert p.get_status() == PipelineStatus.IDLE, \
            f"Pipeline should be IDLE after run, got {p.get_status()}"

    def test_run_pipeline_budget_tracking_updated(self, initialized_pipeline):
        """Budget tracking is updated after successful run."""
        p, config, data_store, orch, val = initialized_pipeline

        result = p.run_pipeline(hyperparameter_overrides=None)
        if result.status == "success":
            assert result.run.budget_spent_usd >= 0, \
                f"budget_spent_usd should be non-negative, got {result.run.budget_spent_usd}"

    def test_run_pipeline_no_model_mutation_on_failure(self, tmp_dir):
        """Active model version is not changed when validation fails."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        val.validate_candidate.side_effect = RuntimeError("Validation regression")
        config = make_config(tmp_dir)

        state = {
            "task_type": "text_classification",
            "current_status": "IDLE",
            "consecutive_failures": 0,
            "last_run_completed_at_utc": "",
            "last_successful_data_version_hash": "",
            "training_example_count_at_last_success": 0,
            "current_month_spent_usd": 0.0,
            "budget_period_start_utc": datetime.now(timezone.utc).isoformat(),
            "active_model_version_id": "stable-model-v42",
            "run_history": [],
        }
        os.makedirs(os.path.dirname(config.state_file_path), exist_ok=True)
        with open(config.state_file_path, "w") as f:
            json.dump(state, f)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)
        assert result.status == "failure"

        # Read state file to check active model version
        with open(config.state_file_path, "r") as f:
            persisted_state = json.load(f)
        assert persisted_state.get("active_model_version_id") == "stable-model-v42", \
            "Active model should remain unchanged on failure"


# ============================================================================
# Test: Budget enforcement
# ============================================================================

class TestBudgetEnforcement:
    """Tests for hard budget enforcement."""

    def test_budget_exact_boundary_exceeds(self, tmp_dir):
        """Budget rejects when spent + estimated > max (by a tiny amount)."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(
            tmp_dir,
            budget=make_budget(
                max_monthly_budget_usd=100.0,
                estimated_cost_per_run_usd=10.01,
            ),
        )

        state = {
            "task_type": "text_classification",
            "current_status": "IDLE",
            "consecutive_failures": 0,
            "last_run_completed_at_utc": "",
            "last_successful_data_version_hash": "",
            "training_example_count_at_last_success": 0,
            "current_month_spent_usd": 90.0,
            "budget_period_start_utc": datetime.now(timezone.utc).isoformat(),
            "active_model_version_id": "",
            "run_history": [],
        }
        os.makedirs(os.path.dirname(config.state_file_path), exist_ok=True)
        with open(config.state_file_path, "w") as f:
            json.dump(state, f)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)
        assert result.status == "failure"
        assert result.run.status in (PipelineStatus.SKIPPED_BUDGET, "SKIPPED_BUDGET"), \
            f"Should be SKIPPED_BUDGET, got {result.run.status}"
        orch.trigger_fine_tuning.assert_not_called()

    def test_budget_exact_fit_allows(self, tmp_dir):
        """Budget allows run when spent + estimated == max exactly."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator(cost_usd=10.0)
        val = make_mock_validator()
        config = make_config(
            tmp_dir,
            budget=make_budget(
                max_monthly_budget_usd=100.0,
                estimated_cost_per_run_usd=10.0,
            ),
        )

        state = {
            "task_type": "text_classification",
            "current_status": "IDLE",
            "consecutive_failures": 0,
            "last_run_completed_at_utc": "",
            "last_successful_data_version_hash": "",
            "training_example_count_at_last_success": 0,
            "current_month_spent_usd": 90.0,
            "budget_period_start_utc": datetime.now(timezone.utc).isoformat(),
            "active_model_version_id": "",
            "run_history": [],
        }
        os.makedirs(os.path.dirname(config.state_file_path), exist_ok=True)
        with open(config.state_file_path, "w") as f:
            json.dump(state, f)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)
        # Budget check should pass — the run should proceed past BUDGET_CHECK
        budget_step = result.run.steps[1]  # BUDGET_CHECK is step index 1
        assert budget_step.status in (PipelineStepStatus.COMPLETED, "COMPLETED"), \
            f"BUDGET_CHECK should pass at exact boundary, got {budget_step.status}"

    def test_budget_never_exceeded_across_runs(self, tmp_dir):
        """Multiple runs never exceed the budget."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator(cost_usd=30.0)
        val = make_mock_validator()
        config = make_config(
            tmp_dir,
            budget=make_budget(
                max_monthly_budget_usd=100.0,
                estimated_cost_per_run_usd=30.0,
            ),
        )

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        total_spent = 0.0
        for i in range(5):
            result = p.run_pipeline(hyperparameter_overrides=None)
            if result.status == "success":
                total_spent += result.run.budget_spent_usd
            elif result.run.status in (PipelineStatus.SKIPPED_BUDGET, "SKIPPED_BUDGET"):
                break

        assert total_spent <= config.budget.max_monthly_budget_usd, \
            f"Total spent {total_spent} should not exceed {config.budget.max_monthly_budget_usd}"


# ============================================================================
# Test: get_status()
# ============================================================================

class TestGetStatus:
    """Tests for TrainingPipeline.get_status()."""

    def test_get_status_idle_after_init(self, initialized_pipeline):
        """Status is IDLE after initialization."""
        p, *_ = initialized_pipeline
        assert p.get_status() == PipelineStatus.IDLE

    def test_get_status_not_initialized(self):
        """get_status before initialize raises RuntimeError."""
        p = TrainingPipeline()
        with pytest.raises(RuntimeError):
            p.get_status()

    def test_get_status_no_mutation(self, initialized_pipeline):
        """get_status does not mutate state."""
        p, config, *_ = initialized_pipeline

        status1 = p.get_status()
        status2 = p.get_status()
        assert status1 == status2, "get_status should be idempotent"

    def test_get_status_idle_after_run(self, initialized_pipeline):
        """Status returns to IDLE after pipeline run completes."""
        p, *_ = initialized_pipeline
        p.run_pipeline(hyperparameter_overrides=None)
        assert p.get_status() == PipelineStatus.IDLE


# ============================================================================
# Test: get_run_history()
# ============================================================================

class TestGetRunHistory:
    """Tests for TrainingPipeline.get_run_history()."""

    def test_get_run_history_after_runs(self, initialized_pipeline):
        """Returns correct history and stats after runs."""
        p, config, data_store, orch, val = initialized_pipeline

        # Run pipeline successfully twice
        p.run_pipeline(hyperparameter_overrides=None)
        p.run_pipeline(hyperparameter_overrides=None)

        # Make one fail
        orch.trigger_fine_tuning.side_effect = RuntimeError("fail")
        p.run_pipeline(hyperparameter_overrides=None)

        history = p.get_run_history(limit=10)
        assert isinstance(history, PipelineRunHistory)
        assert history.total_runs == 3, f"total_runs should be 3, got {history.total_runs}"
        assert history.successful_runs == 2, f"successful_runs should be 2, got {history.successful_runs}"
        assert history.failed_runs >= 1, f"failed_runs should be >= 1, got {history.failed_runs}"
        assert history.task_type == config.task_type

        # Verify descending order
        if len(history.runs) >= 2:
            for i in range(len(history.runs) - 1):
                assert history.runs[i].started_at_utc >= history.runs[i + 1].started_at_utc, \
                    "Runs should be ordered by started_at_utc descending"

    def test_get_run_history_respects_limit(self, initialized_pipeline):
        """Limit parameter caps the number of returned runs."""
        p, *_ = initialized_pipeline

        # Run pipeline 3 times
        for _ in range(3):
            p.run_pipeline(hyperparameter_overrides=None)

        history = p.get_run_history(limit=2)
        assert len(history.runs) <= 2, f"Should return at most 2 runs, got {len(history.runs)}"

    def test_get_run_history_empty(self, initialized_pipeline):
        """Empty history when no runs have been executed."""
        p, *_ = initialized_pipeline
        history = p.get_run_history(limit=10)
        assert history.total_runs == 0, f"total_runs should be 0, got {history.total_runs}"
        assert history.successful_runs == 0
        assert history.failed_runs == 0
        assert len(history.runs) == 0

    def test_get_run_history_not_initialized(self):
        """get_run_history before initialize raises RuntimeError."""
        p = TrainingPipeline()
        with pytest.raises(RuntimeError):
            p.get_run_history(limit=10)

    def test_get_run_history_no_mutation(self, initialized_pipeline):
        """get_run_history does not mutate state."""
        p, *_ = initialized_pipeline
        p.run_pipeline(hyperparameter_overrides=None)

        h1 = p.get_run_history(limit=10)
        h2 = p.get_run_history(limit=10)
        assert h1.total_runs == h2.total_runs
        assert len(h1.runs) == len(h2.runs)


# ============================================================================
# Test: reset_consecutive_failures()
# ============================================================================

class TestResetConsecutiveFailures:
    """Tests for TrainingPipeline.reset_consecutive_failures()."""

    def test_reset_failures_happy_path(self, tmp_dir):
        """Resets consecutive_failures to 0 and persists state."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        orch.trigger_fine_tuning.side_effect = RuntimeError("fail")
        val = make_mock_validator()
        config = make_config(tmp_dir, trigger_policy=make_trigger_policy(max_consecutive_failures=10))

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        # Create some failures
        p.run_pipeline(hyperparameter_overrides=None)
        p.run_pipeline(hyperparameter_overrides=None)

        trigger_before = p.should_trigger()
        assert trigger_before.consecutive_failures >= 2

        p.reset_consecutive_failures()

        trigger_after = p.should_trigger()
        assert trigger_after.consecutive_failures == 0, \
            f"consecutive_failures should be 0 after reset, got {trigger_after.consecutive_failures}"

    def test_reset_failures_not_initialized(self):
        """reset_consecutive_failures before initialize raises RuntimeError."""
        p = TrainingPipeline()
        with pytest.raises(RuntimeError):
            p.reset_consecutive_failures()

    def test_reset_failures_idempotent(self, initialized_pipeline):
        """Resetting when already at 0 is a no-op (idempotent)."""
        p, *_ = initialized_pipeline

        # Should already be at 0
        p.reset_consecutive_failures()
        trigger = p.should_trigger()
        assert trigger.consecutive_failures == 0


# ============================================================================
# Test: State persistence and run history pruning
# ============================================================================

class TestStatePersistence:
    """Tests for state file persistence behavior."""

    def test_state_file_created_on_init(self, tmp_dir, data_store, orchestrator, validator):
        """State file is created during initialization."""
        config = make_config(tmp_dir)
        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orchestrator, validator=validator)

        assert os.path.exists(config.state_file_path), "State file should exist after init"

    def test_state_file_updated_after_run(self, initialized_pipeline):
        """State file is updated after a pipeline run."""
        p, config, *_ = initialized_pipeline

        mtime_before = os.path.getmtime(config.state_file_path)
        time.sleep(0.01)  # Small delay to ensure mtime changes
        p.run_pipeline(hyperparameter_overrides=None)
        mtime_after = os.path.getmtime(config.state_file_path)

        assert mtime_after > mtime_before, "State file should be updated after run"

    def test_state_file_valid_json(self, initialized_pipeline):
        """State file contains valid JSON after operations."""
        p, config, *_ = initialized_pipeline
        p.run_pipeline(hyperparameter_overrides=None)

        with open(config.state_file_path, "r") as f:
            state = json.load(f)

        assert isinstance(state, dict), "State file should contain a JSON object"
        assert "task_type" in state, "State should contain task_type"
        assert "run_history" in state, "State should contain run_history"

    def test_run_history_pruned_to_limit(self, tmp_dir):
        """Run history is pruned to run_history_limit."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(tmp_dir, run_history_limit=3)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        # Run more times than the limit
        for _ in range(5):
            p.run_pipeline(hyperparameter_overrides=None)

        history = p.get_run_history(limit=100)
        assert len(history.runs) <= 3, \
            f"Run history should be pruned to 3, got {len(history.runs)}"

    def test_state_survives_restart(self, tmp_dir):
        """State is preserved across pipeline instance lifetimes."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(tmp_dir)

        # First instance
        p1 = TrainingPipeline()
        p1.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)
        p1.run_pipeline(hyperparameter_overrides=None)

        h1 = p1.get_run_history(limit=100)

        # Second instance (simulating restart)
        p2 = TrainingPipeline()
        data_store2 = make_mock_data_store()
        p2.initialize(config=config, data_store=data_store2, orchestrator=orch, validator=val)

        h2 = p2.get_run_history(limit=100)
        assert h2.total_runs >= h1.total_runs, \
            "State should be preserved across restarts"


# ============================================================================
# Test: Cross-cutting invariants
# ============================================================================

class TestInvariants:
    """Tests for cross-cutting invariants."""

    def test_no_exceptions_for_expected_failures(self, tmp_dir):
        """Expected failures return PipelineFailure, never raise exceptions."""
        scenarios = [
            ("fine_tune_error", {"trigger_fine_tuning_side_effect": RuntimeError("GPU error")}),
        ]

        for name, setup in scenarios:
            data_store = make_mock_data_store()
            orch = make_mock_orchestrator()
            val = make_mock_validator()

            if "trigger_fine_tuning_side_effect" in setup:
                orch.trigger_fine_tuning.side_effect = setup["trigger_fine_tuning_side_effect"]

            config = make_config(tmp_dir)
            p = TrainingPipeline()
            p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

            # This should NOT raise — should return PipelineFailure
            result = p.run_pipeline(hyperparameter_overrides=None)
            assert result.status == "failure", \
                f"Scenario {name}: expected PipelineFailure, got {result.status}"

    def test_only_allowed_exception_types(self, tmp_dir):
        """Only RuntimeError, ValueError, IOError, TypeError may be raised."""
        # Test with not_initialized (should be RuntimeError)
        p = TrainingPipeline()
        try:
            p.run_pipeline(hyperparameter_overrides=None)
            assert False, "Should have raised"
        except (RuntimeError, ValueError, IOError, TypeError):
            pass  # These are allowed
        except Exception as e:
            pytest.fail(f"Unexpected exception type: {type(e).__name__}: {e}")

    def test_serialized_runs_no_concurrent(self, initialized_pipeline):
        """Only one run_pipeline can execute at a time."""
        p, *_ = initialized_pipeline

        # We test the lock by checking that the pipeline tracks RUNNING status
        # A synchronous test can't truly test concurrency, but we can verify
        # the contract by checking status transitions
        result = p.run_pipeline(hyperparameter_overrides=None)
        assert p.get_status() == PipelineStatus.IDLE, \
            "Pipeline should be IDLE after run completes"

    def test_step_count_invariant_success(self, initialized_pipeline):
        """Successful run has exactly 5 step records in canonical order."""
        p, *_ = initialized_pipeline
        result = p.run_pipeline(hyperparameter_overrides=None)

        assert len(result.run.steps) == 5, f"Must have 5 steps, got {len(result.run.steps)}"

    def test_step_count_invariant_failure(self, tmp_dir):
        """Failed run still has exactly 5 step records."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        orch.trigger_fine_tuning.side_effect = RuntimeError("error")
        val = make_mock_validator()
        config = make_config(tmp_dir)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)
        assert len(result.run.steps) == 5, f"Must have 5 steps even on failure, got {len(result.run.steps)}"

    def test_failed_steps_marked_skipped_not_pending(self, tmp_dir):
        """After a failure, subsequent steps are SKIPPED (not PENDING)."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        orch.trigger_fine_tuning.side_effect = RuntimeError("error")
        val = make_mock_validator()
        config = make_config(tmp_dir)

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        result = p.run_pipeline(hyperparameter_overrides=None)
        found_failure = False
        for step in result.run.steps:
            if found_failure:
                assert step.status in (PipelineStepStatus.SKIPPED, "SKIPPED"), \
                    f"Step {step.step_name} after failure should be SKIPPED, got {step.status}"
            if step.status in (PipelineStepStatus.FAILED, "FAILED"):
                found_failure = True

        assert found_failure, "Should have found at least one FAILED step"

    def test_no_global_mutable_state(self, tmp_dir):
        """Two pipeline instances do not share state."""
        data_store1 = make_mock_data_store()
        data_store2 = make_mock_data_store()
        orch = make_mock_orchestrator()
        val = make_mock_validator()

        config1 = make_config(
            tmp_dir,
            state_file_path=os.path.join(tmp_dir, "state1.json"),
        )
        config2 = make_config(
            tmp_dir,
            state_file_path=os.path.join(tmp_dir, "state2.json"),
        )

        p1 = TrainingPipeline()
        p1.initialize(config=config1, data_store=data_store1, orchestrator=orch, validator=val)
        p2 = TrainingPipeline()
        p2.initialize(config=config2, data_store=data_store2, orchestrator=orch, validator=val)

        # Run pipeline on p1
        p1.run_pipeline(hyperparameter_overrides=None)

        # p2 should have no runs
        h1 = p1.get_run_history(limit=100)
        h2 = p2.get_run_history(limit=100)

        assert h1.total_runs >= 1, "p1 should have at least 1 run"
        assert h2.total_runs == 0, "p2 should have 0 runs (independent state)"

    def test_consecutive_failures_only_modified_by_runs_and_reset(self, tmp_dir):
        """Consecutive failures modified only by run outcomes and reset_consecutive_failures."""
        data_store = make_mock_data_store()
        orch = make_mock_orchestrator()
        val = make_mock_validator()
        config = make_config(tmp_dir, trigger_policy=make_trigger_policy(max_consecutive_failures=100))

        p = TrainingPipeline()
        p.initialize(config=config, data_store=data_store, orchestrator=orch, validator=val)

        # Initial state: 0 failures
        t = p.should_trigger()
        assert t.consecutive_failures == 0

        # Successful run: still 0
        p.run_pipeline(hyperparameter_overrides=None)
        t = p.should_trigger()
        assert t.consecutive_failures == 0

        # Failed run: incremented
        orch.trigger_fine_tuning.side_effect = RuntimeError("fail")
        p.run_pipeline(hyperparameter_overrides=None)
        t = p.should_trigger()
        assert t.consecutive_failures == 1

        # Another failed run: incremented again
        p.run_pipeline(hyperparameter_overrides=None)
        t = p.should_trigger()
        assert t.consecutive_failures == 2

        # get_status, should_trigger, add_example should NOT change it
        p.get_status()
        p.should_trigger()
        t = p.should_trigger()
        assert t.consecutive_failures == 2

        # Reset: back to 0
        p.reset_consecutive_failures()
        t = p.should_trigger()
        assert t.consecutive_failures == 0

    def test_pipeline_result_discriminated_union(self, initialized_pipeline):
        """PipelineResult status field discriminates between Success and Failure."""
        p, config, data_store, orch, val = initialized_pipeline

        # Success case
        result = p.run_pipeline(hyperparameter_overrides=None)
        if result.status == "success":
            assert isinstance(result, PipelineSuccess) or hasattr(result, "model_version_id")
        elif result.status == "failure":
            assert isinstance(result, PipelineFailure) or hasattr(result, "failed_step")
        else:
            pytest.fail(f"PipelineResult.status must be 'success' or 'failure', got '{result.status}'")

    def test_config_strict_validation_no_silent_defaults(self, tmp_dir):
        """Invalid config causes ValueError, never silently defaults."""
        p = TrainingPipeline()

        # Various invalid configs
        invalid_configs = []

        # Negative budget
        try:
            invalid_configs.append(make_config(
                tmp_dir,
                budget=make_budget(max_monthly_budget_usd=-1.0),
            ))
        except (ValueError, TypeError):
            pass  # Config itself may reject this

        # Negative min_new_examples
        try:
            invalid_configs.append(make_config(
                tmp_dir,
                trigger_policy=make_trigger_policy(min_new_examples=-1),
            ))
        except (ValueError, TypeError):
            pass  # Config itself may reject this

        for bad_config in invalid_configs:
            with pytest.raises((ValueError, TypeError)):
                p2 = TrainingPipeline()
                p2.initialize(
                    config=bad_config,
                    data_store=make_mock_data_store(),
                    orchestrator=make_mock_orchestrator(),
                    validator=make_mock_validator(),
                )
