"""
Training Pipeline composition - wires together training data store, fine-tuning orchestrator, and model validator.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Import child components with src. prefix for robustness
import src.training_data_store as training_data_store
import src.fine_tuning_orchestrator as fine_tuning_orchestrator
import src.model_validator as model_validator


# ============================================================================
# Re-export all types from the contract
# ============================================================================

# Enums
class PipelineStatus(str, Enum):
    """Current state of the training pipeline."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    SKIPPED_BUDGET = "SKIPPED_BUDGET"
    SKIPPED_COOLDOWN = "SKIPPED_COOLDOWN"
    SKIPPED_ALREADY_RUNNING = "SKIPPED_ALREADY_RUNNING"


class PipelineStepName(str, Enum):
    """Pipeline step identifier."""
    DATA_SNAPSHOT = "DATA_SNAPSHOT"
    BUDGET_CHECK = "BUDGET_CHECK"
    FINE_TUNE = "FINE_TUNE"
    VALIDATE = "VALIDATE"
    PROMOTE = "PROMOTE"


class PipelineStepStatus(str, Enum):
    """Status of an individual pipeline step."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# Data Models
class BudgetConstraint(BaseModel):
    """Budget constraints for the pipeline."""
    max_monthly_budget_usd: float = Field(..., gt=0.0)
    current_month_spent_usd: float = Field(default=0.0, ge=0.0)
    estimated_cost_per_run_usd: float = Field(..., ge=0.0)
    budget_reset_day: int = Field(default=1, ge=1, le=28)


class PipelineTriggerPolicy(BaseModel):
    """Policy configuration for when pipeline should trigger."""
    min_new_examples: int = Field(..., ge=1)
    min_interval_seconds: float = Field(default=3600.0, ge=0.0)
    max_consecutive_failures: int = Field(default=3, ge=1, le=100)
    confidence_ceiling: float = Field(default=0.95, ge=0.0, le=1.0)


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""
    task_type: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-]+$")
    base_model: str = Field(..., min_length=1)
    trigger_policy: PipelineTriggerPolicy
    budget: BudgetConstraint
    state_file_path: str = Field(..., min_length=1)
    run_history_limit: int = Field(default=100, ge=1, le=10000)


class DataSnapshot(BaseModel):
    """Immutable record of training data at run start."""
    model_config = ConfigDict(frozen=True)

    task_type: str
    data_version_hash: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    total_examples: int = Field(..., ge=0)
    training_examples: int = Field(..., ge=0)
    validation_examples: int = Field(..., ge=0)
    snapshot_timestamp_utc: str


class PipelineStepRecord(BaseModel):
    """Record of a single step execution."""
    step_name: PipelineStepName
    status: PipelineStepStatus
    started_at_utc: str = ""
    completed_at_utc: str = ""
    duration_seconds: float = 0.0
    error_detail: str = ""
    result_summary: dict = Field(default_factory=dict)


class PipelineRun(BaseModel):
    """Immutable record of a single pipeline execution."""
    run_id: str = Field(..., min_length=1)
    task_type: str
    status: PipelineStatus
    data_snapshot: Optional[DataSnapshot] = None
    steps: List[PipelineStepRecord]
    started_at_utc: str
    completed_at_utc: str = ""
    duration_seconds: float = 0.0
    fine_tune_run_id: str = ""
    model_version_id: str = ""
    promotion_decision: str = ""
    error_detail: str = ""
    budget_spent_usd: float = 0.0
    metadata: dict = Field(default_factory=dict)


class PipelineState(BaseModel):
    """Internal mutable state of the pipeline."""
    task_type: str
    current_status: PipelineStatus
    consecutive_failures: int = Field(..., ge=0)
    last_run_completed_at_utc: str = ""
    last_successful_data_version_hash: str = ""
    training_example_count_at_last_success: int = Field(default=0, ge=0)
    current_month_spent_usd: float = Field(default=0.0, ge=0.0)
    budget_period_start_utc: str = ""
    active_model_version_id: str = ""
    run_history: List[PipelineRun] = Field(default_factory=list)


class TriggerEvaluation(BaseModel):
    """Result of evaluating whether pipeline should trigger."""
    should_trigger: bool
    reason: str
    new_examples_since_last_run: int = Field(..., ge=0)
    seconds_since_last_run: float = Field(..., ge=-1.0)
    consecutive_failures: int = Field(..., ge=0)
    budget_remaining_usd: float = Field(..., ge=0.0)
    evaluated_at_utc: str


class AddExampleInput(BaseModel):
    """Input for add_example function."""
    input: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)
    task_type: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-]+$")
    timestamp: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
    metadata: dict = Field(default_factory=dict)


class AddExampleOutput(BaseModel):
    """Result of adding an example."""
    task_type: str
    example_count: int = Field(..., ge=1)
    training_count: int = Field(..., ge=0)
    validation_count: int = Field(..., ge=0)
    batch_ready: bool
    trigger_evaluation: TriggerEvaluation


class HyperparameterOverrides(BaseModel):
    """Optional hyperparameter overrides."""
    model_config = ConfigDict(frozen=True)

    learning_rate: Optional[float] = Field(None, gt=0.0, le=1.0)
    num_epochs: Optional[int] = Field(None, ge=1, le=100)
    batch_size: Optional[int] = Field(None, ge=1, le=512)
    lora_rank: Optional[int] = Field(None, ge=1, le=256)
    lora_alpha: Optional[int] = Field(None, ge=1, le=512)


class PipelineSuccess(BaseModel):
    """Successful pipeline result."""
    status: str = Field("success", pattern="^success$")
    run: PipelineRun
    model_version_id: str = Field(..., min_length=1)
    promotion_decision: str
    aggregate_validation_score: float = Field(..., ge=0.0, le=1.0)


class PipelineFailure(BaseModel):
    """Failed pipeline result."""
    status: str = Field("failure", pattern="^failure$")
    run: PipelineRun
    failed_step: PipelineStepName
    error_detail: str
    retry_recommended: bool


# Union type
PipelineResult = PipelineSuccess | PipelineFailure


class PipelineRunHistory(BaseModel):
    """Aggregated pipeline run history with summary statistics."""
    task_type: str
    runs: List[PipelineRun]
    total_runs: int = Field(..., ge=0)
    successful_runs: int = Field(..., ge=0)
    failed_runs: int = Field(..., ge=0)
    consecutive_failures: int = Field(..., ge=0)
    total_budget_spent_usd: float = Field(..., ge=0.0)
    last_successful_run_id: str = ""
    last_run_completed_at_utc: str = ""


# ============================================================================
# Training Pipeline Implementation
# ============================================================================

class TrainingPipeline:
    """Orchestrates the full training data → fine-tuning → validation → promotion lifecycle."""

    def __init__(self):
        """Initialize an empty pipeline. Must call initialize() before use."""
        self._config: Optional[PipelineConfig] = None
        self._data_store: Any = None
        self._orchestrator: Any = None
        self._validator: Any = None
        self._state: Optional[PipelineState] = None
        self._initialized = False

    def initialize(
        self,
        config: PipelineConfig,
        data_store: Any,
        orchestrator: Any,
        validator: Any,
    ) -> None:
        """Initialize the pipeline with config and dependencies."""
        if self._initialized:
            raise RuntimeError("Pipeline has already been initialized")

        # Validate dependencies
        if data_store is None or orchestrator is None or validator is None:
            raise TypeError("All dependencies (data_store, orchestrator, validator) must be non-None")

        # Store config and dependencies
        self._config = config
        self._data_store = data_store
        self._orchestrator = orchestrator
        self._validator = validator

        # Create state file directory
        state_dir = os.path.dirname(config.state_file_path)
        if state_dir:
            try:
                os.makedirs(state_dir, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise IOError(f"Cannot create state file directory '{state_dir}': {e}")

        # Load or initialize state
        self._state = self._load_or_initialize_state()

        # Initialize training data store
        try:
            # Convert to training_data_store config
            tds_config = training_data_store.TrainingDataStoreConfig(
                base_dir=os.path.join(state_dir, "training_data"),
                holdout_percentage=0.10,
                fine_tune_batch_size=100,
            )
            data_store.initialize(tds_config)
        except Exception as e:
            raise IOError(f"Data store initialization failed: {e}")

        # Persist initial state
        self._persist_state()

        self._initialized = True

    def add_example(self, example: AddExampleInput) -> AddExampleOutput:
        """Add a training example and return trigger evaluation."""
        if not self._initialized:
            raise RuntimeError("Pipeline has not been initialized. Call initialize() first.")

        # Validate task type
        if example.task_type != self._config.task_type:
            raise ValueError(f"Example task_type '{example.task_type}' does not match pipeline task_type '{self._config.task_type}'")

        # Convert to training_data_store format
        tds_example = training_data_store.TrainingExample(
            input=example.input,
            output=example.output,
            task_type=example.task_type,
            timestamp=example.timestamp,
            metadata=example.metadata,
        )

        # Delegate to data store
        try:
            result = self._data_store.add_example(tds_example)
        except Exception as e:
            raise IOError(f"Data store write error: {e}")

        # Evaluate trigger
        trigger_eval = self.should_trigger()

        # Return AddExampleOutput
        return AddExampleOutput(
            task_type=result.task_type,
            example_count=result.example_count,
            training_count=result.training_count,
            validation_count=result.validation_count,
            batch_ready=result.batch_ready,
            trigger_evaluation=trigger_eval,
        )

    def should_trigger(self) -> TriggerEvaluation:
        """Evaluate whether pipeline should trigger."""
        if not self._initialized:
            raise RuntimeError("Pipeline has not been initialized. Call initialize() first.")

        evaluated_at_utc = datetime.now(timezone.utc).isoformat()

        # Get current example count
        try:
            count_result = self._data_store.get_example_count(self._config.task_type)
            current_training_count = count_result.training
        except Exception:
            current_training_count = 0

        # Calculate new examples since last run
        new_examples = max(0, current_training_count - self._state.training_example_count_at_last_success)

        # Calculate time since last run
        if self._state.last_run_completed_at_utc:
            last_run_time = datetime.fromisoformat(self._state.last_run_completed_at_utc.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            seconds_since_last = (now - last_run_time).total_seconds()
        else:
            seconds_since_last = -1.0

        # Calculate budget remaining
        budget_remaining = self._config.budget.max_monthly_budget_usd - self._state.current_month_spent_usd

        # Check all trigger conditions
        reasons = []
        should_trigger = True

        # 1. Check new examples
        if new_examples < self._config.trigger_policy.min_new_examples:
            should_trigger = False
            reasons.append(f"Insufficient new examples: {new_examples} < {self._config.trigger_policy.min_new_examples}")

        # 2. Check interval
        if seconds_since_last >= 0 and seconds_since_last < self._config.trigger_policy.min_interval_seconds:
            should_trigger = False
            reasons.append(f"Cooldown period not elapsed: {seconds_since_last:.1f}s < {self._config.trigger_policy.min_interval_seconds}s")

        # 3. Check consecutive failures
        if self._state.consecutive_failures >= self._config.trigger_policy.max_consecutive_failures:
            should_trigger = False
            reasons.append(f"Max consecutive failures reached: {self._state.consecutive_failures} >= {self._config.trigger_policy.max_consecutive_failures}")

        # 4. Check budget
        if budget_remaining < self._config.budget.estimated_cost_per_run_usd:
            should_trigger = False
            reasons.append(f"Insufficient budget: {budget_remaining:.2f} < {self._config.budget.estimated_cost_per_run_usd:.2f}")

        # 5. Check if already running
        if self._state.current_status == PipelineStatus.RUNNING:
            should_trigger = False
            reasons.append("Pipeline is already running")

        if should_trigger:
            reason = f"All trigger conditions met: {new_examples} new examples, budget available"
        else:
            reason = "; ".join(reasons)

        return TriggerEvaluation(
            should_trigger=should_trigger,
            reason=reason,
            new_examples_since_last_run=new_examples,
            seconds_since_last_run=seconds_since_last,
            consecutive_failures=self._state.consecutive_failures,
            budget_remaining_usd=budget_remaining,
            evaluated_at_utc=evaluated_at_utc,
        )

    def run_pipeline(self, hyperparameter_overrides: Optional[HyperparameterOverrides] = None) -> PipelineResult:
        """Execute the full pipeline: DATA_SNAPSHOT → BUDGET_CHECK → FINE_TUNE → VALIDATE → PROMOTE."""
        if not self._initialized:
            raise RuntimeError("Pipeline has not been initialized. Call initialize() first.")

        # Generate run ID
        import uuid
        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)

        # Initialize run record
        run = PipelineRun(
            run_id=run_id,
            task_type=self._config.task_type,
            status=PipelineStatus.RUNNING,
            steps=[
                PipelineStepRecord(step_name=PipelineStepName.DATA_SNAPSHOT, status=PipelineStepStatus.PENDING),
                PipelineStepRecord(step_name=PipelineStepName.BUDGET_CHECK, status=PipelineStepStatus.PENDING),
                PipelineStepRecord(step_name=PipelineStepName.FINE_TUNE, status=PipelineStepStatus.PENDING),
                PipelineStepRecord(step_name=PipelineStepName.VALIDATE, status=PipelineStepStatus.PENDING),
                PipelineStepRecord(step_name=PipelineStepName.PROMOTE, status=PipelineStepStatus.PENDING),
            ],
            started_at_utc=started_at.isoformat(),
        )

        self._state.current_status = PipelineStatus.RUNNING

        try:
            # STEP 1: DATA_SNAPSHOT
            run.steps[0] = self._execute_data_snapshot_step(run.steps[0])
            if run.steps[0].status == PipelineStepStatus.FAILED:
                return self._handle_pipeline_failure(run, 0, "Data snapshot failed", run.steps[0].error_detail)

            # Check minimum training examples
            training_examples = self._data_store.iter_training_examples(self._config.task_type)
            if len(training_examples) < 10:  # Minimum threshold
                return self._handle_pipeline_failure(
                    run, 0, "INSUFFICIENT_DATA",
                    f"Insufficient training examples: {len(training_examples)} < 10"
                )

            # STEP 2: BUDGET_CHECK
            run.steps[1] = self._execute_budget_check_step(run.steps[1])
            if run.steps[1].status == PipelineStepStatus.FAILED:
                return self._handle_pipeline_failure(run, 1, "SKIPPED_BUDGET", run.steps[1].error_detail)

            # STEP 3: FINE_TUNE
            run.steps[2] = self._execute_fine_tune_step(run.steps[2], training_examples, hyperparameter_overrides, run)
            if run.steps[2].status == PipelineStepStatus.FAILED:
                return self._handle_pipeline_failure(run, 2, "Fine-tuning failed", run.steps[2].error_detail)

            # STEP 4: VALIDATE
            run.steps[3] = self._execute_validate_step(run.steps[3], run.model_version_id)
            if run.steps[3].status == PipelineStepStatus.FAILED:
                return self._handle_pipeline_failure(run, 3, "Validation failed", run.steps[3].error_detail)

            # STEP 5: PROMOTE
            run.steps[4] = self._execute_promote_step(run.steps[4], run.steps[3])
            if run.steps[4].status == PipelineStepStatus.FAILED:
                return self._handle_pipeline_failure(run, 4, "Promotion failed", run.steps[4].error_detail)

            # Success!
            return self._handle_pipeline_success(run)

        except Exception as e:
            # Unexpected error
            return self._handle_pipeline_failure(run, 0, "Unexpected error", str(e))

    def get_status(self) -> PipelineStatus:
        """Get current pipeline status."""
        if not self._initialized:
            raise RuntimeError("Pipeline has not been initialized. Call initialize() first.")

        return self._state.current_status

    def get_run_history(self, limit: int = 50) -> PipelineRunHistory:
        """Get pipeline run history."""
        if not self._initialized:
            raise RuntimeError("Pipeline has not been initialized. Call initialize() first.")

        # Sort runs by started_at descending
        sorted_runs = sorted(
            self._state.run_history,
            key=lambda r: r.started_at_utc,
            reverse=True
        )

        # Apply limit
        limited_runs = sorted_runs[:limit]

        # Calculate stats
        total_runs = len(self._state.run_history)
        successful_runs = sum(1 for r in self._state.run_history if r.status == PipelineStatus.COMPLETED)
        failed_runs = sum(1 for r in self._state.run_history if r.status == PipelineStatus.FAILED)
        total_budget_spent = sum(r.budget_spent_usd for r in self._state.run_history)

        last_successful_run_id = ""
        for r in sorted_runs:
            if r.status == PipelineStatus.COMPLETED:
                last_successful_run_id = r.run_id
                break

        last_run_completed_at = sorted_runs[0].completed_at_utc if sorted_runs else ""

        return PipelineRunHistory(
            task_type=self._config.task_type,
            runs=limited_runs,
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            consecutive_failures=self._state.consecutive_failures,
            total_budget_spent_usd=total_budget_spent,
            last_successful_run_id=last_successful_run_id,
            last_run_completed_at_utc=last_run_completed_at,
        )

    def reset_consecutive_failures(self) -> None:
        """Manually reset consecutive failure counter."""
        if not self._initialized:
            raise RuntimeError("Pipeline has not been initialized. Call initialize() first.")

        self._state.consecutive_failures = 0
        self._persist_state()

    # ========================================================================
    # Internal Helper Methods
    # ========================================================================

    def _load_or_initialize_state(self) -> PipelineState:
        """Load state from file or create fresh state."""
        if os.path.exists(self._config.state_file_path):
            try:
                with open(self._config.state_file_path, 'r') as f:
                    state_data = json.load(f)
                    state = PipelineState(**state_data)

                    # Mark incomplete runs as FAILED
                    for run in state.run_history:
                        if run.status == PipelineStatus.RUNNING:
                            run.status = PipelineStatus.FAILED
                            run.error_detail = "Pipeline crashed during execution (recovered on restart)"

                    # Reset status to IDLE if it was RUNNING (crash recovery)
                    if state.current_status == PipelineStatus.RUNNING:
                        state.current_status = PipelineStatus.IDLE

                    # Check if budget period should be reset
                    state = self._maybe_reset_budget_period(state)

                    return state
            except Exception as e:
                raise IOError(f"Failed to load state file: {e}")
        else:
            # Create fresh state
            return PipelineState(
                task_type=self._config.task_type,
                current_status=PipelineStatus.IDLE,
                consecutive_failures=0,
                budget_period_start_utc=datetime.now(timezone.utc).isoformat(),
            )

    def _maybe_reset_budget_period(self, state: PipelineState) -> PipelineState:
        """Reset budget spending if the current date is in a new budget period."""
        if not state.budget_period_start_utc:
            state.budget_period_start_utc = datetime.now(timezone.utc).isoformat()
            return state

        period_start = datetime.fromisoformat(state.budget_period_start_utc.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        reset_day = self._config.budget.budget_reset_day

        # Determine the most recent reset boundary
        # Start from current month's reset day and check if we've passed it
        try:
            current_month_reset = now.replace(day=reset_day, hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            # If reset_day > days in current month, use last day of month
            import calendar
            last_day = calendar.monthrange(now.year, now.month)[1]
            current_month_reset = now.replace(day=min(reset_day, last_day), hour=0, minute=0, second=0, microsecond=0)

        if now >= current_month_reset and period_start < current_month_reset:
            # We've crossed a budget reset boundary since the last period start
            state.current_month_spent_usd = 0.0
            state.budget_period_start_utc = now.isoformat()

        return state

    def _persist_state(self) -> None:
        """Persist state to file."""
        try:
            # Prune run history to limit
            if len(self._state.run_history) > self._config.run_history_limit:
                # Keep most recent runs
                sorted_runs = sorted(
                    self._state.run_history,
                    key=lambda r: r.started_at_utc,
                    reverse=True
                )
                self._state.run_history = sorted_runs[:self._config.run_history_limit]

            with open(self._config.state_file_path, 'w') as f:
                json.dump(self._state.model_dump(), f, indent=2)
        except Exception as e:
            raise IOError(f"Failed to persist state: {e}")

    def _execute_data_snapshot_step(self, step: PipelineStepRecord) -> PipelineStepRecord:
        """Execute DATA_SNAPSHOT step."""
        step.status = PipelineStepStatus.RUNNING
        step.started_at_utc = datetime.now(timezone.utc).isoformat()

        try:
            # Get training data
            count_result = self._data_store.get_example_count(self._config.task_type)

            # Compute data version hash
            training_examples = self._data_store.iter_training_examples(self._config.task_type)
            hash_input = "".join([ex.input for ex in training_examples])
            data_version_hash = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

            step.status = PipelineStepStatus.COMPLETED
            step.completed_at_utc = datetime.now(timezone.utc).isoformat()
            step.result_summary = {
                "data_version_hash": data_version_hash,
                "total_examples": count_result.total,
                "training_examples": count_result.training,
                "validation_examples": count_result.validation,
            }
        except Exception as e:
            step.status = PipelineStepStatus.FAILED
            step.error_detail = str(e)

        return step

    def _execute_budget_check_step(self, step: PipelineStepRecord) -> PipelineStepRecord:
        """Execute BUDGET_CHECK step."""
        step.status = PipelineStepStatus.RUNNING
        step.started_at_utc = datetime.now(timezone.utc).isoformat()

        budget_remaining = self._config.budget.max_monthly_budget_usd - self._state.current_month_spent_usd
        estimated_cost = self._config.budget.estimated_cost_per_run_usd

        if budget_remaining < estimated_cost:
            step.status = PipelineStepStatus.FAILED
            step.error_detail = f"Budget exhausted: remaining={budget_remaining:.2f}, estimated={estimated_cost:.2f}"
        else:
            step.status = PipelineStepStatus.COMPLETED

        step.completed_at_utc = datetime.now(timezone.utc).isoformat()
        return step

    def _execute_fine_tune_step(
        self,
        step: PipelineStepRecord,
        training_examples: List,
        hyperparameter_overrides: Optional[HyperparameterOverrides],
        run: PipelineRun
    ) -> PipelineStepRecord:
        """Execute FINE_TUNE step."""
        step.status = PipelineStepStatus.RUNNING
        step.started_at_utc = datetime.now(timezone.utc).isoformat()

        try:
            # Convert to orchestrator format
            ft_examples = []
            for ex in training_examples:
                # Handle both real and mock timestamp
                timestamp = getattr(ex, 'timestamp', None)
                if timestamp is None or not isinstance(timestamp, str):
                    timestamp = datetime.now(timezone.utc).isoformat()

                ft_ex = fine_tuning_orchestrator.TrainingExample(
                    example_id=hashlib.sha256(ex.input.encode()).hexdigest()[:16],
                    user_prompt=ex.input,
                    assistant_response=ex.output,
                    collected_at=timestamp,
                    source="training_pipeline",
                )
                ft_examples.append(ft_ex)

            # Convert hyperparameter overrides
            ft_overrides = None
            if hyperparameter_overrides:
                ft_overrides = fine_tuning_orchestrator.HyperparameterOverrides(
                    learning_rate=hyperparameter_overrides.learning_rate,
                    num_epochs=hyperparameter_overrides.num_epochs,
                    batch_size=hyperparameter_overrides.batch_size,
                    lora_rank=hyperparameter_overrides.lora_rank,
                    lora_alpha=hyperparameter_overrides.lora_alpha,
                )

            # Call orchestrator
            result = self._orchestrator.trigger_fine_tuning(
                training_examples=ft_examples,
                base_model=self._config.base_model,
                hyperparameter_overrides=ft_overrides,
            )

            # Check result
            if hasattr(result, 'status') and result.status == "error":
                step.status = PipelineStepStatus.FAILED
                step.error_detail = result.message
            else:
                step.status = PipelineStepStatus.COMPLETED
                # Convert to strings to handle both real and mock values
                run.model_version_id = str(getattr(result, 'version_id', ''))
                run.fine_tune_run_id = str(getattr(result, 'run_id', ''))
                run.budget_spent_usd = self._config.budget.estimated_cost_per_run_usd

        except Exception as e:
            step.status = PipelineStepStatus.FAILED
            step.error_detail = str(e)

        step.completed_at_utc = datetime.now(timezone.utc).isoformat()
        return step

    def _execute_validate_step(self, step: PipelineStepRecord, model_version_id: str) -> PipelineStepRecord:
        """Execute VALIDATE step."""
        step.status = PipelineStepStatus.RUNNING
        step.started_at_utc = datetime.now(timezone.utc).isoformat()

        try:
            # Call validator
            validation_result = self._validator.validate_candidate(
                candidate_model_id=model_version_id,
                test_set_id=None,
            )

            step.status = PipelineStepStatus.COMPLETED
            # Store only JSON-serializable data
            step.result_summary = {
                "aggregate_score": getattr(validation_result, 'aggregate_score', 0.0),
                "decision": getattr(validation_result, 'decision', 'unknown'),
                # Store a reference, not the whole object
                "model_id": model_version_id,
            }
            # Store the full validation result as an attribute (not serialized)
            step._validation_result = validation_result

        except Exception as e:
            step.status = PipelineStepStatus.FAILED
            step.error_detail = str(e)

        step.completed_at_utc = datetime.now(timezone.utc).isoformat()
        return step

    def _execute_promote_step(self, step: PipelineStepRecord, validate_step: PipelineStepRecord) -> PipelineStepRecord:
        """Execute PROMOTE step."""
        step.status = PipelineStepStatus.RUNNING
        step.started_at_utc = datetime.now(timezone.utc).isoformat()

        try:
            # Get validation result from the validate step
            validation_result = getattr(validate_step, '_validation_result', None)
            if validation_result is None:
                raise RuntimeError("Validation result not available from VALIDATE step")

            # Call validator promote
            promotion_record = self._validator.promote_candidate(validation_result)

            step.status = PipelineStepStatus.COMPLETED
            # Store only JSON-serializable data
            step.result_summary = {
                "promotion_decision": str(getattr(promotion_record, 'final_decision', 'unknown')),
            }

        except Exception as e:
            step.status = PipelineStepStatus.FAILED
            step.error_detail = str(e)

        step.completed_at_utc = datetime.now(timezone.utc).isoformat()
        return step

    def _handle_pipeline_success(self, run: PipelineRun) -> PipelineSuccess:
        """Handle successful pipeline completion."""
        run.status = PipelineStatus.COMPLETED
        run.completed_at_utc = datetime.now(timezone.utc).isoformat()

        # Update state
        self._state.current_status = PipelineStatus.IDLE
        self._state.consecutive_failures = 0
        self._state.last_run_completed_at_utc = run.completed_at_utc
        self._state.current_month_spent_usd += run.budget_spent_usd
        self._state.active_model_version_id = str(run.model_version_id)

        # Add to history
        self._state.run_history.append(run)
        self._persist_state()

        # Get validation score
        validate_step = run.steps[3]
        aggregate_score = validate_step.result_summary.get("aggregate_score", 0.0)

        # Get promotion decision
        promote_step = run.steps[4]
        promotion_decision = promote_step.result_summary.get("promotion_decision", "unknown")

        return PipelineSuccess(
            status="success",
            run=run,
            model_version_id=run.model_version_id,
            promotion_decision=promotion_decision,
            aggregate_validation_score=aggregate_score,
        )

    def _handle_pipeline_failure(
        self,
        run: PipelineRun,
        failed_step_idx: int,
        status_override: str,
        error_detail: str
    ) -> PipelineFailure:
        """Handle pipeline failure."""
        # Mark status
        if status_override == "INSUFFICIENT_DATA":
            run.status = PipelineStatus.INSUFFICIENT_DATA
        elif status_override == "SKIPPED_BUDGET":
            run.status = PipelineStatus.SKIPPED_BUDGET
        else:
            run.status = PipelineStatus.FAILED

        run.completed_at_utc = datetime.now(timezone.utc).isoformat()
        run.error_detail = error_detail

        # Mark subsequent steps as SKIPPED
        for i in range(failed_step_idx + 1, len(run.steps)):
            run.steps[i].status = PipelineStepStatus.SKIPPED

        # Update state
        self._state.current_status = PipelineStatus.IDLE
        self._state.consecutive_failures += 1
        self._state.last_run_completed_at_utc = run.completed_at_utc

        # Add to history
        self._state.run_history.append(run)
        self._persist_state()

        return PipelineFailure(
            status="failure",
            run=run,
            failed_step=run.steps[failed_step_idx].step_name,
            error_detail=error_detail,
            retry_recommended=True,
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "TrainingPipeline",
    "PipelineConfig",
    "PipelineTriggerPolicy",
    "BudgetConstraint",
    "PipelineStatus",
    "PipelineStepName",
    "PipelineStepStatus",
    "PipelineState",
    "PipelineRun",
    "PipelineStepRecord",
    "PipelineResult",
    "PipelineSuccess",
    "PipelineFailure",
    "PipelineRunHistory",
    "TriggerEvaluation",
    "AddExampleInput",
    "AddExampleOutput",
    "DataSnapshot",
    "HyperparameterOverrides",
]
