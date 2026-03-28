"""
Request Router component for the Apprentice system.

This module implements the central decision-making engine that routes requests
through the phase × sampling × budget matrix, coordinating all collaborators
to make intelligent routing decisions.
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# Enums
# ============================================================================

class Phase(str, Enum):
    """The current operational phase of the system, derived from confidence scores."""
    PHASE_1 = "PHASE_1"
    PHASE_2 = "PHASE_2"
    PHASE_3 = "PHASE_3"


class RoutingAction(str, Enum):
    """The routing action the router intends to take or actually took."""
    REMOTE_ONLY = "REMOTE_ONLY"
    BOTH_RETURN_REMOTE = "BOTH_RETURN_REMOTE"
    BOTH_RETURN_LOCAL = "BOTH_RETURN_LOCAL"
    LOCAL_ONLY = "LOCAL_ONLY"
    LOCAL_FALLBACK_TO_REMOTE = "LOCAL_FALLBACK_TO_REMOTE"


class DualSendOutcome(str, Enum):
    """The four possible outcomes when sending to both backends concurrently."""
    BOTH_SUCCEEDED = "BOTH_SUCCEEDED"
    LOCAL_FAILED_REMOTE_SUCCEEDED = "LOCAL_FAILED_REMOTE_SUCCEEDED"
    LOCAL_SUCCEEDED_REMOTE_FAILED = "LOCAL_SUCCEEDED_REMOTE_FAILED"
    BOTH_FAILED = "BOTH_FAILED"


class TrainingExampleType(str, Enum):
    """Indicates the purpose of a collected training data record."""
    REMOTE_RESPONSE_EXAMPLE = "REMOTE_RESPONSE_EXAMPLE"
    COACHING_EVALUATION_PAIR = "COACHING_EVALUATION_PAIR"


# ============================================================================
# Data Models
# ============================================================================

class TaskRequest(BaseModel):
    """Inbound request to be routed."""
    request_id: str
    task_type: str
    prompt: str
    metadata: dict = Field(default_factory=dict)
    timestamp_utc: str


class ModelResponse(BaseModel):
    """Response from a single model backend (local or remote)."""
    content: str
    model_id: str
    latency_ms: float = Field(ge=0.0)
    cost: float = Field(ge=0.0)
    token_count: int = Field(ge=0)


OptionalModelResponse = Optional[ModelResponse]


class BudgetSnapshot(BaseModel):
    """Point-in-time snapshot of budget state."""
    total_budget_usd: float
    spent_usd: float
    remaining_usd: float
    is_exhausted: bool


class PhaseContext(BaseModel):
    """Phase and confidence information."""
    phase: Phase
    confidence: float = Field(ge=0.0, le=1.0)


class RoutingDecision(BaseModel):
    """Internal intermediate type capturing the pre-execution routing decision."""
    intended_action: RoutingAction
    phase_context: PhaseContext
    is_coaching_sample: bool
    budget_snapshot: BudgetSnapshot


class RoutingResult(BaseModel):
    """The complete result of routing a request."""
    request_id: str
    response: ModelResponse
    intended_action: RoutingAction
    actual_action: RoutingAction
    phase: Phase
    confidence: float
    is_coaching_sample: bool
    total_cost_usd: float = Field(ge=0.0)
    total_latency_ms: float = Field(ge=0.0)
    is_degraded: bool
    is_fallback: bool
    local_response: OptionalModelResponse = None
    remote_response: OptionalModelResponse = None


class RoutingAuditEntry(BaseModel):
    """Complete audit record for a single routing decision."""
    request_id: str
    timestamp_utc: str
    task_type: str
    phase: Phase
    confidence: float
    is_coaching_sample: bool
    budget_snapshot: BudgetSnapshot
    intended_action: RoutingAction
    actual_action: RoutingAction
    total_cost_usd: float
    total_latency_ms: float
    is_degraded: bool
    is_fallback: bool
    local_model_id: Optional[str] = None
    remote_model_id: Optional[str] = None
    error_detail: Optional[str] = None


class DualSendResult(BaseModel):
    """Result of a concurrent dual-send to both backends."""
    outcome: DualSendOutcome
    local_response: OptionalModelResponse = None
    remote_response: OptionalModelResponse = None
    local_error: Optional[str] = None
    remote_error: Optional[str] = None

    model_config = {"frozen": True}


class TrainingExample(BaseModel):
    """A training data record to be persisted."""
    request_id: str
    task_type: str
    prompt: str
    example_type: TrainingExampleType
    remote_response: OptionalModelResponse = None
    local_response: OptionalModelResponse = None
    phase: Phase
    confidence: float
    collected_at_utc: str


class RouterConfig(BaseModel):
    """Configuration for the RequestRouter."""
    max_concurrent_dual_sends: int = Field(default=10, ge=1)
    remote_call_timeout_ms: int = Field(default=30000, ge=1000)
    local_call_timeout_ms: int = Field(default=10000, ge=500)
    enable_training_collection: bool = True
    enable_audit_logging: bool = True


# ============================================================================
# Errors
# ============================================================================

class RoutingError(Exception):
    """Base error for routing failures."""
    pass


class ValidationError(Exception):
    """Validation error."""
    pass


# ============================================================================
# Protocol Interfaces (for dependency injection)
# ============================================================================

class BudgetManager(Protocol):
    """Protocol for budget manager dependency."""
    async def check_budget(self) -> BudgetSnapshot: ...
    async def record_spend(self, cost: float) -> None: ...


class ConfidenceEngine(Protocol):
    """Protocol for confidence engine dependency."""
    async def get_phase_context(self, task_type: str) -> PhaseContext: ...


class SamplingScheduler(Protocol):
    """Protocol for sampling scheduler dependency."""
    async def should_coach(self) -> bool: ...


class LocalModelBackend(Protocol):
    """Protocol for local model backend dependency."""
    async def call(self, request: TaskRequest) -> ModelResponse: ...


class RemoteModelBackend(Protocol):
    """Protocol for remote model backend dependency."""
    async def call(self, request: TaskRequest) -> ModelResponse: ...


class TrainingDataCollector(Protocol):
    """Protocol for training data collector dependency."""
    async def collect(self, example: TrainingExample) -> None: ...


class AuditLogger(Protocol):
    """Protocol for audit logger dependency."""
    async def log(self, entry: RoutingAuditEntry) -> None: ...


# ============================================================================
# Pure Functions
# ============================================================================

def compute_routing_action(
    phase: Phase,
    is_coaching_sample: bool,
    budget_available: bool,
) -> RoutingAction:
    """
    Pure function that computes the intended RoutingAction from the current
    phase, coaching sample status, and budget availability.

    Decision matrix:
    - Phase 1 + budget → REMOTE_ONLY
    - Phase 1 + no budget → LOCAL_ONLY (degraded)
    - Phase 2 + budget → BOTH_RETURN_REMOTE (regardless of coaching)
    - Phase 2 + no budget → LOCAL_ONLY (degraded)
    - Phase 3 + coaching + budget → BOTH_RETURN_LOCAL
    - Phase 3 + no coaching → LOCAL_ONLY (regardless of budget)
    - Phase 3 + coaching + no budget → LOCAL_ONLY (degraded, skips coaching)
    """
    match phase:
        case Phase.PHASE_1:
            if budget_available:
                return RoutingAction.REMOTE_ONLY
            else:
                return RoutingAction.LOCAL_ONLY

        case Phase.PHASE_2:
            if budget_available:
                return RoutingAction.BOTH_RETURN_REMOTE
            else:
                return RoutingAction.LOCAL_ONLY

        case Phase.PHASE_3:
            if is_coaching_sample and budget_available:
                return RoutingAction.BOTH_RETURN_LOCAL
            else:
                return RoutingAction.LOCAL_ONLY

    # This should never be reached due to exhaustive match
    return RoutingAction.LOCAL_ONLY


def build_audit_entry(
    request: TaskRequest,
    decision: RoutingDecision,
    result: RoutingResult,
    error_detail: Optional[str] = None,
) -> RoutingAuditEntry:
    """
    Build a complete RoutingAuditEntry from the request, routing decision,
    and execution result. Pure function that assembles the audit record.
    """
    local_model_id = None
    remote_model_id = None

    if result.local_response:
        local_model_id = result.local_response.model_id
    if result.remote_response:
        remote_model_id = result.remote_response.model_id

    return RoutingAuditEntry(
        request_id=request.request_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        task_type=request.task_type,
        phase=decision.phase_context.phase,
        confidence=decision.phase_context.confidence,
        is_coaching_sample=decision.is_coaching_sample,
        budget_snapshot=decision.budget_snapshot,
        intended_action=decision.intended_action,
        actual_action=result.actual_action,
        total_cost_usd=result.total_cost_usd,
        total_latency_ms=result.total_latency_ms,
        is_degraded=result.is_degraded,
        is_fallback=result.is_fallback,
        local_model_id=local_model_id,
        remote_model_id=remote_model_id,
        error_detail=error_detail,
    )


# ============================================================================
# RequestRouter
# ============================================================================

class RequestRouter:
    """
    The central decision-maker for each request. Routes requests through
    the phase × sampling × budget decision matrix.
    """

    def __init__(
        self,
        config: RouterConfig,
        budget_manager: BudgetManager,
        confidence_engine: ConfidenceEngine,
        sampling_scheduler: SamplingScheduler,
        local_model_backend: LocalModelBackend,
        remote_model_backend: RemoteModelBackend,
        training_data_collector: TrainingDataCollector,
        audit_logger: AuditLogger,
    ):
        self.config = config
        self.budget_manager = budget_manager
        self.confidence_engine = confidence_engine
        self.sampling_scheduler = sampling_scheduler
        self.local_model_backend = local_model_backend
        self.remote_model_backend = remote_model_backend
        self.training_data_collector = training_data_collector
        self.audit_logger = audit_logger
        self._semaphore = asyncio.Semaphore(config.max_concurrent_dual_sends)

    async def route(self, request: TaskRequest) -> RoutingResult:
        """
        Route a single task request through the phase × sampling × budget
        decision matrix. This is the sole public method of RequestRouter.
        """
        start_time = asyncio.get_event_loop().time()
        error_detail = None
        decision = None
        result = None
        budget_snapshot = None
        phase_context = None

        try:
            # Validate request
            if not request.request_id or len(request.request_id) == 0:
                raise ValidationError("invalid_request: request_id must be non-empty")
            if not request.prompt or len(request.prompt) == 0:
                raise ValidationError("invalid_request: prompt must be non-empty")
            if not request.task_type or len(request.task_type) == 0:
                raise ValidationError("invalid_request: task_type must be non-empty")

            # Optional register normalization via Transmogrifier
            try:
                from transmogrifier.core import Transmogrifier
                _transmog = Transmogrifier()
                tr = _transmog.translate(request.prompt)
                if not tr.skipped and tr.output_text:
                    request = request.model_copy(update={"prompt": tr.output_text})
            except ImportError:
                pass
            except Exception:
                pass

            # Check budget
            try:
                budget_snapshot = await self.budget_manager.check_budget()
            except Exception as e:
                error_detail = f"budget_check_failed: {str(e)}"
                raise RoutingError(error_detail)

            # Get phase context
            try:
                phase_context = await self.confidence_engine.get_phase_context(request.task_type)
            except Exception as e:
                error_detail = f"confidence_engine_unavailable: {str(e)}"
                raise RoutingError(error_detail)

            # Get sampling decision
            try:
                is_coaching_sample = await self.sampling_scheduler.should_coach()
            except Exception as e:
                error_detail = f"sampling_scheduler_unavailable: {str(e)}"
                raise RoutingError(error_detail)

            # Compute intended routing action
            intended_action = compute_routing_action(
                phase_context.phase,
                is_coaching_sample,
                not budget_snapshot.is_exhausted,
            )

            # Create decision object
            decision = RoutingDecision(
                intended_action=intended_action,
                phase_context=phase_context,
                is_coaching_sample=is_coaching_sample,
                budget_snapshot=budget_snapshot,
            )

            # Execute the routing action
            result = await self._execute_routing(request, decision)

            # Record spend
            await self.budget_manager.record_spend(result.total_cost_usd)

            # Collect training data if enabled
            if self.config.enable_training_collection:
                await self._collect_training_data(request, result, phase_context)

            # Log audit entry on success
            if self.config.enable_audit_logging:
                await self._log_audit_safe(build_audit_entry(request, decision, result, error_detail))

            return result

        except (ValidationError, RoutingError) as e:
            # Log audit entry before re-raising
            if self.config.enable_audit_logging:
                try:
                    # Build audit entry based on what we have
                    if decision is not None:
                        # Full decision was made
                        audit_entry = RoutingAuditEntry(
                            request_id=request.request_id,
                            timestamp_utc=datetime.now(timezone.utc).isoformat(),
                            task_type=request.task_type,
                            phase=decision.phase_context.phase,
                            confidence=decision.phase_context.confidence,
                            is_coaching_sample=decision.is_coaching_sample,
                            budget_snapshot=decision.budget_snapshot,
                            intended_action=decision.intended_action,
                            actual_action=decision.intended_action,
                            total_cost_usd=0.0,
                            total_latency_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                            is_degraded=True,
                            is_fallback=False,
                            error_detail=error_detail,
                        )
                    elif phase_context is not None and budget_snapshot is not None:
                        # Have phase and budget but no decision yet
                        audit_entry = RoutingAuditEntry(
                            request_id=request.request_id,
                            timestamp_utc=datetime.now(timezone.utc).isoformat(),
                            task_type=request.task_type,
                            phase=phase_context.phase,
                            confidence=phase_context.confidence,
                            is_coaching_sample=False,
                            budget_snapshot=budget_snapshot,
                            intended_action=RoutingAction.LOCAL_ONLY,  # Default
                            actual_action=RoutingAction.LOCAL_ONLY,
                            total_cost_usd=0.0,
                            total_latency_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                            is_degraded=True,
                            is_fallback=False,
                            error_detail=error_detail,
                        )
                    elif budget_snapshot is not None:
                        # Have budget but no phase
                        audit_entry = RoutingAuditEntry(
                            request_id=request.request_id,
                            timestamp_utc=datetime.now(timezone.utc).isoformat(),
                            task_type=request.task_type,
                            phase=Phase.PHASE_1,  # Default
                            confidence=0.0,
                            is_coaching_sample=False,
                            budget_snapshot=budget_snapshot,
                            intended_action=RoutingAction.LOCAL_ONLY,
                            actual_action=RoutingAction.LOCAL_ONLY,
                            total_cost_usd=0.0,
                            total_latency_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                            is_degraded=True,
                            is_fallback=False,
                            error_detail=error_detail,
                        )
                    else:
                        # No decision context at all - still log something minimal
                        audit_entry = RoutingAuditEntry(
                            request_id=request.request_id,
                            timestamp_utc=datetime.now(timezone.utc).isoformat(),
                            task_type=request.task_type,
                            phase=Phase.PHASE_1,
                            confidence=0.0,
                            is_coaching_sample=False,
                            budget_snapshot=BudgetSnapshot(
                                total_budget_usd=0.0,
                                spent_usd=0.0,
                                remaining_usd=0.0,
                                is_exhausted=False,
                            ),
                            intended_action=RoutingAction.LOCAL_ONLY,
                            actual_action=RoutingAction.LOCAL_ONLY,
                            total_cost_usd=0.0,
                            total_latency_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                            is_degraded=True,
                            is_fallback=False,
                            error_detail=error_detail,
                        )
                    await self._log_audit_safe(audit_entry)
                except Exception:
                    # Silently ignore audit logging failures
                    pass
            raise

    async def _log_audit_safe(self, entry: RoutingAuditEntry) -> None:
        """Safely log audit entry, catching any exceptions."""
        try:
            await self.audit_logger.log(entry)
        except Exception:
            # Silently ignore
            pass

    async def _execute_routing(
        self,
        request: TaskRequest,
        decision: RoutingDecision,
    ) -> RoutingResult:
        """Execute the actual routing based on the decision."""
        start_time = asyncio.get_event_loop().time()

        intended_action = decision.intended_action
        actual_action = intended_action
        is_fallback = False
        is_degraded = False
        local_response = None
        remote_response = None
        total_cost = 0.0

        try:
            match intended_action:
                case RoutingAction.REMOTE_ONLY:
                    # Call remote only
                    remote_response = await self.remote_model_backend.call(request)
                    total_cost = remote_response.cost
                    response = remote_response

                case RoutingAction.LOCAL_ONLY:
                    # Call local only
                    try:
                        local_response = await self.local_model_backend.call(request)
                        total_cost = local_response.cost
                        response = local_response
                        # Check if this is degraded (Phase 1 or 2 forced to local)
                        if decision.phase_context.phase in [Phase.PHASE_1, Phase.PHASE_2]:
                            is_degraded = True
                    except Exception as local_error:
                        # Local failed, try fallback to remote if budget allows
                        if not decision.budget_snapshot.is_exhausted:
                            remote_response = await self.remote_model_backend.call(request)
                            total_cost = remote_response.cost
                            response = remote_response
                            actual_action = RoutingAction.LOCAL_FALLBACK_TO_REMOTE
                            is_fallback = True
                        else:
                            # No fallback possible
                            raise RoutingError(
                                f"both_backends_unavailable: local_error={str(local_error)}, "
                                f"budget_exhausted=True"
                            )

                case RoutingAction.BOTH_RETURN_REMOTE | RoutingAction.BOTH_RETURN_LOCAL:
                    # Dual-send
                    dual_result = await self.execute_dual_send(request)
                    local_response = dual_result.local_response
                    remote_response = dual_result.remote_response

                    # Calculate cost
                    if local_response:
                        total_cost += local_response.cost
                    if remote_response:
                        total_cost += remote_response.cost

                    # Handle different outcomes
                    match dual_result.outcome:
                        case DualSendOutcome.BOTH_SUCCEEDED:
                            # Return appropriate response based on action
                            if intended_action == RoutingAction.BOTH_RETURN_REMOTE:
                                response = remote_response
                            else:  # BOTH_RETURN_LOCAL
                                response = local_response

                        case DualSendOutcome.LOCAL_FAILED_REMOTE_SUCCEEDED:
                            # Use remote response
                            response = remote_response
                            if intended_action == RoutingAction.BOTH_RETURN_LOCAL:
                                is_degraded = True

                        case DualSendOutcome.LOCAL_SUCCEEDED_REMOTE_FAILED:
                            # Use local response
                            response = local_response
                            if intended_action == RoutingAction.BOTH_RETURN_REMOTE:
                                is_degraded = True

                        case DualSendOutcome.BOTH_FAILED:
                            raise RoutingError(
                                f"both_backends_unavailable: local_error={dual_result.local_error}, "
                                f"remote_error={dual_result.remote_error}"
                            )

            # Calculate total latency
            end_time = asyncio.get_event_loop().time()
            total_latency_ms = (end_time - start_time) * 1000

            return RoutingResult(
                request_id=request.request_id,
                response=response,
                intended_action=intended_action,
                actual_action=actual_action,
                phase=decision.phase_context.phase,
                confidence=decision.phase_context.confidence,
                is_coaching_sample=decision.is_coaching_sample,
                total_cost_usd=total_cost,
                total_latency_ms=total_latency_ms,
                is_degraded=is_degraded,
                is_fallback=is_fallback,
                local_response=local_response,
                remote_response=remote_response,
            )

        except RoutingError:
            raise
        except Exception as e:
            raise RoutingError(f"Routing execution failed: {str(e)}")

    async def execute_dual_send(self, request: TaskRequest) -> DualSendResult:
        """
        Execute a dual-send to both local and remote backends concurrently
        using asyncio.gather(return_exceptions=True).
        """
        async def call_local():
            return await self.local_model_backend.call(request)

        async def call_remote():
            return await self.remote_model_backend.call(request)

        # Execute both calls concurrently
        results = await asyncio.gather(
            call_local(),
            call_remote(),
            return_exceptions=True,
        )

        local_result = results[0]
        remote_result = results[1]

        # Determine outcome
        local_succeeded = not isinstance(local_result, Exception)
        remote_succeeded = not isinstance(remote_result, Exception)

        local_response = local_result if local_succeeded else None
        remote_response = remote_result if remote_succeeded else None
        local_error = str(local_result) if not local_succeeded else None
        remote_error = str(remote_result) if not remote_succeeded else None

        if local_succeeded and remote_succeeded:
            outcome = DualSendOutcome.BOTH_SUCCEEDED
        elif not local_succeeded and remote_succeeded:
            outcome = DualSendOutcome.LOCAL_FAILED_REMOTE_SUCCEEDED
        elif local_succeeded and not remote_succeeded:
            outcome = DualSendOutcome.LOCAL_SUCCEEDED_REMOTE_FAILED
        else:
            outcome = DualSendOutcome.BOTH_FAILED

        return DualSendResult(
            outcome=outcome,
            local_response=local_response,
            remote_response=remote_response,
            local_error=local_error,
            remote_error=remote_error,
        )

    async def _collect_training_data(
        self,
        request: TaskRequest,
        result: RoutingResult,
        phase_context: PhaseContext,
    ) -> None:
        """Collect training data based on the routing result."""
        try:
            # Collect remote response example in Phase 1 and Phase 2
            if result.remote_response and phase_context.phase in [Phase.PHASE_1, Phase.PHASE_2]:
                example = TrainingExample(
                    request_id=request.request_id,
                    task_type=request.task_type,
                    prompt=request.prompt,
                    example_type=TrainingExampleType.REMOTE_RESPONSE_EXAMPLE,
                    remote_response=result.remote_response,
                    local_response=None,
                    phase=phase_context.phase,
                    confidence=phase_context.confidence,
                    collected_at_utc=datetime.now(timezone.utc).isoformat(),
                )
                await self.training_data_collector.collect(example)

            # Collect coaching evaluation pair when both backends succeeded
            if result.local_response and result.remote_response and result.is_coaching_sample:
                example = TrainingExample(
                    request_id=request.request_id,
                    task_type=request.task_type,
                    prompt=request.prompt,
                    example_type=TrainingExampleType.COACHING_EVALUATION_PAIR,
                    remote_response=result.remote_response,
                    local_response=result.local_response,
                    phase=phase_context.phase,
                    confidence=phase_context.confidence,
                    collected_at_utc=datetime.now(timezone.utc).isoformat(),
                )
                await self.training_data_collector.collect(example)

        except Exception:
            # Silently ignore training collection failures
            pass


# ── Auto-injected export aliases (Pact export gate) ──
route = RouterConfig
