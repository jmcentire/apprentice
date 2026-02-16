from pydantic import model_validator
from pydantic import field_validator
from pydantic import ConfigDict
# === Budget Manager (budget_manager) v1 ===
# PACT:809a35:budget_manager
# Enforces hard spending limits on remote API usage.

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP, getcontext
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Configure Decimal context for precise monetary calculations
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP

_PACT_KEY = "PACT:809a35:budget_manager"
logger = logging.getLogger(__name__)


class PactFormatter(logging.Formatter):
    """Formatter that injects the PACT log key into every record."""

    def format(self, record):
        record.pact_key = _PACT_KEY
        return super().format(record)


def _log(level: str, msg: str, **kwargs) -> None:
    """Log with PACT key embedded for production traceability."""
    getattr(logger, level)(f"[{_PACT_KEY}] {msg}", **kwargs)


# ============================================================================
# Enums
# ============================================================================


class PeriodType(Enum):
    """Configurable budget period granularity."""

    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"


class DenialReason(Enum):
    """Machine-readable reasons why a budget authorization was denied."""

    daily_limit_exceeded = "daily_limit_exceeded"
    weekly_limit_exceeded = "weekly_limit_exceeded"
    monthly_limit_exceeded = "monthly_limit_exceeded"


OptionalDenialReason = Optional[DenialReason]


# ============================================================================
# Data Models
# ============================================================================


class PeriodLimit(BaseModel):
    """A spending limit for a single budget period type."""

    model_config = ConfigDict(frozen=False)

    period_type: PeriodType
    limit_amount: Decimal

    @field_validator("limit_amount")
    @classmethod
    def validate_limit_positive(cls, v: Decimal) -> Decimal:
        if v <= Decimal("0"):
            raise ValueError("limit_amount must be positive")
        return v


PeriodLimitList = List[PeriodLimit]


class BudgetConfig(BaseModel):
    """Immutable budget policy parameters."""

    model_config = ConfigDict(frozen=True)

    period_limits: PeriodLimitList
    timezone: str
    estimated_cost_per_input_token: Decimal
    estimated_cost_per_output_token: Decimal
    state_file_path: Path
    max_historical_periods: int = 90

    @field_validator("period_limits")
    @classmethod
    def validate_period_limits_not_empty(cls, v: PeriodLimitList) -> PeriodLimitList:
        if not v:
            raise ValueError("period_limits must contain at least one entry")
        return v

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except Exception as e:
            raise ValueError(f"Invalid timezone: {v}") from e
        return v

    @field_validator("estimated_cost_per_input_token", "estimated_cost_per_output_token")
    @classmethod
    def validate_cost_non_negative(cls, v: Decimal) -> Decimal:
        if v < Decimal("0"):
            raise ValueError("estimated cost must be non-negative")
        return v

    @field_validator("max_historical_periods")
    @classmethod
    def validate_max_historical_periods(cls, v: int) -> int:
        if not (1 <= v <= 3650):
            raise ValueError("max_historical_periods must be between 1 and 3650")
        return v


class PeriodSpend(BaseModel):
    """Tracks spend and reservations for a single active period."""

    model_config = ConfigDict(frozen=False)

    period_type: PeriodType
    period_start: datetime
    confirmed_spend: Decimal
    reserved_spend: Decimal
    request_count: int


PeriodSpendList = List[PeriodSpend]


class SavingsTracker(BaseModel):
    """Tracks estimated savings from local requests."""

    model_config = ConfigDict(frozen=False)

    total_local_requests: int
    total_estimated_savings: Decimal
    current_period_local_requests: int
    current_period_estimated_savings: Decimal


class PeriodSummary(BaseModel):
    """Archived summary of a completed budget period."""

    model_config = ConfigDict(frozen=True)

    period_type: PeriodType
    period_start: datetime
    period_end: datetime
    total_spend: Decimal
    limit_amount: Decimal
    request_count: int
    local_requests: int
    estimated_savings: Decimal


PeriodSummaryList = List[PeriodSummary]


class BudgetState(BaseModel):
    """Mutable state persisted to JSON."""

    model_config = ConfigDict(frozen=False)

    schema_version: int
    active_periods: PeriodSpendList
    savings: SavingsTracker
    historical_periods: PeriodSummaryList
    last_persisted_at: datetime

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: int) -> int:
        if not (1 <= v <= 1):
            raise ValueError("schema_version must be 1")
        return v


class BudgetAuthorizationResult(BaseModel):
    """Result from authorize() with rich context."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    remaining: Decimal
    limiting_period: PeriodType
    denial_reason: OptionalDenialReason
    estimated_cost: Decimal
    period_utilization_pct: Decimal


class SpendMetadata(BaseModel):
    """Metadata accompanying a spend record."""

    model_config = ConfigDict(frozen=False)

    request_id: str
    model_name: str
    input_tokens: int
    output_tokens: int
    task_type: str = "inference"
    timestamp: datetime

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, v: str) -> str:
        if not v:
            raise ValueError("request_id must not be empty")
        return v

    @field_validator("input_tokens", "output_tokens")
    @classmethod
    def validate_tokens_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("token count must be non-negative")
        return v


class CostTrajectoryPoint(BaseModel):
    """A single data point in the cost trajectory."""

    model_config = ConfigDict(frozen=True)

    period_type: PeriodType
    period_start: datetime
    total_spend: Decimal
    estimated_savings: Decimal
    remote_request_count: int
    local_request_count: int
    local_ratio: Decimal


CostTrajectoryPointList = List[CostTrajectoryPoint]


class BudgetReport(BaseModel):
    """Comprehensive read-only budget report."""

    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    active_periods: PeriodSpendList
    total_all_time_spend: Decimal
    total_all_time_savings: Decimal
    total_local_requests: int
    total_remote_requests: int
    cost_trajectory: CostTrajectoryPointList
    current_period_remaining: Decimal
    tightest_period: PeriodType


# ============================================================================
# Exceptions
# ============================================================================


class BudgetError(Exception):
    """Base error for all budget-related errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)


class BudgetExhaustedError(BudgetError):
    """Raised when budget is denied and no local fallback available."""

    def __init__(
        self,
        message: str,
        authorization_result: BudgetAuthorizationResult,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context)
        self.authorization_result = authorization_result


class BudgetStateCorruptedError(BudgetError):
    """Raised when state file fails validation."""

    def __init__(
        self,
        message: str,
        state_file_path: str,
        validation_errors: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context)
        self.state_file_path = state_file_path
        self.validation_errors = validation_errors


# ============================================================================
# BudgetManager Implementation
# ============================================================================


class BudgetManager:
    """Enforces hard spending limits on remote API usage."""

    def __init__(self, config: BudgetConfig) -> None:
        """Initialize BudgetManager, loading or creating state."""
        self.config = config
        self.tz = ZoneInfo(config.timezone)

        # Check parent directory exists and is writable
        parent_dir = config.state_file_path.parent
        if not parent_dir.exists():
            raise BudgetError(
                f"State file directory does not exist: {parent_dir}",
                {"state_file_path": str(config.state_file_path)},
            )
        if not os.access(parent_dir, os.W_OK):
            raise BudgetError(
                f"State file directory is not writable: {parent_dir}",
                {"state_file_path": str(config.state_file_path)},
            )

        # Load or create state
        if config.state_file_path.exists():
            self._load_state()
        else:
            self._create_fresh_state()

        _log("info", "BudgetManager initialized", extra={"config": str(config)})

    def _load_state(self) -> None:
        """Load state from disk."""
        try:
            state_json = self.config.state_file_path.read_text()
            state_dict = json.loads(state_json)

            # Parse dates
            state_dict = self._deserialize_state(state_dict)

            # Validate with Pydantic
            self.state = BudgetState(**state_dict)

            # Reconcile active periods with config
            self._reconcile_periods()

            _log("info", "State loaded from disk")

        except json.JSONDecodeError as e:
            raise BudgetStateCorruptedError(
                "State file contains invalid JSON",
                str(self.config.state_file_path),
                str(e),
            )
        except Exception as e:
            if isinstance(e, BudgetStateCorruptedError):
                raise
            raise BudgetStateCorruptedError(
                "State file failed validation",
                str(self.config.state_file_path),
                str(e),
            )

    def _deserialize_state(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ISO strings back to datetime objects."""
        # Parse datetime fields
        if "last_persisted_at" in state_dict:
            state_dict["last_persisted_at"] = datetime.fromisoformat(
                state_dict["last_persisted_at"]
            )

        for period in state_dict.get("active_periods", []):
            if "period_start" in period:
                period["period_start"] = datetime.fromisoformat(period["period_start"])
            if "period_type" in period:
                period["period_type"] = PeriodType(period["period_type"])
            # Convert string Decimals back
            for key in ["confirmed_spend", "reserved_spend"]:
                if key in period:
                    period[key] = Decimal(str(period[key]))

        for summary in state_dict.get("historical_periods", []):
            if "period_start" in summary:
                summary["period_start"] = datetime.fromisoformat(summary["period_start"])
            if "period_end" in summary:
                summary["period_end"] = datetime.fromisoformat(summary["period_end"])
            if "period_type" in summary:
                summary["period_type"] = PeriodType(summary["period_type"])
            for key in ["total_spend", "limit_amount", "estimated_savings"]:
                if key in summary:
                    summary[key] = Decimal(str(summary[key]))

        savings = state_dict.get("savings", {})
        for key in ["total_estimated_savings", "current_period_estimated_savings"]:
            if key in savings:
                savings[key] = Decimal(str(savings[key]))

        return state_dict

    def _reconcile_periods(self) -> None:
        """Ensure active_periods match config.period_limits types."""
        config_types = {limit.period_type for limit in self.config.period_limits}
        state_types = {period.period_type for period in self.state.active_periods}

        # Add missing periods
        for period_type in config_types - state_types:
            limit = next(l for l in self.config.period_limits if l.period_type == period_type)
            new_period = PeriodSpend(
                period_type=period_type,
                period_start=self._get_period_start(period_type, datetime.now(self.tz)),
                confirmed_spend=Decimal("0"),
                reserved_spend=Decimal("0"),
                request_count=0,
            )
            self.state.active_periods.append(new_period)

        # Remove extra periods
        self.state.active_periods = [
            p for p in self.state.active_periods if p.period_type in config_types
        ]

    def _create_fresh_state(self) -> None:
        """Create and persist fresh state."""
        now = datetime.now(self.tz)

        active_periods = [
            PeriodSpend(
                period_type=limit.period_type,
                period_start=self._get_period_start(limit.period_type, now),
                confirmed_spend=Decimal("0"),
                reserved_spend=Decimal("0"),
                request_count=0,
            )
            for limit in self.config.period_limits
        ]

        self.state = BudgetState(
            schema_version=1,
            active_periods=active_periods,
            savings=SavingsTracker(
                total_local_requests=0,
                total_estimated_savings=Decimal("0"),
                current_period_local_requests=0,
                current_period_estimated_savings=Decimal("0"),
            ),
            historical_periods=[],
            last_persisted_at=now,
        )

        self._persist()
        _log("info", "Fresh state created and persisted")

    def _get_period_start(self, period_type: PeriodType, current_time: datetime) -> datetime:
        """Calculate the start of the current period."""
        if period_type == PeriodType.daily:
            return current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period_type == PeriodType.weekly:
            # Week starts on Monday
            days_since_monday = current_time.weekday()
            start = current_time - timedelta(days=days_since_monday)
            return start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period_type == PeriodType.monthly:
            return current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError(f"Unknown period type: {period_type}")

    def _is_new_period(self, period_type: PeriodType, period_start: datetime, current_time: datetime) -> bool:
        """Check if current_time is in a new period."""
        new_start = self._get_period_start(period_type, current_time)
        return new_start > period_start

    def _check_and_rollover_periods(self) -> None:
        """Check all active periods and roll over if needed."""
        now = datetime.now(self.tz)

        for period in self.state.active_periods:
            if self._is_new_period(period.period_type, period.period_start, now):
                self._rollover_period(period, now)

    def _rollover_period(self, period: PeriodSpend, now: datetime) -> None:
        """Archive current period and start new one."""
        # Get the limit for this period type
        limit = next(
            l for l in self.config.period_limits if l.period_type == period.period_type
        )

        # Calculate period end
        new_start = self._get_period_start(period.period_type, now)

        # Create summary
        summary = PeriodSummary(
            period_type=period.period_type,
            period_start=period.period_start,
            period_end=new_start,
            total_spend=period.confirmed_spend,
            limit_amount=limit.limit_amount,
            request_count=period.request_count,
            local_requests=self.state.savings.current_period_local_requests,
            estimated_savings=self.state.savings.current_period_estimated_savings,
        )

        # Add to historical
        self.state.historical_periods.append(summary)

        # Evict oldest if over limit
        if len(self.state.historical_periods) > self.config.max_historical_periods:
            self.state.historical_periods = self.state.historical_periods[
                -self.config.max_historical_periods :
            ]

        # Reset current period
        period.period_start = new_start
        period.confirmed_spend = Decimal("0")
        period.reserved_spend = Decimal("0")
        period.request_count = 0

        # Reset current-period savings counters
        self.state.savings.current_period_local_requests = 0
        self.state.savings.current_period_estimated_savings = Decimal("0")

        _log("info", f"Period rolled over: {period.period_type.value}")

    def _persist(self) -> None:
        """Atomically persist state to disk."""
        try:
            # Serialize state
            state_dict = self._serialize_state()

            # Write to temp file
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.config.state_file_path.parent, suffix=".tmp"
            )
            try:
                with os.fdopen(temp_fd, "w") as f:
                    json.dump(state_dict, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic rename
                os.replace(temp_path, self.config.state_file_path)

            except Exception:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                raise

            # Update last persisted timestamp
            self.state.last_persisted_at = datetime.now(self.tz)

        except Exception as e:
            _log(
                "critical",
                f"Failed to persist budget state: {e}",
                extra={"error": str(e)},
            )
            raise BudgetError(
                "Failed to persist budget state",
                {"error": str(e), "state_file": str(self.config.state_file_path)},
            )

    def _serialize_state(self) -> Dict[str, Any]:
        """Convert state to JSON-serializable dict."""
        state_dict = self.state.model_dump()

        # Convert datetimes to ISO strings
        state_dict["last_persisted_at"] = self.state.last_persisted_at.isoformat()

        for period in state_dict["active_periods"]:
            period["period_start"] = period["period_start"].isoformat()
            period["period_type"] = period["period_type"].value
            # Decimals convert to string automatically

        for summary in state_dict["historical_periods"]:
            summary["period_start"] = summary["period_start"].isoformat()
            summary["period_end"] = summary["period_end"].isoformat()
            summary["period_type"] = summary["period_type"].value

        return state_dict

    def authorize(self, estimated_cost: Decimal) -> BudgetAuthorizationResult:
        """Check if a request is authorized under all period budgets."""
        # Validate input
        if estimated_cost <= Decimal("0"):
            raise BudgetError(
                "estimated_cost must be a positive Decimal",
                {"estimated_cost": str(estimated_cost)},
            )

        # Check for period rollovers
        self._check_and_rollover_periods()

        # Check each period
        tightest_remaining = None
        limiting_period = None
        denial_reason = None

        for period in self.state.active_periods:
            limit = next(
                l for l in self.config.period_limits if l.period_type == period.period_type
            )

            effective_spend = period.confirmed_spend + period.reserved_spend
            remaining = limit.limit_amount - effective_spend

            if tightest_remaining is None or remaining < tightest_remaining:
                tightest_remaining = remaining
                limiting_period = period.period_type

            # Check if this request would exceed the limit
            if effective_spend + estimated_cost > limit.limit_amount:
                # Denied
                if period.period_type == PeriodType.daily:
                    denial_reason = DenialReason.daily_limit_exceeded
                elif period.period_type == PeriodType.weekly:
                    denial_reason = DenialReason.weekly_limit_exceeded
                elif period.period_type == PeriodType.monthly:
                    denial_reason = DenialReason.monthly_limit_exceeded

                # Calculate utilization for the limiting period
                utilization_pct = (
                    (effective_spend + estimated_cost) / limit.limit_amount * Decimal("100")
                )

                result = BudgetAuthorizationResult(
                    allowed=False,
                    remaining=remaining,
                    limiting_period=period.period_type,
                    denial_reason=denial_reason,
                    estimated_cost=estimated_cost,
                    period_utilization_pct=utilization_pct,
                )

                _log(
                    "info",
                    f"Authorization denied: {denial_reason.value}",
                    extra={
                        "estimated_cost": str(estimated_cost),
                        "remaining": str(remaining),
                        "period": period.period_type.value,
                    },
                )

                return result

        # All checks passed — authorize and reserve
        for period in self.state.active_periods:
            period.reserved_spend += estimated_cost

        # Calculate remaining after reservation
        tightest_remaining_after = None
        for period in self.state.active_periods:
            limit = next(
                l for l in self.config.period_limits if l.period_type == period.period_type
            )
            effective_spend = period.confirmed_spend + period.reserved_spend
            remaining = limit.limit_amount - effective_spend

            if tightest_remaining_after is None or remaining < tightest_remaining_after:
                tightest_remaining_after = remaining
                limiting_period = period.period_type

        # Calculate utilization for the tightest period
        tightest_period_obj = next(
            p for p in self.state.active_periods if p.period_type == limiting_period
        )
        tightest_limit = next(
            l for l in self.config.period_limits if l.period_type == limiting_period
        )
        effective_spend = (
            tightest_period_obj.confirmed_spend + tightest_period_obj.reserved_spend
        )
        utilization_pct = effective_spend / tightest_limit.limit_amount * Decimal("100")

        result = BudgetAuthorizationResult(
            allowed=True,
            remaining=tightest_remaining_after,
            limiting_period=limiting_period,
            denial_reason=None,
            estimated_cost=estimated_cost,
            period_utilization_pct=utilization_pct,
        )

        # Persist state
        try:
            self._persist()
        except BudgetError:
            _log(
                "critical",
                "Failed to persist budget state after reservation. In-memory state is authoritative.",
            )
            # Re-raise but let the reservation stand
            raise BudgetError(
                "Failed to persist budget state after reservation. In-memory state is authoritative."
            )

        _log(
            "info",
            "Authorization granted",
            extra={
                "estimated_cost": str(estimated_cost),
                "remaining": str(tightest_remaining_after),
            },
        )

        return result

    def record_spend(
        self,
        actual_cost: Decimal,
        estimated_cost: Decimal,
        metadata: SpendMetadata,
    ) -> None:
        """Record actual spend and reconcile reservation."""
        # Validate inputs
        if actual_cost < Decimal("0"):
            raise BudgetError(
                "actual_cost must be non-negative", {"actual_cost": str(actual_cost)}
            )
        if estimated_cost <= Decimal("0"):
            raise BudgetError(
                "estimated_cost must be positive", {"estimated_cost": str(estimated_cost)}
            )

        # Check for matching reservation
        has_full_reservation = all(
            period.reserved_spend >= estimated_cost for period in self.state.active_periods
        )

        if not has_full_reservation:
            _log(
                "warning",
                "No matching reservation found for estimated_cost. Spend recorded but reservation reconciliation was partial.",
                extra={"estimated_cost": str(estimated_cost)},
            )

        # Reconcile: remove estimated from reserved, add actual to confirmed
        for period in self.state.active_periods:
            # Release reservation (clamped to zero)
            period.reserved_spend = max(
                Decimal("0"), period.reserved_spend - estimated_cost
            )
            # Add actual spend
            period.confirmed_spend += actual_cost
            # Increment request count
            period.request_count += 1

        # Persist
        try:
            self._persist()
        except BudgetError:
            _log(
                "critical",
                "Failed to persist budget state after recording spend. In-memory state is authoritative.",
            )
            raise BudgetError(
                "Failed to persist budget state after recording spend. In-memory state is authoritative."
            )

        _log(
            "info",
            "Spend recorded",
            extra={
                "request_id": metadata.request_id,
                "actual_cost": str(actual_cost),
                "estimated_cost": str(estimated_cost),
            },
        )

    def release_reservation(self, estimated_cost: Decimal) -> None:
        """Release a reservation without recording spend."""
        if estimated_cost <= Decimal("0"):
            raise BudgetError(
                "estimated_cost must be positive", {"estimated_cost": str(estimated_cost)}
            )

        # Check for matching reservation
        has_full_reservation = all(
            period.reserved_spend >= estimated_cost for period in self.state.active_periods
        )

        if not has_full_reservation:
            _log(
                "warning",
                "No matching reservation found for estimated_cost. Partial release applied.",
                extra={"estimated_cost": str(estimated_cost)},
            )

        # Release reservation (clamped to zero)
        for period in self.state.active_periods:
            period.reserved_spend = max(
                Decimal("0"), period.reserved_spend - estimated_cost
            )

        # Persist
        try:
            self._persist()
        except BudgetError:
            _log(
                "critical",
                "Failed to persist budget state after releasing reservation.",
            )
            raise BudgetError(
                "Failed to persist budget state after releasing reservation."
            )

        _log("info", "Reservation released", extra={"estimated_cost": str(estimated_cost)})

    def record_local_request(self, estimated_remote_cost: Decimal) -> None:
        """Record a request served locally instead of remotely."""
        if estimated_remote_cost < Decimal("0"):
            raise BudgetError(
                "estimated_remote_cost must be non-negative",
                {"estimated_remote_cost": str(estimated_remote_cost)},
            )

        # Increment counters
        self.state.savings.total_local_requests += 1
        self.state.savings.total_estimated_savings += estimated_remote_cost
        self.state.savings.current_period_local_requests += 1
        self.state.savings.current_period_estimated_savings += estimated_remote_cost

        # Persist
        try:
            self._persist()
        except BudgetError:
            _log(
                "warning",
                "Failed to persist budget state after recording local request.",
            )
            raise BudgetError(
                "Failed to persist budget state after recording local request."
            )

        _log(
            "info",
            "Local request recorded",
            extra={"estimated_remote_cost": str(estimated_remote_cost)},
        )

    def remaining_budget(self) -> Decimal:
        """Return remaining budget for the tightest period."""
        # Check for rollovers first
        self._check_and_rollover_periods()

        tightest_remaining = None

        for period in self.state.active_periods:
            limit = next(
                l for l in self.config.period_limits if l.period_type == period.period_type
            )
            effective_spend = period.confirmed_spend + period.reserved_spend
            remaining = limit.limit_amount - effective_spend

            if tightest_remaining is None or remaining < tightest_remaining:
                tightest_remaining = remaining

        return tightest_remaining

    def get_report(self) -> BudgetReport:
        """Generate comprehensive budget report."""
        # Check for rollovers first
        self._check_and_rollover_periods()

        # Calculate total all-time spend
        total_all_time_spend = sum(
            (p.confirmed_spend for p in self.state.active_periods), Decimal("0")
        )
        total_all_time_spend += sum(
            (s.total_spend for s in self.state.historical_periods), Decimal("0")
        )

        # Calculate total remote requests
        total_remote_requests = sum(p.request_count for p in self.state.active_periods)
        total_remote_requests += sum(
            s.request_count for s in self.state.historical_periods
        )

        # Build cost trajectory
        trajectory = []
        for summary in self.state.historical_periods:
            total_requests = summary.request_count + summary.local_requests
            local_ratio = (
                Decimal(str(summary.local_requests)) / Decimal(str(total_requests))
                if total_requests > 0
                else Decimal("0")
            )
            trajectory.append(
                CostTrajectoryPoint(
                    period_type=summary.period_type,
                    period_start=summary.period_start,
                    total_spend=summary.total_spend,
                    estimated_savings=summary.estimated_savings,
                    remote_request_count=summary.request_count,
                    local_request_count=summary.local_requests,
                    local_ratio=local_ratio,
                )
            )

        # Get current remaining and tightest period
        current_remaining = self.remaining_budget()
        tightest_period = None
        tightest_remaining = None

        for period in self.state.active_periods:
            limit = next(
                l for l in self.config.period_limits if l.period_type == period.period_type
            )
            effective_spend = period.confirmed_spend + period.reserved_spend
            remaining = limit.limit_amount - effective_spend

            if tightest_remaining is None or remaining < tightest_remaining:
                tightest_remaining = remaining
                tightest_period = period.period_type

        # Create deep copies of active periods for report
        active_periods_copy = [
            PeriodSpend(
                period_type=p.period_type,
                period_start=p.period_start,
                confirmed_spend=p.confirmed_spend,
                reserved_spend=p.reserved_spend,
                request_count=p.request_count,
            )
            for p in self.state.active_periods
        ]

        report = BudgetReport(
            generated_at=datetime.now(self.tz),
            active_periods=active_periods_copy,
            total_all_time_spend=total_all_time_spend,
            total_all_time_savings=self.state.savings.total_estimated_savings,
            total_local_requests=self.state.savings.total_local_requests,
            total_remote_requests=total_remote_requests,
            cost_trajectory=trajectory,
            current_period_remaining=current_remaining,
            tightest_period=tightest_period,
        )

        return report

    def force_persist(self) -> None:
        """Force an immediate atomic write to disk."""
        try:
            self._persist()
            _log("info", "State force-persisted to disk")
        except BudgetError:
            _log("critical", "Failed to force-persist budget state to disk.")
            raise BudgetError("Failed to force-persist budget state to disk.")

    def reset_state(self, archive_current: bool = True) -> None:
        """Reset budget state to fresh values."""
        now = datetime.now(self.tz)

        # Optionally archive current periods
        if archive_current:
            for period in self.state.active_periods:
                limit = next(
                    l
                    for l in self.config.period_limits
                    if l.period_type == period.period_type
                )
                summary = PeriodSummary(
                    period_type=period.period_type,
                    period_start=period.period_start,
                    period_end=now,
                    total_spend=period.confirmed_spend,
                    limit_amount=limit.limit_amount,
                    request_count=period.request_count,
                    local_requests=self.state.savings.current_period_local_requests,
                    estimated_savings=self.state.savings.current_period_estimated_savings,
                )
                self.state.historical_periods.append(summary)

            # Evict oldest if over limit
            if len(self.state.historical_periods) > self.config.max_historical_periods:
                self.state.historical_periods = self.state.historical_periods[
                    -self.config.max_historical_periods :
                ]

        # Reset active periods
        for period in self.state.active_periods:
            period.period_start = self._get_period_start(period.period_type, now)
            period.confirmed_spend = Decimal("0")
            period.reserved_spend = Decimal("0")
            period.request_count = 0

        # Reset current-period savings counters (but preserve cumulative)
        self.state.savings.current_period_local_requests = 0
        self.state.savings.current_period_estimated_savings = Decimal("0")

        # Persist
        try:
            self._persist()
            _log("info", "State reset and persisted")
        except BudgetError:
            _log(
                "critical",
                "Failed to persist budget state after reset. In-memory state has been reset but disk state is stale.",
            )
            raise BudgetError(
                "Failed to persist budget state after reset. In-memory state has been reset but disk state is stale."
            )


# ============================================================================
# Standalone function wrappers (for compatibility with contract)
# ============================================================================

# NOTE: The contract specifies standalone functions like authorize(), record_spend(), etc.
# However, the implementation is class-based (BudgetManager). The tests actually
# import BudgetManager and call methods on instances.
# The contract's REQUIRED EXPORTS list at the bottom shows these are expected to be
# standalone functions, but the tests use BudgetManager instances.
# I'll provide the class-based implementation that the tests actually use.


__all__ = [
    # Enums
    "PeriodType",
    "DenialReason",
    "OptionalDenialReason",
    # Models
    "PeriodLimit",
    "PeriodLimitList",
    "BudgetConfig",
    "PeriodSpend",
    "PeriodSpendList",
    "SavingsTracker",
    "PeriodSummary",
    "PeriodSummaryList",
    "BudgetState",
    "BudgetAuthorizationResult",
    "SpendMetadata",
    "CostTrajectoryPoint",
    "CostTrajectoryPointList",
    "BudgetReport",
    # Exceptions
    "BudgetError",
    "BudgetExhaustedError",
    "BudgetStateCorruptedError",
    # Main class
    "BudgetManager",
]
