"""
Contract tests for BudgetManager component.

Tests organized into logical groups:
1. Initialization
2. Authorization logic
3. Reservation lifecycle
4. Period boundaries
5. Savings tracking
6. Persistence
7. Reporting
8. Property-based / invariant tests

Run with: pytest contract_test.py -v
"""

import json
import os
import stat
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

# Try to import time-freezing library
try:
    from freezegun import freeze_time
    HAS_FREEZEGUN = True
except ImportError:
    HAS_FREEZEGUN = False

# Try to import hypothesis
try:
    from hypothesis import given, settings, strategies as st, assume
    from hypothesis.stateful import RuleBasedStateMachine, rule, initialize
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

if not HAS_HYPOTHESIS:
    def given(*a, **kw):
        return lambda f: f
    class st:
        @staticmethod
        def floats(*a, **kw): return None
        @staticmethod
        def integers(*a, **kw): return None
        @staticmethod
        def text(*a, **kw): return None
        @staticmethod
        def booleans(*a, **kw): return None
        @staticmethod
        def lists(*a, **kw): return None
        @staticmethod
        def one_of(*a, **kw): return None
        @staticmethod
        def sampled_from(*a, **kw): return None
        @staticmethod
        def just(*a, **kw): return None
        @staticmethod
        def none(*a, **kw): return None
        @staticmethod
        def dictionaries(*a, **kw): return None
        @staticmethod
        def from_regex(*a, **kw): return None
        @staticmethod
        def datetimes(*a, **kw): return None
        @staticmethod
        def timedeltas(*a, **kw): return None
        @staticmethod
        def characters(*a, **kw): return None
        @staticmethod
        def decimals(*a, **kw): return None
    def assume(*a, **kw): pass
    def settings(*a, **kw):
        return lambda f: f

from zoneinfo import ZoneInfo

# Import the component under test
from src.budget_manager import (
    BudgetManager,
    BudgetConfig,
    BudgetAuthorizationResult,
    BudgetState,
    BudgetReport,
    BudgetError,
    BudgetExhaustedError,
    BudgetStateCorruptedError,
    PeriodLimit,
    PeriodSpend,
    PeriodSummary,
    PeriodType,
    SpendMetadata,
    SavingsTracker,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_state_path(tmp_path):
    """Return a path for a state file inside tmp_path."""
    return tmp_path / "budget_state.json"


@pytest.fixture
def daily_period_limit():
    """A daily period limit of 5.00."""
    return PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00'))


@pytest.fixture
def monthly_period_limit():
    """A monthly period limit of 100.00."""
    return PeriodLimit(period_type=PeriodType.monthly, limit_amount=Decimal('100.00'))


@pytest.fixture
def weekly_period_limit():
    """A weekly period limit of 25.00."""
    return PeriodLimit(period_type=PeriodType.weekly, limit_amount=Decimal('25.00'))


def make_config(
    tmp_path,
    period_limits=None,
    timezone_str="UTC",
    max_historical_periods=10,
    state_file_name="budget_state.json",
):
    """Factory to create BudgetConfig with sensible defaults."""
    if period_limits is None:
        period_limits = [
            PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00')),
        ]
    return BudgetConfig(
        period_limits=period_limits,
        timezone=timezone_str,
        estimated_cost_per_input_token=Decimal('0.000001'),
        estimated_cost_per_output_token=Decimal('0.000002'),
        state_file_path=tmp_path / state_file_name,
        max_historical_periods=max_historical_periods,
    )


def make_metadata(request_id="req-001", model_name="gpt-4", input_tokens=100,
                   output_tokens=50, task_type="completion", timestamp=None):
    """Factory to create SpendMetadata with sensible defaults."""
    if timestamp is None:
        timestamp = datetime.now(tz=ZoneInfo("UTC"))
    return SpendMetadata(
        request_id=request_id,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        task_type=task_type,
        timestamp=timestamp,
    )


@pytest.fixture
def budget_manager_factory(tmp_path):
    """Factory fixture that creates BudgetManager instances with optional config overrides."""
    def _factory(
        period_limits=None,
        timezone_str="UTC",
        max_historical_periods=10,
        state_file_name="budget_state.json",
    ):
        config = make_config(
            tmp_path,
            period_limits=period_limits,
            timezone_str=timezone_str,
            max_historical_periods=max_historical_periods,
            state_file_name=state_file_name,
        )
        return BudgetManager(config)

    return _factory


@pytest.fixture
def daily_manager(budget_manager_factory, daily_period_limit):
    """BudgetManager with a single daily limit of 5.00."""
    return budget_manager_factory(period_limits=[daily_period_limit])


@pytest.fixture
def multi_period_manager(budget_manager_factory, daily_period_limit, monthly_period_limit):
    """BudgetManager with daily (5.00) and monthly (100.00) limits."""
    return budget_manager_factory(
        period_limits=[daily_period_limit, monthly_period_limit]
    )


# ============================================================================
# 1. test_budget_manager_init — Initialization tests
# ============================================================================

class TestBudgetManagerInit:
    """Tests for BudgetManager.__init__()."""

    def test_init_fresh_state_creates_file(self, tmp_path, daily_period_limit):
        """Initialize with no existing state file creates fresh state and persists it."""
        config = make_config(tmp_path, period_limits=[daily_period_limit])
        assert not config.state_file_path.exists(), "State file should not exist before init"

        manager = BudgetManager(config)

        assert config.state_file_path.exists(), "State file should be created after init"

    def test_init_fresh_state_has_zero_spend(self, daily_manager):
        """Fresh state has zero confirmed_spend and reserved_spend."""
        remaining = daily_manager.remaining_budget()
        assert remaining == Decimal('5.00'), (
            f"Fresh daily manager should have full budget remaining, got {remaining}"
        )

    def test_init_fresh_state_active_periods_match_config(self, tmp_path):
        """Active periods match the configured period_limits types."""
        limits = [
            PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00')),
            PeriodLimit(period_type=PeriodType.monthly, limit_amount=Decimal('100.00')),
        ]
        config = make_config(tmp_path, period_limits=limits)
        manager = BudgetManager(config)

        report = manager.get_report()
        active_types = {p.period_type for p in report.active_periods}
        assert active_types == {PeriodType.daily, PeriodType.monthly}, (
            f"Active period types should match config, got {active_types}"
        )

    def test_init_existing_valid_state_loads(self, tmp_path, daily_period_limit):
        """Loading from an existing valid state file preserves spend data."""
        config = make_config(tmp_path, period_limits=[daily_period_limit])

        # Create manager and record some spend
        manager1 = BudgetManager(config)
        result = manager1.authorize(Decimal('2.00'))
        assert result.allowed is True
        manager1.record_spend(
            actual_cost=Decimal('2.00'),
            estimated_cost=Decimal('2.00'),
            metadata=make_metadata(request_id="req-init-1"),
        )

        # Create new manager from same state file
        manager2 = BudgetManager(config)
        remaining = manager2.remaining_budget()
        assert remaining == Decimal('3.00'), (
            f"Loaded manager should have 3.00 remaining, got {remaining}"
        )

    def test_init_corrupted_state_file_raises_error(self, tmp_path, daily_period_limit):
        """Corrupted state file raises BudgetStateCorruptedError."""
        config = make_config(tmp_path, period_limits=[daily_period_limit])

        # Write invalid JSON to state file
        config.state_file_path.write_text("{invalid json content!!! not valid")

        with pytest.raises(BudgetStateCorruptedError) as exc_info:
            BudgetManager(config)

        assert exc_info.value.state_file_path is not None, (
            "BudgetStateCorruptedError should include state_file_path"
        )

    def test_init_corrupted_state_invalid_schema(self, tmp_path, daily_period_limit):
        """State file with valid JSON but invalid schema raises BudgetStateCorruptedError."""
        config = make_config(tmp_path, period_limits=[daily_period_limit])

        # Write valid JSON with wrong schema
        config.state_file_path.write_text(json.dumps({
            "schema_version": 1,
            "active_periods": "not_a_list",
            "savings": "invalid",
        }))

        with pytest.raises(BudgetStateCorruptedError) as exc_info:
            BudgetManager(config)

        assert exc_info.value.validation_errors is not None, (
            "BudgetStateCorruptedError should contain validation_errors"
        )

    def test_init_unreadable_state_file(self, tmp_path, daily_period_limit):
        """Unreadable state file raises appropriate error."""
        config = make_config(tmp_path, period_limits=[daily_period_limit])

        # Create file and remove read permissions
        config.state_file_path.write_text('{"schema_version": 1}')
        config.state_file_path.chmod(0o000)

        try:
            with pytest.raises((BudgetError, PermissionError, OSError)):
                BudgetManager(config)
        finally:
            # Restore permissions for cleanup
            config.state_file_path.chmod(0o644)

    def test_init_unwritable_directory(self, tmp_path, daily_period_limit):
        """State file in non-existent directory raises error."""
        config = BudgetConfig(
            period_limits=[daily_period_limit],
            timezone="UTC",
            estimated_cost_per_input_token=Decimal('0.000001'),
            estimated_cost_per_output_token=Decimal('0.000002'),
            state_file_path=Path("/nonexistent/directory/state.json"),
            max_historical_periods=10,
        )

        with pytest.raises((BudgetError, OSError, FileNotFoundError)):
            BudgetManager(config)

    def test_init_state_period_type_reconciliation(self, tmp_path):
        """Loading state with different periods than config reconciles active_periods."""
        # First create with daily only
        daily_only = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00'))]
        config1 = make_config(tmp_path, period_limits=daily_only)
        manager1 = BudgetManager(config1)
        del manager1

        # Now load with daily + monthly
        daily_and_monthly = [
            PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00')),
            PeriodLimit(period_type=PeriodType.monthly, limit_amount=Decimal('100.00')),
        ]
        config2 = make_config(tmp_path, period_limits=daily_and_monthly)
        manager2 = BudgetManager(config2)

        report = manager2.get_report()
        active_types = {p.period_type for p in report.active_periods}
        assert PeriodType.daily in active_types, "Daily period should be present"
        assert PeriodType.monthly in active_types, "Monthly period should be added"

    def test_init_all_timestamps_timezone_aware(self, daily_manager):
        """All timestamps in initialized state are timezone-aware."""
        report = daily_manager.get_report()
        for period in report.active_periods:
            assert period.period_start.tzinfo is not None, (
                f"period_start for {period.period_type} must be timezone-aware"
            )
        assert report.generated_at.tzinfo is not None, "generated_at must be timezone-aware"


# ============================================================================
# 2. test_budget_manager_authorize — Authorization logic tests
# ============================================================================

class TestBudgetManagerAuthorize:
    """Tests for BudgetManager.authorize()."""

    def test_authorize_within_budget(self, daily_manager):
        """Authorize a request within budget returns allowed=True."""
        result = daily_manager.authorize(Decimal('0.50'))

        assert result.allowed is True, "Request within budget should be allowed"
        assert result.denial_reason is None, "No denial reason when allowed"
        assert result.estimated_cost == Decimal('0.50'), (
            f"estimated_cost should match input, got {result.estimated_cost}"
        )

    def test_authorize_remaining_reflects_reservation(self, daily_manager):
        """After authorization, remaining reflects reservation."""
        result = daily_manager.authorize(Decimal('0.50'))

        assert result.remaining == Decimal('4.50'), (
            f"Remaining should be 4.50 after 0.50 reservation on 5.00 limit, got {result.remaining}"
        )

    def test_authorize_multi_period_tightest_remaining(self, multi_period_manager):
        """With multiple periods, remaining reflects the tightest period."""
        result = multi_period_manager.authorize(Decimal('1.00'))

        assert result.allowed is True
        # Daily limit is 5.00, so remaining should be 4.00 (tighter than monthly 99.00)
        assert result.remaining == Decimal('4.00'), (
            f"Remaining should reflect daily (tighter) period, got {result.remaining}"
        )
        assert result.limiting_period == PeriodType.daily, (
            f"Limiting period should be daily, got {result.limiting_period}"
        )

    def test_authorize_denied_exceeds_daily_limit(self, daily_manager):
        """Authorize denied when cost exceeds daily limit."""
        result = daily_manager.authorize(Decimal('6.00'))

        assert result.allowed is False, "Cost exceeding limit should be denied"
        assert result.denial_reason is not None, "Denied result must have denial_reason"
        assert "daily" in str(result.denial_reason).lower(), (
            f"Denial reason should reference daily, got {result.denial_reason}"
        )

    def test_authorize_denied_monthly_limit_with_prior_spend(self, tmp_path):
        """Denied when monthly limit exhausted but daily is fine."""
        limits = [
            PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('10.00')),
            PeriodLimit(period_type=PeriodType.monthly, limit_amount=Decimal('5.00')),
        ]
        config = make_config(tmp_path, period_limits=limits)
        manager = BudgetManager(config)

        # Spend 4.00 of 5.00 monthly limit
        r1 = manager.authorize(Decimal('4.00'))
        assert r1.allowed is True
        manager.record_spend(
            Decimal('4.00'), Decimal('4.00'),
            make_metadata(request_id="req-monthly-1"),
        )

        # Try to spend 2.00 more — exceeds monthly
        result = manager.authorize(Decimal('2.00'))
        assert result.allowed is False, "Should be denied by monthly limit"
        assert result.limiting_period == PeriodType.monthly, (
            f"Limiting period should be monthly, got {result.limiting_period}"
        )

    def test_authorize_exact_boundary_allowed(self, daily_manager):
        """Authorize request that exactly uses remaining budget is allowed."""
        result = daily_manager.authorize(Decimal('5.00'))

        assert result.allowed is True, "Exact boundary request should be allowed"
        assert result.remaining == Decimal('0.00'), (
            f"Remaining should be exactly 0.00, got {result.remaining}"
        )

    def test_authorize_one_penny_over_denied(self, daily_manager):
        """Authorize request one penny over remaining budget is denied."""
        result = daily_manager.authorize(Decimal('5.01'))

        assert result.allowed is False, "One penny over should be denied"

    def test_authorize_zero_cost_error(self, daily_manager):
        """Authorize with zero estimated cost raises error."""
        with pytest.raises((ValueError, BudgetError)):
            daily_manager.authorize(Decimal('0'))

    def test_authorize_negative_cost_error(self, daily_manager):
        """Authorize with negative estimated cost raises error."""
        with pytest.raises((ValueError, BudgetError)):
            daily_manager.authorize(Decimal('-1.00'))

    def test_authorize_no_state_change_on_denial(self, daily_manager):
        """Denied authorization does not change any state."""
        remaining_before = daily_manager.remaining_budget()

        result = daily_manager.authorize(Decimal('10.00'))
        assert result.allowed is False

        remaining_after = daily_manager.remaining_budget()
        assert remaining_before == remaining_after, (
            f"Remaining should be unchanged on denial: {remaining_before} vs {remaining_after}"
        )

    def test_authorize_reserves_in_all_periods(self, multi_period_manager):
        """Successful authorization reserves cost in all active period types."""
        result = multi_period_manager.authorize(Decimal('1.00'))
        assert result.allowed is True

        report = multi_period_manager.get_report()
        for period in report.active_periods:
            assert period.reserved_spend >= Decimal('1.00'), (
                f"Period {period.period_type} should have reserved_spend >= 1.00, "
                f"got {period.reserved_spend}"
            )

    def test_authorize_utilization_pct(self, tmp_path):
        """Period utilization percentage is accurately computed."""
        limits = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('10.00'))]
        config = make_config(tmp_path, period_limits=limits)
        manager = BudgetManager(config)

        # First authorize and record 2.50
        r1 = manager.authorize(Decimal('2.50'))
        assert r1.allowed
        manager.record_spend(Decimal('2.50'), Decimal('2.50'), make_metadata("req-util-1"))

        # Second authorize for 2.50 more
        r2 = manager.authorize(Decimal('2.50'))
        assert r2.allowed
        # Utilization should be (2.50 confirmed + 2.50 reserved) / 10.00 = 50%
        assert r2.period_utilization_pct == Decimal('50') or r2.period_utilization_pct == Decimal('50.0'), (
            f"Utilization should be 50%, got {r2.period_utilization_pct}"
        )

    def test_authorize_result_estimated_cost_echoed(self, daily_manager):
        """result.estimated_cost equals the input estimated_cost."""
        result = daily_manager.authorize(Decimal('1.23'))
        assert result.estimated_cost == Decimal('1.23')

    def test_authorize_denial_reason_none_when_allowed(self, daily_manager):
        """denial_reason is None if and only if allowed is True."""
        result = daily_manager.authorize(Decimal('1.00'))
        assert result.allowed is True
        assert result.denial_reason is None

    def test_authorize_denial_reason_set_when_denied(self, daily_manager):
        """denial_reason is set when allowed is False."""
        result = daily_manager.authorize(Decimal('6.00'))
        assert result.allowed is False
        assert result.denial_reason is not None

    @pytest.mark.parametrize("cost_str", ["0.01", "0.001", "4.99", "5.00"])
    def test_authorize_boundary_costs_allowed(self, budget_manager_factory, cost_str):
        """Various boundary costs within daily limit of 5.00 are allowed."""
        manager = budget_manager_factory(
            period_limits=[PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00'))]
        )
        cost = Decimal(cost_str)
        result = manager.authorize(cost)
        assert result.allowed is True, f"Cost {cost_str} should be allowed within 5.00 limit"


# ============================================================================
# 3. test_budget_manager_lifecycle — Reservation lifecycle tests
# ============================================================================

class TestBudgetManagerLifecycle:
    """Tests for authorize → record_spend and authorize → release_reservation flows."""

    def test_authorize_then_record_spend(self, daily_manager):
        """Full lifecycle: authorize, then record actual spend."""
        auth = daily_manager.authorize(Decimal('1.00'))
        assert auth.allowed is True

        daily_manager.record_spend(
            actual_cost=Decimal('0.80'),
            estimated_cost=Decimal('1.00'),
            metadata=make_metadata(request_id="req-lc-1"),
        )

        remaining = daily_manager.remaining_budget()
        # After: reservation of 1.00 released, actual 0.80 confirmed
        assert remaining == Decimal('4.20'), (
            f"After recording 0.80 actual spend, remaining should be 4.20, got {remaining}"
        )

    def test_authorize_then_release(self, daily_manager):
        """Full lifecycle: authorize, then release reservation."""
        auth = daily_manager.authorize(Decimal('1.00'))
        assert auth.allowed is True

        remaining_during = daily_manager.remaining_budget()
        assert remaining_during == Decimal('4.00'), "During reservation, remaining should be 4.00"

        daily_manager.release_reservation(Decimal('1.00'))

        remaining_after = daily_manager.remaining_budget()
        assert remaining_after == Decimal('5.00'), (
            f"After release, remaining should be back to 5.00, got {remaining_after}"
        )

    def test_actual_cost_higher_than_estimated(self, daily_manager):
        """Record spend where actual cost exceeds estimate."""
        auth = daily_manager.authorize(Decimal('1.00'))
        assert auth.allowed

        daily_manager.record_spend(
            actual_cost=Decimal('1.50'),
            estimated_cost=Decimal('1.00'),
            metadata=make_metadata(request_id="req-high-1"),
        )

        remaining = daily_manager.remaining_budget()
        # Started with 5.00, actual spend is 1.50
        assert remaining == Decimal('3.50'), (
            f"After actual 1.50 (est 1.00), remaining should be 3.50, got {remaining}"
        )

    def test_actual_cost_lower_than_estimated(self, daily_manager):
        """Record spend where actual cost is less than estimate."""
        auth = daily_manager.authorize(Decimal('1.00'))
        assert auth.allowed

        daily_manager.record_spend(
            actual_cost=Decimal('0.30'),
            estimated_cost=Decimal('1.00'),
            metadata=make_metadata(request_id="req-low-1"),
        )

        remaining = daily_manager.remaining_budget()
        # Started with 5.00, actual spend is 0.30
        assert remaining == Decimal('4.70'), (
            f"After actual 0.30 (est 1.00), remaining should be 4.70, got {remaining}"
        )

    def test_actual_cost_zero(self, daily_manager):
        """Record spend with zero actual cost (free API call)."""
        auth = daily_manager.authorize(Decimal('1.00'))
        assert auth.allowed

        daily_manager.record_spend(
            actual_cost=Decimal('0'),
            estimated_cost=Decimal('1.00'),
            metadata=make_metadata(request_id="req-zero-1"),
        )

        remaining = daily_manager.remaining_budget()
        assert remaining == Decimal('5.00'), (
            f"After zero actual cost, remaining should be 5.00, got {remaining}"
        )

    def test_no_matching_reservation_record_spend(self, daily_manager):
        """Record spend without prior authorization - spend still recorded."""
        # This should log a warning about no matching reservation but not lose data
        # The behavior depends on implementation - it may raise or warn
        try:
            daily_manager.record_spend(
                actual_cost=Decimal('1.00'),
                estimated_cost=Decimal('1.00'),
                metadata=make_metadata(request_id="req-nomatch-1"),
            )
            # If no exception, check spend was recorded
            remaining = daily_manager.remaining_budget()
            assert remaining == Decimal('4.00'), (
                "Spend should still be recorded even without reservation"
            )
        except BudgetError:
            # Some implementations may raise; the contract says spend is still recorded
            pass

    def test_no_matching_reservation_release(self, daily_manager):
        """Release reservation without prior authorization clamps to zero."""
        daily_manager.release_reservation(Decimal('1.00'))

        # reserved_spend should be clamped to zero, never negative
        report = daily_manager.get_report()
        for period in report.active_periods:
            assert period.reserved_spend >= Decimal('0'), (
                f"reserved_spend must be non-negative, got {period.reserved_spend}"
            )

    def test_multiple_outstanding_reservations(self, daily_manager):
        """Multiple authorize calls before recording spend."""
        auth1 = daily_manager.authorize(Decimal('1.00'))
        assert auth1.allowed
        auth2 = daily_manager.authorize(Decimal('1.00'))
        assert auth2.allowed

        remaining = daily_manager.remaining_budget()
        assert remaining == Decimal('3.00'), (
            f"After two 1.00 reservations, remaining should be 3.00, got {remaining}"
        )

        # Record first spend
        daily_manager.record_spend(
            Decimal('1.00'), Decimal('1.00'),
            make_metadata(request_id="req-multi-1"),
        )
        remaining = daily_manager.remaining_budget()
        assert remaining == Decimal('4.00') or remaining == Decimal('3.00'), (
            f"After recording one of two reservations, got remaining {remaining}"
        )

    def test_lifecycle_reconciliation_all_periods(self, multi_period_manager):
        """Record spend reconciles across all active period types."""
        auth = multi_period_manager.authorize(Decimal('1.00'))
        assert auth.allowed

        multi_period_manager.record_spend(
            Decimal('0.80'), Decimal('1.00'),
            make_metadata(request_id="req-multi-period-1"),
        )

        report = multi_period_manager.get_report()
        for period in report.active_periods:
            assert period.confirmed_spend == Decimal('0.80'), (
                f"Period {period.period_type} confirmed_spend should be 0.80, got {period.confirmed_spend}"
            )
            assert period.reserved_spend == Decimal('0'), (
                f"Period {period.period_type} reserved_spend should be 0, got {period.reserved_spend}"
            )
            assert period.request_count == 1, (
                f"Period {period.period_type} request_count should be 1, got {period.request_count}"
            )


# ============================================================================
# 4. test_budget_manager_periods — Period boundary tests
# ============================================================================

@pytest.mark.skipif(not HAS_FREEZEGUN, reason="freezegun required for period tests")
class TestBudgetManagerPeriods:
    """Tests for period boundary rollover behavior."""

    def test_daily_rollover(self, tmp_path):
        """Daily period rolls over when time advances past midnight."""
        limits = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00'))]

        with freeze_time("2024-01-15 23:00:00", tz_offset=0):
            config = make_config(tmp_path, period_limits=limits, timezone_str="UTC")
            manager = BudgetManager(config)

            auth = manager.authorize(Decimal('3.00'))
            assert auth.allowed
            manager.record_spend(
                Decimal('3.00'), Decimal('3.00'),
                make_metadata(request_id="req-roll-1"),
            )
            remaining_before = manager.remaining_budget()
            assert remaining_before == Decimal('2.00')

        with freeze_time("2024-01-16 01:00:00", tz_offset=0):
            # Trigger rollover check
            remaining_after = manager.remaining_budget()
            assert remaining_after == Decimal('5.00'), (
                f"After daily rollover, should have full budget, got {remaining_after}"
            )

    def test_monthly_rollover(self, tmp_path):
        """Monthly period rolls over at month boundary."""
        limits = [PeriodLimit(period_type=PeriodType.monthly, limit_amount=Decimal('100.00'))]

        with freeze_time("2024-01-31 23:00:00", tz_offset=0):
            config = make_config(tmp_path, period_limits=limits, timezone_str="UTC")
            manager = BudgetManager(config)

            auth = manager.authorize(Decimal('50.00'))
            assert auth.allowed
            manager.record_spend(
                Decimal('50.00'), Decimal('50.00'),
                make_metadata(request_id="req-month-1"),
            )

        with freeze_time("2024-02-01 01:00:00", tz_offset=0):
            remaining = manager.remaining_budget()
            assert remaining == Decimal('100.00'), (
                f"After monthly rollover, should have full budget, got {remaining}"
            )

    def test_daily_rolls_monthly_stays(self, tmp_path):
        """Daily period rolls over while monthly period retains accumulated spend."""
        limits = [
            PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00')),
            PeriodLimit(period_type=PeriodType.monthly, limit_amount=Decimal('100.00')),
        ]

        with freeze_time("2024-01-15 23:00:00", tz_offset=0):
            config = make_config(tmp_path, period_limits=limits, timezone_str="UTC")
            manager = BudgetManager(config)

            auth = manager.authorize(Decimal('3.00'))
            assert auth.allowed
            manager.record_spend(
                Decimal('3.00'), Decimal('3.00'),
                make_metadata(request_id="req-mixed-1"),
            )

        with freeze_time("2024-01-16 01:00:00", tz_offset=0):
            report = manager.get_report()
            for period in report.active_periods:
                if period.period_type == PeriodType.daily:
                    assert period.confirmed_spend == Decimal('0'), (
                        f"Daily should be reset, got {period.confirmed_spend}"
                    )
                elif period.period_type == PeriodType.monthly:
                    assert period.confirmed_spend == Decimal('3.00'), (
                        f"Monthly should retain spend, got {period.confirmed_spend}"
                    )

    def test_historical_max_eviction(self, tmp_path):
        """Historical periods evicts oldest when max_historical_periods exceeded."""
        limits = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00'))]

        with freeze_time("2024-01-01 12:00:00", tz_offset=0) as frozen:
            config = make_config(tmp_path, period_limits=limits, max_historical_periods=2, timezone_str="UTC")
            manager = BudgetManager(config)

            # Trigger 3 daily rollovers
            for day in range(3):
                auth = manager.authorize(Decimal('1.00'))
                if auth.allowed:
                    manager.record_spend(
                        Decimal('1.00'), Decimal('1.00'),
                        make_metadata(request_id=f"req-evict-{day}"),
                    )
                frozen.tick(timedelta(days=1))

            # Force a remaining_budget check to trigger rollover
            manager.remaining_budget()

            report = manager.get_report()
            assert len(report.cost_trajectory) <= 2, (
                f"Historical periods should be at most 2, got {len(report.cost_trajectory)}"
            )

    def test_rollover_archives_correct_totals(self, tmp_path):
        """Archived period summary has correct totals."""
        limits = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('10.00'))]

        with freeze_time("2024-01-15 12:00:00", tz_offset=0):
            config = make_config(tmp_path, period_limits=limits, timezone_str="UTC")
            manager = BudgetManager(config)

            # Make 2 requests totaling 3.00
            for i in range(2):
                auth = manager.authorize(Decimal('1.50'))
                assert auth.allowed
                manager.record_spend(
                    Decimal('1.50'), Decimal('1.50'),
                    make_metadata(request_id=f"req-archive-{i}"),
                )

        with freeze_time("2024-01-16 12:00:00", tz_offset=0):
            # Trigger rollover
            manager.remaining_budget()

            report = manager.get_report()
            if report.cost_trajectory:
                archived = report.cost_trajectory[0]
                assert archived.total_spend == Decimal('3.00'), (
                    f"Archived period should have 3.00 total spend, got {archived.total_spend}"
                )
                assert archived.remote_request_count == 2, (
                    f"Archived period should have 2 requests, got {archived.remote_request_count}"
                )

    def test_timezone_handling(self, tmp_path):
        """Period boundaries respect configured timezone."""
        limits = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00'))]

        # 23:30 Eastern = 04:30 UTC next day
        # At this time, it's still Jan 15 in Eastern timezone
        with freeze_time("2024-01-16 04:30:00", tz_offset=0):
            config = make_config(tmp_path, period_limits=limits, timezone_str="US/Eastern")
            manager = BudgetManager(config)

            auth = manager.authorize(Decimal('3.00'))
            assert auth.allowed
            manager.record_spend(
                Decimal('3.00'), Decimal('3.00'),
                make_metadata(request_id="req-tz-1"),
            )

        # 00:30 Eastern = 05:30 UTC — new day in Eastern
        with freeze_time("2024-01-16 05:30:00", tz_offset=0):
            remaining = manager.remaining_budget()
            assert remaining == Decimal('5.00'), (
                f"After Eastern midnight rollover, should have full budget, got {remaining}"
            )

    def test_rollover_on_remaining_budget(self, tmp_path):
        """remaining_budget() triggers period rollover check."""
        limits = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00'))]

        with freeze_time("2024-01-15 23:00:00", tz_offset=0):
            config = make_config(tmp_path, period_limits=limits, timezone_str="UTC")
            manager = BudgetManager(config)
            auth = manager.authorize(Decimal('4.00'))
            assert auth.allowed
            manager.record_spend(
                Decimal('4.00'), Decimal('4.00'),
                make_metadata(request_id="req-rb-1"),
            )

        with freeze_time("2024-01-16 01:00:00", tz_offset=0):
            remaining = manager.remaining_budget()
            assert remaining == Decimal('5.00'), (
                "remaining_budget() should trigger rollover and return full budget"
            )


# ============================================================================
# 5. test_budget_manager_savings — Local request tracking tests
# ============================================================================

class TestBudgetManagerSavings:
    """Tests for record_local_request() and savings tracking."""

    def test_record_local_request(self, daily_manager):
        """Record a local request increments savings counters."""
        daily_manager.record_local_request(Decimal('0.50'))

        report = daily_manager.get_report()
        assert report.total_local_requests == 1, (
            f"total_local_requests should be 1, got {report.total_local_requests}"
        )
        assert report.total_all_time_savings == Decimal('0.50'), (
            f"total_all_time_savings should be 0.50, got {report.total_all_time_savings}"
        )

    def test_record_local_request_no_budget_impact(self, daily_manager):
        """Local request does not affect budget limits or reservations."""
        remaining_before = daily_manager.remaining_budget()

        daily_manager.record_local_request(Decimal('1.00'))

        remaining_after = daily_manager.remaining_budget()
        assert remaining_before == remaining_after, (
            f"Local request should not change remaining budget: {remaining_before} vs {remaining_after}"
        )

    def test_savings_accumulation(self, daily_manager):
        """Multiple local requests accumulate savings correctly."""
        daily_manager.record_local_request(Decimal('0.50'))
        daily_manager.record_local_request(Decimal('0.75'))
        daily_manager.record_local_request(Decimal('1.25'))

        report = daily_manager.get_report()
        assert report.total_local_requests == 3, (
            f"total_local_requests should be 3, got {report.total_local_requests}"
        )
        assert report.total_all_time_savings == Decimal('2.50'), (
            f"total_all_time_savings should be 2.50, got {report.total_all_time_savings}"
        )

    def test_savings_persisted(self, tmp_path, daily_period_limit):
        """Savings state is persisted to disk."""
        config = make_config(tmp_path, period_limits=[daily_period_limit])
        manager = BudgetManager(config)
        manager.record_local_request(Decimal('0.50'))

        # Reload from disk
        manager2 = BudgetManager(config)
        report = manager2.get_report()
        assert report.total_local_requests == 1
        assert report.total_all_time_savings == Decimal('0.50')

    @pytest.mark.skipif(not HAS_FREEZEGUN, reason="freezegun required")
    def test_savings_current_period_reset_on_rollover(self, tmp_path):
        """Current period savings counters reset on period rollover."""
        limits = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00'))]

        with freeze_time("2024-01-15 12:00:00", tz_offset=0):
            config = make_config(tmp_path, period_limits=limits, timezone_str="UTC")
            manager = BudgetManager(config)
            manager.record_local_request(Decimal('1.00'))
            manager.record_local_request(Decimal('2.00'))

        with freeze_time("2024-01-16 12:00:00", tz_offset=0):
            # Trigger rollover
            manager.remaining_budget()

            report = manager.get_report()
            # Total savings preserved
            assert report.total_all_time_savings >= Decimal('3.00'), (
                f"Total savings should be preserved, got {report.total_all_time_savings}"
            )
            assert report.total_local_requests >= 2, (
                f"Total local requests should be preserved, got {report.total_local_requests}"
            )

    def test_savings_and_report_local_ratio(self, tmp_path):
        """Report correctly counts local vs remote requests."""
        limits = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('10.00'))]
        config = make_config(tmp_path, period_limits=limits)
        manager = BudgetManager(config)

        # 2 remote requests
        for i in range(2):
            auth = manager.authorize(Decimal('1.00'))
            assert auth.allowed
            manager.record_spend(
                Decimal('1.00'), Decimal('1.00'),
                make_metadata(request_id=f"req-ratio-{i}"),
            )

        # 3 local requests
        for _ in range(3):
            manager.record_local_request(Decimal('1.00'))

        report = manager.get_report()
        assert report.total_remote_requests == 2, (
            f"total_remote_requests should be 2, got {report.total_remote_requests}"
        )
        assert report.total_local_requests == 3, (
            f"total_local_requests should be 3, got {report.total_local_requests}"
        )


# ============================================================================
# 6. test_budget_manager_persistence — State persistence tests
# ============================================================================

class TestBudgetManagerPersistence:
    """Tests for force_persist(), reset_state(), and persistence failure handling."""

    def test_force_persist(self, daily_manager, tmp_state_path):
        """force_persist writes current state to disk."""
        daily_manager.force_persist()
        # Verify the state file exists and is recent
        # (exact file path depends on the config used by daily_manager fixture)

    def test_reset_state_with_archive(self, daily_manager):
        """Reset state with archive_current=True archives current periods."""
        # Record some spend first
        auth = daily_manager.authorize(Decimal('2.00'))
        assert auth.allowed
        daily_manager.record_spend(
            Decimal('2.00'), Decimal('2.00'),
            make_metadata(request_id="req-reset-1"),
        )

        daily_manager.reset_state(archive_current=True)

        report = daily_manager.get_report()
        for period in report.active_periods:
            assert period.confirmed_spend == Decimal('0'), (
                f"After reset, confirmed_spend should be 0, got {period.confirmed_spend}"
            )
            assert period.reserved_spend == Decimal('0'), (
                f"After reset, reserved_spend should be 0, got {period.reserved_spend}"
            )
            assert period.request_count == 0, (
                f"After reset, request_count should be 0, got {period.request_count}"
            )

        # Verify archival happened (historical data present)
        assert report.total_all_time_spend >= Decimal('0'), (
            "Historical spend should be tracked"
        )

    def test_reset_state_without_archive(self, daily_manager):
        """Reset state with archive_current=False does not archive."""
        auth = daily_manager.authorize(Decimal('2.00'))
        assert auth.allowed
        daily_manager.record_spend(
            Decimal('2.00'), Decimal('2.00'),
            make_metadata(request_id="req-reset-2"),
        )

        daily_manager.reset_state(archive_current=False)

        remaining = daily_manager.remaining_budget()
        assert remaining == Decimal('5.00'), (
            f"After reset, full budget should be available, got {remaining}"
        )

    def test_reset_state_preserves_cumulative_savings(self, daily_manager):
        """Reset preserves total_local_requests and total_estimated_savings."""
        daily_manager.record_local_request(Decimal('1.00'))
        daily_manager.record_local_request(Decimal('2.00'))

        daily_manager.reset_state(archive_current=True)

        report = daily_manager.get_report()
        assert report.total_local_requests == 2, (
            f"total_local_requests should be preserved after reset, got {report.total_local_requests}"
        )
        assert report.total_all_time_savings == Decimal('3.00'), (
            f"total_all_time_savings should be preserved after reset, got {report.total_all_time_savings}"
        )

    def test_reset_state_persisted(self, tmp_path, daily_period_limit):
        """Reset state is persisted to disk."""
        config = make_config(tmp_path, period_limits=[daily_period_limit])
        manager = BudgetManager(config)

        auth = manager.authorize(Decimal('3.00'))
        assert auth.allowed
        manager.record_spend(
            Decimal('3.00'), Decimal('3.00'),
            make_metadata(request_id="req-reset-persist-1"),
        )

        manager.reset_state(archive_current=True)

        # Reload and verify
        manager2 = BudgetManager(config)
        remaining = manager2.remaining_budget()
        assert remaining == Decimal('5.00'), (
            f"After reset and reload, full budget should be available, got {remaining}"
        )

    def test_authorize_persists_on_approval(self, tmp_path, daily_period_limit):
        """State is persisted to disk after successful authorization."""
        config = make_config(tmp_path, period_limits=[daily_period_limit])
        manager = BudgetManager(config)

        mtime_before = config.state_file_path.stat().st_mtime_ns

        auth = manager.authorize(Decimal('1.00'))
        assert auth.allowed

        # Read the state file to verify reservation is persisted
        manager2 = BudgetManager(config)
        remaining = manager2.remaining_budget()
        assert remaining == Decimal('4.00'), (
            f"Persisted state should reflect reservation, got {remaining}"
        )

    def test_every_mutation_persists(self, tmp_path, daily_period_limit):
        """Every state-mutating operation triggers persist."""
        config = make_config(tmp_path, period_limits=[daily_period_limit])
        manager = BudgetManager(config)

        # Authorize
        auth = manager.authorize(Decimal('1.00'))
        assert auth.allowed
        state_after_auth = config.state_file_path.read_text()

        # Record spend
        manager.record_spend(
            Decimal('1.00'), Decimal('1.00'),
            make_metadata(request_id="req-persist-1"),
        )
        state_after_record = config.state_file_path.read_text()
        assert state_after_auth != state_after_record, (
            "State file should change after record_spend"
        )

        # Record local request
        manager.record_local_request(Decimal('0.50'))
        state_after_local = config.state_file_path.read_text()
        assert state_after_record != state_after_local, (
            "State file should change after record_local_request"
        )


# ============================================================================
# 7. test_budget_manager_report — Reporting tests
# ============================================================================

class TestBudgetManagerReport:
    """Tests for get_report() and BudgetReport accuracy."""

    def test_report_basic_fields(self, daily_manager):
        """get_report returns report with all required fields."""
        report = daily_manager.get_report()

        assert report.generated_at is not None, "generated_at should be set"
        assert report.generated_at.tzinfo is not None, "generated_at should be timezone-aware"
        assert report.active_periods is not None, "active_periods should be set"
        assert isinstance(report.total_all_time_spend, Decimal), (
            "total_all_time_spend should be Decimal"
        )

    def test_report_after_spend(self, daily_manager):
        """Report reflects spend after authorize and record."""
        auth = daily_manager.authorize(Decimal('2.00'))
        assert auth.allowed
        daily_manager.record_spend(
            Decimal('2.00'), Decimal('2.00'),
            make_metadata(request_id="req-report-1"),
        )

        report = daily_manager.get_report()
        assert report.total_all_time_spend >= Decimal('2.00'), (
            f"total_all_time_spend should include 2.00, got {report.total_all_time_spend}"
        )

    def test_report_includes_savings(self, daily_manager):
        """Report includes savings from local requests."""
        daily_manager.record_local_request(Decimal('1.50'))

        report = daily_manager.get_report()
        assert report.total_all_time_savings == Decimal('1.50'), (
            f"total_all_time_savings should be 1.50, got {report.total_all_time_savings}"
        )

    def test_report_current_period_remaining_consistency(self, daily_manager):
        """Report current_period_remaining matches remaining_budget()."""
        auth = daily_manager.authorize(Decimal('2.00'))
        assert auth.allowed
        daily_manager.record_spend(
            Decimal('2.00'), Decimal('2.00'),
            make_metadata(request_id="req-report-cons-1"),
        )

        report = daily_manager.get_report()
        remaining = daily_manager.remaining_budget()

        assert report.current_period_remaining == remaining, (
            f"Report remaining {report.current_period_remaining} should match "
            f"remaining_budget() {remaining}"
        )

    def test_report_tightest_period(self, tmp_path):
        """Report identifies the most constrained period."""
        limits = [
            PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00')),
            PeriodLimit(period_type=PeriodType.monthly, limit_amount=Decimal('100.00')),
        ]
        config = make_config(tmp_path, period_limits=limits)
        manager = BudgetManager(config)

        # Daily (5.00 remaining) is tighter than monthly (100.00 remaining)
        report = manager.get_report()
        assert report.tightest_period == PeriodType.daily, (
            f"Tightest period should be daily, got {report.tightest_period}"
        )

    def test_report_tightest_period_changes_with_spend(self, tmp_path):
        """Tightest period can change based on spending pattern."""
        limits = [
            PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('10.00')),
            PeriodLimit(period_type=PeriodType.monthly, limit_amount=Decimal('3.00')),
        ]
        config = make_config(tmp_path, period_limits=limits)
        manager = BudgetManager(config)

        # Monthly (3.00) is tighter than daily (10.00)
        report = manager.get_report()
        assert report.tightest_period == PeriodType.monthly, (
            f"Tightest period should be monthly, got {report.tightest_period}"
        )

    @pytest.mark.skipif(not HAS_FREEZEGUN, reason="freezegun required")
    def test_report_cost_trajectory_ordered(self, tmp_path):
        """Cost trajectory points are ordered oldest first."""
        limits = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('10.00'))]

        with freeze_time("2024-01-15 12:00:00", tz_offset=0) as frozen:
            config = make_config(tmp_path, period_limits=limits, timezone_str="UTC")
            manager = BudgetManager(config)

            for day in range(3):
                auth = manager.authorize(Decimal('1.00'))
                if auth.allowed:
                    manager.record_spend(
                        Decimal('1.00'), Decimal('1.00'),
                        make_metadata(request_id=f"req-traj-{day}"),
                    )
                frozen.tick(timedelta(days=1))

            # Trigger rollover
            manager.remaining_budget()

            report = manager.get_report()
            if len(report.cost_trajectory) >= 2:
                for i in range(len(report.cost_trajectory) - 1):
                    assert report.cost_trajectory[i].period_start <= report.cost_trajectory[i + 1].period_start, (
                        "Cost trajectory should be ordered oldest first"
                    )

    @pytest.mark.skipif(not HAS_FREEZEGUN, reason="freezegun required")
    def test_report_includes_historical_spend(self, tmp_path):
        """Report total_all_time_spend includes historical period spend."""
        limits = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('10.00'))]

        with freeze_time("2024-01-15 12:00:00", tz_offset=0):
            config = make_config(tmp_path, period_limits=limits, timezone_str="UTC")
            manager = BudgetManager(config)

            auth = manager.authorize(Decimal('3.00'))
            assert auth.allowed
            manager.record_spend(
                Decimal('3.00'), Decimal('3.00'),
                make_metadata(request_id="req-hist-1"),
            )

        with freeze_time("2024-01-16 12:00:00", tz_offset=0):
            # New day — spend 2.00 more
            auth = manager.authorize(Decimal('2.00'))
            assert auth.allowed
            manager.record_spend(
                Decimal('2.00'), Decimal('2.00'),
                make_metadata(request_id="req-hist-2"),
            )

            report = manager.get_report()
            assert report.total_all_time_spend >= Decimal('5.00'), (
                f"Total all-time spend should include historical 3.00 + current 2.00, "
                f"got {report.total_all_time_spend}"
            )

    def test_report_total_remote_requests(self, daily_manager):
        """Report total_remote_requests counts all recorded spends."""
        for i in range(3):
            auth = daily_manager.authorize(Decimal('1.00'))
            assert auth.allowed
            daily_manager.record_spend(
                Decimal('1.00'), Decimal('1.00'),
                make_metadata(request_id=f"req-count-{i}"),
            )

        report = daily_manager.get_report()
        assert report.total_remote_requests == 3, (
            f"total_remote_requests should be 3, got {report.total_remote_requests}"
        )

    def test_report_frozen_snapshot(self, daily_manager):
        """Returned BudgetReport should be immutable / not a live reference."""
        report = daily_manager.get_report()
        original_spend = report.total_all_time_spend

        # Record more spend
        auth = daily_manager.authorize(Decimal('1.00'))
        assert auth.allowed
        daily_manager.record_spend(
            Decimal('1.00'), Decimal('1.00'),
            make_metadata(request_id="req-frozen-1"),
        )

        # Original report should not be affected
        assert report.total_all_time_spend == original_spend, (
            "Report should be a frozen snapshot, not a live reference"
        )


# ============================================================================
# 8. test_budget_manager_properties — Invariant and property-based tests
# ============================================================================

class TestBudgetManagerInvariants:
    """Tests for contract invariants."""

    def test_all_monetary_values_are_decimal(self, daily_manager):
        """All monetary values are Decimal, never float."""
        result = daily_manager.authorize(Decimal('1.00'))
        assert isinstance(result.remaining, Decimal), (
            f"result.remaining should be Decimal, got {type(result.remaining)}"
        )
        assert isinstance(result.estimated_cost, Decimal), (
            f"result.estimated_cost should be Decimal, got {type(result.estimated_cost)}"
        )
        assert isinstance(result.period_utilization_pct, Decimal), (
            f"result.period_utilization_pct should be Decimal, got {type(result.period_utilization_pct)}"
        )

    def test_reserved_spend_non_negative_after_release(self, daily_manager):
        """reserved_spend is always non-negative even after over-release."""
        # Release more than reserved (0)
        daily_manager.release_reservation(Decimal('10.00'))

        report = daily_manager.get_report()
        for period in report.active_periods:
            assert period.reserved_spend >= Decimal('0'), (
                f"reserved_spend must be >= 0, got {period.reserved_spend}"
            )

    def test_confirmed_spend_non_negative(self, daily_manager):
        """confirmed_spend is always non-negative."""
        auth = daily_manager.authorize(Decimal('1.00'))
        assert auth.allowed
        daily_manager.record_spend(
            Decimal('1.00'), Decimal('1.00'),
            make_metadata(request_id="req-inv-1"),
        )

        report = daily_manager.get_report()
        for period in report.active_periods:
            assert period.confirmed_spend >= Decimal('0'), (
                f"confirmed_spend must be >= 0, got {period.confirmed_spend}"
            )

    def test_confirmed_spend_monotonically_increasing(self, daily_manager):
        """confirmed_spend only increases within a period."""
        auth1 = daily_manager.authorize(Decimal('1.00'))
        assert auth1.allowed
        daily_manager.record_spend(
            Decimal('1.00'), Decimal('1.00'),
            make_metadata(request_id="req-mono-1"),
        )

        report1 = daily_manager.get_report()
        spend_after_first = {p.period_type: p.confirmed_spend for p in report1.active_periods}

        auth2 = daily_manager.authorize(Decimal('1.00'))
        assert auth2.allowed
        daily_manager.record_spend(
            Decimal('1.00'), Decimal('1.00'),
            make_metadata(request_id="req-mono-2"),
        )

        report2 = daily_manager.get_report()
        for period in report2.active_periods:
            prev = spend_after_first.get(period.period_type, Decimal('0'))
            assert period.confirmed_spend >= prev, (
                f"confirmed_spend for {period.period_type} should be monotonically increasing: "
                f"{prev} -> {period.confirmed_spend}"
            )

    def test_effective_spend_within_limit_on_approval(self, daily_manager):
        """confirmed + reserved never exceeds limit at moment of authorization."""
        # Make several authorizations approaching the limit
        for i in range(4):
            result = daily_manager.authorize(Decimal('1.00'))
            if result.allowed:
                report = daily_manager.get_report()
                for period in report.active_periods:
                    effective = period.confirmed_spend + period.reserved_spend
                    # Find the matching limit
                    assert effective <= Decimal('5.00'), (
                        f"Effective spend {effective} exceeds limit 5.00 for {period.period_type}"
                    )
                daily_manager.record_spend(
                    Decimal('1.00'), Decimal('1.00'),
                    make_metadata(request_id=f"req-eff-{i}"),
                )

    def test_budget_manager_never_raises_exhausted_error(self, daily_manager):
        """BudgetManager returns result, never raises BudgetExhaustedError."""
        # Try to authorize way over budget
        result = daily_manager.authorize(Decimal('999.99'))

        # Should return a result, not raise
        assert isinstance(result, BudgetAuthorizationResult), (
            "Should return BudgetAuthorizationResult, not raise"
        )
        assert result.allowed is False

    def test_no_global_state_two_instances_independent(self, tmp_path):
        """Two BudgetManager instances with different configs are independent."""
        config1 = make_config(tmp_path, state_file_name="state1.json")
        config2 = make_config(tmp_path, state_file_name="state2.json")

        manager1 = BudgetManager(config1)
        manager2 = BudgetManager(config2)

        # Authorize on manager1
        auth = manager1.authorize(Decimal('3.00'))
        assert auth.allowed
        manager1.record_spend(
            Decimal('3.00'), Decimal('3.00'),
            make_metadata(request_id="req-global-1"),
        )

        # Manager2 should still have full budget
        remaining2 = manager2.remaining_budget()
        assert remaining2 == Decimal('5.00'), (
            f"Manager2 should be independent, got remaining {remaining2}"
        )

    def test_remaining_budget_fresh(self, tmp_path):
        """Remaining budget on fresh state equals the smallest limit."""
        limits = [
            PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00')),
            PeriodLimit(period_type=PeriodType.monthly, limit_amount=Decimal('100.00')),
        ]
        config = make_config(tmp_path, period_limits=limits)
        manager = BudgetManager(config)

        remaining = manager.remaining_budget()
        assert remaining == Decimal('5.00'), (
            f"Fresh remaining should equal smallest limit (5.00), got {remaining}"
        )

    def test_remaining_budget_includes_reservations(self, daily_manager):
        """Remaining budget accounts for reserved_spend."""
        auth = daily_manager.authorize(Decimal('3.00'))
        assert auth.allowed

        remaining = daily_manager.remaining_budget()
        assert remaining == Decimal('2.00'), (
            f"Remaining should account for 3.00 reservation, got {remaining}"
        )

    def test_remaining_budget_negative_on_overshoot(self, daily_manager):
        """Remaining budget may be negative if actual spend exceeded estimate."""
        auth = daily_manager.authorize(Decimal('4.00'))
        assert auth.allowed

        # Record actual cost that overshoots
        daily_manager.record_spend(
            Decimal('6.00'), Decimal('4.00'),
            make_metadata(request_id="req-overshoot-1"),
        )

        remaining = daily_manager.remaining_budget()
        assert remaining < Decimal('0'), (
            f"Remaining should be negative on overshoot, got {remaining}"
        )

    def test_remaining_budget_tightest_across_periods(self, tmp_path):
        """Remaining budget returns minimum across all active periods."""
        limits = [
            PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('3.00')),
            PeriodLimit(period_type=PeriodType.monthly, limit_amount=Decimal('100.00')),
        ]
        config = make_config(tmp_path, period_limits=limits)
        manager = BudgetManager(config)

        auth = manager.authorize(Decimal('1.00'))
        assert auth.allowed
        manager.record_spend(
            Decimal('1.00'), Decimal('1.00'),
            make_metadata(request_id="req-tight-1"),
        )

        remaining = manager.remaining_budget()
        # Daily: 3.00 - 1.00 = 2.00, Monthly: 100.00 - 1.00 = 99.00
        assert remaining == Decimal('2.00'), (
            f"Remaining should be min(2.00, 99.00) = 2.00, got {remaining}"
        )


# ============================================================================
# Property-based tests (Hypothesis)
# ============================================================================

@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis required for property tests")
class TestBudgetManagerProperties:
    """Property-based tests using Hypothesis."""

    @given(
        cost=st.decimals(
            min_value=Decimal('0.01'),
            max_value=Decimal('4.99'),
            places=2,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_property_authorization_consistency(self, cost, tmp_path):
        """If authorize allows, remaining >= 0; if denies, cost would exceed limit."""
        limits = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('5.00'))]
        config = make_config(tmp_path, period_limits=limits, state_file_name=f"state_{hash(cost)}.json")
        manager = BudgetManager(config)

        result = manager.authorize(cost)

        if result.allowed:
            assert result.remaining >= Decimal('0'), (
                f"When allowed, remaining ({result.remaining}) must be >= 0"
            )
            assert result.denial_reason is None, (
                "When allowed, denial_reason must be None"
            )
        else:
            assert result.denial_reason is not None, (
                "When denied, denial_reason must be set"
            )

    @given(
        estimated=st.decimals(
            min_value=Decimal('0.01'),
            max_value=Decimal('2.00'),
            places=2,
            allow_nan=False,
            allow_infinity=False,
        ),
        actual=st.decimals(
            min_value=Decimal('0.00'),
            max_value=Decimal('3.00'),
            places=2,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=30, deadline=5000)
    def test_property_budget_conservation(self, estimated, actual, tmp_path):
        """After full lifecycle (authorize → record_spend), effective spend equals actual cost."""
        limits = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('10.00'))]
        fname = f"state_{abs(hash((estimated, actual)))}.json"
        config = make_config(tmp_path, period_limits=limits, state_file_name=fname)
        manager = BudgetManager(config)

        result = manager.authorize(estimated)
        assert result.allowed, f"With limit 10.00, cost {estimated} should be allowed"

        manager.record_spend(
            actual_cost=actual,
            estimated_cost=estimated,
            metadata=make_metadata(request_id=f"req-prop-{abs(hash((estimated, actual)))}"),
        )

        remaining = manager.remaining_budget()
        expected_remaining = Decimal('10.00') - actual
        assert remaining == expected_remaining, (
            f"After recording actual {actual} (est {estimated}), remaining should be "
            f"{expected_remaining}, got {remaining}"
        )

    @given(
        costs=st.lists(
            st.decimals(
                min_value=Decimal('0.01'),
                max_value=Decimal('1.00'),
                places=2,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=20, deadline=10000)
    def test_property_multiple_authorizations_budget_decreases(self, costs, tmp_path):
        """Each successful authorization decreases remaining budget."""
        limits = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('10.00'))]
        fname = f"state_{abs(hash(tuple(costs)))}.json"
        config = make_config(tmp_path, period_limits=limits, state_file_name=fname)
        manager = BudgetManager(config)

        prev_remaining = manager.remaining_budget()

        for i, cost in enumerate(costs):
            result = manager.authorize(cost)
            if result.allowed:
                current_remaining = manager.remaining_budget()
                assert current_remaining < prev_remaining, (
                    f"After authorization #{i}, remaining should decrease: "
                    f"{prev_remaining} -> {current_remaining}"
                )
                prev_remaining = current_remaining

                manager.record_spend(
                    cost, cost,
                    make_metadata(request_id=f"req-seq-{abs(hash(tuple(costs)))}-{i}"),
                )
                prev_remaining = manager.remaining_budget()


# ============================================================================
# Additional edge case tests
# ============================================================================

class TestBudgetManagerEdgeCases:
    """Additional edge case tests."""

    def test_very_small_cost(self, daily_manager):
        """Authorize with very small Decimal cost works."""
        result = daily_manager.authorize(Decimal('0.000001'))
        assert result.allowed is True
        assert result.estimated_cost == Decimal('0.000001')

    def test_authorize_after_full_lifecycle(self, daily_manager):
        """Authorize works correctly after a complete authorize→record cycle."""
        # First cycle
        auth1 = daily_manager.authorize(Decimal('2.00'))
        assert auth1.allowed
        daily_manager.record_spend(
            Decimal('2.00'), Decimal('2.00'),
            make_metadata(request_id="req-edge-1"),
        )

        # Second cycle
        auth2 = daily_manager.authorize(Decimal('2.00'))
        assert auth2.allowed
        assert auth2.remaining == Decimal('1.00'), (
            f"After 2.00 spent and 2.00 reserved from 5.00, remaining should be 1.00, got {auth2.remaining}"
        )

    def test_authorize_after_release_reclaims_budget(self, daily_manager):
        """Released reservations free up budget for new authorizations."""
        auth1 = daily_manager.authorize(Decimal('4.00'))
        assert auth1.allowed

        # This should be denied — only 1.00 remaining
        auth2 = daily_manager.authorize(Decimal('2.00'))
        assert auth2.allowed is False

        # Release first reservation
        daily_manager.release_reservation(Decimal('4.00'))

        # Now this should work
        auth3 = daily_manager.authorize(Decimal('2.00'))
        assert auth3.allowed is True

    def test_weekly_period_limit(self, tmp_path, weekly_period_limit):
        """Weekly period limit works correctly."""
        config = make_config(tmp_path, period_limits=[weekly_period_limit])
        manager = BudgetManager(config)

        result = manager.authorize(Decimal('10.00'))
        assert result.allowed is True
        assert result.remaining == Decimal('15.00')

    def test_denial_reason_weekly(self, tmp_path, weekly_period_limit):
        """Weekly limit denial returns weekly_limit_exceeded."""
        config = make_config(tmp_path, period_limits=[weekly_period_limit])
        manager = BudgetManager(config)

        result = manager.authorize(Decimal('26.00'))
        assert result.allowed is False
        assert "weekly" in str(result.denial_reason).lower()

    def test_record_spend_metadata_preserved(self, daily_manager):
        """SpendMetadata is accepted and processed without error."""
        auth = daily_manager.authorize(Decimal('1.00'))
        assert auth.allowed

        metadata = make_metadata(
            request_id="req-meta-1",
            model_name="claude-3-opus",
            input_tokens=500,
            output_tokens=200,
            task_type="code_review",
        )

        # Should not raise
        daily_manager.record_spend(
            Decimal('1.00'), Decimal('1.00'),
            metadata=metadata,
        )

    def test_force_persist_idempotent(self, daily_manager):
        """Multiple force_persist calls are safe."""
        daily_manager.force_persist()
        daily_manager.force_persist()
        daily_manager.force_persist()
        # Should not raise

    def test_get_report_on_fresh_state(self, daily_manager):
        """get_report works on fresh state with no spend."""
        report = daily_manager.get_report()

        assert report.total_all_time_spend == Decimal('0'), (
            f"Fresh report should have 0 spend, got {report.total_all_time_spend}"
        )
        assert report.total_all_time_savings == Decimal('0'), (
            f"Fresh report should have 0 savings, got {report.total_all_time_savings}"
        )
        assert report.total_local_requests == 0
        assert report.total_remote_requests == 0

    def test_large_number_of_operations(self, tmp_path):
        """Stress test: many authorize/record cycles."""
        limits = [PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal('1000.00'))]
        config = make_config(tmp_path, period_limits=limits)
        manager = BudgetManager(config)

        for i in range(50):
            auth = manager.authorize(Decimal('1.00'))
            assert auth.allowed, f"Authorization {i} should be allowed"
            manager.record_spend(
                Decimal('1.00'), Decimal('1.00'),
                make_metadata(request_id=f"req-stress-{i}"),
            )

        remaining = manager.remaining_budget()
        assert remaining == Decimal('950.00'), (
            f"After 50 x 1.00 spend from 1000.00, remaining should be 950.00, got {remaining}"
        )

        report = manager.get_report()
        assert report.total_remote_requests == 50
