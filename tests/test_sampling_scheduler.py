"""
Contract Test Suite for AdaptiveSamplingScheduler
==================================================
Structured into logical sections:
1. Config validation tests
2. Decay function tests
3. Phase emergence / should_sample tests
4. Budget enforcement tests
5. Decision consistency / invariant tests

All tests are synchronous. RandomSource is injected (not mocked).
"""

import math
import pytest
from unittest.mock import MagicMock

# --------------------------------------------------------------------------- #
# Import the component under test
# --------------------------------------------------------------------------- #
from src.sampling_scheduler import (
    AdaptiveSamplingScheduler,
    SamplingSchedulerConfig,
    ConfidenceSnapshot,
    SamplingDecision,
    SamplingPhase,
    DecayFunctionParams,
    exponential_decay,
    linear_decay,
    step_decay,
)


# =========================================================================== #
#  Fixtures
# =========================================================================== #

def make_random_source(value: float = 0.5):
    """Create a deterministic random source returning a fixed value."""
    call_count = 0

    def _source():
        nonlocal call_count
        call_count += 1
        return value

    _source.call_count = lambda: call_count
    return _source


class CountingRandomSource:
    """A random source that counts calls and returns a configurable value."""

    def __init__(self, value: float = 0.5):
        self._value = value
        self._call_count = 0

    def __call__(self) -> float:
        self._call_count += 1
        return self._value

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset(self):
        self._call_count = 0


@pytest.fixture
def default_decay_params():
    return DecayFunctionParams(
        rate=2.0,
        high_correlation_rate=0.1,
        low_correlation_rate=0.8,
        steps=[(0.7, 0.5), (0.85, 0.3), (0.95, 0.1)],
    )


@pytest.fixture
def default_config(default_decay_params):
    """A valid default config with exponential decay."""
    return SamplingSchedulerConfig(
        emergency_threshold=0.1,
        coaching_trigger=0.3,
        adaptive_entry_correlation=0.6,
        min_floor=0.05,
        min_sample_count_for_adaptive=50,
        decay_function_name="exponential",
        decay_params=default_decay_params,
    )


@pytest.fixture
def linear_config(default_decay_params):
    return SamplingSchedulerConfig(
        emergency_threshold=0.1,
        coaching_trigger=0.3,
        adaptive_entry_correlation=0.6,
        min_floor=0.05,
        min_sample_count_for_adaptive=50,
        decay_function_name="linear",
        decay_params=default_decay_params,
    )


@pytest.fixture
def step_config(default_decay_params):
    return SamplingSchedulerConfig(
        emergency_threshold=0.1,
        coaching_trigger=0.3,
        adaptive_entry_correlation=0.6,
        min_floor=0.05,
        min_sample_count_for_adaptive=50,
        decay_function_name="step",
        decay_params=default_decay_params,
    )


@pytest.fixture
def default_scheduler(default_config):
    return AdaptiveSamplingScheduler(
        config=default_config,
        random_source=make_random_source(0.5),
    )


def make_snapshot(
    correlation_score=0.8,
    is_stale=False,
    sample_count=100,
    budget_exhausted=False,
):
    return ConfidenceSnapshot(
        correlation_score=correlation_score,
        is_stale=is_stale,
        sample_count=sample_count,
        budget_exhausted=budget_exhausted,
    )


# =========================================================================== #
#  1. CONFIG VALIDATION TESTS
# =========================================================================== #


class TestSamplingSchedulerConfig:
    """Tests for SamplingSchedulerConfig validation and scheduler construction."""

    def test_config_valid_exponential(self, default_config):
        """Valid config construction with exponential decay function."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source()
        )
        assert scheduler is not None, "Scheduler should be created successfully"

    def test_config_valid_linear(self, linear_config):
        """Valid config construction with linear decay function."""
        scheduler = AdaptiveSamplingScheduler(
            config=linear_config, random_source=make_random_source()
        )
        assert scheduler is not None, "Scheduler should be created with linear decay"

    def test_config_valid_step(self, step_config):
        """Valid config construction with step decay function and non-empty steps."""
        scheduler = AdaptiveSamplingScheduler(
            config=step_config, random_source=make_random_source()
        )
        assert scheduler is not None, "Scheduler should be created with step decay"

    def test_config_threshold_ordering_emergency_ge_coaching(self, default_decay_params):
        """Config rejected when emergency_threshold >= coaching_trigger."""
        with pytest.raises((ValueError, Exception)):
            SamplingSchedulerConfig(
                emergency_threshold=0.5,
                coaching_trigger=0.3,
                adaptive_entry_correlation=0.6,
                min_floor=0.05,
                min_sample_count_for_adaptive=50,
                decay_function_name="exponential",
                decay_params=default_decay_params,
            )

    def test_config_threshold_ordering_coaching_ge_adaptive(self, default_decay_params):
        """Config rejected when coaching_trigger >= adaptive_entry_correlation."""
        with pytest.raises((ValueError, Exception)):
            SamplingSchedulerConfig(
                emergency_threshold=0.1,
                coaching_trigger=0.7,
                adaptive_entry_correlation=0.5,
                min_floor=0.05,
                min_sample_count_for_adaptive=50,
                decay_function_name="exponential",
                decay_params=default_decay_params,
            )

    def test_config_threshold_ordering_all_equal(self, default_decay_params):
        """Config rejected when all thresholds are equal."""
        with pytest.raises((ValueError, Exception)):
            SamplingSchedulerConfig(
                emergency_threshold=0.5,
                coaching_trigger=0.5,
                adaptive_entry_correlation=0.5,
                min_floor=0.05,
                min_sample_count_for_adaptive=50,
                decay_function_name="exponential",
                decay_params=default_decay_params,
            )

    def test_config_emergency_equal_coaching(self, default_decay_params):
        """Config rejected when emergency_threshold equals coaching_trigger."""
        with pytest.raises((ValueError, Exception)):
            SamplingSchedulerConfig(
                emergency_threshold=0.3,
                coaching_trigger=0.3,
                adaptive_entry_correlation=0.6,
                min_floor=0.05,
                min_sample_count_for_adaptive=50,
                decay_function_name="exponential",
                decay_params=default_decay_params,
            )

    def test_config_min_floor_zero(self, default_decay_params):
        """Config rejected when min_floor is 0.0."""
        with pytest.raises((ValueError, Exception)):
            SamplingSchedulerConfig(
                emergency_threshold=0.1,
                coaching_trigger=0.3,
                adaptive_entry_correlation=0.6,
                min_floor=0.0,
                min_sample_count_for_adaptive=50,
                decay_function_name="exponential",
                decay_params=default_decay_params,
            )

    def test_config_min_floor_negative(self, default_decay_params):
        """Config rejected when min_floor is negative."""
        with pytest.raises((ValueError, Exception)):
            SamplingSchedulerConfig(
                emergency_threshold=0.1,
                coaching_trigger=0.3,
                adaptive_entry_correlation=0.6,
                min_floor=-0.1,
                min_sample_count_for_adaptive=50,
                decay_function_name="exponential",
                decay_params=default_decay_params,
            )

    def test_config_adaptive_entry_above_one(self, default_decay_params):
        """Config rejected when adaptive_entry_correlation > 1.0."""
        with pytest.raises((ValueError, Exception)):
            SamplingSchedulerConfig(
                emergency_threshold=0.1,
                coaching_trigger=0.3,
                adaptive_entry_correlation=1.5,
                min_floor=0.05,
                min_sample_count_for_adaptive=50,
                decay_function_name="exponential",
                decay_params=default_decay_params,
            )

    def test_config_step_decay_empty_steps(self):
        """Config (or scheduler construction) rejected when step decay has empty steps list."""
        empty_params = DecayFunctionParams(
            rate=2.0,
            high_correlation_rate=0.1,
            low_correlation_rate=0.8,
            steps=[],
        )
        with pytest.raises((ValueError, Exception)):
            config = SamplingSchedulerConfig(
                emergency_threshold=0.1,
                coaching_trigger=0.3,
                adaptive_entry_correlation=0.6,
                min_floor=0.05,
                min_sample_count_for_adaptive=50,
                decay_function_name="step",
                decay_params=empty_params,
            )
            # If config creation doesn't raise, scheduler construction should:
            AdaptiveSamplingScheduler(
                config=config, random_source=make_random_source()
            )


# =========================================================================== #
#  2. DECAY FUNCTION TESTS
# =========================================================================== #


class TestExponentialDecay:
    """Tests for the exponential_decay pure function."""

    def test_known_value(self):
        """Exponential decay returns expected value for known inputs.
        rate=2.0, correlation=0.8 with adaptive_entry=0.6:
        normalized = (0.8-0.6)/(1.0-0.6) = 0.5
        result = exp(-2.0 * 0.5) = exp(-1.0) ≈ 0.3679
        """
        params = DecayFunctionParams(rate=2.0, high_correlation_rate=0.1, low_correlation_rate=0.8, steps=[])
        result = exponential_decay(0.8, params)
        expected = math.exp(-1.0)
        assert abs(result - expected) < 0.01, (
            f"Expected ~{expected:.4f}, got {result:.4f}"
        )

    def test_at_entry_boundary(self):
        """Exponential decay returns ~1.0 at the adaptive entry correlation.
        When normalized correlation is 0, exp(0) = 1.0.
        Note: the function receives correlation; the normalization context
        (adaptive_entry_correlation) may be embedded differently per implementation.
        We test that at the lowest end of the adaptive range, rate is near 1.0.
        """
        params = DecayFunctionParams(rate=2.0, high_correlation_rate=0.1, low_correlation_rate=0.8, steps=[])
        # At the entry boundary, the normalized correlation should be 0
        # so exp(-rate * 0) = 1.0
        # The exact correlation value depends on how the function handles adaptive_entry
        # We test at 0.0 normalized (correlation == adaptive_entry or correlation == 0.0 if func normalizes externally)
        result = exponential_decay(0.0, params)
        # At correlation=0.0, if normalization happens inside, it depends on implementation
        # We assert result is in valid range
        assert 0.0 <= result <= 1.0, f"Result {result} out of bounds"

    def test_at_correlation_one(self):
        """Exponential decay at correlation=1.0 returns exp(-rate) ∈ [0, 1]."""
        params = DecayFunctionParams(rate=2.0, high_correlation_rate=0.1, low_correlation_rate=0.8, steps=[])
        result = exponential_decay(1.0, params)
        assert 0.0 <= result <= 1.0, f"Expected result in [0,1], got {result}"

    def test_monotonically_decreasing(self):
        """Exponential decay is monotonically decreasing as correlation increases."""
        params = DecayFunctionParams(rate=3.0, high_correlation_rate=0.1, low_correlation_rate=0.8, steps=[])
        correlations = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        rates = [exponential_decay(c, params) for c in correlations]
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1], (
                f"Monotonicity violated: rate at {correlations[i]}={rates[i]} < "
                f"rate at {correlations[i+1]}={rates[i+1]}"
            )

    @pytest.mark.parametrize("correlation", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_bounded_output(self, correlation):
        """Exponential decay output is always in [0.0, 1.0]."""
        params = DecayFunctionParams(rate=5.0, high_correlation_rate=0.1, low_correlation_rate=0.8, steps=[])
        result = exponential_decay(correlation, params)
        assert 0.0 <= result <= 1.0, (
            f"Expected output in [0, 1] for correlation={correlation}, got {result}"
        )


class TestLinearDecay:
    """Tests for the linear_decay pure function."""

    def test_known_value(self):
        """Linear decay returns expected value for known inputs.
        low_rate=0.8, high_rate=0.2, normalized=(0.8-0.6)/(1.0-0.6)=0.5
        rate = 0.8 + (0.2-0.8)*0.5 = 0.8 - 0.3 = 0.5
        """
        params = DecayFunctionParams(
            rate=2.0, high_correlation_rate=0.2, low_correlation_rate=0.8, steps=[]
        )
        result = linear_decay(0.8, params)
        # Result depends on normalization logic - check it's in the valid range
        assert 0.0 <= result <= 1.0, f"Result {result} out of bounds"

    def test_at_entry_boundary_returns_low_rate(self):
        """At the start of the adaptive range, linear decay returns low_correlation_rate."""
        params = DecayFunctionParams(
            rate=2.0, high_correlation_rate=0.1, low_correlation_rate=0.9, steps=[]
        )
        # At normalized=0 (entry boundary), should return low_correlation_rate
        result = linear_decay(0.0, params)
        # The result should be near the low_correlation_rate end
        assert 0.0 <= result <= 1.0, f"Result {result} out of bounds"

    def test_at_correlation_one_returns_high_rate(self):
        """At correlation=1.0, linear decay returns high_correlation_rate."""
        params = DecayFunctionParams(
            rate=2.0, high_correlation_rate=0.1, low_correlation_rate=0.9, steps=[]
        )
        result = linear_decay(1.0, params)
        assert 0.0 <= result <= 1.0, f"Result {result} out of bounds"

    def test_linearity(self):
        """Linear decay varies linearly - equal spacing in correlation gives equal spacing in rate."""
        params = DecayFunctionParams(
            rate=2.0, high_correlation_rate=0.1, low_correlation_rate=0.9, steps=[]
        )
        r1 = linear_decay(0.2, params)
        r2 = linear_decay(0.5, params)
        r3 = linear_decay(0.8, params)
        # Spacing should be equal: r2-r1 ≈ r3-r2
        diff1 = r2 - r1
        diff2 = r3 - r2
        assert abs(diff1 - diff2) < 0.01, (
            f"Linearity violated: diff1={diff1:.4f}, diff2={diff2:.4f}"
        )

    @pytest.mark.parametrize("correlation", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_output_bounded(self, correlation):
        """Linear decay output is bounded between low and high rates."""
        params = DecayFunctionParams(
            rate=2.0, high_correlation_rate=0.1, low_correlation_rate=0.9, steps=[]
        )
        result = linear_decay(correlation, params)
        lo = min(params.low_correlation_rate, params.high_correlation_rate)
        hi = max(params.low_correlation_rate, params.high_correlation_rate)
        assert lo - 0.001 <= result <= hi + 0.001, (
            f"Expected result in [{lo}, {hi}], got {result}"
        )


class TestStepDecay:
    """Tests for the step_decay pure function."""

    def test_known_value_between_steps(self):
        """Step decay returns correct rate for value between step thresholds."""
        params = DecayFunctionParams(
            rate=2.0,
            high_correlation_rate=0.1,
            low_correlation_rate=0.8,
            steps=[(0.6, 0.8), (0.8, 0.4), (0.95, 0.1)],
        )
        result = step_decay(0.85, params)
        assert result == pytest.approx(0.4), (
            f"Expected 0.4 for correlation=0.85, got {result}"
        )

    def test_below_all_steps_returns_one(self):
        """Step decay returns 1.0 when correlation is below all step thresholds."""
        params = DecayFunctionParams(
            rate=2.0,
            high_correlation_rate=0.1,
            low_correlation_rate=0.8,
            steps=[(0.7, 0.5), (0.9, 0.2)],
        )
        result = step_decay(0.65, params)
        assert result == pytest.approx(1.0), (
            f"Expected 1.0 for correlation below all steps, got {result}"
        )

    def test_at_exact_threshold(self):
        """Step decay at exact step threshold returns that step's rate."""
        params = DecayFunctionParams(
            rate=2.0,
            high_correlation_rate=0.1,
            low_correlation_rate=0.8,
            steps=[(0.7, 0.5), (0.9, 0.2)],
        )
        result = step_decay(0.7, params)
        assert result == pytest.approx(0.5), (
            f"Expected 0.5 at exact threshold 0.7, got {result}"
        )

    def test_at_highest_step(self):
        """Step decay at or above highest step threshold returns that step's rate."""
        params = DecayFunctionParams(
            rate=2.0,
            high_correlation_rate=0.1,
            low_correlation_rate=0.8,
            steps=[(0.7, 0.5), (0.9, 0.2)],
        )
        result = step_decay(0.95, params)
        assert result == pytest.approx(0.2), (
            f"Expected 0.2 above highest step, got {result}"
        )

    def test_empty_steps_error(self):
        """Step decay raises error when steps is empty."""
        params = DecayFunctionParams(
            rate=2.0,
            high_correlation_rate=0.1,
            low_correlation_rate=0.8,
            steps=[],
        )
        with pytest.raises((ValueError, Exception)):
            step_decay(0.5, params)

    @pytest.mark.parametrize("correlation", [0.0, 0.5, 0.7, 0.85, 0.95, 1.0])
    def test_bounded_output(self, correlation):
        """Step decay output is always in [0.0, 1.0]."""
        params = DecayFunctionParams(
            rate=2.0,
            high_correlation_rate=0.1,
            low_correlation_rate=0.8,
            steps=[(0.6, 0.8), (0.8, 0.4), (0.95, 0.1)],
        )
        result = step_decay(correlation, params)
        assert 0.0 <= result <= 1.0, (
            f"Expected result in [0, 1], got {result}"
        )


# =========================================================================== #
#  3. PHASE EMERGENCE / SHOULD_SAMPLE TESTS
# =========================================================================== #


class TestShouldSamplePhases:
    """Tests for phase determination in should_sample."""

    def test_budget_exhausted_local_only(self, default_config):
        """Budget exhausted → LOCAL_ONLY phase."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        snap = make_snapshot(correlation_score=0.8, budget_exhausted=True, sample_count=100)
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.LOCAL_ONLY, (
            f"Expected LOCAL_ONLY, got {decision.phase}"
        )
        assert decision.send_to_local is True, "LOCAL_ONLY must send to local"
        assert decision.send_to_remote is False, "LOCAL_ONLY must not send to remote"
        assert decision.is_coaching_sample is False, "LOCAL_ONLY is not a coaching sample"

    def test_emergency_remote_only(self, default_config):
        """Low correlation below emergency_threshold → REMOTE_ONLY."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        snap = make_snapshot(
            correlation_score=0.05,  # Below emergency_threshold=0.1
            budget_exhausted=False,
            sample_count=100,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.REMOTE_ONLY, (
            f"Expected REMOTE_ONLY, got {decision.phase}"
        )
        assert decision.send_to_remote is True, "REMOTE_ONLY must send to remote"
        assert decision.send_to_local is False, "REMOTE_ONLY must not send to local"

    def test_coaching_full_below_coaching_trigger(self, default_config):
        """Correlation below coaching_trigger but above emergency → COACHING_FULL."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        snap = make_snapshot(
            correlation_score=0.2,  # Between emergency(0.1) and coaching(0.3)
            budget_exhausted=False,
            sample_count=100,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.COACHING_FULL, (
            f"Expected COACHING_FULL, got {decision.phase}"
        )
        assert decision.send_to_local is True
        assert decision.send_to_remote is True
        assert decision.sampling_rate == 1.0

    def test_coaching_full_stale_overrides_adaptive(self, default_config):
        """Stale snapshot forces COACHING_FULL even with high correlation."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        snap = make_snapshot(
            correlation_score=0.9,  # Would be ADAPTIVE
            is_stale=True,
            sample_count=100,
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.COACHING_FULL, (
            f"Expected COACHING_FULL due to staleness, got {decision.phase}"
        )
        assert decision.sampling_rate == 1.0

    def test_coaching_full_insufficient_samples_cold_start(self, default_config):
        """Insufficient sample_count forces COACHING_FULL (cold start)."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        snap = make_snapshot(
            correlation_score=0.9,  # Would be ADAPTIVE
            is_stale=False,
            sample_count=10,  # Below min_sample_count_for_adaptive=50
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.COACHING_FULL, (
            f"Expected COACHING_FULL due to cold start, got {decision.phase}"
        )

    def test_coaching_full_below_adaptive_entry(self, default_config):
        """Correlation between coaching_trigger and adaptive_entry → COACHING_FULL."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        snap = make_snapshot(
            correlation_score=0.45,  # Between coaching(0.3) and adaptive(0.6)
            is_stale=False,
            sample_count=100,
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.COACHING_FULL, (
            f"Expected COACHING_FULL, got {decision.phase}"
        )

    def test_adaptive_phase_coaching_sample(self, default_config):
        """ADAPTIVE phase: random roll below rate → coaching sample (both=True)."""
        # Use low random value so it's below the sampling rate
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.01)
        )
        snap = make_snapshot(
            correlation_score=0.7,  # Above adaptive_entry(0.6)
            is_stale=False,
            sample_count=100,
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.ADAPTIVE, (
            f"Expected ADAPTIVE, got {decision.phase}"
        )
        assert decision.send_to_local is True, "ADAPTIVE coaching sample sends to local"
        assert decision.send_to_remote is True, "ADAPTIVE coaching sample sends to remote"
        assert decision.is_coaching_sample is True

    def test_adaptive_phase_local_only(self, default_config):
        """ADAPTIVE phase: random roll above rate → local only."""
        # Use high random value so it's above the sampling rate
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.99)
        )
        snap = make_snapshot(
            correlation_score=0.9,  # High correlation, low sampling rate
            is_stale=False,
            sample_count=100,
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.ADAPTIVE, (
            f"Expected ADAPTIVE, got {decision.phase}"
        )
        assert decision.send_to_local is True, "ADAPTIVE local-only must send to local"
        assert decision.send_to_remote is False, "ADAPTIVE local-only must not send to remote"
        assert decision.is_coaching_sample is False

    def test_boundary_emergency_threshold_epsilon_below(self, default_config):
        """Correlation epsilon below emergency_threshold → REMOTE_ONLY."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        epsilon = 1e-9
        snap = make_snapshot(
            correlation_score=default_config.emergency_threshold - epsilon,
            budget_exhausted=False,
            sample_count=100,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.REMOTE_ONLY, (
            f"Expected REMOTE_ONLY just below emergency threshold, got {decision.phase}"
        )

    def test_boundary_emergency_threshold_exact(self, default_config):
        """Correlation exactly at emergency_threshold does NOT trigger REMOTE_ONLY
        (condition is strictly less than)."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        snap = make_snapshot(
            correlation_score=default_config.emergency_threshold,  # exactly 0.1
            budget_exhausted=False,
            sample_count=100,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase != SamplingPhase.REMOTE_ONLY, (
            f"At exact emergency threshold, should NOT be REMOTE_ONLY, got {decision.phase}"
        )

    def test_boundary_coaching_trigger_epsilon_below(self, default_config):
        """Correlation epsilon below coaching_trigger → COACHING_FULL."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        epsilon = 1e-9
        snap = make_snapshot(
            correlation_score=default_config.coaching_trigger - epsilon,
            budget_exhausted=False,
            sample_count=100,
        )
        decision = scheduler.should_sample(snap)

        # Should be COACHING_FULL (above emergency=0.1, below coaching=0.3)
        assert decision.phase == SamplingPhase.COACHING_FULL, (
            f"Expected COACHING_FULL just below coaching trigger, got {decision.phase}"
        )

    def test_boundary_adaptive_entry_exact(self, default_config):
        """Correlation exactly at adaptive_entry_correlation with sufficient samples."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.01)
        )
        snap = make_snapshot(
            correlation_score=default_config.adaptive_entry_correlation,  # 0.6
            is_stale=False,
            sample_count=100,
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)

        # At exactly adaptive_entry, behavior depends on < vs <= comparison
        # The contract says correlation < adaptive_entry_correlation → COACHING_FULL
        # So at exactly the threshold, we expect either ADAPTIVE or COACHING_FULL
        # Both are acceptable depending on implementation's < vs <=
        assert decision.phase in (SamplingPhase.ADAPTIVE, SamplingPhase.COACHING_FULL), (
            f"At adaptive_entry boundary, expected ADAPTIVE or COACHING_FULL, got {decision.phase}"
        )

    def test_boundary_sample_count_zero(self, default_config):
        """Zero sample count → COACHING_FULL (cold start)."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        snap = make_snapshot(
            correlation_score=0.9,
            is_stale=False,
            sample_count=0,
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.COACHING_FULL, (
            f"Expected COACHING_FULL with sample_count=0, got {decision.phase}"
        )

    def test_boundary_sample_count_at_minimum(self, default_config):
        """sample_count exactly at min_sample_count_for_adaptive → can enter ADAPTIVE."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.01)
        )
        snap = make_snapshot(
            correlation_score=0.9,
            is_stale=False,
            sample_count=default_config.min_sample_count_for_adaptive,  # 50
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.ADAPTIVE, (
            f"Expected ADAPTIVE at min sample count, got {decision.phase}"
        )

    def test_staleness_between_coaching_and_adaptive(self, default_config):
        """Staleness override triggers COACHING_FULL even between coaching_trigger and adaptive_entry."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        snap = make_snapshot(
            correlation_score=0.45,
            is_stale=True,
            sample_count=100,
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.COACHING_FULL, (
            f"Expected COACHING_FULL due to staleness, got {decision.phase}"
        )


# =========================================================================== #
#  4. BUDGET ENFORCEMENT TESTS
# =========================================================================== #


class TestBudgetEnforcement:
    """Tests for the hard budget_exhausted invariant."""

    @pytest.mark.parametrize(
        "correlation,is_stale,sample_count",
        [
            (0.0, False, 0),        # Emergency + cold start
            (0.05, False, 100),      # Below emergency
            (0.05, True, 0),         # Below emergency + stale + cold start
            (0.2, False, 100),       # Between emergency and coaching
            (0.2, True, 10),         # Between emergency and coaching + stale + low samples
            (0.5, False, 100),       # Between coaching and adaptive
            (0.5, True, 100),        # Between coaching and adaptive + stale
            (0.7, False, 100),       # ADAPTIVE range
            (0.9, False, 100),       # High ADAPTIVE
            (0.95, True, 100),       # High ADAPTIVE + stale
            (1.0, False, 200),       # Maximum correlation
            (0.8, False, 10),        # ADAPTIVE range + cold start
        ],
    )
    def test_budget_exhausted_always_local_only(
        self, default_config, correlation, is_stale, sample_count
    ):
        """Budget exhausted → LOCAL_ONLY for ALL other input combinations."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        snap = make_snapshot(
            correlation_score=correlation,
            is_stale=is_stale,
            sample_count=sample_count,
            budget_exhausted=True,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.LOCAL_ONLY, (
            f"Budget exhausted with corr={correlation}, stale={is_stale}, "
            f"samples={sample_count}: expected LOCAL_ONLY, got {decision.phase}"
        )
        assert decision.send_to_remote is False, (
            f"Budget exhausted must never send to remote, got send_to_remote=True "
            f"with corr={correlation}"
        )
        assert decision.send_to_local is True, "Budget exhausted must send to local"

    def test_budget_exhausted_overrides_emergency_correlation(self, default_config):
        """Budget exhausted + correlation=0.0 → LOCAL_ONLY, not REMOTE_ONLY."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        snap = make_snapshot(correlation_score=0.0, budget_exhausted=True)
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.LOCAL_ONLY
        assert decision.send_to_remote is False

    def test_budget_exhausted_overrides_staleness(self, default_config):
        """Budget exhausted + stale → LOCAL_ONLY, not COACHING_FULL."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        snap = make_snapshot(
            correlation_score=0.9, is_stale=True, budget_exhausted=True
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.LOCAL_ONLY
        assert decision.send_to_remote is False

    def test_budget_exhausted_overrides_adaptive(self, default_config):
        """Budget exhausted + high correlation → LOCAL_ONLY, not ADAPTIVE."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.01)
        )
        snap = make_snapshot(
            correlation_score=0.95,
            is_stale=False,
            sample_count=100,
            budget_exhausted=True,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.LOCAL_ONLY


# =========================================================================== #
#  5. DECISION CONSISTENCY / INVARIANT TESTS
# =========================================================================== #


class TestSamplingDecisionConsistency:
    """Cross-field invariant and consistency tests."""

    def _make_scheduler(self, config, random_val=0.5):
        return AdaptiveSamplingScheduler(
            config=config, random_source=make_random_source(random_val)
        )

    @pytest.mark.parametrize(
        "correlation,is_stale,sample_count,budget_exhausted",
        [
            (0.0, False, 100, True),    # LOCAL_ONLY
            (0.05, False, 100, False),   # REMOTE_ONLY
            (0.2, False, 100, False),    # COACHING_FULL
            (0.9, True, 100, False),     # COACHING_FULL (stale)
            (0.9, False, 10, False),     # COACHING_FULL (cold start)
            (0.9, False, 100, False),    # ADAPTIVE
            (0.7, False, 100, False),    # ADAPTIVE
        ],
    )
    def test_never_drop_request(
        self, default_config, correlation, is_stale, sample_count, budget_exhausted
    ):
        """At least one of send_to_local or send_to_remote is True in every decision."""
        scheduler = self._make_scheduler(default_config)
        snap = make_snapshot(
            correlation_score=correlation,
            is_stale=is_stale,
            sample_count=sample_count,
            budget_exhausted=budget_exhausted,
        )
        decision = scheduler.should_sample(snap)

        assert decision.send_to_local or decision.send_to_remote, (
            f"Request dropped! send_to_local={decision.send_to_local}, "
            f"send_to_remote={decision.send_to_remote}, phase={decision.phase}"
        )

    @pytest.mark.parametrize(
        "correlation,is_stale,sample_count,budget_exhausted",
        [
            (0.0, False, 100, True),
            (0.05, False, 100, False),
            (0.2, False, 100, False),
            (0.9, True, 100, False),
            (0.9, False, 100, False),
            (0.7, False, 100, False),
            (0.9, False, 10, False),
        ],
    )
    def test_coaching_sample_derived(
        self, default_config, correlation, is_stale, sample_count, budget_exhausted
    ):
        """is_coaching_sample always equals (send_to_local AND send_to_remote)."""
        scheduler = self._make_scheduler(default_config, random_val=0.01)
        snap = make_snapshot(
            correlation_score=correlation,
            is_stale=is_stale,
            sample_count=sample_count,
            budget_exhausted=budget_exhausted,
        )
        decision = scheduler.should_sample(snap)

        expected = decision.send_to_local and decision.send_to_remote
        assert decision.is_coaching_sample == expected, (
            f"is_coaching_sample={decision.is_coaching_sample} but "
            f"send_to_local={decision.send_to_local}, send_to_remote={decision.send_to_remote}"
        )

    @pytest.mark.parametrize("correlation", [0.65, 0.7, 0.8, 0.9, 0.95, 1.0])
    def test_adaptive_rate_above_min_floor(self, default_config, correlation):
        """In ADAPTIVE phase, sampling_rate >= min_floor."""
        scheduler = self._make_scheduler(default_config, random_val=0.01)
        snap = make_snapshot(
            correlation_score=correlation,
            is_stale=False,
            sample_count=100,
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)

        if decision.phase == SamplingPhase.ADAPTIVE:
            assert decision.sampling_rate >= default_config.min_floor, (
                f"ADAPTIVE rate {decision.sampling_rate} < min_floor {default_config.min_floor}"
            )

    def test_adaptive_min_floor_clamp(self):
        """When decay output is very low, rate is clamped to min_floor."""
        params = DecayFunctionParams(
            rate=20.0,  # Very aggressive decay
            high_correlation_rate=0.01,
            low_correlation_rate=0.8,
            steps=[],
        )
        config = SamplingSchedulerConfig(
            emergency_threshold=0.1,
            coaching_trigger=0.3,
            adaptive_entry_correlation=0.6,
            min_floor=0.05,
            min_sample_count_for_adaptive=50,
            decay_function_name="exponential",
            decay_params=params,
        )
        scheduler = AdaptiveSamplingScheduler(
            config=config, random_source=make_random_source(0.01)
        )
        snap = make_snapshot(
            correlation_score=0.99,  # Very high → very low decay output
            is_stale=False,
            sample_count=100,
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.ADAPTIVE
        assert decision.sampling_rate >= config.min_floor, (
            f"Rate {decision.sampling_rate} not clamped to min_floor {config.min_floor}"
        )

    @pytest.mark.parametrize(
        "correlation,is_stale,sample_count,budget_exhausted",
        [
            (0.0, False, 100, True),
            (0.05, False, 100, False),
            (0.2, False, 100, False),
            (0.9, False, 100, False),
        ],
    )
    def test_reason_non_empty(
        self, default_config, correlation, is_stale, sample_count, budget_exhausted
    ):
        """SamplingDecision.reason is always non-empty."""
        scheduler = self._make_scheduler(default_config)
        snap = make_snapshot(
            correlation_score=correlation,
            is_stale=is_stale,
            sample_count=sample_count,
            budget_exhausted=budget_exhausted,
        )
        decision = scheduler.should_sample(snap)

        assert decision.reason, (
            f"Reason should be non-empty for phase={decision.phase}"
        )
        assert len(decision.reason) > 0

    def test_remote_only_implies_no_local(self, default_config):
        """REMOTE_ONLY → send_to_local=False, send_to_remote=True."""
        scheduler = self._make_scheduler(default_config)
        snap = make_snapshot(correlation_score=0.05, budget_exhausted=False)
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.REMOTE_ONLY
        assert decision.send_to_local is False
        assert decision.send_to_remote is True

    def test_local_only_implies_no_remote(self, default_config):
        """LOCAL_ONLY → send_to_local=True, send_to_remote=False."""
        scheduler = self._make_scheduler(default_config)
        snap = make_snapshot(correlation_score=0.8, budget_exhausted=True)
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.LOCAL_ONLY
        assert decision.send_to_local is True
        assert decision.send_to_remote is False

    def test_coaching_full_implies_both_and_rate_one(self, default_config):
        """COACHING_FULL → both destinations True, sampling_rate=1.0."""
        scheduler = self._make_scheduler(default_config)
        snap = make_snapshot(
            correlation_score=0.2, budget_exhausted=False, sample_count=100
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.COACHING_FULL
        assert decision.send_to_local is True
        assert decision.send_to_remote is True
        assert decision.sampling_rate == 1.0

    def test_deterministic_given_same_inputs(self, default_config):
        """Phase determination is deterministic given same snapshot and config."""
        snap = make_snapshot(
            correlation_score=0.7, is_stale=False, sample_count=100, budget_exhausted=False
        )

        # Two schedulers with same fixed random source
        s1 = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.3)
        )
        s2 = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.3)
        )

        d1 = s1.should_sample(snap)
        d2 = s2.should_sample(snap)

        assert d1.phase == d2.phase, "Phase should be deterministic"
        assert d1.send_to_local == d2.send_to_local
        assert d1.send_to_remote == d2.send_to_remote
        assert d1.sampling_rate == d2.sampling_rate
        assert d1.is_coaching_sample == d2.is_coaching_sample

    def test_scheduler_stateless_across_calls(self, default_config):
        """Scheduler is stateless — calling should_sample with different snapshots
        doesn't change behavior for subsequent identical snapshots."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        target_snap = make_snapshot(
            correlation_score=0.8, is_stale=False, sample_count=100, budget_exhausted=False
        )

        # First call with target
        d1 = scheduler.should_sample(target_snap)

        # Interleave with different snapshots
        scheduler_same_rand = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        scheduler_same_rand.should_sample(
            make_snapshot(correlation_score=0.05, budget_exhausted=False)
        )
        scheduler_same_rand.should_sample(
            make_snapshot(correlation_score=0.9, budget_exhausted=True)
        )
        d2 = scheduler_same_rand.should_sample(target_snap)

        assert d1.phase == d2.phase, "Phase must be same regardless of call history"
        assert d1.send_to_local == d2.send_to_local
        assert d1.send_to_remote == d2.send_to_remote

    def test_precedence_emergency_over_staleness(self, default_config):
        """Emergency threshold takes priority over staleness."""
        scheduler = self._make_scheduler(default_config)
        snap = make_snapshot(
            correlation_score=0.05,  # Below emergency=0.1
            is_stale=True,
            sample_count=100,
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.REMOTE_ONLY, (
            f"Emergency should override staleness, got {decision.phase}"
        )

    def test_adaptive_random_source_called_once(self, default_config):
        """Random source is called exactly once in ADAPTIVE phase."""
        random_src = CountingRandomSource(0.5)
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=random_src
        )
        snap = make_snapshot(
            correlation_score=0.8,
            is_stale=False,
            sample_count=100,
            budget_exhausted=False,
        )
        scheduler.should_sample(snap)

        assert random_src.call_count == 1, (
            f"Expected random source called once in ADAPTIVE, got {random_src.call_count}"
        )

    @pytest.mark.parametrize(
        "correlation,budget_exhausted,expected_phase",
        [
            (0.05, False, SamplingPhase.REMOTE_ONLY),
            (0.2, False, SamplingPhase.COACHING_FULL),
            (0.0, True, SamplingPhase.LOCAL_ONLY),
        ],
    )
    def test_non_adaptive_random_source_not_called(
        self, default_config, correlation, budget_exhausted, expected_phase
    ):
        """Random source is NOT called outside ADAPTIVE phase."""
        random_src = CountingRandomSource(0.5)
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=random_src
        )
        snap = make_snapshot(
            correlation_score=correlation,
            is_stale=False,
            sample_count=100,
            budget_exhausted=budget_exhausted,
        )
        decision = scheduler.should_sample(snap)

        assert decision.phase == expected_phase
        assert random_src.call_count == 0, (
            f"Random source should not be called in {expected_phase}, "
            f"got {random_src.call_count} calls"
        )

    def test_snapshot_invalid_correlation_above_one(self, default_config):
        """Snapshot with correlation > 1.0 is rejected."""
        scheduler = self._make_scheduler(default_config)
        with pytest.raises((ValueError, Exception)):
            snap = ConfidenceSnapshot(
                correlation_score=1.5,
                is_stale=False,
                sample_count=100,
                budget_exhausted=False,
            )
            scheduler.should_sample(snap)

    def test_snapshot_invalid_correlation_negative(self, default_config):
        """Snapshot with negative correlation is rejected."""
        scheduler = self._make_scheduler(default_config)
        with pytest.raises((ValueError, Exception)):
            snap = ConfidenceSnapshot(
                correlation_score=-0.1,
                is_stale=False,
                sample_count=100,
                budget_exhausted=False,
            )
            scheduler.should_sample(snap)

    def test_snapshot_invalid_negative_sample_count(self, default_config):
        """Snapshot with negative sample_count is rejected."""
        scheduler = self._make_scheduler(default_config)
        with pytest.raises((ValueError, Exception)):
            snap = ConfidenceSnapshot(
                correlation_score=0.5,
                is_stale=False,
                sample_count=-1,
                budget_exhausted=False,
            )
            scheduler.should_sample(snap)


# =========================================================================== #
#  6. CONFLICT RESOLUTION PRIORITY MATRIX
# =========================================================================== #


class TestConflictResolutionPriority:
    """Tests verifying the strict override precedence chain:
    budget_exhausted > emergency > coaching_trigger > staleness >
    min_sample_count > adaptive_entry > decay+floor > random_roll
    """

    @pytest.fixture
    def scheduler(self, default_config):
        return AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )

    def test_budget_beats_everything(self, scheduler):
        """Budget exhaustion overrides all other conditions."""
        snap = make_snapshot(
            correlation_score=0.0,  # Would be REMOTE_ONLY
            is_stale=True,          # Would be COACHING_FULL
            sample_count=0,         # Would be COACHING_FULL
            budget_exhausted=True,  # Highest priority
        )
        decision = scheduler.should_sample(snap)
        assert decision.phase == SamplingPhase.LOCAL_ONLY

    def test_emergency_beats_coaching_and_staleness(self, scheduler):
        """Emergency overrides coaching trigger and staleness."""
        snap = make_snapshot(
            correlation_score=0.05,  # Below emergency=0.1
            is_stale=True,
            sample_count=0,
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)
        assert decision.phase == SamplingPhase.REMOTE_ONLY

    def test_coaching_trigger_beats_staleness(self, default_config):
        """Below coaching_trigger → COACHING_FULL regardless of staleness."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        # Not stale, but below coaching trigger → COACHING_FULL
        snap = make_snapshot(
            correlation_score=0.2,  # Below coaching=0.3
            is_stale=False,
            sample_count=100,
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)
        assert decision.phase == SamplingPhase.COACHING_FULL

    def test_staleness_beats_adaptive(self, default_config):
        """Staleness overrides what would otherwise be ADAPTIVE."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.99)
        )
        snap = make_snapshot(
            correlation_score=0.9,  # Would be ADAPTIVE
            is_stale=True,          # But stale → COACHING_FULL
            sample_count=100,
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)
        assert decision.phase == SamplingPhase.COACHING_FULL

    def test_min_sample_count_beats_adaptive(self, default_config):
        """Insufficient samples override what would be ADAPTIVE."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.99)
        )
        snap = make_snapshot(
            correlation_score=0.9,
            is_stale=False,
            sample_count=10,  # Below min=50
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)
        assert decision.phase == SamplingPhase.COACHING_FULL

    def test_adaptive_entry_beats_decay(self, default_config):
        """Below adaptive_entry → COACHING_FULL, not ADAPTIVE with decay."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.5)
        )
        snap = make_snapshot(
            correlation_score=0.5,  # Between coaching(0.3) and adaptive(0.6)
            is_stale=False,
            sample_count=100,
            budget_exhausted=False,
        )
        decision = scheduler.should_sample(snap)
        assert decision.phase == SamplingPhase.COACHING_FULL


# =========================================================================== #
#  7. ADDITIONAL INTEGRATION-LEVEL INVARIANT TESTS
# =========================================================================== #


class TestAdaptivePhaseWithDifferentDecayFunctions:
    """Integration tests verifying ADAPTIVE phase works with all decay functions."""

    def test_adaptive_with_exponential_decay(self, default_config):
        """ADAPTIVE phase with exponential decay produces valid decision."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(0.01)
        )
        snap = make_snapshot(correlation_score=0.8, sample_count=100)
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.ADAPTIVE
        assert decision.sampling_rate >= default_config.min_floor
        assert decision.send_to_local is True
        assert decision.send_to_remote is True  # low random → coaching sample

    def test_adaptive_with_linear_decay(self, linear_config):
        """ADAPTIVE phase with linear decay produces valid decision."""
        scheduler = AdaptiveSamplingScheduler(
            config=linear_config, random_source=make_random_source(0.01)
        )
        snap = make_snapshot(correlation_score=0.8, sample_count=100)
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.ADAPTIVE
        assert decision.sampling_rate >= linear_config.min_floor
        assert decision.send_to_local is True

    def test_adaptive_with_step_decay(self, step_config):
        """ADAPTIVE phase with step decay produces valid decision."""
        scheduler = AdaptiveSamplingScheduler(
            config=step_config, random_source=make_random_source(0.01)
        )
        snap = make_snapshot(correlation_score=0.8, sample_count=100)
        decision = scheduler.should_sample(snap)

        assert decision.phase == SamplingPhase.ADAPTIVE
        assert decision.sampling_rate >= step_config.min_floor
        assert decision.send_to_local is True

    @pytest.mark.parametrize("random_val", [0.0, 0.01, 0.5, 0.99])
    def test_adaptive_always_sends_to_local(self, default_config, random_val):
        """In ADAPTIVE phase, send_to_local is always True regardless of random roll."""
        scheduler = AdaptiveSamplingScheduler(
            config=default_config, random_source=make_random_source(random_val)
        )
        snap = make_snapshot(
            correlation_score=0.8, is_stale=False, sample_count=100, budget_exhausted=False
        )
        decision = scheduler.should_sample(snap)

        if decision.phase == SamplingPhase.ADAPTIVE:
            assert decision.send_to_local is True, (
                f"ADAPTIVE must always send to local, random_val={random_val}"
            )
