"""
Adaptive Sampling Scheduler
============================
Determines sampling decisions for coaching samples based on confidence state.
"""

import math
import random
from enum import Enum
from typing import Callable, Any
from pydantic import BaseModel, Field, field_validator, model_validator


# =========================================================================== #
#  Enums
# =========================================================================== #


class SamplingPhase(str, Enum):
    """Emergent phase label derived from the current ConfidenceSnapshot."""
    REMOTE_ONLY = "REMOTE_ONLY"
    COACHING_FULL = "COACHING_FULL"
    ADAPTIVE = "ADAPTIVE"
    LOCAL_ONLY = "LOCAL_ONLY"


class DecayFunctionName(str, Enum):
    """The name of a registered decay function."""
    exponential = "exponential"
    linear = "linear"
    step = "step"


# =========================================================================== #
#  Pydantic Models
# =========================================================================== #


class DecayFunctionParams(BaseModel):
    """Parameters for the configurable decay function."""
    rate: float = 5.0
    high_correlation_rate: float = 0.02
    low_correlation_rate: float = 1.0
    steps: list = Field(default_factory=list)

    @field_validator('rate')
    @classmethod
    def validate_rate(cls, v):
        if v <= 0.0:
            raise ValueError("rate must be greater than 0.0")
        return v

    @field_validator('high_correlation_rate', 'low_correlation_rate')
    @classmethod
    def validate_rates(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("correlation rates must be in [0.0, 1.0]")
        return v

    @field_validator('steps')
    @classmethod
    def validate_steps(cls, v):
        if not isinstance(v, list):
            raise ValueError("steps must be a list")

        for step in v:
            if not isinstance(step, (list, tuple)) or len(step) != 2:
                raise ValueError("each step must be a two-element list [threshold, rate]")
            threshold, rate = step
            if not (0.0 <= threshold <= 1.0):
                raise ValueError("step threshold must be in [0.0, 1.0]")
            if not (0.0 <= rate <= 1.0):
                raise ValueError("step rate must be in [0.0, 1.0]")

        # Check sorted order
        for i in range(len(v) - 1):
            if v[i][0] >= v[i + 1][0]:
                raise ValueError("steps must be sorted in ascending order by threshold")

        return v


class SamplingSchedulerConfig(BaseModel):
    """Configuration for the Adaptive Sampling Scheduler."""
    emergency_threshold: float = 0.3
    coaching_trigger: float = 0.6
    adaptive_entry_correlation: float = 0.8
    min_floor: float = 0.02
    min_sample_count_for_adaptive: int = 50
    decay_function_name: DecayFunctionName = DecayFunctionName.exponential
    decay_params: DecayFunctionParams = Field(default_factory=DecayFunctionParams)

    @field_validator('emergency_threshold', 'coaching_trigger', 'adaptive_entry_correlation')
    @classmethod
    def validate_thresholds(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("thresholds must be in [0.0, 1.0]")
        return v

    @field_validator('min_floor')
    @classmethod
    def validate_min_floor(cls, v):
        if v <= 0.0:
            raise ValueError("min_floor must be greater than 0.0")
        if v > 1.0:
            raise ValueError("min_floor must be <= 1.0")
        return v

    @field_validator('min_sample_count_for_adaptive')
    @classmethod
    def validate_min_sample_count(cls, v):
        if v < 1:
            raise ValueError("min_sample_count_for_adaptive must be >= 1")
        return v

    @model_validator(mode='after')
    def validate_threshold_ordering(self):
        if self.emergency_threshold >= self.coaching_trigger:
            raise ValueError(
                "Threshold ordering violated: emergency_threshold must be < coaching_trigger"
            )
        if self.coaching_trigger >= self.adaptive_entry_correlation:
            raise ValueError(
                "Threshold ordering violated: coaching_trigger must be < adaptive_entry_correlation"
            )

        # Check for step decay with empty steps
        if self.decay_function_name == DecayFunctionName.step:
            if not self.decay_params.steps:
                raise ValueError(
                    "Step decay function requires at least one step in decay_params.steps"
                )

        return self


class ConfidenceSnapshot(BaseModel):
    """Immutable snapshot of the current confidence state."""
    correlation_score: float
    is_stale: bool
    sample_count: int
    budget_exhausted: bool

    @field_validator('correlation_score')
    @classmethod
    def validate_correlation(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("correlation_score must be in [0.0, 1.0]")
        return v

    @field_validator('sample_count')
    @classmethod
    def validate_sample_count(cls, v):
        if v < 0:
            raise ValueError("sample_count must be >= 0")
        return v


class SamplingDecision(BaseModel):
    """Frozen (immutable) model representing the scheduler's decision."""
    send_to_local: bool
    send_to_remote: bool
    is_coaching_sample: bool
    sampling_rate: float
    phase: SamplingPhase
    reason: str

    model_config = {"frozen": True}

    @field_validator('sampling_rate')
    @classmethod
    def validate_sampling_rate(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("sampling_rate must be in [0.0, 1.0]")
        return v

    @field_validator('reason')
    @classmethod
    def validate_reason(cls, v):
        if not v or len(v) < 1:
            raise ValueError("reason must be non-empty")
        return v

    @model_validator(mode='after')
    def validate_routing(self):
        # At least one destination must be True
        if not (self.send_to_local or self.send_to_remote):
            raise ValueError("At least one of send_to_local or send_to_remote must be True")

        # is_coaching_sample must be derived correctly
        expected_coaching = self.send_to_local and self.send_to_remote
        if self.is_coaching_sample != expected_coaching:
            raise ValueError(
                f"is_coaching_sample must equal (send_to_local AND send_to_remote)"
            )

        return self


# =========================================================================== #
#  Decay Functions
# =========================================================================== #


def exponential_decay(correlation: float, params: DecayFunctionParams) -> float:
    """
    Computes sampling rate using exponential decay.

    Maps correlation from [0.0, 1.0] with exponential decay.
    The normalization assumes adaptive_entry_correlation = 0.6 as the baseline.
    At correlation=0.6, normalized=0 and rate~=1.0.
    As correlation increases to 1.0, rate decays exponentially.
    """
    # Normalize correlation assuming adaptive entry at 0.6
    # This maps [0.6, 1.0] -> [0, 1]
    # Below 0.6, we still normalize but the result will be > 1.0 before clamping
    adaptive_entry = 0.6
    if correlation <= adaptive_entry:
        normalized = 0.0
    else:
        normalized = (correlation - adaptive_entry) / (1.0 - adaptive_entry)

    # Exponential decay: rate = exp(-params.rate * normalized)
    rate = math.exp(-params.rate * normalized)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, rate))


def linear_decay(correlation: float, params: DecayFunctionParams) -> float:
    """
    Computes sampling rate using linear interpolation.

    Maps correlation from [0.0, 1.0] linearly between
    low_correlation_rate and high_correlation_rate.
    """
    # Linear interpolation
    rate = params.low_correlation_rate + (
        params.high_correlation_rate - params.low_correlation_rate
    ) * correlation

    # Clamp to [0, 1]
    return max(0.0, min(1.0, rate))


def step_decay(correlation: float, params: DecayFunctionParams) -> float:
    """
    Computes sampling rate using a step function.

    Finds the highest threshold that is <= current correlation
    and returns the corresponding rate.
    """
    if not params.steps:
        raise ValueError("Step decay function requires at least one step in params.steps")

    # Find the highest step threshold <= correlation
    rate = 1.0  # Default if below all thresholds
    for threshold, step_rate in params.steps:
        if correlation >= threshold:
            rate = step_rate
        else:
            break

    return rate


# =========================================================================== #
#  Adaptive Sampling Scheduler
# =========================================================================== #


class AdaptiveSamplingScheduler:
    """
    The Adaptive Sampling Scheduler determines whether each request
    should be a coaching sample based on the current confidence state.
    """

    def __init__(
        self,
        config: SamplingSchedulerConfig,
        random_source: Callable[[], float] = None,
    ):
        """
        Construct the scheduler with validated config and an injectable random source.
        """
        self._config = config
        self._random_source = random_source if random_source is not None else random.random

        # Resolve decay function
        self._decay_function = self._resolve_decay_function(config.decay_function_name)

    def _resolve_decay_function(self, name: DecayFunctionName) -> Callable:
        """Resolve the decay function from the config name."""
        if name == DecayFunctionName.exponential:
            return exponential_decay
        elif name == DecayFunctionName.linear:
            return linear_decay
        elif name == DecayFunctionName.step:
            return step_decay
        else:
            raise ValueError(f"Unknown decay function name: {name}")

    def should_sample(self, snapshot: ConfidenceSnapshot) -> SamplingDecision:
        """
        Determines whether this specific request should be a coaching sample.

        Override precedence (highest to lowest):
        1. budget_exhausted → LOCAL_ONLY
        2. correlation < emergency_threshold → REMOTE_ONLY
        3. correlation < coaching_trigger → COACHING_FULL
        4. is_stale → COACHING_FULL
        5. sample_count < min_sample_count_for_adaptive → COACHING_FULL
        6. correlation < adaptive_entry_correlation → COACHING_FULL
        7. ADAPTIVE phase: compute rate, roll random, route accordingly
        """
        # Priority 1: Budget exhausted
        if snapshot.budget_exhausted:
            return SamplingDecision(
                send_to_local=True,
                send_to_remote=False,
                is_coaching_sample=False,
                sampling_rate=1.0,
                phase=SamplingPhase.LOCAL_ONLY,
                reason="Budget exhausted - serving from local model only"
            )

        # Priority 2: Emergency threshold
        if snapshot.correlation_score < self._config.emergency_threshold:
            return SamplingDecision(
                send_to_local=False,
                send_to_remote=True,
                is_coaching_sample=False,
                sampling_rate=1.0,
                phase=SamplingPhase.REMOTE_ONLY,
                reason=f"Emergency: correlation {snapshot.correlation_score:.3f} < emergency_threshold {self._config.emergency_threshold:.3f} - local model too unreliable"
            )

        # Priority 3: Coaching trigger
        if snapshot.correlation_score < self._config.coaching_trigger:
            return SamplingDecision(
                send_to_local=True,
                send_to_remote=True,
                is_coaching_sample=True,
                sampling_rate=1.0,
                phase=SamplingPhase.COACHING_FULL,
                reason=f"Coaching full: correlation {snapshot.correlation_score:.3f} < coaching_trigger {self._config.coaching_trigger:.3f}"
            )

        # Priority 4: Staleness override
        if snapshot.is_stale:
            return SamplingDecision(
                send_to_local=True,
                send_to_remote=True,
                is_coaching_sample=True,
                sampling_rate=1.0,
                phase=SamplingPhase.COACHING_FULL,
                reason="Staleness override - forcing coaching sample to refresh correlation data"
            )

        # Priority 5: Insufficient samples (cold start)
        if snapshot.sample_count < self._config.min_sample_count_for_adaptive:
            return SamplingDecision(
                send_to_local=True,
                send_to_remote=True,
                is_coaching_sample=True,
                sampling_rate=1.0,
                phase=SamplingPhase.COACHING_FULL,
                reason=f"Cold start: sample_count {snapshot.sample_count} < min_sample_count_for_adaptive {self._config.min_sample_count_for_adaptive}"
            )

        # Priority 6: Below adaptive entry
        if snapshot.correlation_score < self._config.adaptive_entry_correlation:
            return SamplingDecision(
                send_to_local=True,
                send_to_remote=True,
                is_coaching_sample=True,
                sampling_rate=1.0,
                phase=SamplingPhase.COACHING_FULL,
                reason=f"Below adaptive entry: correlation {snapshot.correlation_score:.3f} < adaptive_entry_correlation {self._config.adaptive_entry_correlation:.3f}"
            )

        # Priority 7: ADAPTIVE phase
        # Compute sampling rate via decay function
        raw_rate = self._decay_function(snapshot.correlation_score, self._config.decay_params)

        # Apply floor clamp
        effective_rate = max(raw_rate, self._config.min_floor)

        # Roll random
        random_roll = self._random_source()

        if random_roll < effective_rate:
            # Coaching sample
            return SamplingDecision(
                send_to_local=True,
                send_to_remote=True,
                is_coaching_sample=True,
                sampling_rate=effective_rate,
                phase=SamplingPhase.ADAPTIVE,
                reason=f"Adaptive coaching sample: random_roll {random_roll:.3f} < effective_rate {effective_rate:.3f} (correlation={snapshot.correlation_score:.3f})"
            )
        else:
            # Local only
            return SamplingDecision(
                send_to_local=True,
                send_to_remote=False,
                is_coaching_sample=False,
                sampling_rate=effective_rate,
                phase=SamplingPhase.ADAPTIVE,
                reason=f"Adaptive local-only: random_roll {random_roll:.3f} >= effective_rate {effective_rate:.3f} (correlation={snapshot.correlation_score:.3f})"
            )


# Type aliases for protocols
RandomSource = Callable[[], float]
DecayFunction = Callable[[float, DecayFunctionParams], float]
