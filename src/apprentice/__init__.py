"""Apprentice — Adaptive Model Distillation with Coaching.

A framework that starts with frontier API models, progressively trains
a local model, then withdraws the expensive dependency while maintaining
quality guarantees through adaptive sampling and confidence tracking.
"""

from apprentice.apprentice_class import Apprentice
from apprentice.config_loader import load_config, ApprenticeConfig
from apprentice.data_models import (
    TaskResponse,
    TrainingExample,
    ConfidenceSnapshot,
    BudgetState,
    EvaluationResult,
    SamplingDecision,
)

__version__ = "0.1.0"
__all__ = [
    "Apprentice",
    "load_config",
    "ApprenticeConfig",
    "TaskResponse",
    "TrainingExample",
    "ConfidenceSnapshot",
    "BudgetState",
    "EvaluationResult",
    "SamplingDecision",
]
