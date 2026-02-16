"""Apprentice — Adaptive Model Distillation with Coaching.

A framework that starts with frontier API models, progressively trains
a local model, then withdraws the expensive dependency while maintaining
quality guarantees through adaptive sampling and confidence tracking.
"""

from apprentice.data_models import *  # noqa: F401,F403
from apprentice.apprentice_class import Apprentice

__version__ = "0.1.0"
__all__ = ["Apprentice"]
