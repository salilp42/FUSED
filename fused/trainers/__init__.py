"""
Trainer module for FUSED.

This module contains training utilities for the FUSED framework.
"""

from fused.trainers.trainer import Trainer
from fused.trainers.pretraining_objectives import (
    TemporalContrastiveTask,
    MultimodalContrastiveTask,
    MaskedModelingTask,
    FuturePretraining
)
