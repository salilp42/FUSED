"""
Model implementations for FUSED.
"""

from fused.models.base import (
    TimeSeriesEncoder,
    FusionModule,
    TemporalModel,
    PretrainingTask
)

from fused.models.sequential_encoder import SequentialEncoder
from fused.models.spectral_encoder import SpectralEncoder
from fused.models.tabular_encoder import TabularEncoder
from fused.models.fusion import (
    ConcatenationFusion,
    CrossAttentionFusion,
    GatedFusion
)
from fused.models.neural_ode import (
    ContinuousTemporalFlow,
    ProbabilisticTemporalFlow
)
from fused.models.temporal_fusion import MultiScaleProcessor
from fused.models.model import FUSEDModel

__all__ = [
    # Base classes
    "TimeSeriesEncoder",
    "FusionModule",
    "TemporalModel",
    "PretrainingTask",
    
    # Encoder implementations
    "SequentialEncoder",
    "SpectralEncoder",
    "TabularEncoder",
    
    # Fusion implementations
    "ConcatenationFusion",
    "CrossAttentionFusion",
    "GatedFusion",
    
    # Temporal models
    "ContinuousTemporalFlow",
    "MultiScaleProcessor",
]
