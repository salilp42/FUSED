"""
FUSED: Foundation-based Unified Sequential Embedding Design.

An open-source framework for multimodal time series modeling.
"""

__version__ = '0.1.0'

# Import core modules
from fused.models import FUSEDModel
from fused.trainers import Trainer
from fused.configs import (
    get_unimodal_config,
    get_multimodal_config,
    get_pretraining_config
)
