"""
Base classes for FUSED model components.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

class TimeSeriesEncoder(nn.Module, ABC):
    """
    Abstract base class for all time series encoders.
    
    TimeSeriesEncoder processes a time series input and produces
    a fixed-dimensional embedding or a sequence of embeddings.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the encoder.
        
        Args:
            config: Configuration dictionary for the encoder
        """
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process time series data.
        
        Args:
            x: Time series data of shape [batch_size, seq_len, features]
            mask: Optional mask tensor of shape [batch_size, seq_len]
            
        Returns:
            Encoded representation
        """
        pass
        
    @abstractmethod
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the encoder.
        
        Returns:
            Output dimension
        """
        pass


class FusionModule(nn.Module, ABC):
    """
    Abstract base class for fusion modules.
    
    FusionModule combines representations from different modalities.
    """
    
    def __init__(self, 
                 input_dims: Dict[str, int], 
                 output_dim: int,
                 config: Dict):
        """
        Initialize the fusion module.
        
        Args:
            input_dims: Dictionary mapping modality names to their dimensions
            output_dim: Output dimension after fusion
            config: Configuration dictionary for the fusion module
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.config = config
        
    @abstractmethod
    def forward(self, 
                inputs: Dict[str, torch.Tensor], 
                masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Fuse inputs from different modalities.
        
        Args:
            inputs: Dictionary mapping modality names to their representations
            masks: Optional dictionary mapping modality names to their masks
            
        Returns:
            Fused representation
        """
        pass


class TemporalModel(nn.Module, ABC):
    """
    Abstract base class for temporal models.
    
    TemporalModel processes a sequence of representations over time.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the temporal model.
        
        Args:
            config: Configuration dictionary for the temporal model
        """
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, 
                z: torch.Tensor, 
                times: torch.Tensor) -> torch.Tensor:
        """
        Process a sequence of representations.
        
        Args:
            z: Sequence of representations [batch_size, seq_len, hidden_dim]
            times: Time points corresponding to each representation [batch_size, seq_len]
            
        Returns:
            Processed sequence
        """
        pass


class PretrainingTask(nn.Module, ABC):
    """
    Abstract base class for pretraining tasks.
    
    PretrainingTask defines a self-supervised learning objective.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the pretraining task.
        
        Args:
            config: Configuration dictionary for the pretraining task
        """
        super().__init__()
        self.config = config
        
    @abstractmethod
    def compute_loss(self, 
                     outputs: Dict, 
                     batch: Dict, 
                     model: nn.Module) -> torch.Tensor:
        """
        Compute the task-specific loss.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            model: Reference to the full model
            
        Returns:
            Loss value
        """
        pass
        
    @abstractmethod
    def prepare_batch(self, batch: Dict) -> Dict:
        """
        Prepare a batch for this specific task.
        
        This can involve creating augmentations, masking, etc.
        
        Args:
            batch: Input batch
            
        Returns:
            Prepared batch
        """
        pass
