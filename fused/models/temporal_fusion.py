"""
Multi-scale temporal fusion module for FUSED.

This module contains implementations for processing time series at multiple scales
and fusing the representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from fused.models.base import TemporalModel


class TimeScaleProcessor(nn.Module):
    """
    Processor for a specific time scale.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 scale: int, 
                 hidden_dim: Optional[int] = None):
        """
        Initialize the time scale processor.
        
        Args:
            input_dim: Dimension of input features
            scale: Time scale factor (1 = original scale)
            hidden_dim: Dimension of hidden layers (default: same as input_dim)
        """
        super().__init__()
        
        self.scale = scale
        self.hidden_dim = hidden_dim or input_dim
        
        # Skip connection if scale is 1
        self.use_skip = (scale == 1)
        
        # Downsampling if scale > 1
        if scale > 1:
            self.downsample = nn.Conv1d(
                in_channels=input_dim,
                out_channels=input_dim,
                kernel_size=scale,
                stride=scale,
                padding=0,
                groups=input_dim
            )
        else:
            self.downsample = nn.Identity()
            
        # Processing at this scale
        self.process = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, input_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process input at specific time scale.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len]
            
        Returns:
            Processed tensor of shape [batch_size, seq_len / scale, input_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Apply mask if provided
        if mask is not None:
            x = x * (~mask).unsqueeze(-1).float()
        
        # Skip connection for original scale
        identity = x if self.use_skip else None
        
        # Downsample if scale > 1
        if self.scale > 1:
            # Transpose for 1D convolution
            x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
            
            # Pad if needed
            pad_len = (self.scale - (seq_len % self.scale)) % self.scale
            if pad_len > 0:
                x = F.pad(x, (0, pad_len))
                
            # Apply downsampling
            x = self.downsample(x)  # [batch_size, input_dim, seq_len / scale]
            
            # Transpose back
            x = x.transpose(1, 2)  # [batch_size, seq_len / scale, input_dim]
            
            # Update mask if provided
            if mask is not None:
                # Downsample mask
                mask = mask.view(batch_size, -1, self.scale).any(dim=2)
        
        # Process at this scale
        x = self.process(x)
        
        # Add skip connection if at original scale
        if identity is not None:
            x = x + identity
            
        return x


class TimeScaleFusion(nn.Module):
    """
    Fusion module for multiple time scales.
    """
    
    def __init__(self, 
                 num_scales: int, 
                 feature_dim: int,
                 method: str = 'attention'):
        """
        Initialize the time scale fusion module.
        
        Args:
            num_scales: Number of time scales
            feature_dim: Dimension of features
            method: Fusion method ('attention', 'concat', or 'weighted')
        """
        super().__init__()
        
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        self.method = method
        
        if method == 'attention':
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            )
            self.norm = nn.LayerNorm(feature_dim)
        elif method == 'concat':
            # Concatenation-based fusion
            self.projection = nn.Sequential(
                nn.Linear(feature_dim * num_scales, feature_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        elif method == 'weighted':
            # Weighted fusion
            self.weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
            
    def forward(self, 
                inputs: List[torch.Tensor], 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse representations from multiple time scales.
        
        Args:
            inputs: List of tensors from different time scales
                Each tensor should have shape [batch_size, seq_len_i, feature_dim]
                where seq_len_i depends on the scale
            mask: Optional mask tensor for the original scale
            
        Returns:
            Fused representation of shape [batch_size, seq_len, feature_dim]
            where seq_len is the sequence length of the original scale
        """
        # Ensure we have the expected number of inputs
        if len(inputs) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} inputs, got {len(inputs)}"
            )
            
        # Reference scale (original)
        ref_input = inputs[0]  # Original scale
        batch_size, ref_seq_len, feature_dim = ref_input.shape
        
        if self.method == 'attention':
            # Upsample all inputs to reference scale
            upsampled = []
            for i, x in enumerate(inputs):
                if i == 0:  # Original scale
                    upsampled.append(x)
                else:
                    # Get scale factor
                    scale = 2 ** i
                    
                    # Repeat each element
                    x_flat = x.reshape(batch_size, -1, feature_dim)
                    x_repeated = x_flat.repeat_interleave(scale, dim=1)
                    
                    # Truncate if needed
                    x_upsampled = x_repeated[:, :ref_seq_len, :]
                    
                    upsampled.append(x_upsampled)
            
            # Concatenate along batch dimension
            concat = torch.cat(upsampled, dim=0)  # [batch_size * num_scales, seq_len, feature_dim]
            
            # Create queries from original scale
            queries = ref_input
            
            # Apply attention
            attended, _ = self.attention(
                query=queries,
                key=concat,
                value=concat
            )
            
            # Add residual connection and normalize
            fused = queries + attended
            fused = self.norm(fused)
            
        elif self.method == 'concat':
            # Upsample all inputs to reference scale
            upsampled = []
            for i, x in enumerate(inputs):
                if i == 0:  # Original scale
                    upsampled.append(x)
                else:
                    # Get scale factor
                    scale = 2 ** i
                    
                    # Repeat each element
                    x_flat = x.reshape(batch_size, -1, feature_dim)
                    x_repeated = x_flat.repeat_interleave(scale, dim=1)
                    
                    # Truncate if needed
                    x_upsampled = x_repeated[:, :ref_seq_len, :]
                    
                    upsampled.append(x_upsampled)
            
            # Concatenate along feature dimension
            concat = torch.cat(upsampled, dim=-1)  # [batch_size, seq_len, feature_dim * num_scales]
            
            # Project to output dimension
            fused = self.projection(concat)
            
        elif self.method == 'weighted':
            # Upsample all inputs to reference scale
            upsampled = []
            for i, x in enumerate(inputs):
                if i == 0:  # Original scale
                    upsampled.append(x)
                else:
                    # Get scale factor
                    scale = 2 ** i
                    
                    # Repeat each element
                    x_flat = x.reshape(batch_size, -1, feature_dim)
                    x_repeated = x_flat.repeat_interleave(scale, dim=1)
                    
                    # Truncate if needed
                    x_upsampled = x_repeated[:, :ref_seq_len, :]
                    
                    upsampled.append(x_upsampled)
            
            # Apply softmax to weights
            weights = F.softmax(self.weights, dim=0)
            
            # Weighted sum
            fused = torch.zeros_like(ref_input)
            for i, x in enumerate(upsampled):
                fused += weights[i] * x
        
        return fused


class MultiScaleProcessor(TemporalModel):
    """
    Processes time series at multiple temporal scales.
    
    This module applies different processing at different time scales
    and fuses the results.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the multi-scale processor.
        
        Args:
            config: Configuration dictionary with the following keys:
                input_dim: Dimension of input features
                scales: List of scale factors
                hidden_dim: Dimension of hidden layers
                fusion_method: Method for fusing scales ('attention', 'concat', or 'weighted')
        """
        super().__init__(config)
        
        # Extract configuration
        input_dim = config.get("input_dim", 128)
        scales = config.get("scales", [1, 2, 4, 8])
        hidden_dim = config.get("hidden_dim", input_dim)
        fusion_method = config.get("fusion_method", "attention")
        
        # Create processors for each scale
        self.processors = nn.ModuleList([
            TimeScaleProcessor(
                input_dim=input_dim,
                scale=scale,
                hidden_dim=hidden_dim
            ) for scale in scales
        ])
        
        # Create fusion module
        self.fusion = TimeScaleFusion(
            num_scales=len(scales),
            feature_dim=input_dim,
            method=fusion_method
        )
        
        # Track input dimension
        self.input_dim = input_dim
        
    def forward(self, 
                x: torch.Tensor, 
                times: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process time series at multiple scales.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            times: Optional tensor of times (not used in this implementation)
            
        Returns:
            Processed tensor of shape [batch_size, seq_len, input_dim]
        """
        # Process at each scale
        multi_scale_features = []
        for processor in self.processors:
            # Process current scale
            features = processor(x)
            multi_scale_features.append(features)
        
        # Fuse features from different scales
        fused = self.fusion(multi_scale_features)
        
        return fused
