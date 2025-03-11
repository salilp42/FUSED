"""
Spectral Encoder implementation for FUSED.

This module contains the implementation of the Spectral Encoder,
which is designed to process frequency-domain time series data
(e.g., spectrograms, Fourier transforms, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from fused.models.base import TimeSeriesEncoder


class FrequencyAttention(nn.Module):
    """
    Attention mechanism for frequency bands.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        """
        Initialize the frequency attention module.
        
        Args:
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention across frequency bands.
        
        Args:
            x: Input tensor of shape [batch_size, num_bands, hidden_dim]
            mask: Optional mask tensor
            
        Returns:
            Attended tensor of shape [batch_size, num_bands, hidden_dim]
        """
        # Apply attention
        attended, _ = self.attention(x, x, x, key_padding_mask=mask)
        
        return attended


class Conv2DBlock(nn.Module):
    """
    2D convolutional block for processing spectrograms.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3, 
                 stride: int = 1,
                 padding: int = 1,
                 dropout: float = 0.1):
        """
        Initialize the 2D convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolutions (square)
            stride: Stride for convolutions
            padding: Padding for convolutions
            dropout: Dropout probability
        """
        super().__init__()
        
        # Define layers
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        
        # Residual connection if dimensions match
        self.has_residual = (in_channels == out_channels and stride == 1)
        if not self.has_residual and in_channels != out_channels:
            self.projection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0
            )
        else:
            self.projection = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through the 2D convolutional block.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Processed tensor
        """
        # Store residual
        residual = x
        
        # Apply convolution
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Add residual if dimensions match
        if self.has_residual:
            x = x + residual
        elif self.projection is not None:
            x = x + self.projection(residual)
            
        return x


class SpectralEncoder(TimeSeriesEncoder):
    """
    Encoder for spectral time series data.
    
    This encoder uses convolutional layers to process 
    spectrograms or other 2D time-frequency representations.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the spectral encoder.
        
        Args:
            config: Configuration dictionary with the following keys:
                input_channels: Number of input channels
                hidden_dim: Dimension of hidden layers
                num_layers: Number of convolutional blocks
                freq_attention_heads: Number of heads for frequency attention
                channel_growth_factor: Factor by which channels grow in each layer
                dropout: Dropout probability
        """
        super().__init__(config)
        
        # Extract configuration
        input_channels = config.get("input_channels", 1)
        hidden_dim = config.get("hidden_dim", 128)
        num_layers = config.get("num_layers", 4)
        freq_attention_heads = config.get("freq_attention_heads", 8)
        channel_growth_factor = config.get("channel_growth_factor", 2)
        dropout = config.get("dropout", 0.1)
        
        # Convolutional blocks
        conv_layers = []
        in_channels = input_channels
        out_channels = hidden_dim
        
        for i in range(num_layers):
            conv_layers.append(
                Conv2DBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2 if i < num_layers - 1 else 1,  # Downsample except last layer
                    padding=1,
                    dropout=dropout
                )
            )
            in_channels = out_channels
            if i < num_layers - 1:  # Increase channels except for last layer
                out_channels = min(out_channels * channel_growth_factor, 512)
        
        self.conv_layers = nn.ModuleList(conv_layers)
        
        # Adaptive pooling to handle variable-sized inputs
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # Collapse frequency dimension
        
        # Frequency attention
        self.frequency_attention = FrequencyAttention(
            hidden_dim=out_channels,
            num_heads=freq_attention_heads
        )
        
        # Final processing
        self.output_norm = nn.LayerNorm(out_channels)
        
        # Hidden dimension
        self.hidden_dim = out_channels
        
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process spectral data.
        
        Args:
            x: Input tensor of shape [batch_size, channels, freq_bins, time_steps]
            mask: Optional mask tensor of shape [batch_size, time_steps]
            
        Returns:
            Encoded representation of shape [batch_size, time_steps, hidden_dim]
        """
        # Apply convolutional blocks
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Get dimensions
        batch_size, channels, freq_bins, time_steps = x.shape
        
        # Adaptive pooling over frequency dimension
        x = self.adaptive_pool(x)  # [batch_size, channels, 1, time_steps]
        x = x.squeeze(2)  # [batch_size, channels, time_steps]
        
        # Transpose for attention
        x = x.transpose(1, 2)  # [batch_size, time_steps, channels]
        
        # Apply output normalization
        x = self.output_norm(x)
        
        return x
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the encoder.
        
        Returns:
            Output dimension
        """
        return self.hidden_dim
