"""
Sequential Encoder implementation for FUSED.

This module contains the implementation of the Sequential Encoder,
which is designed to process sequential time series data
(e.g., sensor readings, physiological signals, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

from fused.models.base import TimeSeriesEncoder


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-based models.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class ConvBlock(nn.Module):
    """
    Convolutional block for processing time series.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3, 
                 dilation: int = 1,
                 dropout: float = 0.1):
        """
        Initialize the convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolutions
            dilation: Dilation factor for convolutions
            dropout: Dropout probability
        """
        super().__init__()
        
        # Compute padding to maintain sequence length
        padding = (kernel_size - 1) * dilation // 2
        
        # Define layers
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection if dimensions match
        self.has_residual = (in_channels == out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through the convolutional block.
        
        Args:
            x: Input tensor of shape [batch_size, channels, seq_len]
            
        Returns:
            Processed tensor
        """
        # Apply convolution
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Add residual if dimensions match
        if self.has_residual:
            x = x + residual
            
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer implementation.
    """
    
    def __init__(self, 
                 d_model: int, 
                 nhead: int = 8, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1):
        """
        Initialize the transformer encoder layer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Activation
        self.activation = nn.GELU()
        
    def forward(self, 
                src: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process input through the transformer encoder layer.
        
        Args:
            src: Input tensor of shape [batch_size, seq_len, d_model]
            src_mask: Optional mask for the input
            
        Returns:
            Processed tensor
        """
        # Self-attention block
        src2, _ = self.self_attn(src, src, src, key_padding_mask=src_mask)
        src = src + self.dropout(src2)  # Residual connection
        src = self.norm1(src)  # Layer normalization
        
        # Feedforward block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)  # Residual connection
        src = self.norm2(src)  # Layer normalization
        
        return src


class SequentialEncoder(TimeSeriesEncoder):
    """
    Encoder for sequential time series data.
    
    This encoder uses a combination of convolutional layers and
    transformer layers to process sequential data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the sequential encoder.
        
        Args:
            config: Configuration dictionary with the following keys:
                input_dim: Dimension of input features
                hidden_dim: Dimension of hidden layers
                num_layers: Number of transformer layers
                num_heads: Number of attention heads
                ff_dim: Dimension of feedforward layers
                conv_kernel_size: Kernel size for convolutional layers
                conv_dilations: List of dilation factors for convolutional layers
                dropout: Dropout probability
                max_len: Maximum sequence length
        """
        super().__init__(config)
        
        # Extract configuration
        input_dim = config.get("input_dim", 1)
        hidden_dim = config.get("hidden_dim", 128)
        num_layers = config.get("num_layers", 4)
        num_heads = config.get("num_heads", 8)
        ff_dim = config.get("ff_dim", hidden_dim * 4)
        conv_kernel_size = config.get("conv_kernel_size", 3)
        conv_dilations = config.get("conv_dilations", [1, 2, 4, 8])
        dropout = config.get("dropout", 0.1)
        max_len = config.get("max_len", 5000)
        
        # Initial projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Convolutional feature extraction
        conv_layers = []
        in_channels = hidden_dim
        for dilation in conv_dilations:
            conv_layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=conv_kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
            in_channels = hidden_dim
        self.conv_layers = nn.ModuleList(conv_layers)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len)
        
        # Transformer layers
        transformer_layers = []
        for _ in range(num_layers):
            transformer_layers.append(
                TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout
                )
            )
        self.transformer_layers = nn.ModuleList(transformer_layers)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        # Hidden dimension
        self.hidden_dim = hidden_dim
        
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process sequential data.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len]
            
        Returns:
            Encoded representation of shape [batch_size, seq_len, hidden_dim]
        """
        # Initial projection
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # Apply convolutional layers
        x_conv = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        for conv_layer in self.conv_layers:
            x_conv = conv_layer(x_conv)
        x = x_conv.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer layers
        attention_mask = mask
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, attention_mask)
        
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
