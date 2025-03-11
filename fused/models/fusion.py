"""
Fusion module implementations for FUSED.

This module contains implementations of different fusion strategies
for combining representations from multiple modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from fused.models.base import FusionModule


class ConcatenationFusion(FusionModule):
    """
    Simple concatenation-based fusion.
    
    This module concatenates representations from different modalities
    and projects them to the desired output dimension.
    """
    
    def __init__(self, 
                 input_dims: Dict[str, int], 
                 output_dim: int,
                 config: Dict):
        """
        Initialize the concatenation fusion module.
        
        Args:
            input_dims: Dictionary mapping modality names to their dimensions
            output_dim: Output dimension after fusion
            config: Configuration dictionary with the following keys:
                dropout: Dropout probability
                use_layernorm: Whether to use layer normalization
        """
        super().__init__(input_dims, output_dim, config)
        
        # Extract configuration
        dropout = config.get("dropout", 0.1)
        use_layernorm = config.get("use_layernorm", True)
        
        # Compute total input dimension
        total_input_dim = sum(input_dims.values())
        
        # Define projection layer
        self.projection = nn.Sequential()
        
        if use_layernorm:
            self.projection.add_module("norm", nn.LayerNorm(total_input_dim))
            
        self.projection.add_module("dropout", nn.Dropout(dropout))
        self.projection.add_module("linear", nn.Linear(total_input_dim, output_dim))
        self.projection.add_module("activation", nn.GELU())
        
    def forward(self, 
                inputs: Dict[str, torch.Tensor], 
                masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Fuse inputs from different modalities.
        
        Args:
            inputs: Dictionary mapping modality names to their representations
                Each representation should have shape [batch_size, seq_len, dim]
            masks: Optional dictionary mapping modality names to their masks
            
        Returns:
            Fused representation of shape [batch_size, seq_len, output_dim]
        """
        # Check if inputs are provided for all modalities
        for modality in self.input_dims:
            if modality not in inputs:
                raise ValueError(f"Input for modality {modality} is missing")
        
        # Get sequence length and batch size from first input
        first_modality = next(iter(inputs.values()))
        batch_size, seq_len, _ = first_modality.shape
        
        # Concatenate inputs
        concat_inputs = []
        for modality, dim in self.input_dims.items():
            modality_input = inputs[modality]
            
            # Ensure all inputs have same sequence length
            if modality_input.shape[1] != seq_len:
                raise ValueError(
                    f"Sequence length mismatch: expected {seq_len}, "
                    f"got {modality_input.shape[1]} for modality {modality}"
                )
            
            concat_inputs.append(modality_input)
        
        # Concatenate along feature dimension
        concat = torch.cat(concat_inputs, dim=-1)  # [batch_size, seq_len, sum(dims)]
        
        # Project to output dimension
        output = self.projection(concat)  # [batch_size, seq_len, output_dim]
        
        return output


class CrossAttentionFusion(FusionModule):
    """
    Cross-attention based fusion.
    
    This module uses cross-attention to fuse representations
    from different modalities.
    """
    
    def __init__(self, 
                 input_dims: Dict[str, int], 
                 output_dim: int,
                 config: Dict):
        """
        Initialize the cross-attention fusion module.
        
        Args:
            input_dims: Dictionary mapping modality names to their dimensions
            output_dim: Output dimension after fusion
            config: Configuration dictionary with the following keys:
                num_heads: Number of attention heads
                dropout: Dropout probability
                use_layernorm: Whether to use layer normalization
        """
        super().__init__(input_dims, output_dim, config)
        
        # Extract configuration
        num_heads = config.get("num_heads", 8)
        dropout = config.get("dropout", 0.1)
        use_layernorm = config.get("use_layernorm", True)
        
        # Ensure all inputs have same dimension for cross-attention
        self.projections = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.projections[modality] = nn.Linear(dim, output_dim)
        
        # Cross-attention layers
        self.cross_attentions = nn.ModuleDict()
        for modality in input_dims:
            self.cross_attentions[modality] = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Layer normalization
        if use_layernorm:
            self.norms = nn.ModuleDict()
            for modality in input_dims:
                self.norms[modality] = nn.LayerNorm(output_dim)
        else:
            self.norms = None
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim * len(input_dims), output_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        
    def forward(self, 
                inputs: Dict[str, torch.Tensor], 
                masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Fuse inputs from different modalities.
        
        Args:
            inputs: Dictionary mapping modality names to their representations
                Each representation should have shape [batch_size, seq_len, dim]
            masks: Optional dictionary mapping modality names to their masks
            
        Returns:
            Fused representation of shape [batch_size, seq_len, output_dim]
        """
        # Check if inputs are provided for all modalities
        for modality in self.input_dims:
            if modality not in inputs:
                raise ValueError(f"Input for modality {modality} is missing")
        
        # Project all inputs to same dimension
        projected_inputs = {}
        for modality, input_tensor in inputs.items():
            projected_inputs[modality] = self.projections[modality](input_tensor)
        
        # Apply cross-attention
        attended_features = []
        for target_modality in self.input_dims:
            target_input = projected_inputs[target_modality]
            
            # Compute mask for target modality
            target_mask = None
            if masks is not None and target_modality in masks:
                target_mask = masks[target_modality]
            
            # Apply cross-attention with all other modalities
            cross_attended = target_input.clone()
            for source_modality in self.input_dims:
                if source_modality != target_modality:
                    source_input = projected_inputs[source_modality]
                    
                    # Compute mask for source modality
                    source_mask = None
                    if masks is not None and source_modality in masks:
                        source_mask = masks[source_modality]
                    
                    # Apply cross-attention
                    attended, _ = self.cross_attentions[target_modality](
                        query=cross_attended,
                        key=source_input,
                        value=source_input,
                        key_padding_mask=source_mask
                    )
                    
                    # Add residual connection
                    cross_attended = cross_attended + attended
            
            # Apply layer normalization if enabled
            if self.norms is not None:
                cross_attended = self.norms[target_modality](cross_attended)
            
            attended_features.append(cross_attended)
        
        # Concatenate attended features
        concat_features = torch.cat(attended_features, dim=-1)
        
        # Apply final projection
        output = self.output_projection(concat_features)
        
        return output


class GatedFusion(FusionModule):
    """
    Gated fusion.
    
    This module uses gating mechanisms to fuse representations
    from different modalities.
    """
    
    def __init__(self, 
                 input_dims: Dict[str, int], 
                 output_dim: int,
                 config: Dict):
        """
        Initialize the gated fusion module.
        
        Args:
            input_dims: Dictionary mapping modality names to their dimensions
            output_dim: Output dimension after fusion
            config: Configuration dictionary with the following keys:
                hidden_dim: Dimension of hidden layers
                dropout: Dropout probability
                use_layernorm: Whether to use layer normalization
        """
        super().__init__(input_dims, output_dim, config)
        
        # Extract configuration
        hidden_dim = config.get("hidden_dim", output_dim)
        dropout = config.get("dropout", 0.1)
        use_layernorm = config.get("use_layernorm", True)
        
        # Projections for each modality
        self.projections = nn.ModuleDict()
        for modality, dim in input_dims.items():
            self.projections[modality] = nn.Linear(dim, hidden_dim)
        
        # Gate networks for each modality
        self.gates = nn.ModuleDict()
        for modality in input_dims:
            self.gates[modality] = nn.Sequential(
                nn.Linear(sum(input_dims.values()), hidden_dim),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
        
        # Layer normalization if enabled
        if use_layernorm:
            self.norms = nn.ModuleDict()
            for modality in input_dims:
                self.norms[modality] = nn.LayerNorm(hidden_dim)
        else:
            self.norms = None
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        
    def forward(self, 
                inputs: Dict[str, torch.Tensor], 
                masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Fuse inputs from different modalities.
        
        Args:
            inputs: Dictionary mapping modality names to their representations
                Each representation should have shape [batch_size, seq_len, dim]
            masks: Optional dictionary mapping modality names to their masks
            
        Returns:
            Fused representation of shape [batch_size, seq_len, output_dim]
        """
        # Check if inputs are provided for all modalities
        for modality in self.input_dims:
            if modality not in inputs:
                raise ValueError(f"Input for modality {modality} is missing")
        
        # Get sequence length and batch size from first input
        first_modality = next(iter(inputs.values()))
        batch_size, seq_len, _ = first_modality.shape
        
        # Concatenate all inputs for gate computation
        concat_inputs = []
        for modality in self.input_dims:
            concat_inputs.append(inputs[modality])
        
        concat_for_gate = torch.cat(concat_inputs, dim=-1)
        
        # Project each modality
        projected_inputs = {}
        for modality, input_tensor in inputs.items():
            projected_inputs[modality] = self.projections[modality](input_tensor)
        
        # Apply gating
        gated_features = []
        for modality in self.input_dims:
            # Compute gate values
            gate = self.gates[modality](concat_for_gate)
            
            # Apply gate to projected input
            gated = gate * projected_inputs[modality]
            
            # Apply layer normalization if enabled
            if self.norms is not None:
                gated = self.norms[modality](gated)
            
            gated_features.append(gated)
        
        # Sum gated features
        summed_features = sum(gated_features)
        
        # Apply final projection
        output = self.output_projection(summed_features)
        
        return output
