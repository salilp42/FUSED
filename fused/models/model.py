"""
Main model implementation for FUSED.

This module contains the implementation of the main model class
that integrates different components of the FUSED framework.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from fused.models.base import TimeSeriesEncoder, FusionModule, TemporalModel
from fused.models.sequential_encoder import SequentialEncoder
from fused.models.spectral_encoder import SpectralEncoder
from fused.models.tabular_encoder import TabularEncoder
from fused.models.fusion import ConcatenationFusion, CrossAttentionFusion, GatedFusion
from fused.models.neural_ode import ContinuousTemporalFlow
from fused.models.temporal_fusion import MultiScaleProcessor


class FUSEDModel(nn.Module):
    """
    FUSED model for multimodal time series modeling.
    
    This model integrates different components such as encoders,
    fusion modules, and temporal models for multimodal time series tasks.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the FUSED model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        
        # Set up encoders
        self.encoders = nn.ModuleDict()
        
        if "encoder" in config:
            # Single encoder case
            encoder_config = config["encoder"]
            encoder_type = encoder_config.pop("type", "sequential")
            
            if encoder_type == "sequential":
                self.encoders["sequential"] = SequentialEncoder(encoder_config)
            elif encoder_type == "spectral":
                self.encoders["spectral"] = SpectralEncoder(encoder_config)
            elif encoder_type == "tabular":
                self.encoders["tabular"] = TabularEncoder(encoder_config)
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")
                
            # Restore pop'd config
            encoder_config["type"] = encoder_type
            
        elif "encoders" in config:
            # Multiple encoders case
            for modality, encoder_config in config["encoders"].items():
                encoder_type = encoder_config.pop("type", "sequential")
                
                if encoder_type == "sequential":
                    self.encoders[modality] = SequentialEncoder(encoder_config)
                elif encoder_type == "spectral":
                    self.encoders[modality] = SpectralEncoder(encoder_config)
                elif encoder_type == "tabular":
                    self.encoders[modality] = TabularEncoder(encoder_config)
                else:
                    raise ValueError(f"Unknown encoder type: {encoder_type}")
                    
                # Restore pop'd config
                encoder_config["type"] = encoder_type
        else:
            raise ValueError("Either 'encoder' or 'encoders' must be specified in config")
            
        # Set up fusion module if multiple encoders
        if len(self.encoders) > 1 and "fusion" in config:
            fusion_config = config["fusion"]
            fusion_type = fusion_config.pop("type", "concatenation")
            
            # Get output dimensions for each encoder
            input_dims = {}
            for modality, encoder in self.encoders.items():
                input_dims[modality] = encoder.get_output_dim()
                
            # Output dimension
            output_dim = fusion_config.pop("output_dim", 128)
            
            if fusion_type == "concatenation":
                self.fusion = ConcatenationFusion(input_dims, output_dim, fusion_config)
            elif fusion_type == "cross_attention":
                self.fusion = CrossAttentionFusion(input_dims, output_dim, fusion_config)
            elif fusion_type == "gated":
                self.fusion = GatedFusion(input_dims, output_dim, fusion_config)
            else:
                raise ValueError(f"Unknown fusion type: {fusion_type}")
                
            # Restore pop'd config
            fusion_config["type"] = fusion_type
            fusion_config["output_dim"] = output_dim
        else:
            self.fusion = None
            
        # Set up temporal model
        if "temporal_model" in config:
            temporal_config = config["temporal_model"]
            temporal_type = temporal_config.pop("type", "multi_scale")
            
            if temporal_type == "multi_scale":
                self.temporal_model = MultiScaleProcessor(temporal_config)
            elif temporal_type == "neural_ode":
                self.temporal_model = ContinuousTemporalFlow(temporal_config)
            else:
                raise ValueError(f"Unknown temporal model type: {temporal_type}")
                
            # Restore pop'd config
            temporal_config["type"] = temporal_type
        else:
            self.temporal_model = None
            
        # Set up output layer
        if self.fusion is not None:
            input_dim = self.fusion.output_dim
        elif len(self.encoders) == 1:
            input_dim = next(iter(self.encoders.values())).get_output_dim()
        else:
            raise ValueError("Cannot determine input dimension for output layer")
            
        output_dim = config.get("output_dim", input_dim)
        
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
            
    def forward(self, batch: Dict) -> Dict:
        """
        Process a batch of data.
        
        Args:
            batch: Dictionary containing input data
            
        Returns:
            Dictionary containing model outputs
        """
        outputs = {}
        
        # Apply encoders
        encoded = {}
        for modality, encoder in self.encoders.items():
            # Check if input for this modality is available
            if modality in batch:
                # Get input and mask
                x = batch[modality]
                mask = batch.get(f"{modality}_mask", None)
                
                # Apply encoder
                encoded[modality] = encoder(x, mask)
                
                # Store embeddings in outputs
                outputs[f"{modality}_embeddings"] = encoded[modality]
                
        # Apply fusion if multiple encoders
        if self.fusion is not None and len(encoded) > 1:
            # Get masks
            masks = {}
            for modality in encoded:
                if f"{modality}_mask" in batch:
                    masks[modality] = batch[f"{modality}_mask"]
                    
            # Apply fusion
            fused = self.fusion(encoded, masks)
            outputs["fused_embeddings"] = fused
        elif len(encoded) == 1:
            # If only one encoder, use its output directly
            fused = next(iter(encoded.values()))
            outputs["fused_embeddings"] = fused
        else:
            raise ValueError("No encoder outputs available for fusion")
            
        # Apply temporal model if available
        if self.temporal_model is not None:
            # Get times if available
            times = batch.get("times", None)
            
            # Apply temporal model
            temporal = self.temporal_model(fused, times)
            outputs["temporal_embeddings"] = temporal
            
            # Update fused to use temporal output
            fused = temporal
            
        # Apply output layer
        output = self.output_layer(fused)
        outputs["embeddings"] = output
        
        return outputs
        
    @classmethod
    def from_config(cls, config: Dict) -> "FUSEDModel":
        """
        Create a model from configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            FUSED model
        """
        return cls(config)
        
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config
        }, path)
        
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "FUSEDModel":
        """
        Load a model from file.
        
        Args:
            path: Path to model file
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        if device is None:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=device)
            
        config = checkpoint["config"]
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if device is not None:
            model = model.to(device)
            
        return model
