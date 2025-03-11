"""
Tabular Encoder implementation for FUSED.

This module contains the implementation of the Tabular Encoder,
which is designed to process tabular/static data
(e.g., demographic information, clinical measurements, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from fused.models.base import TimeSeriesEncoder


class FeatureEmbedding(nn.Module):
    """
    Embedding layer for categorical features.
    """
    
    def __init__(self, 
                 num_categories: int, 
                 embedding_dim: int):
        """
        Initialize the feature embedding.
        
        Args:
            num_categories: Number of categories
            embedding_dim: Embedding dimension
        """
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=embedding_dim
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed categorical features.
        
        Args:
            x: Input tensor of shape [batch_size, ...]
            
        Returns:
            Embedded tensor
        """
        return self.embedding(x)


class FeatureTokenizer(nn.Module):
    """
    Tokenizer for tabular features.
    
    This module applies different transformations to numeric and categorical features.
    """
    
    def __init__(self, 
                 feature_dims: Dict[str, int],
                 numeric_features: List[str],
                 categorical_features: Dict[str, int],
                 embedding_dim: int):
        """
        Initialize the feature tokenizer.
        
        Args:
            feature_dims: Dictionary mapping feature names to their dimensions
            numeric_features: List of numeric feature names
            categorical_features: Dictionary mapping categorical feature names to their cardinalities
            embedding_dim: Embedding dimension for categorical features
        """
        super().__init__()
        
        self.feature_dims = feature_dims
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.embedding_dim = embedding_dim
        
        # Embedding layers for categorical features
        self.embedding_layers = nn.ModuleDict()
        for feature_name, num_categories in categorical_features.items():
            self.embedding_layers[feature_name] = FeatureEmbedding(
                num_categories=num_categories,
                embedding_dim=embedding_dim
            )
        
        # Linear layers for numeric features
        self.numeric_layers = nn.ModuleDict()
        for feature_name in numeric_features:
            self.numeric_layers[feature_name] = nn.Linear(
                in_features=feature_dims.get(feature_name, 1),
                out_features=embedding_dim
            )
            
    def forward(self, 
                x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Tokenize features.
        
        Args:
            x: Dictionary mapping feature names to their values
            
        Returns:
            Tokenized features of shape [batch_size, num_features, embedding_dim]
        """
        batch_size = next(iter(x.values())).shape[0]
        embedded_features = []
        
        # Process numeric features
        for feature_name in self.numeric_features:
            if feature_name in x:
                numeric_value = x[feature_name]
                embedded = self.numeric_layers[feature_name](numeric_value)
                embedded_features.append(embedded)
        
        # Process categorical features
        for feature_name, _ in self.categorical_features.items():
            if feature_name in x:
                categorical_value = x[feature_name]
                embedded = self.embedding_layers[feature_name](categorical_value)
                embedded_features.append(embedded)
        
        # Concatenate features
        if embedded_features:
            features = torch.stack(embedded_features, dim=1)
        else:
            # Fallback if no features are provided
            features = torch.zeros(batch_size, 0, self.embedding_dim, device=next(iter(x.values())).device)
            
        return features


class TabularEncoder(TimeSeriesEncoder):
    """
    Encoder for tabular data.
    
    This encoder uses a combination of embeddings and transformers
    to process tabular data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the tabular encoder.
        
        Args:
            config: Configuration dictionary with the following keys:
                feature_dims: Dictionary mapping feature names to their dimensions
                numeric_features: List of numeric feature names
                categorical_features: Dictionary mapping categorical feature names to their cardinalities
                embedding_dim: Embedding dimension for features
                hidden_dim: Dimension of hidden layers
                num_layers: Number of transformer layers
                num_heads: Number of attention heads
                ff_dim: Dimension of feedforward layers
                dropout: Dropout probability
        """
        super().__init__(config)
        
        # Extract configuration
        feature_dims = config.get("feature_dims", {})
        numeric_features = config.get("numeric_features", [])
        categorical_features = config.get("categorical_features", {})
        embedding_dim = config.get("embedding_dim", 64)
        hidden_dim = config.get("hidden_dim", 128)
        num_layers = config.get("num_layers", 2)
        num_heads = config.get("num_heads", 4)
        ff_dim = config.get("ff_dim", hidden_dim * 4)
        dropout = config.get("dropout", 0.1)
        
        # Feature tokenizer
        self.feature_tokenizer = FeatureTokenizer(
            feature_dims=feature_dims,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            embedding_dim=embedding_dim
        )
        
        # Projection to hidden dimension
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Transformer layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers
        )
        
        # Output processing
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Hidden dimension
        self.hidden_dim = hidden_dim
        
    def forward(self, 
                x: Dict[str, torch.Tensor], 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process tabular data.
        
        Args:
            x: Dictionary mapping feature names to their values
            mask: Optional mask tensor
            
        Returns:
            Encoded representation of shape [batch_size, 1, hidden_dim]
        """
        # Tokenize features
        features = self.feature_tokenizer(x)  # [batch_size, num_features, embedding_dim]
        
        # Project to hidden dimension
        features = self.projection(features)  # [batch_size, num_features, hidden_dim]
        
        # Add class token
        batch_size = features.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        features = torch.cat([cls_tokens, features], dim=1)  # [batch_size, 1 + num_features, hidden_dim]
        
        # Apply transformer
        if mask is not None:
            # Add mask for class token
            mask = F.pad(mask, (1, 0), value=False)
        
        features = self.transformer(features, src_key_padding_mask=mask)
        
        # Extract class token
        cls_token = features[:, 0]  # [batch_size, hidden_dim]
        
        # Apply output normalization
        cls_token = self.output_norm(cls_token)
        
        # Add sequence dimension back
        cls_token = cls_token.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        return cls_token
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the encoder.
        
        Returns:
            Output dimension
        """
        return self.hidden_dim
