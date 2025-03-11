"""
Tests for interpretability utilities.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from fused.utils.interpretability import (
    FeatureImportance,
    AttentionVisualization
)


class SimpleFeatureModel(nn.Module):
    """Simple model for testing feature importance."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)
        
    def forward(self, x):
        # Ensure x is properly handled whether it's a tensor or dict
        if isinstance(x, dict):
            x = x.get('features', torch.zeros(1, 10))
        
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return {"logits": logits, "embeddings": x}


class SimpleTransformerModel(nn.Module):
    """Simple transformer model for testing attention visualization."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(5, 20)
        self.attn = nn.MultiheadAttention(20, 2, batch_first=True)
        self.fc = nn.Linear(20, 2)
        
    def forward(self, x):
        # Ensure x is properly handled whether it's a tensor or dict
        if isinstance(x, dict):
            x = x.get('sequential', torch.zeros(1, 10, 5))
        
        # Embed sequence
        x_emb = self.embedding(x)
        
        # Apply attention
        attn_output, attn_weights = self.attn(x_emb, x_emb, x_emb)
        
        # Store attention weights for later access
        self.attn_weights = attn_weights
        
        # Pool and classify
        x_pool = torch.mean(attn_output, dim=1)
        logits = self.fc(x_pool)
        
        return {"logits": logits, "embeddings": x_pool, "attention": attn_weights}


@pytest.fixture
def feature_model():
    """Create a simple feature model for testing."""
    model = SimpleFeatureModel()
    model.eval()
    return model


@pytest.fixture
def transformer_model():
    """Create a simple transformer model for testing."""
    model = SimpleTransformerModel()
    model.eval()
    return model


@pytest.fixture
def tabular_data():
    """Create tabular data for testing feature importance."""
    X = torch.randn(10, 10)
    y = torch.randint(0, 2, (10,))
    return X, y


@pytest.fixture
def sequential_data():
    """Create sequential data for testing attention visualization."""
    X = torch.randn(5, 10, 5)  # [batch_size, seq_len, features]
    y = torch.randint(0, 2, (5,))
    return X, y


def test_feature_importance_permutation(feature_model, tabular_data):
    """Test permutation feature importance."""
    X, y = tabular_data
    
    # Create feature importance analyzer
    analyzer = FeatureImportance(feature_model)
    
    # Define metric function
    def accuracy_metric(y_true, y_pred):
        return (torch.argmax(y_pred, dim=1) == y_true).float().mean().item()
    
    # Compute permutation importance
    results = analyzer.permutation_importance(
        X, y, accuracy_metric, n_repeats=2,
        feature_names=["Feature_" + str(i) for i in range(10)]
    )
    
    # Check results
    assert "importances_mean" in results
    assert "importances_std" in results
    assert "feature_names" in results
    assert len(results["importances_mean"]) == 10
    assert len(results["feature_names"]) == 10


def test_feature_importance_integrated_gradients(feature_model, tabular_data):
    """Test integrated gradients feature importance."""
    X, y = tabular_data
    
    # Create feature importance analyzer
    analyzer = FeatureImportance(feature_model)
    
    # Compute integrated gradients
    results = analyzer.integrated_gradients(
        X, n_steps=5,
        feature_names=["Feature_" + str(i) for i in range(10)]
    )
    
    # Check results
    assert "importances" in results
    assert "feature_names" in results
    assert len(results["importances"]) == 10
    assert len(results["feature_names"]) == 10


def test_feature_importance_plot(feature_model, tabular_data):
    """Test feature importance plotting."""
    X, y = tabular_data
    
    # Create feature importance analyzer
    analyzer = FeatureImportance(feature_model)
    
    # Define metric function
    def accuracy_metric(y_true, y_pred):
        return (torch.argmax(y_pred, dim=1) == y_true).float().mean().item()
    
    # Compute permutation importance
    results = analyzer.permutation_importance(
        X, y, accuracy_metric, n_repeats=2,
        feature_names=["Feature_" + str(i) for i in range(10)]
    )
    
    # Create plot
    fig = analyzer.plot_feature_importance(
        results, title="Test Feature Importance", sort=True
    )
    
    # Check plot
    assert isinstance(fig, plt.Figure)
    
    # Close figure to avoid memory leaks
    plt.close(fig)


def test_attention_visualization(transformer_model, sequential_data):
    """Test attention visualization."""
    X, y = sequential_data
    
    # Create attention visualizer
    visualizer = AttentionVisualization(transformer_model)
    
    # Register hooks
    visualizer.register_hooks()
    
    # Get attention maps
    attention_maps = visualizer.get_attention_maps(X)
    
    # Check results
    assert len(attention_maps) > 0
    for layer_name, attn_map in attention_maps.items():
        assert isinstance(attn_map, torch.Tensor)
        assert len(attn_map.shape) >= 3  # Should have at least (batch, seq, seq) dimensions
    
    # Create attention heatmap
    for layer_name in attention_maps:
        fig = visualizer.plot_attention_heatmap(
            attention_maps, layer_name, head_idx=0, sample_idx=0
        )
        
        # Check plot
        assert isinstance(fig, plt.Figure)
        
        # Close figure to avoid memory leaks
        plt.close(fig)
    
    # Remove hooks
    visualizer.remove_hooks()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
