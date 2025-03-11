"""
Unit tests for model components.
"""

import pytest
import torch
import numpy as np
from fused.models.sequential_encoder import SequentialEncoder
from fused.models.tabular_encoder import TabularEncoder
from fused.models.fusion import ConcatenationFusion, CrossAttentionFusion, GatedFusion


@pytest.fixture
def sample_sequential_config():
    """Return a sample configuration for SequentialEncoder."""
    return {
        "input_dim": 5,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 64,
        "conv_kernel_size": 3,
        "conv_dilations": [1, 2],
        "dropout": 0.1,
        "max_len": 100
    }


@pytest.fixture
def sample_tabular_config():
    """Return a sample configuration for TabularEncoder."""
    return {
        "input_dim": 10,
        "hidden_dims": [64, 32],
        "dropout": 0.1,
        "use_batch_norm": True
    }


@pytest.fixture
def sample_fusion_config():
    """Return a sample configuration for fusion modules."""
    return {
        "dropout": 0.1
    }


class TestSequentialEncoder:
    """Test SequentialEncoder."""
    
    def test_initialization(self, sample_sequential_config):
        """Test that the encoder can be initialized."""
        encoder = SequentialEncoder(sample_sequential_config)
        assert isinstance(encoder, SequentialEncoder)
        
    def test_forward_pass(self, sample_sequential_config):
        """Test forward pass with sample input."""
        encoder = SequentialEncoder(sample_sequential_config)
        batch_size, seq_len, input_dim = 8, 50, sample_sequential_config["input_dim"]
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Test without mask
        output = encoder(x)
        assert output.shape == (batch_size, seq_len, sample_sequential_config["hidden_dim"])
        
        # Test with mask
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, seq_len//2:] = False  # Mask second half of sequences
        output = encoder(x, mask)
        assert output.shape == (batch_size, seq_len, sample_sequential_config["hidden_dim"])
        
    def test_output_dim(self, sample_sequential_config):
        """Test output dimension getter."""
        encoder = SequentialEncoder(sample_sequential_config)
        assert encoder.get_output_dim() == sample_sequential_config["hidden_dim"]


class TestTabularEncoder:
    """Test TabularEncoder."""
    
    def test_initialization(self, sample_tabular_config):
        """Test that the encoder can be initialized."""
        encoder = TabularEncoder(sample_tabular_config)
        assert isinstance(encoder, TabularEncoder)
        
    def test_forward_pass(self, sample_tabular_config):
        """Test forward pass with sample input."""
        encoder = TabularEncoder(sample_tabular_config)
        batch_size, input_dim = 8, sample_tabular_config["input_dim"]
        x = torch.randn(batch_size, input_dim)
        
        output = encoder(x)
        assert output.shape == (batch_size, sample_tabular_config["hidden_dims"][-1])
        
    def test_output_dim(self, sample_tabular_config):
        """Test output dimension getter."""
        encoder = TabularEncoder(sample_tabular_config)
        assert encoder.get_output_dim() == sample_tabular_config["hidden_dims"][-1]


class TestFusionModules:
    """Test fusion modules."""
    
    def test_concatenation_fusion(self, sample_fusion_config):
        """Test ConcatenationFusion."""
        input_dims = {"modality1": 32, "modality2": 64}
        output_dim = 96
        fusion = ConcatenationFusion(input_dims, output_dim, sample_fusion_config)
        
        batch_size = 8
        inputs = {
            "modality1": torch.randn(batch_size, 32),
            "modality2": torch.randn(batch_size, 64)
        }
        
        output = fusion(inputs)
        assert output.shape == (batch_size, output_dim)
        
    def test_cross_attention_fusion(self, sample_fusion_config):
        """Test CrossAttentionFusion."""
        input_dims = {"modality1": 32, "modality2": 32}
        output_dim = 32
        fusion_config = sample_fusion_config.copy()
        fusion_config["num_heads"] = 4
        fusion = CrossAttentionFusion(input_dims, output_dim, fusion_config)
        
        batch_size, seq_len = 8, 10
        inputs = {
            "modality1": torch.randn(batch_size, seq_len, 32),
            "modality2": torch.randn(batch_size, seq_len, 32)
        }
        
        output = fusion(inputs)
        assert output.shape == (batch_size, seq_len, output_dim)
        
    def test_gated_fusion(self, sample_fusion_config):
        """Test GatedFusion."""
        input_dims = {"modality1": 32, "modality2": 32}
        output_dim = 32
        fusion = GatedFusion(input_dims, output_dim, sample_fusion_config)
        
        batch_size = 8
        inputs = {
            "modality1": torch.randn(batch_size, 32),
            "modality2": torch.randn(batch_size, 32)
        }
        
        output = fusion(inputs)
        assert output.shape == (batch_size, output_dim)
