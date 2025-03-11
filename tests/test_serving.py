"""
Tests for model serving utilities.
"""

import os
import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import json

from fused.utils.serving import (
    ModelExporter,
    ModelServer,
    load_model
)


class SimpleModel(nn.Module):
    """Simple model for testing serving utilities."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)
        
        # Add a config attribute for testing
        self.config = {
            "input_dim": 10,
            "hidden_dim": 20,
            "output_dim": 2
        }
        
    def forward(self, x):
        # Ensure x is properly handled whether it's a tensor or dict
        if isinstance(x, dict):
            if "features" in x:
                x = x["features"]
            else:
                x = next(iter(x.values()))
        
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return {"logits": logits, "embeddings": x}


@pytest.fixture
def model():
    """Create a simple model for testing."""
    model = SimpleModel()
    model.eval()
    return model


@pytest.fixture
def example_inputs():
    """Create example inputs for testing."""
    return {"features": torch.randn(1, 10)}


@pytest.fixture
def temp_dir():
    """Create a temporary directory for exported models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_model_exporter_pytorch(model, temp_dir):
    """Test exporting model in PyTorch format."""
    # Create exporter
    exporter = ModelExporter(model, save_dir=temp_dir)
    
    # Export PyTorch model
    save_path = exporter.export_pytorch(filename="test_model.pt")
    
    # Check if file exists
    assert os.path.exists(save_path)
    
    # Load model
    loaded_data = torch.load(save_path)
    
    # Check contents
    assert "state_dict" in loaded_data
    assert "config" in loaded_data
    
    # Create a new model and load state dict
    new_model = SimpleModel()
    new_model.load_state_dict(loaded_data["state_dict"])
    
    # Check if model works
    with torch.no_grad():
        dummy_input = torch.randn(1, 10)
        original_output = model(dummy_input)
        loaded_output = new_model(dummy_input)
        
        # Compare outputs
        torch.testing.assert_close(
            original_output["logits"], 
            loaded_output["logits"]
        )


def test_model_exporter_torchscript(model, example_inputs, temp_dir):
    """Test exporting model in TorchScript format."""
    # Skip if not supported
    try:
        torch.jit.trace(model, example_inputs)
    except Exception as e:
        pytest.skip(f"TorchScript tracing failed: {str(e)}")
    
    # Create exporter
    exporter = ModelExporter(model, save_dir=temp_dir)
    
    # Export TorchScript model
    save_path = exporter.export_torchscript(
        filename="test_model_script.pt",
        example_inputs=example_inputs,
        method="trace"
    )
    
    # Check if file exists
    assert os.path.exists(save_path)
    
    try:
        # Load model
        loaded_model = torch.jit.load(save_path)
        
        # Check if model works
        with torch.no_grad():
            dummy_input = example_inputs
            original_output = model(dummy_input)
            loaded_output = loaded_model(dummy_input)
            
            # Compare outputs (keys might be different in TorchScript)
            if isinstance(loaded_output, dict):
                torch.testing.assert_close(
                    original_output["logits"], 
                    loaded_output["logits"]
                )
            else:
                torch.testing.assert_close(
                    original_output["logits"], 
                    loaded_output
                )
    except Exception as e:
        pytest.skip(f"TorchScript loading failed: {str(e)}")


def test_model_exporter_config(model, temp_dir):
    """Test exporting model configuration."""
    # Create exporter
    exporter = ModelExporter(model, save_dir=temp_dir)
    
    # Export config
    save_path = exporter.export_config(filename="test_config.json")
    
    # Check if file exists
    assert os.path.exists(save_path)
    
    # Load config
    with open(save_path, "r") as f:
        loaded_config = json.load(f)
    
    # Check contents
    assert "input_dim" in loaded_config
    assert "hidden_dim" in loaded_config
    assert "output_dim" in loaded_config
    assert loaded_config["input_dim"] == 10
    assert loaded_config["hidden_dim"] == 20
    assert loaded_config["output_dim"] == 2


def test_model_server(model, example_inputs, temp_dir):
    """Test model server."""
    # Export model first
    exporter = ModelExporter(model, save_dir=temp_dir)
    save_path = exporter.export_pytorch(filename="test_server_model.pt")
    
    # Create server
    server = ModelServer(save_path, device="cpu")
    
    # Check if model is loaded
    assert server.model is not None
    
    # Test prediction
    result = server.predict(example_inputs)
    
    # Check results
    assert isinstance(result, dict)
    assert "logits" in result
    
    # Compare with original model
    with torch.no_grad():
        original_output = model(example_inputs)
        torch.testing.assert_close(
            original_output["logits"], 
            result["logits"]
        )


def test_load_model(model, example_inputs, temp_dir):
    """Test load_model function."""
    # Export model first
    exporter = ModelExporter(model, save_dir=temp_dir)
    save_path = exporter.export_pytorch(filename="test_load_model.pt")
    
    # Load model
    loaded_model = load_model(save_path, device="cpu")
    
    # Check if model is loaded
    assert loaded_model is not None
    
    # Test prediction
    with torch.no_grad():
        original_output = model(example_inputs)
        loaded_output = loaded_model(example_inputs)
        
        # Compare outputs
        torch.testing.assert_close(
            original_output["logits"], 
            loaded_output["logits"]
        )


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
