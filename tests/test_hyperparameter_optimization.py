"""
Tests for hyperparameter optimization utilities.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset

from fused.utils.hyperparameter_optimization import (
    OptunaOptimizer,
    HyperparameterTuner
)


class SimpleModel(nn.Module):
    """Simple model for testing hyperparameter optimization."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.get("hidden_dim", 32)
        self.dropout_rate = config.get("dropout_rate", 0.1)
        
        self.fc1 = nn.Linear(10, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def fit(self, train_data, validation_data=None, epochs=10, lr=0.01):
        """Simple training function."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Create DataLoader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=32, shuffle=True
        )
        
        # Training loop
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                x, y = batch
                optimizer.zero_grad()
                outputs = self(x)
                loss = criterion(outputs, y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            
        # Validation
        if validation_data is not None:
            self.eval()
            val_loader = torch.utils.data.DataLoader(
                validation_data, batch_size=64, shuffle=False
            )
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    outputs = self(x)
                    loss = criterion(outputs, y.unsqueeze(1))
                    val_loss += loss.item()
                
                val_loss /= len(val_loader)
            
            return {"loss": epoch_loss, "val_loss": val_loss}
        
        return {"loss": epoch_loss}


@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset for testing."""
    X = torch.randn(100, 10)
    y = torch.sum(X[:, :3], dim=1) / 3.0  # Simple function to learn
    return TensorDataset(X, y)


def test_optuna_optimizer(dummy_dataset):
    """Test the OptunaOptimizer class."""
    # Skip if optuna is not installed
    pytest.importorskip("optuna")
    
    # Create search space
    search_space = {
        "hidden_dim": {
            "type": "int",
            "low": 16,
            "high": 64
        },
        "dropout_rate": {
            "type": "float",
            "low": 0.0,
            "high": 0.5
        }
    }
    
    # Create optimizer configuration
    config = {
        "direction": "minimize",
        "n_trials": 2,  # Use just 2 trials for faster testing
        "study_name": "test_study"
    }
    
    # Create optimizer
    optimizer = OptunaOptimizer(config)
    
    # Define objective function
    def objective(trial):
        model_config = {
            "hidden_dim": trial.suggest_int("hidden_dim", 16, 64),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5)
        }
        
        model = SimpleModel(model_config)
        results = model.fit(dummy_dataset, epochs=2)
        
        return results["loss"]
    
    # Run optimization
    best_params, best_value = optimizer.optimize(objective, search_space)
    
    # Check results
    assert isinstance(best_params, dict)
    assert "hidden_dim" in best_params
    assert "dropout_rate" in best_params
    assert isinstance(best_value, float)


def test_hyperparameter_tuner(dummy_dataset):
    """Test the HyperparameterTuner class."""
    # Skip if optuna is not installed
    pytest.importorskip("optuna")
    
    # Create search space
    search_space = {
        "hidden_dim": {
            "type": "int",
            "low": 16,
            "high": 64
        },
        "dropout_rate": {
            "type": "float",
            "low": 0.0,
            "high": 0.5
        }
    }
    
    # Create optimizer configuration
    optimizer_config = {
        "direction": "minimize",
        "n_trials": 2,  # Use just 2 trials for faster testing
        "study_name": "test_tuner_study"
    }
    
    # Create tuner
    tuner = HyperparameterTuner(
        optimizer_type="optuna",
        optimizer_config=optimizer_config,
        cv_folds=2  # Use just 2 folds for faster testing
    )
    
    # Run tuning
    best_params, best_model = tuner.tune(
        model_class=SimpleModel,
        dataset=dummy_dataset,
        search_space=search_space,
        eval_metric="val_loss",
        direction="minimize",
        epochs=2,
        lr=0.01
    )
    
    # Check results
    assert isinstance(best_params, dict)
    assert "hidden_dim" in best_params
    assert "dropout_rate" in best_params
    assert isinstance(best_model, SimpleModel)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
