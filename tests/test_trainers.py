"""
Unit tests for trainer components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from fused.trainers.trainer import Trainer
from fused.trainers.pretraining_objectives import MaskedModelingTask, TemporalContrastiveTask
from fused.models.model import FUSEDModel


@pytest.fixture
def sample_model_config():
    """Return a sample configuration for a FUSEDModel."""
    return {
        "modalities": {
            "time_series": {
                "type": "sequential",
                "config": {
                    "input_dim": 5,
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "dropout": 0.1
                }
            },
            "static": {
                "type": "tabular",
                "config": {
                    "input_dim": 10,
                    "hidden_dims": [32, 32],
                    "dropout": 0.1
                }
            }
        },
        "fusion": {
            "type": "concatenation",
            "config": {
                "output_dim": 32,
                "dropout": 0.1
            }
        },
        "task_head": {
            "type": "classification",
            "config": {
                "num_classes": 3,
                "hidden_dims": [64]
            }
        }
    }


@pytest.fixture
def sample_trainer_config():
    """Return a sample configuration for a Trainer."""
    return {
        "batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "patience": 5,
        "device": "cpu",
        "model_checkpoint_path": "model.pt"
    }


class MockDataLoader:
    """Mock DataLoader for testing."""
    
    def __init__(self, num_batches=5):
        """Initialize mock dataloader."""
        self.num_batches = num_batches
        self._iter = None
        
    def __iter__(self):
        """Return iterator."""
        self._iter = self._data_generator()
        return self._iter
        
    def _data_generator(self):
        """Generate mock data batches."""
        for _ in range(self.num_batches):
            # Generate batch
            batch = {
                "modalities": {
                    "time_series": torch.randn(8, 20, 5),
                    "static": torch.randn(8, 10)
                },
                "times": {
                    "time_series": torch.arange(0, 20).reshape(1, -1).repeat(8, 1)
                },
                "y": torch.randint(0, 3, (8,))
            }
            yield batch
            
    def __len__(self):
        """Return number of batches."""
        return self.num_batches


class TestTrainer:
    """Test Trainer class."""
    
    def test_initialization(self, sample_model_config, sample_trainer_config):
        """Test that the trainer can be initialized."""
        model = FUSEDModel(sample_model_config)
        trainer = Trainer(model, sample_trainer_config)
        
        assert trainer.model == model
        assert trainer.config == sample_trainer_config
        assert trainer.device == "cpu"
        
    @patch("torch.optim.Adam")
    def test_train_epoch(self, mock_optim, sample_model_config, sample_trainer_config):
        """Test training one epoch."""
        model = MagicMock(spec=FUSEDModel)
        model.return_value = {"logits": torch.randn(8, 3), "loss": torch.tensor(0.5)}
        
        trainer = Trainer(model, sample_trainer_config)
        train_loader = MockDataLoader()
        
        # Mock optimizer
        optimizer = MagicMock()
        mock_optim.return_value = optimizer
        trainer.optimizer = optimizer
        
        # Train one epoch
        epoch_loss = trainer._train_epoch(train_loader)
        
        # Check that loss is a float
        assert isinstance(epoch_loss, float)
        
        # Check that optimizer step was called for each batch
        assert optimizer.step.call_count == len(train_loader)
        
    @patch("torch.optim.Adam")
    def test_validate_epoch(self, mock_optim, sample_model_config, sample_trainer_config):
        """Test validating one epoch."""
        model = MagicMock(spec=FUSEDModel)
        model.return_value = {"logits": torch.randn(8, 3), "loss": torch.tensor(0.5)}
        
        trainer = Trainer(model, sample_trainer_config)
        val_loader = MockDataLoader()
        
        # Validate one epoch
        with patch("torch.no_grad"):
            epoch_loss, metrics = trainer._validate_epoch(val_loader)
        
        # Check that loss is a float
        assert isinstance(epoch_loss, float)
        
        # Check that metrics is a dictionary
        assert isinstance(metrics, dict)
        
    @patch("torch.optim.Adam")
    @patch("torch.save")
    def test_train(self, mock_save, mock_optim, sample_model_config, sample_trainer_config):
        """Test training loop."""
        model = MagicMock(spec=FUSEDModel)
        model.return_value = {"logits": torch.randn(8, 3), "loss": torch.tensor(0.5)}
        
        # Reduce epochs for testing
        sample_trainer_config["num_epochs"] = 2
        
        trainer = Trainer(model, sample_trainer_config)
        train_loader = MockDataLoader()
        val_loader = MockDataLoader()
        
        # Mock optimizer
        optimizer = MagicMock()
        mock_optim.return_value = optimizer
        trainer.optimizer = optimizer
        
        # Train
        with patch.object(trainer, "_train_epoch", return_value=0.5):
            with patch.object(trainer, "_validate_epoch", return_value=(0.6, {})):
                history = trainer.train(train_loader, val_loader)
        
        # Check that history is returned
        assert isinstance(history, dict)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == sample_trainer_config["num_epochs"]
        
        # Check that model was saved
        assert mock_save.called


class TestPretrainingObjectives:
    """Test pretraining objectives."""
    
    def test_masked_modeling_task(self):
        """Test MaskedModelingTask objective."""
        config = {
            "mask_ratio": 0.3,
            "mask_method": "random",
            "mask_token_value": 0.0
        }
        task = MaskedModelingTask(config)
        
        # Create sample batch
        batch = {
            "modalities": {
                "time_series": torch.randn(8, 20, 5)
            }
        }
        
        # Prepare batch for task
        prepared_batch = task.prepare_batch(batch)
        
        # Check that masked input is created
        assert "masked_input" in prepared_batch["modalities"]
        assert "mask" in prepared_batch
        
        # Create mock model and outputs
        model = MagicMock()
        outputs = {
            "reconstructed": torch.randn(8, 20, 5)
        }
        
        # Compute loss
        loss = task.compute_loss(outputs, prepared_batch, model)
        
        # Check that loss is a tensor
        assert torch.is_tensor(loss)
        
    def test_temporal_contrastive_task(self):
        """Test TemporalContrastiveTask objective."""
        config = {
            "temperature": 0.1,
            "time_threshold": 0.5,
            "negative_sample_method": "random",
            "use_cross_batch": True
        }
        task = TemporalContrastiveTask(config)
        
        # Create sample batch
        batch = {
            "modalities": {
                "time_series": torch.randn(8, 20, 5)
            },
            "timestamps": torch.arange(0, 20).reshape(1, -1).repeat(8, 1)
        }
        
        # Prepare batch for task
        prepared_batch = task.prepare_batch(batch)
        
        # Create mock model and outputs
        model = MagicMock()
        outputs = {
            "embeddings": torch.randn(8, 20, 32)
        }
        
        # Compute loss
        loss = task.compute_loss(outputs, prepared_batch, model)
        
        # Check that loss is a tensor
        assert torch.is_tensor(loss)
