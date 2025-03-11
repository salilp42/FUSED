"""
Basic example of using the FUSED framework.

This example demonstrates how to use the FUSED framework for
training a multimodal time series model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from fused.data import TimeSeriesDataset, Normalize, Compose
from fused.models import FUSEDModel
from fused.trainers import Trainer
from fused.configs.model_configs import get_unimodal_config
from fused.evaluation.metrics import forecasting_metrics


def generate_synthetic_data(num_samples=1000, seq_len=100, num_features=10):
    """
    Generate synthetic time series data.
    
    Args:
        num_samples: Number of samples
        seq_len: Sequence length
        num_features: Number of features
        
    Returns:
        Tuple of (data, targets)
    """
    # Generate time steps
    time = np.linspace(0, 1, seq_len)
    
    # Generate data
    data = np.zeros((num_samples, seq_len, num_features))
    targets = np.zeros((num_samples, 1))
    
    for i in range(num_samples):
        # Generate random frequencies and phases
        freqs = np.random.uniform(1, 10, num_features)
        phases = np.random.uniform(0, 2 * np.pi, num_features)
        amplitudes = np.random.uniform(0.5, 2, num_features)
        
        # Generate features
        for j in range(num_features):
            data[i, :, j] = amplitudes[j] * np.sin(2 * np.pi * freqs[j] * time + phases[j])
            
            # Add some noise
            data[i, :, j] += np.random.normal(0, 0.1, seq_len)
            
        # Generate target (binary classification based on average frequency)
        targets[i] = 1 if np.mean(freqs) > 5 else 0
        
    return data, targets


def main():
    """
    Main function to run the example.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    data, targets = generate_synthetic_data()
    
    # Split into train and test
    train_size = int(0.8 * len(data))
    train_data, test_data = data[:train_size], data[train_size:]
    train_targets, test_targets = targets[:train_size], targets[train_size:]
    
    # Create datasets
    print("Creating datasets...")
    transform = Compose([
        Normalize(method='standard')
    ])
    
    train_dataset = TimeSeriesDataset(
        data=train_data,
        targets=train_targets,
        transform=transform
    )
    
    test_dataset = TimeSeriesDataset(
        data=test_data,
        targets=test_targets,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Get model configuration
    config = get_unimodal_config(
        input_dim=10,
        hidden_dim=64,
        output_dim=1,
        seq_len=100,
        task_type='classification'
    )
    
    # Create model
    print("Creating model...")
    model = FUSEDModel(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        learning_rate=0.001,
        num_epochs=10,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train model
    print("Training model...")
    trainer.train()
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test metrics: {test_metrics}")
    
    # Save model
    torch.save(model.state_dict(), "fused_model.pt")
    print("Model saved to fused_model.pt")


if __name__ == "__main__":
    main()
