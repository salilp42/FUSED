"""
Pretraining example for the FUSED framework.

This example demonstrates how to pretrain a model using
self-supervised learning objectives and then fine-tune on a downstream task.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from fused.data import TimeSeriesDataset, Normalize, Compose, RandomMask
from fused.models import FUSEDModel
from fused.trainers import Trainer
from fused.trainers.pretraining_objectives import (
    TemporalContrastiveTask,
    MaskedModelingTask
)
from fused.configs.model_configs import get_pretraining_config, get_unimodal_config


def generate_complex_data(num_samples=2000, seq_len=128, num_features=10):
    """
    Generate synthetic complex time series data.
    
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
        # Define sample type (0: normal, 1: anomaly)
        sample_type = np.random.randint(0, 2)
        targets[i] = sample_type
        
        # Generate base patterns
        if sample_type == 0:
            # Normal pattern: smooth sine waves with different frequencies
            for j in range(num_features):
                freq = np.random.uniform(3, 8)
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = np.random.uniform(0.8, 1.2)
                
                data[i, :, j] = amplitude * np.sin(2 * np.pi * freq * time + phase)
                
                # Add some harmonics
                data[i, :, j] += 0.3 * np.sin(2 * np.pi * 2 * freq * time + phase)
                
                # Add correlations between features
                if j > 0:
                    data[i, :, j] += 0.2 * data[i, :, j-1]
        else:
            # Anomaly pattern: more erratic with sudden changes
            for j in range(num_features):
                freq = np.random.uniform(5, 12)
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = np.random.uniform(0.8, 1.5)
                
                data[i, :, j] = amplitude * np.sin(2 * np.pi * freq * time + phase)
                
                # Add some sharp transitions
                change_points = np.random.choice(seq_len - 10, size=3, replace=False)
                for cp in change_points:
                    # Create sharp transition
                    data[i, cp:cp+10, j] += np.linspace(0, amplitude * 2, 10) * np.sin(
                        2 * np.pi * freq * 2 * time[cp:cp+10] + phase
                    )
        
        # Add noise to all samples
        data[i] += np.random.normal(0, 0.1, (seq_len, num_features))
        
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
    data, targets = generate_complex_data()
    
    # Split into pretraining, fine-tuning, and test sets
    pretrain_size = int(0.6 * len(data))
    finetune_size = int(0.2 * len(data))
    test_size = len(data) - pretrain_size - finetune_size
    
    pretrain_data = data[:pretrain_size]
    finetune_data = data[pretrain_size:pretrain_size+finetune_size]
    finetune_targets = targets[pretrain_size:pretrain_size+finetune_size]
    test_data = data[pretrain_size+finetune_size:]
    test_targets = targets[pretrain_size+finetune_size:]
    
    # Create transforms
    normalize_transform = Compose([
        Normalize(method='standard')
    ])
    
    # For pretraining, we add masking for masked modeling task
    pretrain_transform = Compose([
        Normalize(method='standard'),
        RandomMask(mask_ratio=0.15, mask_value=0.0)
    ])
    
    # Create datasets
    print("Creating datasets...")
    pretrain_dataset = TimeSeriesDataset(
        data=pretrain_data,
        transform=pretrain_transform
    )
    
    finetune_dataset = TimeSeriesDataset(
        data=finetune_data,
        targets=finetune_targets,
        transform=normalize_transform
    )
    
    test_dataset = TimeSeriesDataset(
        data=test_data,
        targets=test_targets,
        transform=normalize_transform
    )
    
    # Create data loaders
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True)
    finetune_loader = DataLoader(finetune_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ----- PRETRAINING PHASE -----
    
    print("\n=== PRETRAINING PHASE ===")
    
    # Create pretraining objectives
    temporal_contrastive = TemporalContrastiveTask(
        temperature=0.1,
        projection_dim=128
    )
    
    masked_modeling = MaskedModelingTask(
        mask_token_value=0.0
    )
    
    # Get pretraining configuration
    pretrain_config = get_pretraining_config(
        input_dim=10,
        hidden_dim=128,
        seq_len=128,
        pretraining_objectives=[
            temporal_contrastive,
            masked_modeling
        ]
    )
    
    # Create model for pretraining
    print("Creating pretraining model...")
    pretrain_model = FUSEDModel(pretrain_config)
    
    # Create trainer for pretraining
    pretrain_trainer = Trainer(
        model=pretrain_model,
        train_loader=pretrain_loader,
        learning_rate=0.001,
        num_epochs=20,
        device=device,
        is_pretraining=True
    )
    
    # Pretrain model
    print("Pretraining model...")
    pretrain_trainer.train()
    
    # Plot pretraining curves
    pretrain_trainer.plot_training_curves(title="Pretraining Curves")
    
    # Save pretrained model weights
    torch.save(pretrain_model.state_dict(), "fused_pretrained.pt")
    print("Pretrained model saved to fused_pretrained.pt")
    
    # ----- FINE-TUNING PHASE -----
    
    print("\n=== FINE-TUNING PHASE ===")
    
    # Get fine-tuning configuration
    finetune_config = get_unimodal_config(
        input_dim=10,
        hidden_dim=128,
        output_dim=1,
        seq_len=128,
        task_type='classification'
    )
    
    # Create model for fine-tuning
    print("Creating fine-tuning model...")
    finetune_model = FUSEDModel(finetune_config)
    
    # Load pretrained weights into fine-tuning model
    # We only load the encoder weights, not the pretraining heads
    print("Loading pretrained weights...")
    pretrained_dict = torch.load("fused_pretrained.pt")
    model_dict = finetune_model.state_dict()
    
    # Filter out pretraining-specific parameters
    # and load only encoder parameters
    filtered_dict = {k: v for k, v in pretrained_dict.items() 
                    if k in model_dict and 'encoder' in k}
    model_dict.update(filtered_dict)
    finetune_model.load_state_dict(model_dict)
    
    # Create trainer for fine-tuning
    finetune_trainer = Trainer(
        model=finetune_model,
        train_loader=finetune_loader,
        val_loader=test_loader,
        learning_rate=0.0001,  # Lower learning rate for fine-tuning
        num_epochs=10,
        device=device
    )
    
    # Fine-tune model
    print("Fine-tuning model...")
    finetune_trainer.train()
    
    # Plot fine-tuning curves
    finetune_trainer.plot_training_curves(title="Fine-tuning Curves")
    
    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model...")
    test_loss, test_metrics = finetune_trainer.evaluate(test_loader)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test metrics: {test_metrics}")
    
    # Save fine-tuned model
    torch.save(finetune_model.state_dict(), "fused_finetuned.pt")
    print("Fine-tuned model saved to fused_finetuned.pt")
    
    # ----- COMPARE WITH TRAINING FROM SCRATCH -----
    
    print("\n=== TRAINING FROM SCRATCH FOR COMPARISON ===")
    
    # Create new model with same configuration but random initialization
    scratch_model = FUSEDModel(finetune_config)
    
    # Create trainer for model from scratch
    scratch_trainer = Trainer(
        model=scratch_model,
        train_loader=finetune_loader,
        val_loader=test_loader,
        learning_rate=0.001,
        num_epochs=10,
        device=device
    )
    
    # Train model from scratch
    print("Training model from scratch...")
    scratch_trainer.train()
    
    # Plot training curves
    scratch_trainer.plot_training_curves(title="Training From Scratch Curves")
    
    # Evaluate model from scratch
    print("Evaluating model trained from scratch...")
    scratch_loss, scratch_metrics = scratch_trainer.evaluate(test_loader)
    print(f"Scratch test loss: {scratch_loss:.4f}")
    print(f"Scratch test metrics: {scratch_metrics}")
    
    # Compare results
    print("\n=== COMPARISON ===")
    print(f"Fine-tuned model test metrics: {test_metrics}")
    print(f"Model from scratch test metrics: {scratch_metrics}")
    print("The pretrained+fine-tuned model should generally perform better, especially with limited labeled data.")


if __name__ == "__main__":
    main()
