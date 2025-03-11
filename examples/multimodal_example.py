"""
Multimodal example of using the FUSED framework.

This example demonstrates how to use the FUSED framework for
training a multimodal time series model with different fusion strategies.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from fused.data import MultimodalTimeSeriesDataset, Normalize, Compose, FFT
from fused.models import FUSEDModel
from fused.trainers import Trainer
from fused.configs.model_configs import get_multimodal_config
from fused.evaluation.metrics import classification_metrics


def generate_multimodal_data(num_samples=1000, seq_lens={'ecg': 128, 'audio': 256}, class_balance=0.5):
    """
    Generate synthetic multimodal time series data.
    
    Args:
        num_samples: Number of samples
        seq_lens: Dictionary mapping modality names to sequence lengths
        class_balance: Proportion of positive samples
        
    Returns:
        Tuple of (modalities, targets)
    """
    # Generate synthetic ECG-like data
    ecg_data = np.zeros((num_samples, seq_lens['ecg'], 1))
    
    # Generate synthetic audio-like data
    audio_data = np.zeros((num_samples, seq_lens['audio'], 1))
    
    # Generate binary targets
    targets = np.random.binomial(1, class_balance, (num_samples, 1))
    
    # Generate time steps
    ecg_time = np.linspace(0, 1, seq_lens['ecg'])
    audio_time = np.linspace(0, 1, seq_lens['audio'])
    
    for i in range(num_samples):
        # Generate ECG-like data
        if targets[i] == 1:
            # Positive class: faster rhythm with occasional anomalies
            base_freq = np.random.uniform(10, 15)
            anomaly_points = np.random.choice(seq_lens['ecg'], size=np.random.randint(1, 5), replace=False)
        else:
            # Negative class: slower, regular rhythm
            base_freq = np.random.uniform(5, 8)
            anomaly_points = []
            
        # Generate base ECG signal
        ecg_signal = np.sin(2 * np.pi * base_freq * ecg_time)
        
        # Add QRS complex-like peaks
        peak_indices = np.arange(0, seq_lens['ecg'], int(seq_lens['ecg'] / (base_freq + 1)))
        for idx in peak_indices:
            if idx < seq_lens['ecg'] - 5:
                ecg_signal[idx:idx+5] += np.array([0.5, 2.0, 0.0, -1.5, -0.5])
                
        # Add anomalies
        for idx in anomaly_points:
            if idx < seq_lens['ecg'] - 5:
                ecg_signal[idx:idx+5] += np.array([1.0, 3.0, 3.0, -2.0, -1.0])
                
        # Add noise
        ecg_signal += np.random.normal(0, 0.1, seq_lens['ecg'])
        
        # Store ECG signal
        ecg_data[i, :, 0] = ecg_signal
        
        # Generate audio-like data
        if targets[i] == 1:
            # Positive class: mix of frequencies
            freqs = [np.random.uniform(5, 10), np.random.uniform(15, 20)]
            amplitudes = [1.0, 0.8]
        else:
            # Negative class: more harmonic
            freqs = [np.random.uniform(5, 8), np.random.uniform(10, 12)]
            amplitudes = [1.0, 0.5]
            
        # Generate base audio signal
        audio_signal = amplitudes[0] * np.sin(2 * np.pi * freqs[0] * audio_time)
        audio_signal += amplitudes[1] * np.sin(2 * np.pi * freqs[1] * audio_time)
        
        # Add some harmonics
        audio_signal += 0.3 * np.sin(2 * np.pi * 2 * freqs[0] * audio_time)
        
        # Add noise
        audio_signal += np.random.normal(0, 0.2, seq_lens['audio'])
        
        # Store audio signal
        audio_data[i, :, 0] = audio_signal
        
    # Create modalities dictionary
    modalities = {
        'ecg': ecg_data,
        'audio': audio_data
    }
    
    return modalities, targets


def main():
    """
    Main function to run the example.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic multimodal data
    print("Generating synthetic multimodal data...")
    modalities, targets = generate_multimodal_data()
    
    # Split into train and test
    train_size = int(0.8 * len(targets))
    
    train_modalities = {
        'ecg': modalities['ecg'][:train_size],
        'audio': modalities['audio'][:train_size]
    }
    test_modalities = {
        'ecg': modalities['ecg'][train_size:],
        'audio': modalities['audio'][train_size:]
    }
    
    train_targets = targets[:train_size]
    test_targets = targets[train_size:]
    
    # Create transforms
    transforms = {
        'ecg': Compose([
            Normalize(method='standard')
        ]),
        'audio': Compose([
            # For audio, let's also add FFT transform
            Normalize(method='standard'),
            # Uncomment to use FFT as an additional preprocessor
            # FFT(return_type='magnitude', normalize=True)
        ])
    }
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MultimodalTimeSeriesDataset(
        modalities=train_modalities,
        targets=train_targets,
        transforms=transforms
    )
    
    test_dataset = MultimodalTimeSeriesDataset(
        modalities=test_modalities,
        targets=test_targets,
        transforms=transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Define fusion strategies to try
    fusion_strategies = ['concatenation', 'cross_attention', 'gated']
    
    for fusion_strategy in fusion_strategies:
        print(f"\nTraining with {fusion_strategy} fusion...")
        
        # Get model configuration
        config = get_multimodal_config(
            modality_dims={
                'ecg': 1,
                'audio': 1
            },
            modality_seq_lens={
                'ecg': 128,
                'audio': 256
            },
            hidden_dim=64,
            output_dim=1,
            task_type='classification',
            fusion_strategy=fusion_strategy
        )
        
        # Create model
        model = FUSEDModel(config)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            learning_rate=0.001,
            num_epochs=5,  # Using fewer epochs for example
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Train model
        trainer.train()
        
        # Plot training curves
        trainer.plot_training_curves(title=f"{fusion_strategy.capitalize()} Fusion Training Curves")
        
        # Evaluate model
        print(f"Evaluating model with {fusion_strategy} fusion...")
        test_loss, test_metrics = trainer.evaluate(test_loader)
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test metrics: {test_metrics}")
        
        # Save model
        torch.save(model.state_dict(), f"fused_model_{fusion_strategy}.pt")
        print(f"Model saved to fused_model_{fusion_strategy}.pt")
        
    # Compare results
    print("\nComparing fusion strategies:")
    # In a real scenario, you would load the saved models and compare their performance
    # Here we just print a message
    print("To compare strategies, examine the saved training curves and test metrics.")


if __name__ == "__main__":
    main()
