"""
Minimal example of using the FUSED framework.

This script demonstrates the most basic usage of the FUSED framework
with minimal code to get started quickly.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

import fused
from fused.data import TimeSeriesDataset, Normalize, Compose
from fused.models import FUSEDModel
from fused.configs import get_unimodal_config


def main():
    # Generate simple synthetic time series data
    seq_len, features = 50, 5
    num_samples = 200
    
    # Generate data with simple sine waves
    times = np.linspace(0, 1, seq_len)
    data = np.zeros((num_samples, seq_len, features))
    labels = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Generate either high or low frequency signals
        is_high_freq = np.random.choice([0, 1])
        labels[i] = is_high_freq
        
        freq_range = [10, 20] if is_high_freq else [2, 5]
        
        # Generate each feature with slightly different frequency
        for j in range(features):
            freq = np.random.uniform(*freq_range)
            phase = np.random.uniform(0, 2 * np.pi)
            data[i, :, j] = np.sin(2 * np.pi * freq * times + phase)
    
    # Split into train/test sets
    train_size = int(0.8 * num_samples)
    train_data = data[:train_size]
    train_labels = labels[:train_size]
    test_data = data[train_size:]
    test_labels = labels[train_size:]
    
    # Create datasets
    transform = Compose([Normalize(method='standard')])
    train_dataset = TimeSeriesDataset(train_data, train_labels, transform=transform)
    test_dataset = TimeSeriesDataset(test_data, test_labels, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Create model configuration for classification
    config = get_unimodal_config(
        input_dim=features,
        hidden_dim=32,
        output_dim=1,
        seq_len=seq_len,
        task_type='classification'
    )
    
    # Create model
    model = FUSEDModel(config)
    
    # Print model summary
    print(f"FUSED model created for {seq_len}-step time series with {features} features")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train for a few epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"Training on {device} for 10 epochs...")
    
    for epoch in range(10):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            # Get data
            x = batch['data'].to(device)
            y = batch['targets'].to(device).float()
            
            # Forward pass
            outputs = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Get data
                x = batch['data'].to(device)
                y = batch['targets'].to(device).float()
                
                # Forward pass
                outputs = model(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, y)
                
                # Compute accuracy
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)
                
                val_loss += loss.item()
        
        # Print epoch results
        print(f"Epoch {epoch+1}/10: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(test_loader):.4f}, "
              f"Accuracy: {100*correct/total:.2f}%")
    
    print("Training complete!")


if __name__ == "__main__":
    main()
