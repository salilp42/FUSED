"""
Trainer implementation for FUSED.

This module contains the implementation of the Trainer class, which
handles the training and evaluation of FUSED models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Callable
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from fused.models.base import PretrainingTask


class Trainer:
    """
    Trainer for FUSED models.
    
    This class handles the training and evaluation of FUSED models,
    including support for pre-training, fine-tuning, and evaluation.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 config: Dict, 
                 pretraining_tasks: Optional[List[PretrainingTask]] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            pretraining_tasks: List of pretraining tasks
            device: Device to train on (defaults to cuda if available)
        """
        self.model = model
        self.config = config
        self.pretraining_tasks = pretraining_tasks or []
        
        # Extract configuration
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.clip_grad_norm = config.get("clip_grad_norm", 1.0)
        self.num_epochs = config.get("num_epochs", 100)
        self.patience = config.get("patience", 10)
        self.scheduler_factor = config.get("scheduler_factor", 0.5)
        self.scheduler_patience = config.get("scheduler_patience", 5)
        
        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Set up learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            verbose=True
        )
        
        # Initialize logging
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = config.get("output_dir", f"./runs/{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """
        Move a batch of data to the device.
        
        Args:
            batch: Batch of data
            
        Returns:
            Batch on device
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
                
        return device_batch
        
    def _compute_pretraining_loss(self, 
                                 outputs: Dict, 
                                 batch: Dict) -> torch.Tensor:
        """
        Compute the pretraining loss.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            Total pretraining loss
        """
        total_loss = 0.0
        
        # Compute loss for each pretraining task
        for task in self.pretraining_tasks:
            task_loss = task.compute_loss(outputs, batch, self.model)
            total_loss += task_loss
            
        return total_loss
        
    def _train_epoch(self, 
                    dataloader: DataLoader, 
                    loss_fn: Optional[Callable] = None) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            loss_fn: Loss function (if None, use pretraining tasks)
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for batch in tqdm(dataloader, desc="Training"):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Prepare batch for pretraining if needed
            if loss_fn is None and self.pretraining_tasks:
                # Apply all pretraining tasks to prepare the batch
                for task in self.pretraining_tasks:
                    batch = task.prepare_batch(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            
            # Compute loss
            if loss_fn is not None:
                loss = loss_fn(outputs, batch)
            elif self.pretraining_tasks:
                loss = self._compute_pretraining_loss(outputs, batch)
            else:
                raise ValueError("Either loss_fn or pretraining_tasks must be provided")
            
            # Backward pass and optimization
            loss.backward()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_norm
                )
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
        # Return average loss
        return total_loss / num_batches
        
    def _validate(self, 
                 dataloader: DataLoader, 
                 loss_fn: Optional[Callable] = None) -> float:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            loss_fn: Loss function (if None, use pretraining tasks)
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Prepare batch for pretraining if needed
                if loss_fn is None and self.pretraining_tasks:
                    # Apply all pretraining tasks to prepare the batch
                    for task in self.pretraining_tasks:
                        batch = task.prepare_batch(batch)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute loss
                if loss_fn is not None:
                    loss = loss_fn(outputs, batch)
                elif self.pretraining_tasks:
                    loss = self._compute_pretraining_loss(outputs, batch)
                else:
                    raise ValueError("Either loss_fn or pretraining_tasks must be provided")
                
                # Accumulate loss
                total_loss += loss.item()
                
        # Return average loss
        return total_loss / num_batches
        
    def train(self, 
              train_dataloader: DataLoader, 
              val_dataloader: Optional[DataLoader] = None, 
              loss_fn: Optional[Callable] = None) -> Dict:
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            loss_fn: Loss function (if None, use pretraining tasks)
            
        Returns:
            Training history
        """
        print(f"Training on {self.device}")
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Train for one epoch
            train_loss = self._train_epoch(train_dataloader, loss_fn)
            self.train_losses.append(train_loss)
            
            # Validate if validation data is provided
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader, loss_fn)
                self.val_losses.append(val_loss)
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    self.save_checkpoint(os.path.join(self.output_dir, "best_model.pt"))
                else:
                    self.epochs_without_improvement += 1
                    
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break
                    
                print(f"Epoch {epoch + 1}/{self.num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch + 1}/{self.num_epochs}: Train Loss: {train_loss:.6f}")
                
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(os.path.join(self.output_dir, f"checkpoint_epoch_{epoch + 1}.pt"))
                
        # Save final model
        self.save_checkpoint(os.path.join(self.output_dir, "final_model.pt"))
        
        # Plot training curve
        self.plot_training_curve()
        
        # Return training history
        history = {
            "train_losses": self.train_losses
        }
        
        if val_dataloader is not None:
            history["val_losses"] = self.val_losses
            
        return history
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load training state
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.epochs_without_improvement = checkpoint["epochs_without_improvement"]
        
        print(f"Checkpoint loaded from {path}")
        
    def plot_training_curve(self) -> None:
        """
        Plot the training curve.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Train Loss")
        
        if self.val_losses:
            plt.plot(self.val_losses, label="Validation Loss")
            
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curve")
        plt.legend()
        plt.grid(True)
        plt.yscale("log")
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "training_curve.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Training curve saved to {plot_path}")
        
    def evaluate(self, 
                dataloader: DataLoader, 
                metrics: Optional[Dict[str, Callable]] = None) -> Dict:
        """
        Evaluate the model.
        
        Args:
            dataloader: Evaluation data loader
            metrics: Dictionary of metric functions
            
        Returns:
            Evaluation results
        """
        self.model.eval()
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Store outputs and targets
                all_outputs.append(outputs)
                all_targets.append(batch)
                
        # Compute metrics
        results = {}
        
        if metrics is not None:
            for name, metric_fn in metrics.items():
                result = metric_fn(all_outputs, all_targets)
                results[name] = result
                
        return results
