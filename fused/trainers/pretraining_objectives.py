"""
Self-supervised pretraining objectives for FUSED.

This module contains implementations of different pretraining objectives
for self-supervised learning on time series data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from fused.models.base import PretrainingTask


class TemporalContrastiveTask(PretrainingTask):
    """
    Temporal contrastive learning task.
    
    This task learns to distinguish between embeddings from temporally close
    points and temporally distant points.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the temporal contrastive task.
        
        Args:
            config: Configuration dictionary with the following keys:
                temperature: Temperature parameter for contrastive loss
                time_threshold: Threshold for defining positive pairs
                negative_sample_method: Method for negative sampling ('random', 'hard', 'distance')
                use_cross_batch: Whether to use samples from other batch elements as negatives
        """
        super().__init__(config)
        
        # Extract configuration
        self.temperature = config.get("temperature", 0.07)
        self.time_threshold = config.get("time_threshold", 0.5)
        self.negative_sample_method = config.get("negative_sample_method", "random")
        self.use_cross_batch = config.get("use_cross_batch", True)
        
    def prepare_batch(self, batch: Dict) -> Dict:
        """
        Prepare a batch for this task.
        
        Args:
            batch: Input batch containing time series and timestamps
            
        Returns:
            Prepared batch
        """
        # This task doesn't require special preparation
        return batch
        
    def compute_loss(self, 
                     outputs: Dict, 
                     batch: Dict, 
                     model: nn.Module) -> torch.Tensor:
        """
        Compute the temporal contrastive loss.
        
        Args:
            outputs: Model outputs containing embeddings
            batch: Input batch containing time series and timestamps
            model: Reference to the full model
            
        Returns:
            Contrastive loss value
        """
        # Extract embeddings and timestamps
        embeddings = outputs["embeddings"]  # [batch_size, seq_len, dim]
        timestamps = batch["timestamps"]  # [batch_size, seq_len]
        
        # Get dimensions
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Reshape embeddings and timestamps
        flat_embeddings = embeddings.reshape(-1, embed_dim)  # [batch_size * seq_len, dim]
        flat_timestamps = timestamps.reshape(-1)  # [batch_size * seq_len]
        
        # Time distances between all pairs
        time_dists = torch.abs(flat_timestamps.unsqueeze(1) - flat_timestamps.unsqueeze(0))
        
        # Create mask for positive pairs (close in time)
        pos_mask = time_dists < self.time_threshold
        
        # Create mask for valid pairs (exclude self-pairs)
        valid_mask = torch.ones_like(pos_mask, dtype=torch.bool)
        valid_mask.fill_diagonal_(False)
        
        # Compute similarity matrix
        similarity = torch.matmul(
            F.normalize(flat_embeddings, dim=1),
            F.normalize(flat_embeddings, dim=1).transpose(0, 1)
        ) / self.temperature
        
        # Apply valid mask (set invalid pairs to large negative value)
        similarity = similarity.masked_fill(~valid_mask, -1e9)
        
        # Compute loss for each anchor
        total_loss = 0.0
        n_valid = 0
        
        for i in range(batch_size * seq_len):
            # Skip if no positive pairs
            if not torch.any(pos_mask[i] & valid_mask[i]):
                continue
                
            # Compute loss for current anchor
            anchor_sim = similarity[i]
            
            # Get positive and negative logits
            pos_logits = anchor_sim[pos_mask[i] & valid_mask[i]]
            neg_logits = anchor_sim[~pos_mask[i] & valid_mask[i]]
            
            # Skip if no negatives
            if neg_logits.numel() == 0:
                continue
                
            # Compute loss (InfoNCE)
            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
            labels[:pos_logits.size(0)] = 1
            
            loss = F.cross_entropy(
                logits.unsqueeze(0),
                labels.unsqueeze(0)
            )
            
            total_loss += loss
            n_valid += 1
            
        # Return average loss
        if n_valid > 0:
            return total_loss / n_valid
        else:
            # Return zero loss if no valid anchors
            return torch.tensor(0.0, device=similarity.device)


class MultimodalContrastiveTask(PretrainingTask):
    """
    Multimodal contrastive learning task.
    
    This task learns to align representations from different modalities.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the multimodal contrastive task.
        
        Args:
            config: Configuration dictionary with the following keys:
                temperature: Temperature parameter for contrastive loss
                modalities: List of modalities to align
                use_cross_batch: Whether to use samples from other batch elements as negatives
        """
        super().__init__(config)
        
        # Extract configuration
        self.temperature = config.get("temperature", 0.07)
        self.modalities = config.get("modalities", ["modality_a", "modality_b"])
        self.use_cross_batch = config.get("use_cross_batch", True)
        
    def prepare_batch(self, batch: Dict) -> Dict:
        """
        Prepare a batch for this task.
        
        Args:
            batch: Input batch containing different modalities
            
        Returns:
            Prepared batch
        """
        # Ensure all required modalities are present
        for modality in self.modalities:
            if modality not in batch:
                raise ValueError(f"Modality {modality} not found in batch")
                
        return batch
        
    def compute_loss(self, 
                     outputs: Dict, 
                     batch: Dict, 
                     model: nn.Module) -> torch.Tensor:
        """
        Compute the multimodal contrastive loss.
        
        Args:
            outputs: Model outputs containing embeddings for each modality
            batch: Input batch containing different modalities
            model: Reference to the full model
            
        Returns:
            Contrastive loss value
        """
        # Extract modality embeddings
        modality_embeddings = {}
        for modality in self.modalities:
            if f"{modality}_embeddings" not in outputs:
                raise ValueError(f"Embeddings for modality {modality} not found in outputs")
                
            modality_embeddings[modality] = outputs[f"{modality}_embeddings"]
            
        # Get sequence lengths for each modality
        seq_lengths = {}
        for modality in self.modalities:
            if f"{modality}_mask" in batch:
                # Compute sequence lengths from mask
                mask = batch[f"{modality}_mask"]
                seq_lengths[modality] = (~mask).sum(dim=1)
            else:
                # Assume all sequences are full length
                seq_lengths[modality] = torch.full(
                    (modality_embeddings[modality].shape[0],),
                    modality_embeddings[modality].shape[1],
                    device=modality_embeddings[modality].device
                )
        
        # Compute pooled embeddings for each modality
        pooled_embeddings = {}
        for modality in self.modalities:
            # Get embeddings and sequence lengths
            embeddings = modality_embeddings[modality]  # [batch_size, seq_len, dim]
            lengths = seq_lengths[modality]  # [batch_size]
            
            # Mean pooling
            pooled = []
            for b in range(embeddings.shape[0]):
                pooled.append(embeddings[b, :lengths[b]].mean(dim=0))
                
            pooled_embeddings[modality] = torch.stack(pooled)  # [batch_size, dim]
        
        # Compute pairwise losses between modalities
        total_loss = 0.0
        n_pairs = 0
        
        for i, modality_i in enumerate(self.modalities):
            for j, modality_j in enumerate(self.modalities):
                if i >= j:  # Only compute for unique pairs
                    continue
                    
                # Get embeddings
                embeddings_i = pooled_embeddings[modality_i]  # [batch_size, dim]
                embeddings_j = pooled_embeddings[modality_j]  # [batch_size, dim]
                
                # Normalize embeddings
                embeddings_i = F.normalize(embeddings_i, dim=1)
                embeddings_j = F.normalize(embeddings_j, dim=1)
                
                # Compute similarity matrix
                similarity = torch.matmul(embeddings_i, embeddings_j.transpose(0, 1)) / self.temperature
                
                # Compute InfoNCE loss
                labels = torch.arange(similarity.shape[0], device=similarity.device)
                loss_i_to_j = F.cross_entropy(similarity, labels)
                loss_j_to_i = F.cross_entropy(similarity.transpose(0, 1), labels)
                
                total_loss += (loss_i_to_j + loss_j_to_i) / 2.0
                n_pairs += 1
        
        # Return average loss
        if n_pairs > 0:
            return total_loss / n_pairs
        else:
            # Return zero loss if no modality pairs
            return torch.tensor(0.0, device=next(iter(pooled_embeddings.values())).device)


class MaskedModelingTask(PretrainingTask):
    """
    Masked modeling task.
    
    This task learns to reconstruct masked portions of the input.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the masked modeling task.
        
        Args:
            config: Configuration dictionary with the following keys:
                mask_ratio: Ratio of tokens to mask
                mask_method: Masking method ('random', 'structured', 'forecast')
                mask_token_value: Value to use for masked tokens
        """
        super().__init__(config)
        
        # Extract configuration
        self.mask_ratio = config.get("mask_ratio", 0.15)
        self.mask_method = config.get("mask_method", "random")
        self.mask_token_value = config.get("mask_token_value", 0.0)
        
    def prepare_batch(self, batch: Dict) -> Dict:
        """
        Prepare a batch for this task by masking portions of the input.
        
        Args:
            batch: Input batch containing time series
            
        Returns:
            Prepared batch with masked inputs
        """
        # Make a copy of the batch
        masked_batch = batch.copy()
        
        # Iterate over modalities
        for key, value in batch.items():
            # Only mask tensors that represent time series
            if not isinstance(value, torch.Tensor) or len(value.shape) < 3:
                continue
                
            # Skip masks
            if key.endswith("_mask"):
                continue
                
            # Get dimensions
            batch_size, seq_len, feature_dim = value.shape
            
            # Create masking tensor
            if self.mask_method == "random":
                # Random masking
                mask_prob = torch.full((batch_size, seq_len), self.mask_ratio, device=value.device)
                mask = torch.bernoulli(mask_prob).bool()
            elif self.mask_method == "structured":
                # Structured masking (mask consecutive tokens)
                mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=value.device)
                
                for b in range(batch_size):
                    # Determine number of mask spans
                    n_masks = max(1, int(seq_len * self.mask_ratio / 5))
                    
                    for _ in range(n_masks):
                        # Randomly select start position
                        start = torch.randint(0, seq_len - 5, (1,)).item()
                        
                        # Randomly select length of mask span
                        length = torch.randint(1, 6, (1,)).item()
                        
                        # Apply mask
                        end = min(start + length, seq_len)
                        mask[b, start:end] = True
            elif self.mask_method == "forecast":
                # Forecast masking (mask future tokens)
                mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=value.device)
                
                # Determine forecast horizon
                horizon = int(seq_len * self.mask_ratio)
                
                # Mask last `horizon` tokens
                mask[:, -horizon:] = True
            else:
                raise ValueError(f"Unknown mask method: {self.mask_method}")
            
            # Apply mask
            masked_value = value.clone()
            masked_value[mask.unsqueeze(-1).expand_as(value)] = self.mask_token_value
            
            # Store masked input and mask
            masked_batch[key] = masked_value
            masked_batch[f"{key}_original"] = value
            masked_batch[f"{key}_mask"] = mask
            
        return masked_batch
        
    def compute_loss(self, 
                     outputs: Dict, 
                     batch: Dict, 
                     model: nn.Module) -> torch.Tensor:
        """
        Compute the masked modeling loss.
        
        Args:
            outputs: Model outputs containing reconstructed inputs
            batch: Input batch containing original inputs and masks
            model: Reference to the full model
            
        Returns:
            Reconstruction loss value
        """
        total_loss = 0.0
        n_modalities = 0
        
        # Iterate over reconstructed outputs
        for key, value in outputs.items():
            # Only consider reconstructed outputs
            if not key.endswith("_reconstructed"):
                continue
                
            # Get original modality name
            modality = key.replace("_reconstructed", "")
            
            # Get original input and mask
            if f"{modality}_original" not in batch:
                continue
                
            original = batch[f"{modality}_original"]
            mask = batch[f"{modality}_mask"]
            
            # Compute reconstruction loss only on masked tokens
            masked_original = original[mask.unsqueeze(-1).expand_as(original)]
            masked_reconstructed = value[mask.unsqueeze(-1).expand_as(value)]
            
            # Compute mean squared error
            loss = F.mse_loss(masked_reconstructed, masked_original)
            
            total_loss += loss
            n_modalities += 1
            
        # Return average loss
        if n_modalities > 0:
            return total_loss / n_modalities
        else:
            # Return zero loss if no reconstructed outputs
            return torch.tensor(0.0, device=next(iter(outputs.values())).device)


class FuturePretraining(PretrainingTask):
    """
    Future prediction task.
    
    This task learns to predict future values of the time series.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the future prediction task.
        
        Args:
            config: Configuration dictionary with the following keys:
                prediction_horizon: Number of future steps to predict
                teacher_forcing_ratio: Ratio of teacher forcing during training
                loss_type: Type of loss function ('mse', 'mae', 'smooth_l1')
        """
        super().__init__(config)
        
        # Extract configuration
        self.prediction_horizon = config.get("prediction_horizon", 10)
        self.teacher_forcing_ratio = config.get("teacher_forcing_ratio", 0.5)
        self.loss_type = config.get("loss_type", "mse")
        
    def prepare_batch(self, batch: Dict) -> Dict:
        """
        Prepare a batch for this task.
        
        Args:
            batch: Input batch containing time series
            
        Returns:
            Prepared batch with inputs and targets
        """
        # Make a copy of the batch
        prepared_batch = batch.copy()
        
        # Iterate over modalities
        for key, value in batch.items():
            # Only process tensors that represent time series
            if not isinstance(value, torch.Tensor) or len(value.shape) < 3:
                continue
                
            # Skip masks
            if key.endswith("_mask"):
                continue
                
            # Get dimensions
            batch_size, seq_len, feature_dim = value.shape
            
            # Skip if sequence is too short
            if seq_len <= self.prediction_horizon:
                continue
                
            # Split into input and target
            input_seq = value[:, :-self.prediction_horizon].clone()
            target_seq = value[:, -self.prediction_horizon:].clone()
            
            # Store input and target
            prepared_batch[f"{key}_input"] = input_seq
            prepared_batch[f"{key}_target"] = target_seq
            
            # Update original tensor
            prepared_batch[key] = input_seq
            
        return prepared_batch
        
    def compute_loss(self, 
                     outputs: Dict, 
                     batch: Dict, 
                     model: nn.Module) -> torch.Tensor:
        """
        Compute the future prediction loss.
        
        Args:
            outputs: Model outputs containing predicted future values
            batch: Input batch containing target future values
            model: Reference to the full model
            
        Returns:
            Prediction loss value
        """
        total_loss = 0.0
        n_modalities = 0
        
        # Iterate over predicted outputs
        for key, value in outputs.items():
            # Only consider predicted outputs
            if not key.endswith("_predicted"):
                continue
                
            # Get original modality name
            modality = key.replace("_predicted", "")
            
            # Get target
            target_key = f"{modality}_target"
            if target_key not in batch:
                continue
                
            target = batch[target_key]
            
            # Compute loss based on selected loss type
            if self.loss_type == "mse":
                loss = F.mse_loss(value, target)
            elif self.loss_type == "mae":
                loss = F.l1_loss(value, target)
            elif self.loss_type == "smooth_l1":
                loss = F.smooth_l1_loss(value, target)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            total_loss += loss
            n_modalities += 1
            
        # Return average loss
        if n_modalities > 0:
            return total_loss / n_modalities
        else:
            # Return zero loss if no predicted outputs
            return torch.tensor(0.0, device=next(iter(outputs.values())).device)
