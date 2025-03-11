"""
Interpretability tools for FUSED models.

This module provides tools for interpreting and explaining FUSED models,
including feature importance analysis and attention visualization.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings


class FeatureImportance:
    """
    Feature importance analysis for FUSED models.
    
    This class provides methods for computing and visualizing feature 
    importance using various techniques including permutation importance,
    integrated gradients, and SHAP values.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize feature importance analyzer.
        
        Args:
            model: FUSED model
        """
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        
    def permutation_importance(self, 
                              X: torch.Tensor,
                              y: torch.Tensor,
                              metric_fn: Callable,
                              n_repeats: int = 10,
                              feature_names: Optional[List[str]] = None) -> Dict:
        """
        Compute permutation feature importance.
        
        Args:
            X: Input features [n_samples, n_features] or [n_samples, seq_len, n_features]
            y: Target values [n_samples]
            metric_fn: Function to compute evaluation metric
            n_repeats: Number of times to permute each feature
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary with feature importance results
        """
        # Make sure everything is on the same device
        device = next(self.model.parameters()).device
        X = X.to(device)
        y = y.to(device)
        
        # Get predictions with original features
        with torch.no_grad():
            original_preds = self.model(X)
            if isinstance(original_preds, dict):
                original_preds = original_preds.get("logits", original_preds.get("predictions", None))
            
            baseline_score = metric_fn(y, original_preds)
        
        # Determine feature dimensionality
        is_sequential = len(X.shape) == 3
        n_features = X.shape[-1]
        
        # Initialize importance scores
        importance_scores = np.zeros((n_repeats, n_features))
        
        # For each feature
        for feature_idx in range(n_features):
            # Repeat multiple times
            for repeat_idx in range(n_repeats):
                # Create a copy of the data
                if is_sequential:
                    X_permuted = X.clone()
                    # Permute the feature across all time steps and samples
                    perm_idx = torch.randperm(X.shape[0] * X.shape[1])
                    permuted_values = X[:, :, feature_idx].reshape(-1)[perm_idx]
                    X_permuted[:, :, feature_idx] = permuted_values.reshape(X.shape[0], X.shape[1])
                else:
                    X_permuted = X.clone()
                    # Permute the feature across samples
                    perm_idx = torch.randperm(X.shape[0])
                    X_permuted[:, feature_idx] = X[perm_idx, feature_idx]
                
                # Get predictions with permuted features
                with torch.no_grad():
                    permuted_preds = self.model(X_permuted)
                    if isinstance(permuted_preds, dict):
                        permuted_preds = permuted_preds.get("logits", permuted_preds.get("predictions", None))
                    
                    permuted_score = metric_fn(y, permuted_preds)
                
                # Compute importance as decrease in performance
                importance_scores[repeat_idx, feature_idx] = baseline_score - permuted_score
        
        # Compute mean and std of importance scores
        mean_importance = np.mean(importance_scores, axis=0)
        std_importance = np.std(importance_scores, axis=0)
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
            
        # Create result dictionary
        results = {
            "importances_mean": mean_importance,
            "importances_std": std_importance,
            "feature_names": feature_names,
            "baseline_score": baseline_score,
            "all_importances": importance_scores
        }
        
        return results
    
    def integrated_gradients(self, 
                            X: torch.Tensor,
                            baseline: Optional[torch.Tensor] = None,
                            n_steps: int = 50,
                            feature_names: Optional[List[str]] = None) -> Dict:
        """
        Compute integrated gradients for feature importance.
        
        Args:
            X: Input features [n_samples, n_features] or [n_samples, seq_len, n_features]
            baseline: Baseline input for integration (zeros by default)
            n_steps: Number of steps for approximating the integral
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary with feature importance results
        """
        # Make sure X requires gradients
        X = X.detach().requires_grad_(True)
        
        # Determine feature dimensionality
        is_sequential = len(X.shape) == 3
        n_features = X.shape[-1]
        
        # Create baseline if not provided
        if baseline is None:
            if is_sequential:
                baseline = torch.zeros_like(X)
            else:
                baseline = torch.zeros_like(X)
        
        # Compute the difference between input and baseline
        difference = X - baseline
        
        # Initialize integrated gradients
        integrated_grads = torch.zeros_like(X)
        
        # Compute integral approximation
        for step in range(n_steps):
            # Compute intermediate point along the path
            alpha = step / (n_steps - 1)
            intermediate_input = baseline + alpha * difference
            intermediate_input = intermediate_input.detach().requires_grad_(True)
            
            # Forward pass
            output = self.model(intermediate_input)
            if isinstance(output, dict):
                output = output.get("logits", output.get("predictions", None))
            
            # For classification, use the predicted class
            if len(output.shape) > 1 and output.shape[1] > 1:
                pred_class = torch.argmax(output, dim=1)
                output = output.gather(1, pred_class.unsqueeze(1)).squeeze()
            
            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=intermediate_input,
                grad_outputs=torch.ones_like(output),
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Accumulate gradients
            integrated_grads += gradients
        
        # Scale integrated gradients
        integrated_grads *= difference / n_steps
        
        # Compute absolute importance per feature
        if is_sequential:
            # Average over time dimension for sequential data
            feature_importance = integrated_grads.abs().mean(dim=1).mean(dim=0)
        else:
            # Average over samples for tabular data
            feature_importance = integrated_grads.abs().mean(dim=0)
        
        # Convert to numpy for easier handling
        feature_importance_np = feature_importance.detach().cpu().numpy()
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
            
        # Create result dictionary
        results = {
            "importances": feature_importance_np,
            "feature_names": feature_names,
            "integrated_gradients": integrated_grads.detach().cpu().numpy()
        }
        
        return results
    
    def shap_values(self, 
                   X: torch.Tensor,
                   n_samples: int = 100,
                   feature_names: Optional[List[str]] = None) -> Dict:
        """
        Compute SHAP values for feature importance.
        
        Args:
            X: Input features [n_samples, n_features] or [n_samples, seq_len, n_features]
            n_samples: Number of background samples for SHAP
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary with feature importance results
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")
            
        # Convert PyTorch model to function
        def model_fn(inputs):
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
            with torch.no_grad():
                outputs = self.model(inputs_tensor)
                if isinstance(outputs, dict):
                    outputs = outputs.get("logits", outputs.get("predictions", None))
            return outputs.numpy()
        
        # Convert to numpy
        X_np = X.detach().cpu().numpy()
        
        # Determine feature dimensionality
        is_sequential = len(X_np.shape) == 3
        n_features = X_np.shape[-1]
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        # Create explainer
        if is_sequential:
            warnings.warn("SHAP for sequential data is experimental and may be slow.")
            # Reshape to 2D for SHAP (samples, seq_len*features)
            X_np_reshaped = X_np.reshape(X_np.shape[0], -1)
            explainer = shap.KernelExplainer(model_fn, X_np_reshaped[:n_samples])
            shap_values_result = explainer.shap_values(X_np_reshaped[:min(100, X_np.shape[0])])
            
            # Reshape back to 3D
            if isinstance(shap_values_result, list):
                # Multi-class output
                shap_values_3d = []
                for class_shap in shap_values_result:
                    shap_values_3d.append(class_shap.reshape(-1, X_np.shape[1], X_np.shape[2]))
                shap_values_result = shap_values_3d
            else:
                # Single output
                shap_values_result = shap_values_result.reshape(-1, X_np.shape[1], X_np.shape[2])
                
            # Compute feature importance by averaging over time and samples
            if isinstance(shap_values_result, list):
                # Multi-class output - take mean over all classes
                all_class_shap = np.stack(shap_values_result, axis=0)
                feature_importance = np.abs(all_class_shap).mean(axis=0).mean(axis=0).mean(axis=0)
            else:
                feature_importance = np.abs(shap_values_result).mean(axis=0).mean(axis=0)
        else:
            explainer = shap.KernelExplainer(model_fn, X_np[:n_samples])
            shap_values_result = explainer.shap_values(X_np[:min(100, X_np.shape[0])])
            
            # Compute feature importance by averaging over samples
            if isinstance(shap_values_result, list):
                # Multi-class output - take mean over all classes
                all_class_shap = np.stack(shap_values_result, axis=0)
                feature_importance = np.abs(all_class_shap).mean(axis=0).mean(axis=0)
            else:
                feature_importance = np.abs(shap_values_result).mean(axis=0)
        
        # Create result dictionary
        results = {
            "importances": feature_importance,
            "feature_names": feature_names,
            "shap_values": shap_values_result
        }
        
        return results
    
    def plot_feature_importance(self, 
                               importance_results: Dict,
                               title: str = "Feature Importance",
                               sort: bool = True,
                               top_k: Optional[int] = None,
                               figsize: Tuple[int, int] = (10, 6)):
        """
        Plot feature importance.
        
        Args:
            importance_results: Results from importance analysis
            title: Plot title
            sort: Whether to sort features by importance
            top_k: Number of top features to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Extract data
        if "importances_mean" in importance_results:
            # Permutation importance
            importance_values = importance_results["importances_mean"]
            if "importances_std" in importance_results:
                std_values = importance_results["importances_std"]
            else:
                std_values = None
        else:
            # Other methods
            importance_values = importance_results["importances"]
            std_values = None
            
        feature_names = importance_results["feature_names"]
        
        # Create indices for sorting
        indices = np.argsort(importance_values)
        if sort:
            if top_k is not None:
                # Get top-k features
                indices = indices[-top_k:]
            
            # Sort features and values
            feature_names = [feature_names[i] for i in indices]
            importance_values = importance_values[indices]
            if std_values is not None:
                std_values = std_values[indices]
        elif top_k is not None:
            # Just limit to top-k without sorting
            feature_names = feature_names[:top_k]
            importance_values = importance_values[:top_k]
            if std_values is not None:
                std_values = std_values[:top_k]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar plot
        y_pos = np.arange(len(feature_names))
        bars = ax.barh(y_pos, importance_values, align="center")
        
        # Add error bars if available
        if std_values is not None:
            ax.errorbar(
                importance_values, y_pos, 
                xerr=std_values,
                fmt="none", ecolor="black", capsize=5
            )
        
        # Add labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # Highest values at the top
        ax.set_xlabel("Importance")
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        return fig


class AttentionVisualization:
    """
    Attention weights visualization for transformer-based FUSED models.
    
    This class provides methods for extracting and visualizing attention
    weights from transformer-based models in the FUSED framework.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize attention visualizer.
        
        Args:
            model: FUSED model with transformer components
        """
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        self.attention_hooks = []
        self.attention_maps = {}
        
    def _attention_hook(self, module, input, output, layer_name):
        """
        Hook to capture attention weights.
        
        Args:
            module: Module being hooked
            input: Input to the module
            output: Output from the module
            layer_name: Name of the layer
        """
        # Extract attention weights
        if isinstance(output, tuple):
            # Some implementations return (attn_output, attn_weights)
            if len(output) > 1:
                attn_weights = output[1]
                self.attention_maps[layer_name] = attn_weights.detach()
        else:
            # Try to extract from the module's state
            if hasattr(module, "attn_weights"):
                self.attention_maps[layer_name] = module.attn_weights.detach()
    
    def register_hooks(self):
        """
        Register hooks to capture attention weights.
        
        Returns:
            Self for chaining
        """
        # Clear previous hooks
        self.remove_hooks()
        
        # Find all attention modules
        for name, module in self.model.named_modules():
            if "attention" in name.lower() and hasattr(module, "forward"):
                if isinstance(module, nn.MultiheadAttention):
                    # Standard PyTorch MultiheadAttention
                    hook = module.register_forward_hook(
                        lambda mod, inp, out, layer=name: self._attention_hook(mod, inp, out, layer)
                    )
                    self.attention_hooks.append(hook)
                elif hasattr(module, "self_attn"):
                    # Transformer encoder layer
                    hook = module.self_attn.register_forward_hook(
                        lambda mod, inp, out, layer=name: self._attention_hook(mod, inp, out, layer)
                    )
                    self.attention_hooks.append(hook)
        
        return self
    
    def remove_hooks(self):
        """
        Remove registered hooks.
        
        Returns:
            Self for chaining
        """
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks = []
        
        return self
    
    def get_attention_maps(self, 
                          X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for a given input.
        
        Args:
            X: Input tensor
            
        Returns:
            Dictionary mapping layer names to attention maps
        """
        # Clear previous attention maps
        self.attention_maps = {}
        
        # Make sure hooks are registered
        if not self.attention_hooks:
            self.register_hooks()
        
        # Forward pass to capture attention weights
        with torch.no_grad():
            self.model(X)
            
        return self.attention_maps
    
    def plot_attention_heatmap(self, 
                              attention_maps: Dict[str, torch.Tensor],
                              layer_name: str,
                              head_idx: int = 0,
                              sample_idx: int = 0,
                              x_labels: Optional[List[str]] = None,
                              y_labels: Optional[List[str]] = None,
                              title: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8)):
        """
        Plot attention heatmap for a specific layer and attention head.
        
        Args:
            attention_maps: Dictionary of attention maps from get_attention_maps
            layer_name: Name of the layer to visualize
            head_idx: Index of the attention head to visualize
            sample_idx: Index of the sample to visualize
            x_labels: Optional labels for x-axis (target tokens)
            y_labels: Optional labels for y-axis (source tokens)
            title: Optional plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if layer_name not in attention_maps:
            raise ValueError(f"Layer {layer_name} not found in attention maps")
            
        # Get attention weights for the specified layer
        attn_weights = attention_maps[layer_name]
        
        # Extract weights for the specified head and sample
        if len(attn_weights.shape) == 4:
            # [batch_size, num_heads, seq_len, seq_len]
            weights = attn_weights[sample_idx, head_idx].cpu().numpy()
        else:
            # [batch_size, seq_len, seq_len]
            weights = attn_weights[sample_idx].cpu().numpy()
            
        # Create default labels if not provided
        seq_len = weights.shape[0]
        if x_labels is None:
            x_labels = [f"Token {i+1}" for i in range(seq_len)]
        if y_labels is None:
            y_labels = [f"Token {i+1}" for i in range(seq_len)]
            
        # Create default title if not provided
        if title is None:
            title = f"Attention Weights - {layer_name} - Head {head_idx}"
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(weights, cmap="viridis")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_yticklabels(y_labels)
        
        # Add title
        ax.set_title(title)
        
        # Add grid
        ax.grid(False)
        
        # Loop over data dimensions and create text annotations
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                text = ax.text(j, i, f"{weights[i, j]:.2f}",
                              ha="center", va="center", 
                              color="white" if weights[i, j] > 0.5 else "black",
                              fontsize=8)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_attention_flow(self, 
                           attention_maps: Dict[str, torch.Tensor],
                           target_token_idx: int,
                           sample_idx: int = 0,
                           head_idx: Optional[int] = None,
                           token_labels: Optional[List[str]] = None,
                           layer_names: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (12, 8)):
        """
        Plot attention flow across layers for a specific token.
        
        Args:
            attention_maps: Dictionary of attention maps from get_attention_maps
            target_token_idx: Index of the token to analyze
            sample_idx: Index of the sample to visualize
            head_idx: Index of the attention head to visualize (average over heads if None)
            token_labels: Optional labels for tokens
            layer_names: Optional list of layer names to include (in order)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Determine layers to visualize
        if layer_names is None:
            layer_names = sorted(attention_maps.keys())
            
        # Create figure
        fig, axes = plt.subplots(1, len(layer_names), figsize=figsize)
        if len(layer_names) == 1:
            axes = [axes]
            
        # For each layer
        for i, layer_name in enumerate(layer_names):
            if layer_name not in attention_maps:
                continue
                
            # Get attention weights
            attn_weights = attention_maps[layer_name]
            
            # Extract weights for the specified token, sample, and head
            if len(attn_weights.shape) == 4:
                # [batch_size, num_heads, seq_len, seq_len]
                if head_idx is not None:
                    # Use specific head
                    weights = attn_weights[sample_idx, head_idx, target_token_idx].cpu().numpy()
                else:
                    # Average over heads
                    weights = attn_weights[sample_idx, :, target_token_idx].mean(dim=0).cpu().numpy()
            else:
                # [batch_size, seq_len, seq_len]
                weights = attn_weights[sample_idx, target_token_idx].cpu().numpy()
                
            # Create default labels if not provided
            seq_len = weights.shape[0]
            if token_labels is None:
                token_labels = [f"Token {i+1}" for i in range(seq_len)]
                
            # Plot attention weights as bar chart
            axes[i].bar(range(seq_len), weights)
            axes[i].set_xticks(range(seq_len))
            axes[i].set_xticklabels(token_labels, rotation=45, ha="right")
            axes[i].set_ylim(0, 1)
            axes[i].set_title(f"Layer: {layer_name.split('.')[-1]}")
            
            # Highlight the target token
            axes[i].axvline(x=target_token_idx, color="red", linestyle="--", alpha=0.5)
            
            # Add grid
            axes[i].grid(True, linestyle="--", alpha=0.3)
            
        # Set common labels
        fig.text(0.5, 0.01, "Token Position", ha="center", va="center")
        fig.text(0.01, 0.5, "Attention Weight", ha="center", va="center", rotation="vertical")
        
        # Add title
        if head_idx is not None:
            fig.suptitle(f"Attention Flow for Token {target_token_idx+1} - Head {head_idx}", fontsize=14)
        else:
            fig.suptitle(f"Attention Flow for Token {target_token_idx+1} - Average over Heads", fontsize=14)
        
        # Tight layout
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        
        return fig
