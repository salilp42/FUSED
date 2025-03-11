"""
Plotting utilities for time series data and model results.

This module provides functions for visualizing time series data,
model predictions, and analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc
from matplotlib.colors import LinearSegmentedColormap

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')


def plot_time_series(data: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                     timestamps: Optional[Union[np.ndarray, torch.Tensor, pd.DatetimeIndex]] = None,
                     feature_names: Optional[List[str]] = None,
                     title: str = 'Time Series Plot',
                     figsize: Tuple[int, int] = (12, 6),
                     alpha: float = 0.8,
                     legend_loc: str = 'best',
                     color_palette: str = 'tab10',
                     highlight_regions: Optional[List[Tuple[int, int, str]]] = None,
                     save_path: Optional[str] = None):
    """
    Plot a single time series with multiple features.
    
    Args:
        data: Time series data
            If numpy array or torch tensor: [seq_len, features] or [batch, seq_len, features]
            If pandas DataFrame: Each row is a time step, columns are features
        timestamps: Optional timestamps for x-axis
        feature_names: Names of features for legend
        title: Plot title
        figsize: Figure size
        alpha: Opacity of lines
        legend_loc: Legend location
        color_palette: Color palette to use
        highlight_regions: List of (start, end, label) tuples to highlight regions
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure and axes objects
    """
    # Convert to numpy array if needed
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
        
    # Convert timestamps if provided
    if timestamps is not None:
        if isinstance(timestamps, torch.Tensor):
            timestamps = timestamps.detach().cpu().numpy()
        elif isinstance(timestamps, pd.DatetimeIndex):
            timestamps = timestamps.values
    
    # Handle different input shapes
    if isinstance(data, pd.DataFrame):
        # Use DataFrame columns as feature names if not provided
        if feature_names is None:
            feature_names = data.columns.tolist()
            
        # Use DataFrame index as timestamps if not provided
        if timestamps is None and isinstance(data.index, pd.DatetimeIndex):
            timestamps = data.index
            
        # Convert to numpy array
        data = data.values
    
    # Handle 3D data (batch dimension)
    if len(data.shape) == 3:
        # Take the first batch for simplicity
        data = data[0]
        if timestamps is not None and len(timestamps.shape) > 1:
            timestamps = timestamps[0]
    
    # Get dimensions
    seq_len, num_features = data.shape
    
    # Create default feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(num_features)]
    
    # Create x-axis values
    if timestamps is None:
        x = np.arange(seq_len)
    else:
        x = timestamps
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get color palette
    colors = sns.color_palette(color_palette, num_features)
    
    # Plot each feature
    for i in range(num_features):
        ax.plot(x, data[:, i], label=feature_names[i], color=colors[i], alpha=alpha)
    
    # Add highlight regions if provided
    if highlight_regions is not None:
        for start, end, label in highlight_regions:
            ax.axvspan(x[start], x[end], alpha=0.2, color='red', label=label)
    
    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    
    # Add legend
    ax.legend(loc=legend_loc)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_multimodal_time_series(modalities: Dict[str, Union[np.ndarray, torch.Tensor]],
                                timestamps: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
                                feature_names: Optional[Dict[str, List[str]]] = None,
                                title: str = 'Multimodal Time Series Plot',
                                figsize: Tuple[int, int] = None,
                                sharex: bool = True,
                                color_palette: str = 'tab10',
                                save_path: Optional[str] = None):
    """
    Plot multiple time series from different modalities.
    
    Args:
        modalities: Dictionary mapping modality names to their data
            Each modality data should be [seq_len, features] or [batch, seq_len, features]
        timestamps: Optional dictionary mapping modality names to their timestamps
        feature_names: Optional dictionary mapping modality names to lists of feature names
        title: Overall plot title
        figsize: Figure size (calculated automatically if None)
        sharex: Whether to share x-axis across subplots
        color_palette: Color palette to use
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure and list of axes objects
    """
    # Get number of modalities
    num_modalities = len(modalities)
    
    # Calculate figure size if not provided
    if figsize is None:
        figsize = (12, 4 * num_modalities)
    
    # Create figure and axes
    fig, axes = plt.subplots(num_modalities, 1, figsize=figsize, sharex=sharex)
    
    # Handle single modality case
    if num_modalities == 1:
        axes = [axes]
    
    # Plot each modality
    for i, (modality_name, modality_data) in enumerate(modalities.items()):
        # Get timestamps for this modality
        modality_timestamps = None if timestamps is None else timestamps.get(modality_name)
        
        # Get feature names for this modality
        modality_feature_names = None if feature_names is None else feature_names.get(modality_name)
        
        # Convert tensor to numpy if needed
        if isinstance(modality_data, torch.Tensor):
            modality_data = modality_data.detach().cpu().numpy()
            
        # Handle 3D data (batch dimension)
        if len(modality_data.shape) == 3:
            modality_data = modality_data[0]
            
        # Get dimensions
        seq_len, num_features = modality_data.shape
        
        # Create default feature names if not provided
        if modality_feature_names is None:
            modality_feature_names = [f'{modality_name} Feature {j+1}' for j in range(num_features)]
        
        # Create x-axis values
        if modality_timestamps is None:
            x = np.arange(seq_len)
        else:
            if isinstance(modality_timestamps, torch.Tensor):
                modality_timestamps = modality_timestamps.detach().cpu().numpy()
            x = modality_timestamps
            if len(x.shape) > 1:
                x = x[0]  # Take first batch if needed
        
        # Get color palette
        colors = sns.color_palette(color_palette, num_features)
        
        # Plot each feature
        for j in range(num_features):
            axes[i].plot(x, modality_data[:, j], 
                         label=modality_feature_names[j], 
                         color=colors[j], 
                         alpha=0.8)
        
        # Set title and labels
        axes[i].set_title(f'{modality_name.capitalize()}', fontsize=12)
        axes[i].set_ylabel('Value', fontsize=10)
        
        # Add legend
        if num_features <= 10:  # Only add legend if not too many features
            axes[i].legend(loc='best')
        
        # Add grid
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Set x-label for bottom subplot only
    axes[-1].set_xlabel('Time', fontsize=12)
    
    # Set overall title
    fig.suptitle(title, fontsize=14)
    
    # Tight layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust for main title
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_feature_importance(importance_scores: Union[np.ndarray, torch.Tensor, List[float]],
                            feature_names: Optional[List[str]] = None,
                            title: str = 'Feature Importance',
                            figsize: Tuple[int, int] = (10, 6),
                            color: str = 'skyblue',
                            horizontal: bool = True,
                            save_path: Optional[str] = None):
    """
    Plot feature importance scores.
    
    Args:
        importance_scores: Feature importance scores
        feature_names: Names of features
        title: Plot title
        figsize: Figure size
        color: Bar color
        horizontal: Whether to make horizontal bar plot
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure and axes objects
    """
    # Convert to numpy if needed
    if isinstance(importance_scores, torch.Tensor):
        importance_scores = importance_scores.detach().cpu().numpy()
    elif isinstance(importance_scores, list):
        importance_scores = np.array(importance_scores)
    
    # Get number of features
    num_features = len(importance_scores)
    
    # Create default feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(num_features)]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort features by importance
    sorted_indices = np.argsort(importance_scores)
    if horizontal:
        sorted_indices = sorted_indices[::-1]  # Reverse for horizontal plot
    
    sorted_scores = importance_scores[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    # Create bar plot
    if horizontal:
        ax.barh(range(num_features), sorted_scores, color=color)
        ax.set_yticks(range(num_features))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance Score')
    else:
        ax.bar(range(num_features), sorted_scores, color=color)
        ax.set_xticks(range(num_features))
        ax.set_xticklabels(sorted_names, rotation=45, ha='right')
        ax.set_ylabel('Importance Score')
    
    # Set title
    ax.set_title(title, fontsize=14)
    
    # Add grid
    ax.grid(True, linestyle='--', axis='x' if horizontal else 'y', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_embedding_space(embeddings: Union[np.ndarray, torch.Tensor],
                         labels: Optional[Union[np.ndarray, torch.Tensor, List[int]]] = None,
                         method: str = 'tsne',
                         feature_names: Optional[List[str]] = None,
                         title: str = 'Embedding Space',
                         figsize: Tuple[int, int] = (10, 8),
                         marker_size: int = 50,
                         alpha: float = 0.7,
                         cmap: str = 'viridis',
                         annotate: bool = False,
                         random_state: int = 42,
                         save_path: Optional[str] = None):
    """
    Plot embeddings in 2D space using dimensionality reduction.
    
    Args:
        embeddings: Embeddings array of shape [num_samples, embedding_dim]
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
        feature_names: Optional names for samples (for annotation)
        title: Plot title
        figsize: Figure size
        marker_size: Size of markers
        alpha: Opacity of markers
        cmap: Colormap for labels
        annotate: Whether to annotate points with feature names
        random_state: Random state for reproducibility
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure and axes objects
    """
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        elif isinstance(labels, list):
            labels = np.array(labels)
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(embeddings)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(embeddings)
    elif method.lower() == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=random_state)
            reduced_embeddings = reducer.fit_transform(embeddings)
        except ImportError:
            print("UMAP not installed. Defaulting to t-SNE.")
            reducer = TSNE(n_components=2, random_state=random_state)
            reduced_embeddings = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne', 'pca', or 'umap'.")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot embeddings
    if labels is not None:
        # Get unique labels
        unique_labels = np.unique(labels)
        
        # If labels are continuous, use scatter with colorbar
        if len(unique_labels) > 10:
            scatter = ax.scatter(reduced_embeddings[:, 0], 
                              reduced_embeddings[:, 1],
                              c=labels, 
                              cmap=cmap,
                              s=marker_size, 
                              alpha=alpha)
            plt.colorbar(scatter, ax=ax, label='Label')
        else:
            # If labels are discrete, use different colors
            for label in unique_labels:
                mask = labels == label
                ax.scatter(reduced_embeddings[mask, 0], 
                          reduced_embeddings[mask, 1],
                          label=f'Class {label}',
                          s=marker_size, 
                          alpha=alpha)
            ax.legend(title='Classes')
    else:
        ax.scatter(reduced_embeddings[:, 0], 
                  reduced_embeddings[:, 1],
                  s=marker_size, 
                  alpha=alpha)
    
    # Annotate points if requested
    if annotate and feature_names is not None:
        for i, txt in enumerate(feature_names):
            ax.annotate(txt, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                      fontsize=8, alpha=0.7)
    
    # Set title and labels
    ax.set_title(f'{title} ({method.upper()})', fontsize=14)
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_confusion_matrix(y_true: Union[np.ndarray, torch.Tensor, List[int]],
                         y_pred: Union[np.ndarray, torch.Tensor, List[int]],
                         class_names: Optional[List[str]] = None,
                         normalize: bool = False,
                         title: str = 'Confusion Matrix',
                         figsize: Tuple[int, int] = (8, 6),
                         cmap: str = 'Blues',
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize by row
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure and axes objects
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    elif isinstance(y_true, list):
        y_true = np.array(y_true)
        
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    elif isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Get number of classes
    n_classes = cm.shape[0]
    
    # Create default class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    plt.colorbar(im, ax=ax)
    
    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Set ticks
    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            text_color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color=text_color, fontsize=10)
    
    # Add grid
    ax.grid(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_roc_curve(y_true: Union[np.ndarray, torch.Tensor, List[int]],
                  y_score: Union[np.ndarray, torch.Tensor, List[float]],
                  class_names: Optional[List[str]] = None,
                  title: str = 'ROC Curve',
                  figsize: Tuple[int, int] = (8, 6),
                  lw: float = 2.0,
                  save_path: Optional[str] = None):
    """
    Plot ROC curve for binary or multiclass classification.
    
    Args:
        y_true: True labels (one-hot or class indices)
        y_score: Predicted scores or probabilities
        class_names: Names of classes
        title: Plot title
        figsize: Figure size
        lw: Line width
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure and axes objects
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    elif isinstance(y_true, list):
        y_true = np.array(y_true)
        
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()
    elif isinstance(y_score, list):
        y_score = np.array(y_score)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if binary or multiclass
    if len(y_score.shape) == 1 or y_score.shape[1] == 1:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, lw=lw,
               label=f'ROC curve (area = {roc_auc:.3f})')
        
    else:
        # Multiclass classification
        n_classes = y_score.shape[1]
        
        # Create default class names if not provided
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
            
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Binarize y_true if needed
        if len(y_true.shape) == 1:
            y_true_bin = np.zeros((len(y_true), n_classes))
            for i in range(len(y_true)):
                y_true_bin[i, y_true[i]] = 1
            y_true = y_true_bin
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot ROC curve for each class
            ax.plot(fpr[i], tpr[i], lw=lw,
                   label=f'{class_names[i]} (area = {roc_auc[i]:.3f})')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    
    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    
    # Set limits and aspect
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_aspect('equal')
    
    # Add legend
    ax.legend(loc="lower right")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_forecasting_results(actual: Union[np.ndarray, torch.Tensor],
                            predicted: Union[np.ndarray, torch.Tensor],
                            timestamps: Optional[Union[np.ndarray, torch.Tensor, pd.DatetimeIndex]] = None,
                            feature_names: Optional[List[str]] = None,
                            title: str = 'Forecasting Results',
                            figsize: Tuple[int, int] = None,
                            alpha: float = 0.8,
                            confint: Optional[Union[np.ndarray, torch.Tensor]] = None,
                            num_samples: int = 5,
                            save_path: Optional[str] = None):
    """
    Plot forecasting results comparing actual and predicted values.
    
    Args:
        actual: Actual values [batch, horizon, features] or [horizon, features]
        predicted: Predicted values [batch, horizon, features] or [horizon, features]
        timestamps: Optional timestamps for x-axis
        feature_names: Names of features for subplot titles
        title: Overall plot title
        figsize: Figure size (calculated automatically if None)
        alpha: Opacity of lines and confidence intervals
        confint: Optional confidence intervals with shape [batch, horizon, features, 2]
                where last dim is (lower, upper)
        num_samples: Number of samples to plot (if batch dimension exists)
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure and list of axes objects
    """
    # Convert to numpy if needed
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if confint is not None and isinstance(confint, torch.Tensor):
        confint = confint.detach().cpu().numpy()
        
    # Convert timestamps if provided
    if timestamps is not None:
        if isinstance(timestamps, torch.Tensor):
            timestamps = timestamps.detach().cpu().numpy()
        elif isinstance(timestamps, pd.DatetimeIndex):
            timestamps = timestamps.values
    
    # Handle different input shapes
    if len(actual.shape) == 2:
        # No batch dimension
        horizon, num_features = actual.shape
        has_batch = False
    else:
        # Has batch dimension
        batch_size, horizon, num_features = actual.shape
        has_batch = True
    
    # Create default feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(num_features)]
    
    # Create default figure size if not provided
    if figsize is None:
        figsize = (12, 4 * min(num_features, 5))  # Limit to 5 features max by default
    
    # Create figure and axes
    fig, axes = plt.subplots(min(num_features, 5), 1, figsize=figsize, sharex=True)
    
    # Handle single feature case
    if num_features == 1:
        axes = [axes]
    elif num_features > 5:
        print(f"Plotting first 5 features out of {num_features}.")
    
    # Create x-axis values
    if timestamps is None:
        x = np.arange(horizon)
    else:
        x = timestamps
        if has_batch and len(x.shape) > 1:
            x = x[0]  # Take first batch
    
    # Plot each feature
    for i, feature_name in enumerate(feature_names[:min(num_features, 5)]):
        if has_batch:
            # Plot multiple samples with different transparencies
            for j in range(min(batch_size, num_samples)):
                # Actual values
                axes[i].plot(x, actual[j, :, i], 
                            color='blue', 
                            alpha=alpha * (1 - 0.5 * j / num_samples),
                            linestyle='--',
                            linewidth=1)
                
                # Predicted values
                axes[i].plot(x, predicted[j, :, i], 
                             color='red', 
                             alpha=alpha * (1 - 0.5 * j / num_samples),
                             linewidth=1)
                
                # Confidence intervals if provided
                if confint is not None:
                    axes[i].fill_between(x, 
                                       confint[j, :, i, 0], 
                                       confint[j, :, i, 1],
                                       color='red', 
                                       alpha=0.1 * (1 - 0.5 * j / num_samples))
        else:
            # Plot single sample
            axes[i].plot(x, actual[:, i], color='blue', alpha=alpha, label='Actual', linestyle='--')
            axes[i].plot(x, predicted[:, i], color='red', alpha=alpha, label='Predicted')
            
            # Confidence intervals if provided
            if confint is not None:
                axes[i].fill_between(x, 
                                   confint[:, i, 0], 
                                   confint[:, i, 1],
                                   color='red', 
                                   alpha=0.1)
        
        # Set title and labels
        axes[i].set_title(feature_name, fontsize=12)
        axes[i].set_ylabel('Value', fontsize=10)
        
        # Add legend (only for first feature to avoid clutter)
        if i == 0:
            axes[i].legend(['Actual', 'Predicted'])
        
        # Add grid
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Set x-label for bottom subplot only
    axes[-1].set_xlabel('Time', fontsize=12)
    
    # Set overall title
    fig.suptitle(title, fontsize=14)
    
    # Tight layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust for main title
    
    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes
