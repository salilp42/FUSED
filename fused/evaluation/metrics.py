"""
Evaluation metrics for FUSED.

This module implements common evaluation metrics for
time series forecasting, classification, and regression tasks.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Any
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def classification_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute common classification metrics.
    
    Args:
        predictions: Predicted class probabilities or labels
        targets: Ground truth labels
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Handle probabilities vs. class labels
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Convert probabilities to class labels
        pred_labels = np.argmax(predictions, axis=1)
        pred_probs = predictions
    else:
        # Already class labels
        pred_labels = predictions.flatten()
        pred_probs = None
    
    # Ensure targets are flattened
    targets = targets.flatten()
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(targets, pred_labels),
        'f1': f1_score(targets, pred_labels, average=average),
        'precision': precision_score(targets, pred_labels, average=average),
        'recall': recall_score(targets, pred_labels, average=average)
    }
    
    # Add AUC if probabilities are available
    if pred_probs is not None:
        # For binary classification
        if pred_probs.shape[1] == 2:
            metrics['auc'] = roc_auc_score(targets, pred_probs[:, 1])
        # For multi-class
        else:
            try:
                metrics['auc'] = roc_auc_score(targets, pred_probs, multi_class='ovr')
            except ValueError:
                # Skip AUC if it can't be computed (e.g., missing classes)
                pass
    
    return metrics


def regression_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray]
) -> Dict[str, float]:
    """
    Compute common regression metrics.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Compute metrics
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
    
    return metrics


def forecasting_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    horizon: int = 1
) -> Dict[str, float]:
    """
    Compute metrics for time series forecasting.
    
    Args:
        predictions: Predicted values [batch_size, horizon, features]
        targets: Ground truth values [batch_size, horizon, features]
        horizon: Forecasting horizon
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Compute metrics for each step in the horizon
    horizon_metrics = []
    
    for h in range(horizon):
        # Extract predictions and targets for current horizon step
        if h < predictions.shape[1] and h < targets.shape[1]:
            h_preds = predictions[:, h]
            h_targets = targets[:, h]
            
            # Compute metrics
            mae = mean_absolute_error(h_targets, h_preds)
            mse = mean_squared_error(h_targets, h_preds)
            rmse = np.sqrt(mse)
            
            horizon_metrics.append({
                'horizon': h + 1,
                'mae': mae,
                'mse': mse,
                'rmse': rmse
            })
    
    # Compute aggregate metrics
    overall_mae = mean_absolute_error(targets.reshape(-1), predictions.reshape(-1))
    overall_mse = mean_squared_error(targets.reshape(-1), predictions.reshape(-1))
    overall_rmse = np.sqrt(overall_mse)
    
    metrics = {
        'overall_mae': overall_mae,
        'overall_mse': overall_mse,
        'overall_rmse': overall_rmse,
        'horizon_metrics': horizon_metrics
    }
    
    return metrics


def multivariate_forecasting_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    feature_names: List[str] = None
) -> Dict[str, Any]:
    """
    Compute metrics for multivariate time series forecasting.
    
    Args:
        predictions: Predicted values [batch_size, horizon, features]
        targets: Ground truth values [batch_size, horizon, features]
        feature_names: Names of features
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Default feature names if not provided
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(predictions.shape[2])]
    
    # Compute overall metrics
    overall_metrics = forecasting_metrics(predictions, targets, horizon=predictions.shape[1])
    
    # Compute metrics for each feature
    feature_metrics = {}
    
    for f in range(len(feature_names)):
        if f < predictions.shape[2] and f < targets.shape[2]:
            f_preds = predictions[:, :, f]
            f_targets = targets[:, :, f]
            
            # Compute metrics for this feature
            f_metrics = forecasting_metrics(
                f_preds[:, :, np.newaxis],
                f_targets[:, :, np.newaxis],
                horizon=predictions.shape[1]
            )
            
            feature_metrics[feature_names[f]] = f_metrics
    
    # Combine metrics
    metrics = {
        'overall': overall_metrics,
        'features': feature_metrics
    }
    
    return metrics
