"""
Experiment tracking utilities for FUSED.

This module provides integration with experiment tracking tools like
MLflow and Weights & Biases.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Union, Any
import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment trackers.
    
    This class defines the interface for all experiment trackers.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the experiment tracker.
        
        Args:
            config: Configuration dictionary for the tracker
        """
        self.config = config
        
    @abstractmethod
    def start_run(self, run_name: Optional[str] = None):
        """
        Start a new experiment run.
        
        Args:
            run_name: Optional name for the run
        """
        pass
        
    @abstractmethod
    def end_run(self):
        """
        End the current experiment run.
        """
        pass
        
    @abstractmethod
    def log_params(self, params: Dict):
        """
        Log parameters for the current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        pass
        
    @abstractmethod
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """
        Log metrics for the current run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        pass
        
    @abstractmethod
    def log_artifact(self, local_path: str):
        """
        Log an artifact (file) for the current run.
        
        Args:
            local_path: Path to the artifact
        """
        pass
        
    @abstractmethod
    def log_figure(self, figure: plt.Figure, filename: str):
        """
        Log a matplotlib figure for the current run.
        
        Args:
            figure: Matplotlib figure
            filename: Name for the saved figure
        """
        pass
        

class MLflowTracker(ExperimentTracker):
    """
    Experiment tracker using MLflow.
    
    This tracker provides integration with MLflow for experiment tracking.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the MLflow tracker.
        
        Args:
            config: Configuration dictionary with the following keys:
                tracking_uri: URI for the MLflow tracking server
                experiment_name: Name of the experiment
                tags: Optional tags for the experiment
        """
        super().__init__(config)
        
        # Import MLflow
        try:
            import mlflow
            self.mlflow = mlflow
        except ImportError:
            raise ImportError(
                "MLflow not installed. Install with: pip install mlflow"
            )
            
        # Set tracking URI if provided
        tracking_uri = config.get("tracking_uri")
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
            
        # Set experiment
        experiment_name = config.get("experiment_name", "FUSED")
        self.mlflow.set_experiment(experiment_name)
        
        # Set tags if provided
        self.tags = config.get("tags", {})
        
    def start_run(self, run_name: Optional[str] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
        """
        self.mlflow.start_run(run_name=run_name, tags=self.tags)
        
    def end_run(self):
        """
        End the current MLflow run.
        """
        self.mlflow.end_run()
        
    def log_params(self, params: Dict):
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        # Handle nested dictionaries by flattening
        flattened_params = self._flatten_dict(params)
        self.mlflow.log_params(flattened_params)
        
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        # Handle nested dictionaries by flattening
        flattened_metrics = self._flatten_dict(metrics)
        
        # Convert numpy and torch types to Python types
        converted_metrics = {}
        for key, value in flattened_metrics.items():
            if isinstance(value, (np.number, np.ndarray)):
                converted_metrics[key] = value.item() if np.isscalar(value) else value.tolist()
            elif isinstance(value, torch.Tensor):
                converted_metrics[key] = value.item() if value.numel() == 1 else value.tolist()
            else:
                converted_metrics[key] = value
        
        self.mlflow.log_metrics(converted_metrics, step=step)
        
    def log_artifact(self, local_path: str):
        """
        Log an artifact to MLflow.
        
        Args:
            local_path: Path to the artifact
        """
        self.mlflow.log_artifact(local_path)
        
    def log_figure(self, figure: plt.Figure, filename: str):
        """
        Log a matplotlib figure to MLflow.
        
        Args:
            figure: Matplotlib figure
            filename: Name for the saved figure
        """
        # Save figure to temporary file
        temp_path = f"temp_{filename}"
        figure.savefig(temp_path)
        
        # Log artifact
        self.log_artifact(temp_path)
        
        # Remove temporary file
        os.remove(temp_path)
        
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """
        Flatten nested dictionaries for MLflow logging.
        
        Args:
            d: Dictionary to flatten
            parent_key: Key of parent dictionary
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class WandbTracker(ExperimentTracker):
    """
    Experiment tracker using Weights & Biases.
    
    This tracker provides integration with Weights & Biases for experiment tracking.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Weights & Biases tracker.
        
        Args:
            config: Configuration dictionary with the following keys:
                project: Name of the project
                entity: Name of the entity (team or username)
                tags: Optional tags for the experiment
                config: Optional configuration dictionary
        """
        super().__init__(config)
        
        # Import wandb
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "Weights & Biases not installed. Install with: pip install wandb"
            )
            
        # Extract configuration
        self.project = config.get("project", "FUSED")
        self.entity = config.get("entity")
        self.tags = config.get("tags", [])
        self.run_config = config.get("config", {})
        
    def start_run(self, run_name: Optional[str] = None):
        """
        Start a new Weights & Biases run.
        
        Args:
            run_name: Optional name for the run
        """
        self.wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            tags=self.tags,
            config=self.run_config
        )
        
    def end_run(self):
        """
        End the current Weights & Biases run.
        """
        self.wandb.finish()
        
    def log_params(self, params: Dict):
        """
        Log parameters to Weights & Biases.
        
        Args:
            params: Dictionary of parameters to log
        """
        # Update config with params
        self.wandb.config.update(params)
        
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """
        Log metrics to Weights & Biases.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        self.wandb.log(metrics, step=step)
        
    def log_artifact(self, local_path: str):
        """
        Log an artifact to Weights & Biases.
        
        Args:
            local_path: Path to the artifact
        """
        self.wandb.save(local_path)
        
    def log_figure(self, figure: plt.Figure, filename: str):
        """
        Log a matplotlib figure to Weights & Biases.
        
        Args:
            figure: Matplotlib figure
            filename: Name for the saved figure (not used)
        """
        self.wandb.log({filename: self.wandb.Image(figure)})


# Factory function to create trackers
def create_tracker(tracker_type: str, config: Dict) -> ExperimentTracker:
    """
    Create an experiment tracker of the specified type.
    
    Args:
        tracker_type: Type of tracker ('mlflow' or 'wandb')
        config: Configuration dictionary for the tracker
        
    Returns:
        ExperimentTracker instance
    """
    if tracker_type.lower() == 'mlflow':
        return MLflowTracker(config)
    elif tracker_type.lower() == 'wandb':
        return WandbTracker(config)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")
