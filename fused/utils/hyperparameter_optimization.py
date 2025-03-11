"""
Hyperparameter optimization utilities for FUSED.

This module provides tools for hyperparameter optimization using
common libraries like Optuna and Ray Tune.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
from abc import ABC, abstractmethod
import logging


class HyperOptimizer(ABC):
    """
    Abstract base class for hyperparameter optimizers.
    
    This class defines the interface for all hyperparameter optimizers.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            config: Configuration dictionary for the optimizer
        """
        self.config = config
        
    @abstractmethod
    def optimize(self, objective_function: Callable, search_space: Dict) -> Tuple[Dict, float]:
        """
        Run hyperparameter optimization.
        
        Args:
            objective_function: Function to minimize/maximize
            search_space: Dictionary defining the search space
            
        Returns:
            Tuple of (best_params, best_value)
        """
        pass


class OptunaOptimizer(HyperOptimizer):
    """
    Hyperparameter optimizer using Optuna.
    
    This optimizer provides integration with Optuna for hyperparameter tuning.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Optuna optimizer.
        
        Args:
            config: Configuration dictionary with the following keys:
                direction: Optimization direction ('minimize' or 'maximize')
                n_trials: Number of trials
                timeout: Optional timeout in seconds
                study_name: Optional name for the study
                storage: Optional storage URL
                sampler: Optional sampler configuration
        """
        super().__init__(config)
        
        # Import Optuna
        try:
            import optuna
            self.optuna = optuna
        except ImportError:
            raise ImportError(
                "Optuna not installed. Install with: pip install optuna"
            )
            
        # Extract configuration
        self.direction = config.get("direction", "minimize")
        self.n_trials = config.get("n_trials", 100)
        self.timeout = config.get("timeout")
        self.study_name = config.get("study_name", "fused_hyperopt")
        self.storage = config.get("storage")
        
        # Configure sampler
        sampler_config = config.get("sampler", {})
        sampler_type = sampler_config.get("type", "tpe")
        if sampler_type.lower() == "tpe":
            self.sampler = optuna.samplers.TPESampler(**sampler_config.get("params", {}))
        elif sampler_type.lower() == "random":
            self.sampler = optuna.samplers.RandomSampler(**sampler_config.get("params", {}))
        elif sampler_type.lower() == "cmaes":
            self.sampler = optuna.samplers.CmaEsSampler(**sampler_config.get("params", {}))
        else:
            self.sampler = optuna.samplers.TPESampler()
            
    def optimize(self, objective_function: Callable, search_space: Dict) -> Tuple[Dict, float]:
        """
        Run hyperparameter optimization with Optuna.
        
        Args:
            objective_function: Function to minimize/maximize.
                Should accept a trial object and return a scalar value.
            search_space: Dictionary defining the search space.
                Not directly used in the function but needed for interface consistency.
            
        Returns:
            Tuple of (best_params, best_value)
        """
        # Create a study
        study = self.optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage,
            sampler=self.sampler,
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(
            objective_function,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Return best parameters and value
        return study.best_params, study.best_value


class RayTuneOptimizer(HyperOptimizer):
    """
    Hyperparameter optimizer using Ray Tune.
    
    This optimizer provides integration with Ray Tune for hyperparameter tuning.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Ray Tune optimizer.
        
        Args:
            config: Configuration dictionary with the following keys:
                mode: Optimization mode ('min' or 'max')
                num_samples: Number of trials
                max_concurrent_trials: Maximum number of concurrent trials
                time_budget_s: Optional time budget in seconds
                scheduler: Optional scheduler configuration
                search_alg: Optional search algorithm configuration
                resources_per_trial: Optional resources per trial
        """
        super().__init__(config)
        
        # Import Ray Tune
        try:
            from ray import tune
            self.tune = tune
        except ImportError:
            raise ImportError(
                "Ray Tune not installed. Install with: pip install ray[tune]"
            )
            
        # Extract configuration
        self.mode = config.get("mode", "min")
        self.num_samples = config.get("num_samples", 100)
        self.max_concurrent_trials = config.get("max_concurrent_trials", 1)
        self.time_budget_s = config.get("time_budget_s")
        
        # Configure scheduler
        scheduler_config = config.get("scheduler", {})
        self.scheduler = self._create_scheduler(scheduler_config)
        
        # Configure search algorithm
        search_alg_config = config.get("search_alg", {})
        self.search_alg = self._create_search_alg(search_alg_config)
        
        # Resource configuration
        self.resources_per_trial = config.get("resources_per_trial", {})
        
    def optimize(self, objective_function: Callable, search_space: Dict) -> Tuple[Dict, float]:
        """
        Run hyperparameter optimization with Ray Tune.
        
        Args:
            objective_function: Function to minimize/maximize.
                Should accept a config dictionary and report results using tune.report.
            search_space: Dictionary defining the search space.
                Should be compatible with Ray Tune's search space specification.
            
        Returns:
            Tuple of (best_params, best_value)
        """
        # Convert objective function to Ray Tune format
        def tune_objective(config):
            return objective_function(config)
        
        # Create experiment name
        experiment_name = self.config.get("experiment_name", "fused_hyperopt")
        
        # Run optimization
        analysis = self.tune.run(
            tune_objective,
            config=search_space,
            num_samples=self.num_samples,
            time_budget_s=self.time_budget_s,
            resources_per_trial=self.resources_per_trial,
            max_concurrent_trials=self.max_concurrent_trials,
            scheduler=self.scheduler,
            search_alg=self.search_alg,
            mode=self.mode,
            name=experiment_name,
            verbose=1
        )
        
        # Get best trial
        best_trial = analysis.get_best_trial(metric="objective", mode=self.mode)
        
        # Return best parameters and value
        return best_trial.config, best_trial.last_result["objective"]
    
    def _create_scheduler(self, scheduler_config: Dict) -> Optional[Any]:
        """
        Create a scheduler based on the configuration.
        
        Args:
            scheduler_config: Scheduler configuration
            
        Returns:
            Scheduler instance or None
        """
        scheduler_type = scheduler_config.get("type")
        if scheduler_type is None:
            return None
            
        params = scheduler_config.get("params", {})
        
        if scheduler_type.lower() == "asha":
            return self.tune.schedulers.ASHAScheduler(**params)
        elif scheduler_type.lower() == "hyperband":
            return self.tune.schedulers.HyperBandScheduler(**params)
        elif scheduler_type.lower() == "pbt":
            return self.tune.schedulers.PopulationBasedTraining(**params)
        else:
            logging.warning(f"Unknown scheduler type: {scheduler_type}. Using None.")
            return None
            
    def _create_search_alg(self, search_alg_config: Dict) -> Optional[Any]:
        """
        Create a search algorithm based on the configuration.
        
        Args:
            search_alg_config: Search algorithm configuration
            
        Returns:
            Search algorithm instance or None
        """
        search_alg_type = search_alg_config.get("type")
        if search_alg_type is None:
            return None
            
        params = search_alg_config.get("params", {})
        
        if search_alg_type.lower() == "hyperopt":
            from ray.tune.search.hyperopt import HyperOptSearch
            return HyperOptSearch(**params)
        elif search_alg_type.lower() == "bayesopt":
            from ray.tune.search.bayesopt import BayesOptSearch
            return BayesOptSearch(**params)
        elif search_alg_type.lower() == "bohb":
            from ray.tune.search.bohb import TuneBOHB
            return TuneBOHB(**params)
        else:
            logging.warning(f"Unknown search algorithm type: {search_alg_type}. Using None.")
            return None


class HyperparameterTuner:
    """
    High-level wrapper for hyperparameter tuning.
    
    This class provides a simplified interface for hyperparameter tuning
    with built-in cross-validation and result tracking.
    """
    
    def __init__(self, 
                 optimizer_type: str, 
                 optimizer_config: Dict,
                 cv_folds: int = 5,
                 tracker: Optional[Any] = None):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            optimizer_type: Type of optimizer ('optuna' or 'raytune')
            optimizer_config: Configuration for the optimizer
            cv_folds: Number of cross-validation folds
            tracker: Optional experiment tracker
        """
        # Create optimizer
        if optimizer_type.lower() == "optuna":
            self.optimizer = OptunaOptimizer(optimizer_config)
        elif optimizer_type.lower() == "raytune":
            self.optimizer = RayTuneOptimizer(optimizer_config)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
        self.cv_folds = cv_folds
        self.tracker = tracker
        
    def tune(self, 
             model_class: Any, 
             dataset,
             search_space: Dict,
             eval_metric: str = "val_loss",
             direction: str = "minimize",
             **kwargs) -> Tuple[Dict, Any]:
        """
        Run hyperparameter tuning with cross-validation.
        
        Args:
            model_class: Model class to instantiate
            dataset: Dataset to use for training/validation
            search_space: Hyperparameter search space
            eval_metric: Metric to optimize
            direction: Optimization direction ('minimize' or 'maximize')
            **kwargs: Additional arguments for model training
            
        Returns:
            Tuple of (best_params, best_model)
        """
        from sklearn.model_selection import KFold
        import torch
        
        # Start tracking if available
        if self.tracker is not None:
            self.tracker.start_run("hyperparameter_tuning")
            self.tracker.log_params({"search_space": search_space})
        
        # Create cross-validation splits
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Define objective function for optimization
        def objective(trial_or_config):
            # Convert trial to config for Optuna
            if hasattr(trial_or_config, 'suggest_float'):
                # This is an Optuna trial
                config = self._sample_from_search_space(search_space, trial_or_config)
            else:
                # This is a Ray Tune config
                config = trial_or_config
            
            # Track scores across folds
            fold_scores = []
            
            # Run cross-validation
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset)):
                # Create train/val splits
                train_subset = torch.utils.data.Subset(dataset, train_idx)
                val_subset = torch.utils.data.Subset(dataset, val_idx)
                
                # Create model with the trial parameters
                model = model_class(config)
                
                # Train model
                train_results = model.fit(
                    train_subset, 
                    validation_data=val_subset,
                    **kwargs
                )
                
                # Get validation score
                val_score = train_results.get(eval_metric, float('inf') if direction == 'minimize' else float('-inf'))
                fold_scores.append(val_score)
            
            # Aggregate scores across folds
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            # Log results if tracker is available
            if self.tracker is not None:
                self.tracker.log_metrics({
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "config": config
                })
            
            # Return mean score (Optuna minimizes by default)
            if direction == 'maximize':
                return -mean_score  # Negate for maximization
            return mean_score
        
        # Run optimization
        best_params, best_value = self.optimizer.optimize(objective, search_space)
        
        # Log best results if tracker is available
        if self.tracker is not None:
            self.tracker.log_metrics({
                "best_score": -best_value if direction == 'maximize' else best_value,
                "best_params": best_params
            })
            self.tracker.end_run()
        
        # Train best model on full dataset
        best_model = model_class(best_params)
        best_model.fit(dataset, **kwargs)
        
        return best_params, best_model
    
    def _sample_from_search_space(self, search_space: Dict, trial) -> Dict:
        """
        Sample parameters from search space using Optuna trial.
        
        Args:
            search_space: Search space definition
            trial: Optuna trial
            
        Returns:
            Parameter configuration
        """
        config = {}
        for param_name, param_config in search_space.items():
            param_type = param_config.get("type", "float")
            
            if param_type == "float":
                low = param_config.get("low", 0.0)
                high = param_config.get("high", 1.0)
                log = param_config.get("log", False)
                config[param_name] = trial.suggest_float(param_name, low, high, log=log)
                
            elif param_type == "int":
                low = param_config.get("low", 0)
                high = param_config.get("high", 10)
                log = param_config.get("log", False)
                config[param_name] = trial.suggest_int(param_name, low, high, log=log)
                
            elif param_type == "categorical":
                choices = param_config.get("choices", [])
                config[param_name] = trial.suggest_categorical(param_name, choices)
                
            elif param_type == "discrete_uniform":
                low = param_config.get("low", 0.0)
                high = param_config.get("high", 1.0)
                q = param_config.get("q", 0.1)
                config[param_name] = trial.suggest_discrete_uniform(param_name, low, high, q)
        
        return config
