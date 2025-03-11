"""
Example model configurations for FUSED.

This module contains predefined model configurations
for various use cases.
"""

from typing import Dict


def get_unimodal_config() -> Dict:
    """
    Get configuration for a unimodal time series model.
    
    Returns:
        Model configuration
    """
    return {
        "encoder": {
            "type": "sequential",
            "input_dim": 64,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "use_cnn": True,
            "kernel_size": 3,
            "positional_encoding": True
        },
        "temporal_model": {
            "type": "multi_scale",
            "input_dim": 128,
            "scales": [1, 2, 4, 8],
            "fusion_method": "attention"
        },
        "output_dim": 128
    }


def get_multimodal_config() -> Dict:
    """
    Get configuration for a multimodal time series model.
    
    Returns:
        Model configuration
    """
    return {
        "encoders": {
            "time_series": {
                "type": "sequential",
                "input_dim": 64,
                "hidden_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1,
                "use_cnn": True,
                "kernel_size": 3,
                "positional_encoding": True
            },
            "spectral": {
                "type": "spectral",
                "input_dim": 64,
                "hidden_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1,
                "freq_dim": 32
            },
            "tabular": {
                "type": "tabular",
                "feature_dims": {
                    "age": 1,
                    "gender": 1,
                    "height": 1,
                    "weight": 1
                },
                "numeric_features": ["age", "height", "weight"],
                "categorical_features": {
                    "gender": 2
                },
                "embedding_dim": 64,
                "hidden_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1
            }
        },
        "fusion": {
            "type": "cross_attention",
            "output_dim": 128,
            "num_heads": 8,
            "dropout": 0.1,
            "use_layernorm": True
        },
        "temporal_model": {
            "type": "neural_ode",
            "latent_dim": 128,
            "hidden_dims": [256, 256],
            "solver": "dopri5"
        },
        "output_dim": 128
    }


def get_pretraining_config() -> Dict:
    """
    Get configuration for pretraining a time series model.
    
    Returns:
        Pretraining configuration
    """
    return {
        "model": get_multimodal_config(),
        "pretraining_tasks": [
            {
                "type": "temporal_contrastive",
                "temperature": 0.07,
                "time_threshold": 0.5,
                "negative_sample_method": "random",
                "use_cross_batch": True
            },
            {
                "type": "multimodal_contrastive",
                "temperature": 0.07,
                "modalities": ["time_series", "spectral", "tabular"],
                "use_cross_batch": True
            },
            {
                "type": "masked_modeling",
                "mask_ratio": 0.15,
                "mask_method": "random",
                "mask_token_value": 0.0
            }
        ],
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "clip_grad_norm": 1.0,
            "num_epochs": 100,
            "patience": 10,
            "scheduler_factor": 0.5,
            "scheduler_patience": 5,
            "output_dir": "./runs/pretraining"
        }
    }
