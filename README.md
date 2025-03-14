# FUSED: Foundation-based Unified Sequential Embedding Design


## Overview

FUSED is an open-source framework for multimodal time series modeling that provides a flexible and extensible architecture for working with sequential data of various types. It is designed to serve as a foundation model framework for researchers and practitioners working with temporal data across different domains.

### Key Features

- **Modality-Agnostic Design**: Work with any type of time series data regardless of source
- **Flexible Multi-Scale Processing**: Capture patterns at different temporal resolutions
- **Neural ODE-based Trajectories**: Model continuous-time dynamics with state-of-the-art methods
- **Self-Supervised Learning**: Multiple pretraining objectives for representation learning
- **Modular Fusion Strategies**: Easily configurable cross-modal fusion techniques
- **Extensible Architecture**: Add custom encoders, fusion methods, and models
- **Advanced Utilities**: Built-in support for hyperparameter optimization, model interpretability, and deployment
- **Comprehensive Documentation**: Extensive guides and API references
- **Example Implementations**: Ready-to-use examples for common time series tasks

## Installation

```bash
# Install from PyPI
pip install fused

# Install from source
git clone https://github.com/fused-project/fused.git
cd fused
pip install -e .

# With advanced utilities
pip install fused[all]  # Install all dependencies
pip install fused[hyperopt,interpretability]  # Install specific components
```

## Quick Start

```python
import torch
from fused import TemporalModel

# Create a model with default configuration
model = TemporalModel()

# Or with custom configuration
model = TemporalModel(
    modalities=["timeseries_a", "timeseries_b", "static"],
    fusion_strategy="cross_attention",
    sequence_length=100
)

# Train with your data
model.fit(train_loader, valid_loader)

# Make predictions
predictions = model.predict(test_loader)

# Extract embeddings
embeddings = model.embed(test_loader)
```

## Examples

FUSED includes a variety of examples to help you get started:

- **Basic Time Series Classification**: Simple example of time series classification
- **Multimodal Fusion**: Combining different types of time series data
- **Transfer Learning**: Using pretrained encoders for downstream tasks
- **Continuous Trajectories**: Modeling smooth trajectories with Neural ODEs
- **Custom Components**: Creating your own encoders and fusion strategies
- **Hyperparameter Optimization**: Automatically tuning model parameters
- **Model Interpretability**: Understanding feature importance and attention patterns
- **Model Deployment**: Exporting and serving models in production environments

## Framework Architecture

FUSED is built around a flexible architecture that allows for easy customization:

1. **Data Processing**: Standardized interfaces for loading and processing time series data
2. **Encoder Modules**: Specialized modules for different types of time series
3. **Fusion Layer**: Combining representations from different modalities
4. **Trajectory Modeling**: Continuous-time modeling of temporal progression
5. **Self-Supervised Learning**: Built-in objectives for representation learning
6. **Evaluation Tools**: Comprehensive evaluation metrics and visualization

## Advanced Utilities

FUSED provides several advanced utilities to enhance your workflow:

### Hyperparameter Optimization

Automatically find the optimal model configuration using state-of-the-art optimization libraries:

```python
from fused.utils import HyperparameterTuner

# Define search space
search_space = {
    "hidden_dim": {"type": "int", "low": 32, "high": 256},
    "dropout_rate": {"type": "float", "low": 0.0, "high": 0.5}
}

# Create tuner
tuner = HyperparameterTuner(
    optimizer_type="optuna",
    optimizer_config={"n_trials": 50}
)

# Find optimal parameters
best_params, best_model = tuner.tune(
    model_class=MyModel,
    dataset=train_dataset,
    search_space=search_space
)
```

### Model Interpretability

Analyze and visualize feature importance and model attention:

```python
from fused.utils import FeatureImportance, AttentionVisualization

# Analyze feature importance
analyzer = FeatureImportance(model)
importance = analyzer.permutation_importance(X, y, metric_fn)
analyzer.plot_feature_importance(importance)

# Visualize attention patterns
visualizer = AttentionVisualization(model)
attention_maps = visualizer.get_attention_maps(X)
visualizer.plot_attention_heatmap(attention_maps, layer_name)
```

### Model Serving

Export and serve your models in various formats:

```python
from fused.utils import ModelExporter

# Export model
exporter = ModelExporter(model)
exporter.export_all(example_inputs=sample_batch)

# Load and serve
from fused.utils import ModelServer
server = ModelServer("exported_models/model.pt")
server.start_http_server(port=8000)
```

## License

FUSED is released under the MIT License. See [LICENSE](LICENSE) for details.
