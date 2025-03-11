"""
Documentation utilities for FUSED.

This module provides tools for generating documentation
for the FUSED framework, including API reference and
example notebooks.
"""

import os
import inspect
import importlib
import re
import glob
import json
import pkgutil
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
import logging
import warnings


class APIDocumentationGenerator:
    """
    Generator for FUSED API documentation.
    
    This class provides methods for extracting docstrings and signatures
    from FUSED modules and generating markdown documentation.
    """
    
    def __init__(self, 
                output_dir: str = "docs/api",
                package_name: str = "fused"):
        """
        Initialize the API documentation generator.
        
        Args:
            output_dir: Directory to save generated documentation
            package_name: Name of the package to document
        """
        self.output_dir = output_dir
        self.package_name = package_name
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_module_doc(self, 
                           module_name: str,
                           filename: Optional[str] = None) -> str:
        """
        Generate documentation for a single module.
        
        Args:
            module_name: Name of the module (e.g., "fused.models.base")
            filename: Optional output filename (default: derived from module name)
            
        Returns:
            Path to the generated documentation file
        """
        # Import the module
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            warnings.warn(f"Could not import module {module_name}")
            return None
        
        # Determine output filename
        if filename is None:
            # Convert module name to path
            rel_name = module_name[len(self.package_name) + 1:] if module_name.startswith(self.package_name) else module_name
            filename = f"{rel_name.replace('.', '/')}.md"
        
        # Create output path
        output_path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Extract module docstring
        module_doc = inspect.getdoc(module)
        
        # Generate markdown content
        content = [f"# {module_name}"]
        
        if module_doc:
            content.append(module_doc)
        
        content.append("\n## Module Contents\n")
        
        # Extract classes
        classes = self._get_public_classes(module)
        if classes:
            content.append("### Classes\n")
            for cls in classes:
                cls_doc = inspect.getdoc(cls)
                content.append(f"#### {cls.__name__}")
                
                if cls_doc:
                    content.append(cls_doc)
                
                # Get method information
                methods = self._get_public_methods(cls)
                if methods:
                    content.append("\n**Methods:**\n")
                    for method in methods:
                        # Get method signature
                        try:
                            signature = inspect.signature(method)
                            content.append(f"- `{method.__name__}{signature}`")
                        except (ValueError, TypeError):
                            content.append(f"- `{method.__name__}(...)`")
                
                content.append("\n")
        
        # Extract functions
        functions = self._get_public_functions(module)
        if functions:
            content.append("### Functions\n")
            for func in functions:
                func_doc = inspect.getdoc(func)
                
                # Get function signature
                try:
                    signature = inspect.signature(func)
                    content.append(f"#### `{func.__name__}{signature}`")
                except (ValueError, TypeError):
                    content.append(f"#### `{func.__name__}(...)`")
                
                if func_doc:
                    content.append(func_doc)
                
                content.append("\n")
        
        # Write content to file
        with open(output_path, "w") as f:
            f.write("\n\n".join(content))
        
        logging.info(f"Generated documentation for {module_name} at {output_path}")
        
        return output_path
    
    def generate_package_doc(self,
                            package_name: Optional[str] = None) -> List[str]:
        """
        Generate documentation for all modules in a package.
        
        Args:
            package_name: Name of the package (default: the initialized package)
            
        Returns:
            List of paths to generated documentation files
        """
        # Use initialized package if none specified
        if package_name is None:
            package_name = self.package_name
        
        # Import the package
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            warnings.warn(f"Could not import package {package_name}")
            return []
        
        # Get package directory
        if not hasattr(package, "__path__"):
            warnings.warn(f"{package_name} is not a package")
            return []
        
        # Create index file
        index_path = os.path.join(self.output_dir, f"{package_name.replace('.', '/')}.md")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Extract package docstring
        package_doc = inspect.getdoc(package)
        
        # Create index content
        index_content = [f"# {package_name} Package"]
        
        if package_doc:
            index_content.append(package_doc)
        
        index_content.append("\n## Subpackages and Modules\n")
        
        # Find all subpackages and modules
        generated_files = []
        subpackages = []
        submodules = []
        
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package_name + "."):
            if is_pkg:
                subpackages.append(name)
            else:
                submodules.append(name)
        
        # Document subpackages
        if subpackages:
            index_content.append("### Subpackages\n")
            for subpackage in sorted(subpackages):
                files = self.generate_package_doc(subpackage)
                generated_files.extend(files)
                
                # Add link to index
                rel_path = subpackage[len(self.package_name) + 1:] if subpackage.startswith(self.package_name) else subpackage
                index_content.append(f"- [{subpackage}]({rel_path.replace('.', '/')}.md)")
        
        # Document modules
        if submodules:
            index_content.append("\n### Modules\n")
            for submodule in sorted(submodules):
                file_path = self.generate_module_doc(submodule)
                if file_path:
                    generated_files.append(file_path)
                    
                    # Add link to index
                    rel_path = submodule[len(self.package_name) + 1:] if submodule.startswith(self.package_name) else submodule
                    index_content.append(f"- [{submodule}]({rel_path.replace('.', '/')}.md)")
        
        # Write index to file
        with open(index_path, "w") as f:
            f.write("\n\n".join(index_content))
        
        generated_files.append(index_path)
        
        logging.info(f"Generated documentation for {package_name} at {index_path}")
        
        return generated_files
    
    def generate_all(self) -> List[str]:
        """
        Generate documentation for the entire FUSED package.
        
        Returns:
            List of paths to generated documentation files
        """
        return self.generate_package_doc()
    
    def _get_public_classes(self, module) -> List[Any]:
        """
        Get public classes from a module.
        
        Args:
            module: Module to extract classes from
            
        Returns:
            List of class objects
        """
        return [
            obj for name, obj in inspect.getmembers(module, inspect.isclass)
            if not name.startswith("_") and obj.__module__ == module.__name__
        ]
    
    def _get_public_functions(self, module) -> List[Any]:
        """
        Get public functions from a module.
        
        Args:
            module: Module to extract functions from
            
        Returns:
            List of function objects
        """
        return [
            obj for name, obj in inspect.getmembers(module, inspect.isfunction)
            if not name.startswith("_") and obj.__module__ == module.__name__
        ]
    
    def _get_public_methods(self, cls) -> List[Any]:
        """
        Get public methods from a class.
        
        Args:
            cls: Class to extract methods from
            
        Returns:
            List of method objects
        """
        return [
            obj for name, obj in inspect.getmembers(cls, inspect.isfunction)
            if not name.startswith("_")
        ]


class ExampleGenerator:
    """
    Generator for FUSED example notebooks.
    
    This class provides methods for generating example Jupyter notebooks
    for the FUSED framework.
    """
    
    def __init__(self, 
                output_dir: str = "docs/examples",
                template_dir: Optional[str] = None):
        """
        Initialize the example generator.
        
        Args:
            output_dir: Directory to save generated examples
            template_dir: Optional directory with notebook templates
        """
        self.output_dir = output_dir
        self.template_dir = template_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_basic_example(self, 
                             filename: str = "basic_example.ipynb",
                             dataset: str = "synthetic") -> str:
        """
        Generate a basic example notebook.
        
        Args:
            filename: Output filename
            dataset: Dataset to use in the example
            
        Returns:
            Path to the generated notebook
        """
        # Check if we have templates
        if self.template_dir:
            template_path = os.path.join(self.template_dir, "basic_example_template.ipynb")
            if os.path.exists(template_path):
                return self._fill_template(template_path, filename, {"dataset": dataset})
        
        # Otherwise, generate from scratch
        notebook = {
            "cells": [
                # Introduction cell
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# FUSED Framework: Basic Example\n",
                        "\n",
                        "This notebook demonstrates the basic usage of the FUSED framework for time series analysis."
                    ]
                },
                # Import cell
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "import torch\n",
                        "import numpy as np\n",
                        "import pandas as pd\n",
                        "import matplotlib.pyplot as plt\n",
                        "\n",
                        "from fused.data import TimeSeriesDataset, MultimodalTimeSeriesDataset\n",
                        "from fused.models import SequentialEncoder, TabularEncoder, FusionModule\n",
                        "from fused.training import Trainer\n",
                        "\n",
                        "%matplotlib inline"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                # Data loading cell
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Data Loading\n",
                        "\n",
                        "First, let's load a dataset for our demonstration."
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        f"# Load {dataset} dataset\n",
                        f"if '{dataset}' == 'synthetic':\n",
                        "    # Generate synthetic data\n",
                        "    n_samples = 1000\n",
                        "    seq_len = 50\n",
                        "    n_features = 10\n",
                        "    \n",
                        "    # Sequential data\n",
                        "    sequential_data = np.random.randn(n_samples, seq_len, n_features)\n",
                        "    \n",
                        "    # Tabular data\n",
                        "    tabular_data = np.random.randn(n_samples, 20)\n",
                        "    \n",
                        "    # Labels (binary classification)\n",
                        "    labels = np.random.randint(0, 2, size=n_samples)\n",
                        "    \n",
                        "    # Create dataset\n",
                        "    dataset = MultimodalTimeSeriesDataset(\n",
                        "        sequential=torch.tensor(sequential_data, dtype=torch.float32),\n",
                        "        tabular=torch.tensor(tabular_data, dtype=torch.float32),\n",
                        "        targets=torch.tensor(labels, dtype=torch.long)\n",
                        "    )\n",
                        "else:\n",
                        "    # Load a real dataset\n",
                        "    raise ValueError(f\"Dataset '{dataset}' not implemented in this example.\")\n",
                        "\n",
                        "# Split dataset\n",
                        "train_size = int(0.8 * len(dataset))\n",
                        "val_size = len(dataset) - train_size\n",
                        "train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])\n",
                        "\n",
                        "print(f\"Training set size: {len(train_dataset)}\")\n",
                        "print(f\"Validation set size: {len(val_dataset)}\")"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                # Model definition cell
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Model Definition\n",
                        "\n",
                        "Now, let's define a FUSED model with sequential and tabular encoders."
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# Define model components\n",
                        "sequential_encoder = SequentialEncoder(\n",
                        "    input_dim=n_features,\n",
                        "    hidden_dim=64,\n",
                        "    output_dim=128,\n",
                        "    num_layers=2,\n",
                        "    dropout=0.1\n",
                        ")\n",
                        "\n",
                        "tabular_encoder = TabularEncoder(\n",
                        "    input_dim=20,\n",
                        "    hidden_dims=[64, 32],\n",
                        "    output_dim=64,\n",
                        "    dropout=0.1\n",
                        ")\n",
                        "\n",
                        "fusion_module = FusionModule(\n",
                        "    input_dims={'sequential': 128, 'tabular': 64},\n",
                        "    output_dim=64,\n",
                        "    fusion_method='concat'\n",
                        ")\n",
                        "\n",
                        "# Create a simple classifier head\n",
                        "classifier = torch.nn.Linear(64, 2)\n",
                        "\n",
                        "# Define complete model\n",
                        "class FUSEDModel(torch.nn.Module):\n",
                        "    def __init__(self):\n",
                        "        super().__init__()\n",
                        "        self.sequential_encoder = sequential_encoder\n",
                        "        self.tabular_encoder = tabular_encoder\n",
                        "        self.fusion_module = fusion_module\n",
                        "        self.classifier = classifier\n",
                        "        \n",
                        "    def forward(self, x):\n",
                        "        # Encode sequential data\n",
                        "        sequential_features = self.sequential_encoder(x['sequential'])\n",
                        "        \n",
                        "        # Encode tabular data\n",
                        "        tabular_features = self.tabular_encoder(x['tabular'])\n",
                        "        \n",
                        "        # Fuse features\n",
                        "        fused_features = self.fusion_module({\n",
                        "            'sequential': sequential_features,\n",
                        "            'tabular': tabular_features\n",
                        "        })\n",
                        "        \n",
                        "        # Classification\n",
                        "        logits = self.classifier(fused_features)\n",
                        "        \n",
                        "        return {\n",
                        "            'logits': logits,\n",
                        "            'embeddings': fused_features\n",
                        "        }\n",
                        "\n",
                        "# Initialize model\n",
                        "model = FUSEDModel()"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                # Training cell
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Model Training\n",
                        "\n",
                        "Let's train the model using the FUSED Trainer."
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# Initialize trainer\n",
                        "trainer = Trainer(\n",
                        "    model=model,\n",
                        "    loss_fn=torch.nn.CrossEntropyLoss(),\n",
                        "    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),\n",
                        "    device='cuda' if torch.cuda.is_available() else 'cpu'\n",
                        ")\n",
                        "\n",
                        "# Train model\n",
                        "history = trainer.fit(\n",
                        "    train_dataset=train_dataset,\n",
                        "    val_dataset=val_dataset,\n",
                        "    batch_size=32,\n",
                        "    num_epochs=10,\n",
                        "    patience=3,\n",
                        "    verbose=True\n",
                        ")"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                # Evaluation cell
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Model Evaluation\n",
                        "\n",
                        "Let's evaluate the model's performance."
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "# Plot training history\n",
                        "plt.figure(figsize=(12, 4))\n",
                        "\n",
                        "plt.subplot(1, 2, 1)\n",
                        "plt.plot(history['loss'], label='Train')\n",
                        "plt.plot(history['val_loss'], label='Validation')\n",
                        "plt.xlabel('Epoch')\n",
                        "plt.ylabel('Loss')\n",
                        "plt.legend()\n",
                        "plt.title('Training and Validation Loss')\n",
                        "\n",
                        "plt.subplot(1, 2, 2)\n",
                        "plt.plot(history['accuracy'], label='Train')\n",
                        "plt.plot(history['val_accuracy'], label='Validation')\n",
                        "plt.xlabel('Epoch')\n",
                        "plt.ylabel('Accuracy')\n",
                        "plt.legend()\n",
                        "plt.title('Training and Validation Accuracy')\n",
                        "\n",
                        "plt.tight_layout()\n",
                        "plt.show()\n",
                        "\n",
                        "# Evaluate on validation set\n",
                        "val_metrics = trainer.evaluate(val_dataset, batch_size=32)\n",
                        "print(f\"Validation metrics: {val_metrics}\")"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                # Feature importance cell
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Feature Importance Analysis\n",
                        "\n",
                        "Let's analyze which features are most important for the model's predictions."
                    ]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "from fused.utils.interpretability import FeatureImportance\n",
                        "\n",
                        "# Initialize feature importance analyzer\n",
                        "analyzer = FeatureImportance(model)\n",
                        "\n",
                        "# Get a batch of data\n",
                        "dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)\n",
                        "batch = next(iter(dataloader))\n",
                        "\n",
                        "# Define metric function\n",
                        "def accuracy_metric(y_true, y_pred):\n",
                        "    return (torch.argmax(y_pred, dim=1) == y_true).float().mean().item()\n",
                        "\n",
                        "# Compute permutation importance for sequential features\n",
                        "seq_importance = analyzer.permutation_importance(\n",
                        "    batch['sequential'], \n",
                        "    batch['targets'],\n",
                        "    accuracy_metric,\n",
                        "    n_repeats=5,\n",
                        "    feature_names=[f'Seq_{i}' for i in range(n_features)]\n",
                        ")\n",
                        "\n",
                        "# Plot feature importance\n",
                        "analyzer.plot_feature_importance(\n",
                        "    seq_importance,\n",
                        "    title='Sequential Feature Importance',\n",
                        "    sort=True\n",
                        ")\n",
                        "plt.show()"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                # Conclusion cell
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Conclusion\n",
                        "\n",
                        "In this notebook, we demonstrated the basic usage of the FUSED framework for multimodal time series analysis. We covered:\n",
                        "\n",
                        "1. Data loading and preprocessing\n",
                        "2. Building a multimodal model with the FUSED architecture\n",
                        "3. Training and evaluation\n",
                        "4. Feature importance analysis\n",
                        "\n",
                        "The FUSED framework provides a flexible and extensible platform for developing advanced time series models."
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.10"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Write notebook to file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w") as f:
            json.dump(notebook, f, indent=2)
        
        logging.info(f"Generated example notebook at {output_path}")
        
        return output_path
    
    def generate_advanced_example(self, 
                                filename: str = "advanced_example.ipynb",
                                feature: str = "pretrain") -> str:
        """
        Generate an advanced example notebook.
        
        Args:
            filename: Output filename
            feature: Advanced feature to showcase
            
        Returns:
            Path to the generated notebook
        """
        # Check if we have templates
        if self.template_dir:
            template_path = os.path.join(self.template_dir, f"advanced_{feature}_template.ipynb")
            if os.path.exists(template_path):
                return self._fill_template(template_path, filename, {"feature": feature})
        
        # Otherwise, generate from scratch
        # Note: This is a simplified version, a real implementation would be more detailed
        notebook = {
            "cells": [
                # Introduction cell
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# FUSED Framework: Advanced Example - {feature.title()}\n",
                        "\n",
                        f"This notebook demonstrates advanced usage of the FUSED framework, focusing on {feature}."
                    ]
                },
                # Import cell
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "import torch\n",
                        "import numpy as np\n",
                        "import pandas as pd\n",
                        "import matplotlib.pyplot as plt\n",
                        "\n",
                        "from fused.data import TimeSeriesDataset, MultimodalTimeSeriesDataset\n",
                        "from fused.models import SequentialEncoder, TabularEncoder, FusionModule\n",
                        "from fused.training import Trainer\n",
                        f"from fused.utils import {feature}\n",
                        "\n",
                        "%matplotlib inline"
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                # Feature-specific cell
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"## {feature.title()} in FUSED\n",
                        "\n",
                        f"Let's explore how to use the {feature} feature in the FUSED framework."
                    ]
                },
                # Placeholder for feature-specific code
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        f"# This is a placeholder for {feature}-specific code.\n",
                        "# A real example would include detailed code for the feature."
                    ],
                    "execution_count": None,
                    "outputs": []
                },
                # Conclusion cell
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Conclusion\n",
                        "\n",
                        f"In this notebook, we demonstrated how to use the {feature} feature in the FUSED framework."
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.10"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Write notebook to file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w") as f:
            json.dump(notebook, f, indent=2)
        
        logging.info(f"Generated advanced example notebook at {output_path}")
        
        return output_path
    
    def generate_all_examples(self) -> List[str]:
        """
        Generate all example notebooks.
        
        Returns:
            List of paths to generated notebooks
        """
        generated_files = []
        
        # Generate basic example
        basic_path = self.generate_basic_example()
        generated_files.append(basic_path)
        
        # Generate advanced examples
        advanced_features = [
            "pretrain",
            "hyperparameter_optimization",
            "interpretability",
            "experiment_tracking",
            "serving"
        ]
        
        for feature in advanced_features:
            advanced_path = self.generate_advanced_example(
                filename=f"advanced_{feature}.ipynb",
                feature=feature
            )
            generated_files.append(advanced_path)
        
        return generated_files
    
    def _fill_template(self, 
                      template_path: str,
                      output_filename: str,
                      replacements: Dict[str, str]) -> str:
        """
        Fill a notebook template with replacements.
        
        Args:
            template_path: Path to template notebook
            output_filename: Output filename
            replacements: Dictionary of replacements
            
        Returns:
            Path to the generated notebook
        """
        # Read template
        with open(template_path, "r") as f:
            notebook = json.load(f)
        
        # Convert notebook to string for replacement
        notebook_str = json.dumps(notebook)
        
        # Apply replacements
        for key, value in replacements.items():
            placeholder = f"{{{{FUSED_{key.upper()}}}}}"
            notebook_str = notebook_str.replace(placeholder, value)
        
        # Convert back to notebook
        notebook = json.loads(notebook_str)
        
        # Write notebook to file
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, "w") as f:
            json.dump(notebook, f, indent=2)
        
        logging.info(f"Generated example notebook from template at {output_path}")
        
        return output_path
