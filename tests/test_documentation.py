"""
Tests for documentation utilities.
"""

import os
import pytest
import json
import tempfile
import importlib

from fused.utils.documentation import (
    APIDocumentationGenerator,
    ExampleGenerator
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for documentation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_api_documentation_generator_module(temp_dir):
    """Test generating documentation for a single module."""
    # Create generator
    generator = APIDocumentationGenerator(output_dir=temp_dir)
    
    # Generate documentation for a module from fused
    # Will test with the experiment_tracking module since we know it exists
    module_path = "fused.utils.experiment_tracking"
    
    try:
        # Try to import the module first
        importlib.import_module(module_path)
        
        # Generate documentation
        output_path = generator.generate_module_doc(module_path)
        
        # Check if file exists
        assert output_path is not None
        assert os.path.exists(output_path)
        
        # Check content
        with open(output_path, "r") as f:
            content = f.read()
            
        # Basic content checks
        assert module_path in content
        assert "## Module Contents" in content
        
    except ImportError:
        pytest.skip(f"Module {module_path} not found")


def test_api_documentation_generator_package(temp_dir):
    """Test generating documentation for a package."""
    # Create generator
    generator = APIDocumentationGenerator(output_dir=temp_dir)
    
    # Generate documentation for a package from fused
    # Will test with the utils package since we know it exists
    package_path = "fused.utils"
    
    try:
        # Try to import the package first
        importlib.import_module(package_path)
        
        # Generate documentation
        output_paths = generator.generate_package_doc(package_path)
        
        # Check if files exist
        assert len(output_paths) > 0
        for path in output_paths:
            assert os.path.exists(path)
        
        # Check if index file was created
        index_path = os.path.join(temp_dir, "utils.md")
        assert os.path.exists(index_path)
        
        # Check content of index
        with open(index_path, "r") as f:
            content = f.read()
            
        # Basic content checks
        assert package_path in content
        assert "## Subpackages and Modules" in content
        
    except ImportError:
        pytest.skip(f"Package {package_path} not found")


def test_example_generator_basic(temp_dir):
    """Test generating a basic example notebook."""
    # Create generator
    generator = ExampleGenerator(output_dir=temp_dir)
    
    # Generate basic example
    output_path = generator.generate_basic_example(
        filename="test_basic_example.ipynb"
    )
    
    # Check if file exists
    assert os.path.exists(output_path)
    
    # Check content
    with open(output_path, "r") as f:
        notebook = json.load(f)
        
    # Check notebook structure
    assert "cells" in notebook
    assert len(notebook["cells"]) > 0
    assert "metadata" in notebook
    
    # Check content of cells
    markdown_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    
    assert len(markdown_cells) > 0
    assert len(code_cells) > 0
    
    # Check if first cell contains title
    assert any("FUSED Framework" in ''.join(cell["source"]) for cell in markdown_cells)
    
    # Check if imports are included
    imports_found = False
    for cell in code_cells:
        source = ''.join(cell["source"])
        if "import torch" in source and "from fused" in source:
            imports_found = True
            break
    assert imports_found


def test_example_generator_advanced(temp_dir):
    """Test generating an advanced example notebook."""
    # Create generator
    generator = ExampleGenerator(output_dir=temp_dir)
    
    # Generate advanced example
    feature = "hyperparameter_optimization"
    output_path = generator.generate_advanced_example(
        filename=f"test_advanced_{feature}.ipynb",
        feature=feature
    )
    
    # Check if file exists
    assert os.path.exists(output_path)
    
    # Check content
    with open(output_path, "r") as f:
        notebook = json.load(f)
        
    # Check notebook structure
    assert "cells" in notebook
    assert len(notebook["cells"]) > 0
    assert "metadata" in notebook
    
    # Check content of cells
    markdown_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    
    assert len(markdown_cells) > 0
    assert len(code_cells) > 0
    
    # Check if first cell contains title and the feature name
    assert any(feature in ''.join(cell["source"]) for cell in markdown_cells)
    
    # Check if feature-specific import is included
    feature_import_found = False
    for cell in code_cells:
        source = ''.join(cell["source"])
        if f"from fused.utils import {feature}" in source:
            feature_import_found = True
            break
    assert feature_import_found


def test_example_generator_all(temp_dir):
    """Test generating all example notebooks."""
    # Create generator
    generator = ExampleGenerator(output_dir=temp_dir)
    
    # Generate all examples
    output_paths = generator.generate_all_examples()
    
    # Check if files exist
    assert len(output_paths) > 0
    for path in output_paths:
        assert os.path.exists(path)
    
    # Check if basic example was created
    basic_path = os.path.join(temp_dir, "basic_example.ipynb")
    assert os.path.exists(basic_path)
    
    # Check if advanced examples were created
    advanced_features = [
        "pretrain",
        "hyperparameter_optimization",
        "interpretability",
        "experiment_tracking",
        "serving"
    ]
    
    for feature in advanced_features:
        advanced_path = os.path.join(temp_dir, f"advanced_{feature}.ipynb")
        assert os.path.exists(advanced_path)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
