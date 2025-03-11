#!/usr/bin/env python
"""
Test script to verify all components can be imported correctly.
This helps ensure the package structure is correct.
"""

import sys
import importlib

def test_imports():
    """Test importing all major components of the FUSED package."""
    modules_to_test = [
        # Core modules
        "fused",
        "fused.models",
        "fused.models.model",
        "fused.models.fusion",
        "fused.models.neural_ode",
        "fused.models.tabular_encoder",
        "fused.models.temporal_fusion",
        
        # Trainers
        "fused.trainers",
        "fused.trainers.trainer",
        "fused.trainers.pretraining_objectives",
        
        # Data handling
        "fused.data",
        "fused.data.dataset",
        "fused.data.transforms",
        
        # Evaluation
        "fused.evaluation",
        "fused.evaluation.metrics",
        
        # Configurations
        "fused.configs",
        "fused.configs.model_configs",
        
        # Examples
        "examples",
        "examples.basic_example",
        "examples.multimodal_example",
        "examples.pretraining_example"
    ]
    
    success = True
    failed_modules = []
    
    print("Testing FUSED module imports...")
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"✓ Successfully imported {module_name}")
        except Exception as e:
            success = False
            failed_modules.append((module_name, str(e)))
            print(f"✗ Failed to import {module_name}: {e}")
    
    # Print summary
    print("\n" + "="*50)
    if success:
        print("All modules imported successfully!")
    else:
        print(f"Failed to import {len(failed_modules)} modules:")
        for module_name, error in failed_modules:
            print(f"  - {module_name}: {error}")
    
    return success

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
