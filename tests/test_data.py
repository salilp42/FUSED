"""
Unit tests for data components.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from fused.data.dataset import TimeSeriesDataset, MultimodalTimeSeriesDataset
from fused.data.transforms import Normalize, RandomCrop, RandomMask


@pytest.fixture
def sample_time_series_data():
    """Return sample time series data."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 10
    seq_len = 20
    n_features = 3
    
    # Time series data
    X = np.random.randn(n_samples, seq_len, n_features)
    
    # Labels
    y = np.random.randint(0, 3, size=n_samples)
    
    # Times
    times = np.arange(0, seq_len).reshape(1, -1).repeat(n_samples, axis=0)
    
    return {"X": X, "y": y, "times": times}


@pytest.fixture
def sample_multimodal_data():
    """Return sample multimodal time series data."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 10
    seq_len = 20
    
    # Modality 1: Time series data
    X1 = np.random.randn(n_samples, seq_len, 3)
    
    # Modality 2: Time series data
    X2 = np.random.randn(n_samples, seq_len, 5)
    
    # Modality 3: Static data
    X3 = np.random.randn(n_samples, 7)
    
    # Labels
    y = np.random.randint(0, 3, size=n_samples)
    
    # Times
    times = np.arange(0, seq_len).reshape(1, -1).repeat(n_samples, axis=0)
    
    modalities = {
        "modality1": X1,
        "modality2": X2,
        "static": X3
    }
    
    times_dict = {
        "modality1": times,
        "modality2": times,
    }
    
    return {"modalities": modalities, "y": y, "times": times_dict}


class TestTimeSeriesDataset:
    """Test TimeSeriesDataset."""
    
    def test_initialization(self, sample_time_series_data):
        """Test that the dataset can be initialized."""
        X, y, times = (
            sample_time_series_data["X"],
            sample_time_series_data["y"],
            sample_time_series_data["times"]
        )
        
        # Initialize dataset with numpy arrays
        dataset = TimeSeriesDataset(X, y, times)
        assert len(dataset) == len(X)
        
        # Initialize dataset with torch tensors
        dataset = TimeSeriesDataset(
            torch.tensor(X), 
            torch.tensor(y),
            torch.tensor(times)
        )
        assert len(dataset) == len(X)
        
    def test_getitem(self, sample_time_series_data):
        """Test __getitem__ method."""
        X, y, times = (
            sample_time_series_data["X"],
            sample_time_series_data["y"],
            sample_time_series_data["times"]
        )
        
        dataset = TimeSeriesDataset(X, y, times)
        
        # Get first item
        item = dataset[0]
        assert isinstance(item, dict)
        assert "X" in item
        assert "y" in item
        assert "times" in item
        assert torch.is_tensor(item["X"])
        assert torch.is_tensor(item["y"])
        assert torch.is_tensor(item["times"])
        assert item["X"].shape == (X.shape[1], X.shape[2])
        
    def test_transforms(self, sample_time_series_data):
        """Test applying transforms."""
        X, y, times = (
            sample_time_series_data["X"],
            sample_time_series_data["y"],
            sample_time_series_data["times"]
        )
        
        # Define transforms
        transforms = [
            Normalize()
        ]
        
        dataset = TimeSeriesDataset(X, y, times, transforms=transforms)
        
        # Get transformed item
        item = dataset[0]
        
        # Check that data has been normalized
        assert -1.0 <= item["X"].min() <= 1.0
        assert -1.0 <= item["X"].max() <= 1.0


class TestMultimodalTimeSeriesDataset:
    """Test MultimodalTimeSeriesDataset."""
    
    def test_initialization(self, sample_multimodal_data):
        """Test that the dataset can be initialized."""
        modalities, y, times = (
            sample_multimodal_data["modalities"],
            sample_multimodal_data["y"],
            sample_multimodal_data["times"]
        )
        
        # Initialize dataset with numpy arrays
        dataset = MultimodalTimeSeriesDataset(modalities, y, times)
        assert len(dataset) == len(y)
        
        # Initialize dataset with torch tensors
        modalities_tensor = {
            k: torch.tensor(v) for k, v in modalities.items()
        }
        times_tensor = {
            k: torch.tensor(v) for k, v in times.items() if k in times
        }
        
        dataset = MultimodalTimeSeriesDataset(
            modalities_tensor, 
            torch.tensor(y),
            times_tensor
        )
        assert len(dataset) == len(y)
        
    def test_getitem(self, sample_multimodal_data):
        """Test __getitem__ method."""
        modalities, y, times = (
            sample_multimodal_data["modalities"],
            sample_multimodal_data["y"],
            sample_multimodal_data["times"]
        )
        
        dataset = MultimodalTimeSeriesDataset(modalities, y, times)
        
        # Get first item
        item = dataset[0]
        assert isinstance(item, dict)
        assert "modalities" in item
        assert "y" in item
        assert "times" in item
        
        # Check modalities
        for modality in modalities:
            assert modality in item["modalities"]
            assert torch.is_tensor(item["modalities"][modality])
        
        # Check static modality shape
        assert item["modalities"]["static"].shape == (7,)
        
        # Check sequential modality shape
        assert item["modalities"]["modality1"].shape == (20, 3)
        
    def test_transforms(self, sample_multimodal_data):
        """Test applying transforms."""
        modalities, y, times = (
            sample_multimodal_data["modalities"],
            sample_multimodal_data["y"],
            sample_multimodal_data["times"]
        )
        
        # Define transforms
        transforms = {
            "modality1": [Normalize()],
            "modality2": [Normalize(), RandomMask(mask_prob=0.3)]
        }
        
        dataset = MultimodalTimeSeriesDataset(
            modalities, 
            y, 
            times, 
            transforms=transforms
        )
        
        # Get transformed item
        item = dataset[0]
        
        # Check that data has been normalized
        assert -1.0 <= item["modalities"]["modality1"].min() <= 1.0
        assert -1.0 <= item["modalities"]["modality1"].max() <= 1.0


class TestTransforms:
    """Test data transforms."""
    
    def test_normalize(self):
        """Test Normalize transform."""
        data = torch.randn(10, 5) * 5 + 2  # Non-standard mean and std
        transform = Normalize()
        
        # Apply transform
        transformed = transform(data)
        
        # Check statistics
        assert -1.0 <= transformed.min() <= 1.0
        assert -1.0 <= transformed.max() <= 1.0
        assert abs(transformed.mean()) < 0.5
        assert abs(transformed.std() - 1.0) < 0.5
        
    def test_random_crop(self):
        """Test RandomCrop transform."""
        data = torch.randn(100, 5)
        target_len = 50
        transform = RandomCrop(target_len)
        
        # Apply transform
        transformed = transform(data)
        
        # Check shape
        assert transformed.shape == (target_len, 5)
        
    def test_random_mask(self):
        """Test RandomMask transform."""
        data = torch.randn(100, 5)
        mask_prob = 0.5
        transform = RandomMask(mask_prob=mask_prob)
        
        # Apply transform
        transformed, mask = transform(data, return_mask=True)
        
        # Check that some elements are masked
        assert (~mask).sum() > 0
        
        # Check that masked elements are zero
        masked_elements = transformed[~mask]
        assert torch.all(masked_elements == 0)
