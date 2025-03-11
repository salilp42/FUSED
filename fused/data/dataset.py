"""
Dataset implementations for FUSED.

This module contains implementations of dataset classes for
handling time series data with multiple modalities.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
import os
import json


class TimeSeriesDataset(Dataset):
    """
    Dataset for single-modality time series data.
    """
    
    def __init__(self, 
                 data: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                 targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 timestamps: Optional[Union[np.ndarray, torch.Tensor, pd.DatetimeIndex]] = None,
                 seq_len: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize the time series dataset.
        
        Args:
            data: Time series data
                If numpy array or torch tensor: [num_samples, seq_len, features]
                If pandas DataFrame: Each row is a time step, columns are features
            targets: Optional target values
                If classification/regression: [num_samples] or [num_samples, num_classes]
                If forecasting: [num_samples, horizon, features]
            timestamps: Optional timestamps for each time step
            seq_len: Optional sequence length (required if data is a DataFrame)
            transform: Optional transform to apply to the data
            target_transform: Optional transform to apply to the targets
        """
        self.transform = transform
        self.target_transform = target_transform
        
        # Process data
        if isinstance(data, pd.DataFrame):
            if seq_len is None:
                raise ValueError("seq_len must be provided when data is a DataFrame")
                
            # Handle DataFrame input
            self.timestamps = data.index if timestamps is None else timestamps
            self.data = self._prepare_df_data(data, seq_len)
        else:
            # Handle numpy array or torch tensor
            self.data = self._prepare_array_data(data)
            
            if timestamps is not None:
                self.timestamps = self._prepare_timestamps(timestamps, self.data.shape[0], self.data.shape[1])
            else:
                # Create synthetic timestamps if not provided
                self.timestamps = torch.arange(self.data.shape[1]).unsqueeze(0).repeat(self.data.shape[0], 1)
        
        # Process targets
        if targets is not None:
            self.targets = self._prepare_targets(targets)
            self.has_targets = True
        else:
            self.targets = None
            self.has_targets = False
            
    def _prepare_df_data(self, 
                         df: pd.DataFrame, 
                         seq_len: int) -> torch.Tensor:
        """
        Prepare data from DataFrame.
        
        Args:
            df: DataFrame where each row is a time step, columns are features
            seq_len: Sequence length for windowing
            
        Returns:
            Tensor of shape [num_samples, seq_len, num_features]
        """
        # Convert DataFrame to numpy array
        values = df.values
        
        # Create windows
        windows = []
        for i in range(len(df) - seq_len + 1):
            window = values[i:i+seq_len]
            windows.append(window)
            
        # Convert to tensor
        if len(windows) > 0:
            windows_tensor = torch.tensor(np.array(windows), dtype=torch.float32)
        else:
            # Handle empty case
            windows_tensor = torch.zeros((0, seq_len, df.shape[1]), dtype=torch.float32)
            
        return windows_tensor
        
    def _prepare_array_data(self, 
                            data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Prepare data from numpy array or torch tensor.
        
        Args:
            data: Array of shape [num_samples, seq_len, features]
            
        Returns:
            Tensor of shape [num_samples, seq_len, features]
        """
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data_tensor = torch.tensor(data, dtype=torch.float32)
        else:
            data_tensor = data.float()
            
        return data_tensor
        
    def _prepare_timestamps(self, 
                           timestamps: Union[np.ndarray, torch.Tensor, pd.DatetimeIndex], 
                           num_samples: int,
                           seq_len: int) -> torch.Tensor:
        """
        Prepare timestamps.
        
        Args:
            timestamps: Timestamps for each time step
            num_samples: Number of samples
            seq_len: Sequence length
            
        Returns:
            Tensor of shape [num_samples, seq_len]
        """
        # Handle DatetimeIndex
        if isinstance(timestamps, pd.DatetimeIndex):
            # Convert to seconds since epoch
            timestamps = timestamps.astype(np.int64) // 10**9
            timestamps = torch.tensor(timestamps.values, dtype=torch.float32)
        elif isinstance(timestamps, np.ndarray):
            timestamps = torch.tensor(timestamps, dtype=torch.float32)
            
        # Ensure correct shape
        if len(timestamps.shape) == 1:
            # Single sequence of timestamps
            if len(timestamps) < num_samples * seq_len:
                raise ValueError(f"Number of timestamps ({len(timestamps)}) is less than required ({num_samples * seq_len})")
                
            # Reshape to match data
            timestamps_tensor = timestamps[:num_samples * seq_len].reshape(num_samples, seq_len)
        elif len(timestamps.shape) == 2:
            # Already in correct shape
            if timestamps.shape[0] != num_samples or timestamps.shape[1] != seq_len:
                raise ValueError(f"Timestamps shape {timestamps.shape} doesn't match data shape {(num_samples, seq_len)}")
                
            timestamps_tensor = timestamps
        else:
            raise ValueError(f"Unexpected timestamps shape: {timestamps.shape}")
            
        return timestamps_tensor
        
    def _prepare_targets(self, 
                         targets: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Prepare targets.
        
        Args:
            targets: Target values
            
        Returns:
            Tensor of targets
        """
        # Convert to tensor if needed
        if isinstance(targets, np.ndarray):
            targets_tensor = torch.tensor(targets)
        else:
            targets_tensor = targets
            
        # Convert to float for regression/forecasting, long for classification
        if targets_tensor.dtype == torch.float32 or targets_tensor.dtype == torch.float64:
            targets_tensor = targets_tensor.float()
        else:
            targets_tensor = targets_tensor.long()
            
        return targets_tensor
        
    def __len__(self) -> int:
        """
        Get the number of samples.
        
        Returns:
            Number of samples
        """
        return self.data.shape[0]
        
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the sample
        """
        # Get data
        data = self.data[idx]
        timestamps = self.timestamps[idx] if self.timestamps is not None else None
        
        # Apply transform
        if self.transform:
            data = self.transform(data)
            
        # Get targets if available
        if self.has_targets:
            targets = self.targets[idx]
            
            # Apply target transform
            if self.target_transform:
                targets = self.target_transform(targets)
                
            return {"data": data, "timestamps": timestamps, "targets": targets}
        else:
            return {"data": data, "timestamps": timestamps}
            
    @classmethod
    def from_csv(cls, 
                 file_path: str, 
                 target_cols: Optional[Union[str, List[str]]] = None,
                 timestamp_col: Optional[str] = None,
                 seq_len: int = 100,
                 **kwargs) -> "TimeSeriesDataset":
        """
        Create a dataset from a CSV file.
        
        Args:
            file_path: Path to CSV file
            target_cols: Column(s) to use as target
            timestamp_col: Column to use as timestamp
            seq_len: Sequence length
            **kwargs: Additional arguments for pandas.read_csv
            
        Returns:
            TimeSeriesDataset
        """
        # Read CSV
        df = pd.read_csv(file_path, **kwargs)
        
        # Extract timestamps if specified
        timestamps = None
        if timestamp_col is not None:
            if timestamp_col in df.columns:
                timestamps = pd.to_datetime(df[timestamp_col])
                df = df.drop(columns=[timestamp_col])
            else:
                raise ValueError(f"Timestamp column '{timestamp_col}' not found in CSV")
                
        # Extract targets if specified
        targets = None
        if target_cols is not None:
            if isinstance(target_cols, str):
                target_cols = [target_cols]
                
            # Check if all target columns exist
            missing_cols = [col for col in target_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Target column(s) {missing_cols} not found in CSV")
                
            # Extract target columns
            target_df = df[target_cols]
            df = df.drop(columns=target_cols)
            
            # Convert targets to numpy array
            targets = target_df.values
            
        # Create dataset
        return cls(data=df, targets=targets, timestamps=timestamps, seq_len=seq_len)


class MultimodalTimeSeriesDataset(Dataset):
    """
    Dataset for multimodal time series data.
    """
    
    def __init__(self, 
                 modalities: Dict[str, Union[np.ndarray, torch.Tensor, pd.DataFrame]],
                 targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 timestamps: Optional[Dict[str, Union[np.ndarray, torch.Tensor, pd.DatetimeIndex]]] = None,
                 seq_lens: Optional[Dict[str, int]] = None,
                 transforms: Optional[Dict[str, Callable]] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize the multimodal time series dataset.
        
        Args:
            modalities: Dictionary mapping modality names to their data
            targets: Optional target values
            timestamps: Optional dictionary mapping modality names to their timestamps
            seq_lens: Optional dictionary mapping modality names to their sequence lengths
            transforms: Optional dictionary mapping modality names to their transforms
            target_transform: Optional transform to apply to the targets
        """
        self.modalities = list(modalities.keys())
        self.transforms = transforms or {}
        self.target_transform = target_transform
        
        # Create separate datasets for each modality
        self.datasets = {}
        
        for modality, data in modalities.items():
            modality_timestamps = None if timestamps is None else timestamps.get(modality)
            modality_seq_len = None if seq_lens is None else seq_lens.get(modality)
            modality_transform = self.transforms.get(modality)
            
            # Create dataset for this modality
            self.datasets[modality] = TimeSeriesDataset(
                data=data,
                targets=None,  # Targets are handled at the multimodal level
                timestamps=modality_timestamps,
                seq_len=modality_seq_len,
                transform=modality_transform
            )
            
        # Ensure all modalities have the same number of samples
        lengths = [len(dataset) for dataset in self.datasets.values()]
        if len(set(lengths)) > 1:
            raise ValueError(f"All modalities must have the same number of samples, got {lengths}")
            
        self.num_samples = lengths[0]
        
        # Process targets
        if targets is not None:
            self.targets = self._prepare_targets(targets)
            self.has_targets = True
        else:
            self.targets = None
            self.has_targets = False
            
    def _prepare_targets(self, 
                         targets: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Prepare targets.
        
        Args:
            targets: Target values
            
        Returns:
            Tensor of targets
        """
        # Convert to tensor if needed
        if isinstance(targets, np.ndarray):
            targets_tensor = torch.tensor(targets)
        else:
            targets_tensor = targets
            
        # Convert to float for regression/forecasting, long for classification
        if targets_tensor.dtype == torch.float32 or targets_tensor.dtype == torch.float64:
            targets_tensor = targets_tensor.float()
        else:
            targets_tensor = targets_tensor.long()
            
        return targets_tensor
        
    def __len__(self) -> int:
        """
        Get the number of samples.
        
        Returns:
            Number of samples
        """
        return self.num_samples
        
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the sample for each modality
        """
        # Get data for each modality
        sample = {}
        
        for modality in self.modalities:
            modality_sample = self.datasets[modality][idx]
            
            # Add to sample with modality prefix
            sample[modality] = modality_sample["data"]
            sample[f"{modality}_timestamps"] = modality_sample["timestamps"]
            
        # Add targets if available
        if self.has_targets:
            targets = self.targets[idx]
            
            # Apply target transform
            if self.target_transform:
                targets = self.target_transform(targets)
                
            sample["targets"] = targets
            
        return sample
        
    @classmethod
    def from_tensors(cls, 
                     modalities: Dict[str, torch.Tensor],
                     targets: Optional[torch.Tensor] = None,
                     **kwargs) -> "MultimodalTimeSeriesDataset":
        """
        Create a dataset from tensors.
        
        Args:
            modalities: Dictionary mapping modality names to their tensors
            targets: Optional target tensor
            **kwargs: Additional arguments for MultimodalTimeSeriesDataset
            
        Returns:
            MultimodalTimeSeriesDataset
        """
        return cls(modalities=modalities, targets=targets, **kwargs)
        
    @classmethod
    def from_csv_dict(cls, 
                      file_paths: Dict[str, str],
                      target_cols: Optional[Union[str, List[str]]] = None,
                      target_file: Optional[str] = None,
                      timestamp_cols: Optional[Dict[str, str]] = None,
                      seq_lens: Optional[Dict[str, int]] = None,
                      **kwargs) -> "MultimodalTimeSeriesDataset":
        """
        Create a dataset from multiple CSV files.
        
        Args:
            file_paths: Dictionary mapping modality names to their CSV file paths
            target_cols: Column(s) to use as target
            target_file: File to read targets from (if not specified, targets are read from the first file)
            timestamp_cols: Dictionary mapping modality names to their timestamp column names
            seq_lens: Dictionary mapping modality names to their sequence lengths
            **kwargs: Additional arguments for pandas.read_csv
            
        Returns:
            MultimodalTimeSeriesDataset
        """
        # Read targets
        targets = None
        if target_cols is not None:
            if target_file is not None:
                # Read targets from specified file
                if target_file not in file_paths.values():
                    target_df = pd.read_csv(target_file, **kwargs)
                    
                    if isinstance(target_cols, str):
                        target_cols = [target_cols]
                        
                    # Check if all target columns exist
                    missing_cols = [col for col in target_cols if col not in target_df.columns]
                    if missing_cols:
                        raise ValueError(f"Target column(s) {missing_cols} not found in target file")
                        
                    # Extract target columns
                    targets = target_df[target_cols].values
            else:
                # Use the first modality's file for targets
                first_modality = next(iter(file_paths))
                target_file = file_paths[first_modality]
        
        # Read data for each modality
        modalities = {}
        timestamps = {}
        
        for modality, file_path in file_paths.items():
            # Read CSV
            df = pd.read_csv(file_path, **kwargs)
            
            # Extract timestamps if specified
            if timestamp_cols is not None and modality in timestamp_cols:
                timestamp_col = timestamp_cols[modality]
                if timestamp_col in df.columns:
                    timestamps[modality] = pd.to_datetime(df[timestamp_col])
                    df = df.drop(columns=[timestamp_col])
                else:
                    raise ValueError(f"Timestamp column '{timestamp_col}' not found in CSV for modality '{modality}'")
                    
            # Extract targets if this is the target file and targets haven't been read yet
            if targets is None and target_cols is not None and file_path == target_file:
                if isinstance(target_cols, str):
                    target_cols = [target_cols]
                    
                # Check if all target columns exist
                missing_cols = [col for col in target_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Target column(s) {missing_cols} not found in CSV for modality '{modality}'")
                    
                # Extract target columns
                target_df = df[target_cols]
                df = df.drop(columns=target_cols)
                
                # Convert targets to numpy array
                targets = target_df.values
                
            # Add data to modalities
            modalities[modality] = df
            
        # Create dataset
        return cls(
            modalities=modalities,
            targets=targets,
            timestamps=timestamps if timestamps else None,
            seq_lens=seq_lens
        )
