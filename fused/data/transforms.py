"""
Data transforms for time series in FUSED.

This module contains transformations for time series data preprocessing
and augmentation.
"""

import torch
import numpy as np
from typing import List, Tuple, Union, Optional, Callable
import scipy.signal as signal
from scipy.fft import fft


class Compose:
    """
    Compose multiple transforms.
    """
    
    def __init__(self, transforms: List[Callable]):
        """
        Initialize the composition of transforms.
        
        Args:
            transforms: List of transforms to apply in sequence
        """
        self.transforms = transforms
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all transforms in sequence.
        
        Args:
            x: Input tensor
            
        Returns:
            Transformed tensor
        """
        for transform in self.transforms:
            x = transform(x)
        return x


class Normalize:
    """
    Normalize time series data.
    """
    
    def __init__(self, 
                 method: str = 'standard',
                 mean: Optional[Union[float, torch.Tensor]] = None,
                 std: Optional[Union[float, torch.Tensor]] = None,
                 min_val: Optional[Union[float, torch.Tensor]] = None,
                 max_val: Optional[Union[float, torch.Tensor]] = None,
                 axis: int = 0):
        """
        Initialize normalization transform.
        
        Args:
            method: Normalization method ('standard', 'minmax', or 'robust')
            mean: Mean value for standard normalization (computed from data if None)
            std: Standard deviation for standard normalization (computed from data if None)
            min_val: Minimum value for minmax normalization (computed from data if None)
            max_val: Maximum value for minmax normalization (computed from data if None)
            axis: Axis along which to normalize (0 for per-feature, None for global)
        """
        self.method = method
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.axis = axis
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor.
        
        Args:
            x: Input tensor of shape [seq_len, features] or [batch, seq_len, features]
            
        Returns:
            Normalized tensor of the same shape
        """
        if self.method == 'standard':
            # Standard normalization (z-score)
            if self.mean is None:
                mean = x.mean(dim=self.axis, keepdim=True)
            else:
                mean = self.mean
                
            if self.std is None:
                std = x.std(dim=self.axis, keepdim=True)
            else:
                std = self.std
                
            # Avoid division by zero
            std = torch.clamp(std, min=1e-8)
            
            return (x - mean) / std
            
        elif self.method == 'minmax':
            # Min-max normalization
            if self.min_val is None:
                min_val = x.min(dim=self.axis, keepdim=True)[0]
            else:
                min_val = self.min_val
                
            if self.max_val is None:
                max_val = x.max(dim=self.axis, keepdim=True)[0]
            else:
                max_val = self.max_val
                
            # Avoid division by zero
            denominator = max_val - min_val
            denominator = torch.clamp(denominator, min=1e-8)
            
            return (x - min_val) / denominator
            
        elif self.method == 'robust':
            # Robust normalization (median and IQR)
            if self.mean is None:
                median = torch.median(x, dim=self.axis, keepdim=True)[0]
            else:
                median = self.mean
                
            if self.std is None:
                q75 = torch.quantile(x, 0.75, dim=self.axis, keepdim=True)
                q25 = torch.quantile(x, 0.25, dim=self.axis, keepdim=True)
                iqr = q75 - q25
            else:
                iqr = self.std
                
            # Avoid division by zero
            iqr = torch.clamp(iqr, min=1e-8)
            
            return (x - median) / iqr
            
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")


class RandomCrop:
    """
    Randomly crop a time series.
    """
    
    def __init__(self, crop_length: int):
        """
        Initialize random crop transform.
        
        Args:
            crop_length: Length of the cropped sequence
        """
        self.crop_length = crop_length
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly crop the input tensor.
        
        Args:
            x: Input tensor of shape [seq_len, features] or [batch, seq_len, features]
            
        Returns:
            Cropped tensor
        """
        # Get sequence length
        if len(x.shape) == 2:
            seq_len = x.shape[0]
        else:
            seq_len = x.shape[1]
            
        # Check if crop is possible
        if seq_len <= self.crop_length:
            return x
            
        # Random start index
        start = torch.randint(0, seq_len - self.crop_length + 1, (1,)).item()
        
        # Crop sequence
        if len(x.shape) == 2:
            return x[start:start+self.crop_length]
        else:
            return x[:, start:start+self.crop_length]


class RandomMask:
    """
    Randomly mask portions of a time series.
    """
    
    def __init__(self, 
                 mask_ratio: float = 0.15,
                 mask_value: float = 0.0):
        """
        Initialize random mask transform.
        
        Args:
            mask_ratio: Proportion of values to mask
            mask_value: Value to use for masked positions
        """
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly mask the input tensor.
        
        Args:
            x: Input tensor of shape [seq_len, features] or [batch, seq_len, features]
            
        Returns:
            Masked tensor
        """
        # Create a copy of the input
        masked = x.clone()
        
        # Generate mask
        if len(x.shape) == 2:
            seq_len, features = x.shape
            mask = torch.rand(seq_len, features) < self.mask_ratio
        else:
            batch, seq_len, features = x.shape
            mask = torch.rand(batch, seq_len, features) < self.mask_ratio
        
        # Apply mask
        masked[mask] = self.mask_value
        
        return masked


class TimeSeriesResample:
    """
    Resample a time series to a different frequency.
    """
    
    def __init__(self, 
                 factor: float,
                 method: str = 'linear'):
        """
        Initialize resampling transform.
        
        Args:
            factor: Resampling factor (>1 for upsampling, <1 for downsampling)
            method: Interpolation method ('linear', 'nearest', or 'cubic')
        """
        self.factor = factor
        self.method = method
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resample the input tensor.
        
        Args:
            x: Input tensor of shape [seq_len, features] or [batch, seq_len, features]
            
        Returns:
            Resampled tensor
        """
        # Get sequence length
        if len(x.shape) == 2:
            seq_len, features = x.shape
            batch = None
        else:
            batch, seq_len, features = x.shape
            
        # Calculate new sequence length
        new_seq_len = int(seq_len * self.factor)
        
        # Original time points
        orig_time = torch.linspace(0, 1, seq_len)
        
        # New time points
        new_time = torch.linspace(0, 1, new_seq_len)
        
        # Resample each feature for each batch
        if batch is None:
            # Handle 2D input
            resampled = torch.zeros((new_seq_len, features), dtype=x.dtype, device=x.device)
            
            for f in range(features):
                # Convert to numpy for interpolation
                x_np = x[:, f].cpu().numpy()
                orig_time_np = orig_time.cpu().numpy()
                new_time_np = new_time.cpu().numpy()
                
                # Interpolate
                if self.method == 'linear':
                    resampled_f = np.interp(new_time_np, orig_time_np, x_np)
                elif self.method == 'nearest':
                    resampled_f = np.interp(new_time_np, orig_time_np, x_np, left=x_np[0], right=x_np[-1])
                elif self.method == 'cubic':
                    from scipy import interpolate
                    f = interpolate.interp1d(orig_time_np, x_np, kind='cubic', bounds_error=False, fill_value="extrapolate")
                    resampled_f = f(new_time_np)
                else:
                    raise ValueError(f"Unknown interpolation method: {self.method}")
                    
                # Convert back to tensor
                resampled[:, f] = torch.tensor(resampled_f, dtype=x.dtype, device=x.device)
                
        else:
            # Handle 3D input
            resampled = torch.zeros((batch, new_seq_len, features), dtype=x.dtype, device=x.device)
            
            for b in range(batch):
                for f in range(features):
                    # Convert to numpy for interpolation
                    x_np = x[b, :, f].cpu().numpy()
                    orig_time_np = orig_time.cpu().numpy()
                    new_time_np = new_time.cpu().numpy()
                    
                    # Interpolate
                    if self.method == 'linear':
                        resampled_f = np.interp(new_time_np, orig_time_np, x_np)
                    elif self.method == 'nearest':
                        resampled_f = np.interp(new_time_np, orig_time_np, x_np, left=x_np[0], right=x_np[-1])
                    elif self.method == 'cubic':
                        from scipy import interpolate
                        f_interp = interpolate.interp1d(orig_time_np, x_np, kind='cubic', bounds_error=False, fill_value="extrapolate")
                        resampled_f = f_interp(new_time_np)
                    else:
                        raise ValueError(f"Unknown interpolation method: {self.method}")
                        
                    # Convert back to tensor
                    resampled[b, :, f] = torch.tensor(resampled_f, dtype=x.dtype, device=x.device)
                    
        return resampled


class FilterNaN:
    """
    Replace NaN values in a time series.
    """
    
    def __init__(self, 
                 replacement: str = 'zero',
                 fill_value: float = 0.0):
        """
        Initialize NaN filtering transform.
        
        Args:
            replacement: Replacement method ('zero', 'mean', 'median', 'interpolate', or 'value')
            fill_value: Value to use when replacement is 'value'
        """
        self.replacement = replacement
        self.fill_value = fill_value
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Replace NaN values in the input tensor.
        
        Args:
            x: Input tensor of shape [seq_len, features] or [batch, seq_len, features]
            
        Returns:
            Tensor with NaN values replaced
        """
        # Create a copy of the input
        filtered = x.clone()
        
        # Get dimensions
        if len(x.shape) == 2:
            seq_len, features = x.shape
            batch = None
        else:
            batch, seq_len, features = x.shape
            
        # Replace NaNs based on chosen method
        if self.replacement == 'zero':
            # Replace with zeros
            filtered = torch.nan_to_num(filtered, nan=0.0)
            
        elif self.replacement == 'mean':
            # Replace with feature means
            if batch is None:
                for f in range(features):
                    feature_mean = torch.nanmean(filtered[:, f])
                    filtered[:, f] = torch.nan_to_num(filtered[:, f], nan=feature_mean.item())
            else:
                for b in range(batch):
                    for f in range(features):
                        feature_mean = torch.nanmean(filtered[b, :, f])
                        filtered[b, :, f] = torch.nan_to_num(filtered[b, :, f], nan=feature_mean.item())
                        
        elif self.replacement == 'median':
            # Replace with feature medians
            if batch is None:
                for f in range(features):
                    feature_median = torch.nanmedian(filtered[:, f])
                    filtered[:, f] = torch.nan_to_num(filtered[:, f], nan=feature_median.item())
            else:
                for b in range(batch):
                    for f in range(features):
                        feature_median = torch.nanmedian(filtered[b, :, f])
                        filtered[b, :, f] = torch.nan_to_num(filtered[b, :, f], nan=feature_median.item())
                        
        elif self.replacement == 'interpolate':
            # Replace with linear interpolation
            if batch is None:
                for f in range(features):
                    # Get indices and values of non-NaN entries
                    mask = ~torch.isnan(filtered[:, f])
                    indices = torch.nonzero(mask).squeeze()
                    values = filtered[indices, f]
                    
                    if len(indices) == 0:
                        # All values are NaN, replace with zeros
                        filtered[:, f] = 0.0
                    else:
                        # Interpolate
                        for i in range(seq_len):
                            if torch.isnan(filtered[i, f]):
                                # Find nearest non-NaN values before and after
                                before = indices[indices < i]
                                after = indices[indices > i]
                                
                                if len(before) == 0 and len(after) == 0:
                                    # No valid values, use fill_value
                                    filtered[i, f] = self.fill_value
                                elif len(before) == 0:
                                    # No values before, use first valid value
                                    filtered[i, f] = values[0]
                                elif len(after) == 0:
                                    # No values after, use last valid value
                                    filtered[i, f] = values[-1]
                                else:
                                    # Interpolate between nearest values
                                    i_before = before[-1].item()
                                    i_after = after[0].item()
                                    v_before = filtered[i_before, f].item()
                                    v_after = filtered[i_after, f].item()
                                    
                                    # Linear interpolation
                                    t = (i - i_before) / (i_after - i_before)
                                    filtered[i, f] = v_before + t * (v_after - v_before)
            else:
                for b in range(batch):
                    for f in range(features):
                        # Get indices and values of non-NaN entries
                        mask = ~torch.isnan(filtered[b, :, f])
                        indices = torch.nonzero(mask).squeeze()
                        values = filtered[b, indices, f]
                        
                        if len(indices) == 0:
                            # All values are NaN, replace with zeros
                            filtered[b, :, f] = 0.0
                        else:
                            # Interpolate
                            for i in range(seq_len):
                                if torch.isnan(filtered[b, i, f]):
                                    # Find nearest non-NaN values before and after
                                    before = indices[indices < i]
                                    after = indices[indices > i]
                                    
                                    if len(before) == 0 and len(after) == 0:
                                        # No valid values, use fill_value
                                        filtered[b, i, f] = self.fill_value
                                    elif len(before) == 0:
                                        # No values before, use first valid value
                                        filtered[b, i, f] = values[0]
                                    elif len(after) == 0:
                                        # No values after, use last valid value
                                        filtered[b, i, f] = values[-1]
                                    else:
                                        # Interpolate between nearest values
                                        i_before = before[-1].item()
                                        i_after = after[0].item()
                                        v_before = filtered[b, i_before, f].item()
                                        v_after = filtered[b, i_after, f].item()
                                        
                                        # Linear interpolation
                                        t = (i - i_before) / (i_after - i_before)
                                        filtered[b, i, f] = v_before + t * (v_after - v_before)
                                        
        elif self.replacement == 'value':
            # Replace with specified value
            filtered = torch.nan_to_num(filtered, nan=self.fill_value)
            
        else:
            raise ValueError(f"Unknown replacement method: {self.replacement}")
            
        return filtered


class FFT:
    """
    Apply Fast Fourier Transform to a time series.
    """
    
    def __init__(self, 
                 return_type: str = 'magnitude',
                 normalize: bool = True):
        """
        Initialize FFT transform.
        
        Args:
            return_type: Type of output ('magnitude', 'phase', 'complex', or 'both')
            normalize: Whether to normalize the output
        """
        self.return_type = return_type
        self.normalize = normalize
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply FFT to the input tensor.
        
        Args:
            x: Input tensor of shape [seq_len, features] or [batch, seq_len, features]
            
        Returns:
            FFT transformed tensor
        """
        # Get dimensions
        if len(x.shape) == 2:
            seq_len, features = x.shape
            batch = None
        else:
            batch, seq_len, features = x.shape
            
        # Apply FFT
        if batch is None:
            # Handle 2D input
            fft_result = torch.zeros_like(x, dtype=torch.complex64)
            
            for f in range(features):
                fft_result[:, f] = torch.fft.fft(x[:, f])
                
            # Extract components based on return type
            if self.return_type == 'magnitude':
                result = torch.abs(fft_result)
            elif self.return_type == 'phase':
                result = torch.angle(fft_result)
            elif self.return_type == 'complex':
                result = fft_result
            elif self.return_type == 'both':
                # Return magnitude and phase as separate features
                magnitude = torch.abs(fft_result)
                phase = torch.angle(fft_result)
                result = torch.cat([magnitude, phase], dim=1)
            else:
                raise ValueError(f"Unknown return type: {self.return_type}")
                
        else:
            # Handle 3D input
            fft_result = torch.zeros((batch, seq_len, features), dtype=torch.complex64, device=x.device)
            
            for b in range(batch):
                for f in range(features):
                    fft_result[b, :, f] = torch.fft.fft(x[b, :, f])
                    
            # Extract components based on return type
            if self.return_type == 'magnitude':
                result = torch.abs(fft_result)
            elif self.return_type == 'phase':
                result = torch.angle(fft_result)
            elif self.return_type == 'complex':
                result = fft_result
            elif self.return_type == 'both':
                # Return magnitude and phase as separate features
                magnitude = torch.abs(fft_result)
                phase = torch.angle(fft_result)
                result = torch.cat([magnitude, phase], dim=2)
            else:
                raise ValueError(f"Unknown return type: {self.return_type}")
                
        # Normalize if requested
        if self.normalize and self.return_type != 'complex':
            if self.return_type == 'both':
                # Normalize magnitude only
                if batch is None:
                    features_half = features
                    max_val = torch.max(result[:, :features_half])
                    if max_val > 0:
                        result[:, :features_half] = result[:, :features_half] / max_val
                else:
                    features_half = features
                    for b in range(batch):
                        max_val = torch.max(result[b, :, :features_half])
                        if max_val > 0:
                            result[b, :, :features_half] = result[b, :, :features_half] / max_val
            else:
                # Normalize the whole result
                if batch is None:
                    max_val = torch.max(result)
                    if max_val > 0:
                        result = result / max_val
                else:
                    for b in range(batch):
                        max_val = torch.max(result[b])
                        if max_val > 0:
                            result[b] = result[b] / max_val
                            
        return result
