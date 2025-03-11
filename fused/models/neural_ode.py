"""
Neural ODE implementation for FUSED.

This module contains the implementation of Neural ODEs for modeling
continuous-time trajectories in latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
try:
    import torchdiffeq
except ImportError:
    print("Warning: torchdiffeq not found. Neural ODE functionality will not be available.")
    print("Install torchdiffeq with: pip install torchdiffeq")
    torchdiffeq = None

from fused.models.base import TemporalModel


class DynamicsNetwork(nn.Module):
    """
    Neural network for modeling dynamics in Neural ODE.
    """
    
    def __init__(self, 
                 latent_dim: int, 
                 hidden_dims: List[int] = [128, 128]):
        """
        Initialize the dynamics network.
        
        Args:
            latent_dim: Dimension of latent space
            hidden_dims: Dimensions of hidden layers
        """
        super().__init__()
        
        # Build network
        layers = []
        input_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, latent_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamics.
        
        Args:
            t: Current time
            z: Current state
            
        Returns:
            Time derivative of state
        """
        return self.network(z)


class ContinuousTemporalFlow(TemporalModel):
    """
    Continuous temporal flow using Neural ODEs.
    
    This model uses Neural ODEs to model continuous-time
    trajectories in latent space.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the continuous temporal flow.
        
        Args:
            config: Configuration dictionary with the following keys:
                latent_dim: Dimension of latent space
                hidden_dims: Dimensions of hidden layers for dynamics network
                solver: ODE solver (e.g., 'dopri5', 'rk4', 'euler')
                solver_options: Options for ODE solver
        """
        super().__init__(config)
        
        if torchdiffeq is None:
            raise ImportError(
                "torchdiffeq is required for Neural ODE. "
                "Install with: pip install torchdiffeq"
            )
        
        # Extract configuration
        latent_dim = config.get("latent_dim", 128)
        hidden_dims = config.get("hidden_dims", [256, 256])
        solver = config.get("solver", "dopri5")
        
        solver_options = config.get("solver_options", {})
        default_options = {
            "rtol": 1e-4,
            "atol": 1e-4,
        }
        solver_options = {**default_options, **solver_options}
        
        # Dynamics network
        self.dynamics = DynamicsNetwork(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        )
        
        # Save configuration
        self.latent_dim = latent_dim
        self.solver = solver
        self.solver_options = solver_options
        
    def forward(self, 
                z: torch.Tensor, 
                times: torch.Tensor) -> torch.Tensor:
        """
        Process a sequence of representations.
        
        Args:
            z: Initial state [batch_size, latent_dim]
            times: Time points at which to evaluate trajectory [batch_size, num_times]
            
        Returns:
            Trajectory points at requested times [batch_size, num_times, latent_dim]
        """
        batch_size, num_times = times.shape
        
        # Ensure z has shape [batch_size, latent_dim]
        if len(z.shape) == 3 and z.shape[1] == 1:
            z = z.squeeze(1)
        
        # Process each batch element separately
        trajectories = []
        
        for b in range(batch_size):
            # Get time points for this batch element
            t = times[b]
            z0 = z[b]
            
            # Solve ODE
            trajectory = torchdiffeq.odeint(
                func=self.dynamics,
                y0=z0,
                t=t,
                method=self.solver,
                options=self.solver_options
            )
            
            # trajectory has shape [num_times, latent_dim]
            trajectories.append(trajectory)
            
        # Stack trajectories
        trajectories = torch.stack(trajectories, dim=0)
        
        # Transpose to get [batch_size, num_times, latent_dim]
        trajectories = trajectories.transpose(0, 1)
        
        return trajectories
    
    
class ProbabilisticTemporalFlow(ContinuousTemporalFlow):
    """
    Probabilistic continuous temporal flow.
    
    This model extends the Neural ODE with uncertainty estimation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the probabilistic temporal flow.
        
        Args:
            config: Configuration dictionary with the following keys:
                latent_dim: Dimension of latent space
                hidden_dims: Dimensions of hidden layers for dynamics network
                solver: ODE solver (e.g., 'dopri5', 'rk4', 'euler')
                solver_options: Options for ODE solver
                num_samples: Number of samples for uncertainty estimation
        """
        super().__init__(config)
        
        # Extract configuration
        self.num_samples = config.get("num_samples", 10)
        
        # Add uncertainty estimation network
        self.uncertainty_network = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.GELU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim)
        )
        
    def forward(self, 
                z: torch.Tensor, 
                times: torch.Tensor,
                return_uncertainty: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process a sequence of representations with uncertainty estimation.
        
        Args:
            z: Initial state [batch_size, latent_dim]
            times: Time points at which to evaluate trajectory [batch_size, num_times]
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Trajectory points at requested times [batch_size, num_times, latent_dim]
            If return_uncertainty is True, also returns uncertainty estimates
        """
        batch_size, num_times = times.shape
        
        # Ensure z has shape [batch_size, latent_dim]
        if len(z.shape) == 3 and z.shape[1] == 1:
            z = z.squeeze(1)
        
        # Estimate uncertainty
        log_variance = self.uncertainty_network(z)
        
        # Generate samples
        samples = []
        
        for _ in range(self.num_samples):
            # Add noise to initial state
            noise = torch.randn_like(z) * torch.exp(0.5 * log_variance)
            z_noisy = z + noise
            
            # Get trajectory for noisy initial state
            trajectory = super().forward(z_noisy, times)
            samples.append(trajectory)
            
        # Stack samples
        samples = torch.stack(samples, dim=1)  # [batch_size, num_samples, num_times, latent_dim]
        
        # Compute mean trajectory
        mean_trajectory = samples.mean(dim=1)  # [batch_size, num_times, latent_dim]
        
        if return_uncertainty:
            # Compute uncertainty (standard deviation across samples)
            uncertainty = samples.std(dim=1)  # [batch_size, num_times, latent_dim]
            return mean_trajectory, uncertainty
        else:
            return mean_trajectory
