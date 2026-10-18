import torch
import torch.nn as nn
from ml_collections import ConfigDict

from algorithms.utils import make_broadcastable


class FlowMatchingScheduler(nn.Module):

    def __init__(self, config: ConfigDict):
        """Rectified-flow scheduler: time sampling and interpolation utilities."""
        super().__init__()
        self.config = config

    def sample_time(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Sample flow times (uniform or configured distribution) for a batch."""
        strategy = self.config.get('timestep_sampling', 'uniform')
        if strategy == 'beta':
            alpha = float(self.config.get('timestep_beta_alpha', 1.5))
            beta = float(self.config.get('timestep_beta_beta', 1.0))
            s = float(self.config.get('timestep_beta_s', 0.999))
            u = torch.distributions.Beta(alpha, beta).sample((batch_size, seq_len)).to(device)
            return s * (1.0 - u)
        return torch.rand(batch_size, seq_len, device=device)

    def get_flow_trajectory(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Interpolate x_t = (1-t) x_0 + t x_1 along the flow."""
        t = make_broadcastable(t, x_0.shape)
        return (1 - t) * x_0 + t * x_1

    def compute_target_velocity(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Target velocity x_1 - x_0 of the rectified flow."""
        return x_1 - x_0
