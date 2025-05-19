from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.distributions
import torch.nn.functional as F

from .wrapped_distributions import VaeDistribution, EuclideanNormal
from ..ops.common import ln_2, ln_pi


class HypersphericalUniform(torch.distributions.Distribution, VaeDistribution):
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    def __init__(self, dim: int, validate_args: Optional[bool] = None,
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__(torch.Size([dim]), validate_args=validate_args)
        self.dim = dim
        self.device = device
        self.normal = EuclideanNormal(0, 1)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        output = self.normal.sample(sample_shape + torch.Size([1, self.dim + 1])).to(self.device)
        return F.normalize(output, dim=-1)

    def entropy(self) -> Tensor:
        return self.__log_surface_area()

    def log_prob(self, x: Tensor) -> Tensor:
        return -torch.ones(x.shape[:-1], device=self.device) * self.__log_surface_area()

    def __log_surface_area(self) -> Tensor:
        n = self.dim
        t = torch.tensor((n + 1.) / 2.)
        ret = ln_2 + t * ln_pi - torch.lgamma(t)  # + n * torch.log(self.radius)
        return ret

    def rsample_with_parts(self, shape: torch.Size = torch.Size()) -> Tuple[Tensor, Optional[Tuple[Tensor, ...]]]:
        z = self.rsample(shape)
        return z, None

    def log_prob_from_parts(self, z: Tensor, data: Optional[Tuple[Tensor, ...]]) -> Tensor:
        log_prob = self.log_prob(z)
        assert torch.isfinite(log_prob).all()
        return log_prob

    def log_normalizer(self) -> Tensor:
        return self.__log_surface_area().to(self.device)
