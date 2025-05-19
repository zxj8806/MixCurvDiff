from typing import Tuple, Optional

import torch
import torch.distributions
from torch import Tensor


class VaeDistribution:

    def rsample_with_parts(self, shape: torch.Size = torch.Size()) -> Tuple[Tensor, Optional[Tuple[Tensor, ...]]]:
        z = self.rsample(shape)
        return z, None

    def log_prob_from_parts(self, z: Tensor, data: Optional[Tuple[Tensor, ...]]) -> Tensor:
        log_prob = self.log_prob(z)
        assert torch.isfinite(log_prob).all()
        return log_prob

    def rsample_log_prob(self, shape: torch.Size = torch.Size()) -> Tuple[Tensor, Tensor]:
        z, data = self.rsample_with_parts(shape)
        return z, self.log_prob_from_parts(z, data)


class EuclideanNormal(torch.distributions.Normal, VaeDistribution):

    def log_prob(self, value: Tensor) -> Tensor:
        return super().log_prob(value).sum(dim=-1)


class EuclideanUniform(torch.distributions.Uniform, VaeDistribution):

    def log_prob(self, value: Tensor) -> Tensor:
        return super().log_prob(value).sum(dim=-1)
