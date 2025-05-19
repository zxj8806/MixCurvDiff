from typing import Tuple

import torch
from torch import Tensor


def rexpand(a: Tensor, *dimensions: Tuple[int]) -> Tensor:
    return a.view(a.shape + (1,) * len(dimensions)).expand(a.shape + tuple(dimensions))


def log_sum_exp_signs(value: Tensor, signs: Tensor, dim: int = 0, keepdim: bool = False) -> Tensor:
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.clamp(torch.sum(signs * torch.exp(value0), dim=dim, keepdim=keepdim), min=1e15))
