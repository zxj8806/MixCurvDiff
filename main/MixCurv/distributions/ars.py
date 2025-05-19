from typing import Callable, Any, Tuple, Optional

import torch
from torch import Tensor

infty = torch.tensor(float("Inf"))


def diff(x: Tensor) -> Tensor:
    return x[:, 1:] - x[:, :-1]


class ARS:

    def __init__(self,
                 logpdf: Callable[[Tensor, Any], Tensor],
                 grad_logpdf: Callable[[Tensor, Any], Tensor],
                 device: torch.device,
                 xi: Tensor,
                 lb: float = -infty,
                 ub: float = infty,
                 use_lower: bool = False,
                 ns: int = 50,
                 **fargs: Any) -> None:

        self.device = device

        self.lb = lb
        self.ub = ub

        self.logpdf = logpdf
        self.grad_logpdf = grad_logpdf
        self.fargs = fargs

        self.ns = ns
        self.xi = xi.to(self.device)
        self.B, self.K = self.xi.size()  # hull size
        self.h = torch.zeros(self.B, ns).to(self.device)
        self.hprime = torch.zeros(self.B, ns).to(self.device)
        self.x = torch.zeros(self.B, ns).to(self.device)
        self.h[:, :self.K] = self.logpdf(self.xi, **self.fargs)
        self.hprime[:, :self.K] = self.grad_logpdf(self.xi, **self.fargs)
        assert torch.isfinite(self.hprime).all()
        self.x[:, :self.K] = self.xi

        self.offset = self.h.max(-1)[0].view(-1, 1)
        self.h = self.h - self.offset

        if not (self.hprime[:, 0] > 0).all():
            raise IOError("Initial anchor points must span mode of PDF (left).")
        if not (self.hprime[:, self.K - 1] < 0).all():
            raise IOError("Initial anchor points must span mode of PDF (right).")
        self.recalculate_hull()

    def sample(self, shape: torch.Size = torch.Size(), max_steps: int = 1e3) -> Tensor:
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
        samples = torch.ones(self.B, *shape).to(self.device)
        bool_mask = (torch.ones(self.B, *shape) == 1).to(self.device)
        count = 0
        while bool_mask.sum() != 0:
            count += 1
            xt, i = self.sample_upper(shape)
            ht = self.logpdf(xt, **self.fargs)
            hprimet = self.grad_logpdf(xt, **self.fargs)
            ht = ht - self.offset
            ut = self.h.gather(1, i) + (xt - self.x.gather(1, i)) * self.hprime.gather(1, i)

            u = torch.rand(shape).to(self.device)
            accept = u < torch.exp(ht - ut)
            reject = 1 - accept
            samples[bool_mask * accept] = xt[bool_mask * accept]
            bool_mask[bool_mask * accept] = reject[bool_mask * accept]
            if self.K < self.ns:
                nb_insert = self.ns - self.K
                self.recalculate_hull(nb_insert, xt[:, :nb_insert], ht[:, :nb_insert], hprimet[:, :nb_insert])

            if count > max_steps:
                raise ValueError(f"ARS did not converge in {max_steps} steps ({bool_mask.sum()}/{bool_mask.shape}).")

        return samples.t().unsqueeze(-1)

    def recalculate_hull(self,
                         nbnew: int = 0,
                         xnew: Optional[Tensor] = None,
                         hnew: Optional[Tensor] = None,
                         hprimenew: Optional[Tensor] = None) -> None:
        if xnew is not None:
            self.x[:, self.K:self.K + nbnew] = xnew
            self.x, idx = self.x.sort()
            self.h[:, self.K:self.K + nbnew] = hnew
            self.h = self.h.gather(1, idx)
            self.hprime[:, self.K:self.K + nbnew] = hprimenew
            self.hprime = self.hprime.gather(1, idx)

            self.K += xnew.size(-1)

        self.z = torch.zeros(self.B, self.K + 1).to(self.device)
        self.z[:, 0] = self.lb
        self.z[:, self.K] = self.ub
        self.z[:, 1:self.K] = (diff(self.h[:, :self.K]) -
                               diff(self.x[:, :self.K] * self.hprime[:, :self.K])) / -diff(self.hprime[:, :self.K])
        idx = [0] + list(range(self.K))
        self.u = self.h[:, idx] + self.hprime[:, idx] * (self.z - self.x[:, idx])
        exp_u = torch.exp(self.u)
        self.s = diff(exp_u) / self.hprime[:, :self.K]
        self.s[self.hprime[:, :self.K] == 0.] = 0.  # should be 0 when gradient is 0
        assert torch.isfinite(self.s).all()
        self.cs = torch.cat((torch.zeros(self.B, 1).to(self.device), torch.cumsum(self.s, dim=-1)), dim=-1)
        self.cu = self.cs[:, -1]
        assert torch.isfinite(self.cu).all()

    def sample_upper(self, shape: torch.Size = torch.Size()) -> Tuple[Tensor, Tensor]:
        u = torch.rand(self.B, *shape, device=self.device)
        i = (self.cs / self.cu.unsqueeze(-1)).unsqueeze(-1) <= u.unsqueeze(1).expand(*self.cs.shape, *shape)
        idx = i.sum(1) - 1

        xt = self.x.gather(1, idx) + (-self.h.gather(1, idx) + torch.log(
            self.hprime.gather(1, idx) * (self.cu.unsqueeze(-1) * u - self.cs.gather(1, idx)) +
            torch.exp(self.u.gather(1, idx)))) / self.hprime.gather(1, idx)
        assert torch.isfinite(xt).all()
        return xt, idx
