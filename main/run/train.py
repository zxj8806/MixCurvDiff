import os
import os.path as osp
import argparse
from typing import Dict, Iterable, List, Tuple, Optional, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import APPNP
from torch_geometric.utils import add_self_loops, remove_self_loops, negative_sampling
from torch_scatter import scatter_add
from sklearn.metrics import roc_auc_score, average_precision_score

from main.MixCurv.components import (
    Component,
    ConstantComponent,
    EuclideanComponent,
    SphericalComponent,
    HyperbolicComponent,
    PoincareComponent,
    StereographicallyProjectedSphereComponent,
    UniversalComponent,
)
from main.MixCurv.sampling import (
    EuclideanNormalProcedure,
    WrappedNormalProcedure,
    SphericalVmfProcedure,
    ProjectedSphericalVmfProcedure,
    RiemannianNormalProcedure,
    UniversalSamplingProcedure,
    EuclideanConstantProcedure,
)

space_creator_map: Dict[str, Callable[..., Component]] = {
    "h": HyperbolicComponent,
    "u": UniversalComponent,
    "s": SphericalComponent,
    "d": StereographicallyProjectedSphereComponent,
    "p": PoincareComponent,
    "c": ConstantComponent,
    "e": EuclideanComponent,
}

sampling_procedure_map = {
    SphericalComponent: WrappedNormalProcedure,
    StereographicallyProjectedSphereComponent: WrappedNormalProcedure,
    EuclideanComponent: EuclideanNormalProcedure,
    ConstantComponent: EuclideanConstantProcedure,
    HyperbolicComponent: WrappedNormalProcedure,
    PoincareComponent: WrappedNormalProcedure,
    UniversalComponent: UniversalSamplingProcedure,
}

def _parse_component_str(space_str: str) -> Tuple[int, str, int]:
    space_str = space_str.split("-")[0]
    i = 0
    while i < len(space_str) and space_str[i].isdigit():
        i += 1
    multiplier = space_str[:i] or "1"
    j = i
    while j < len(space_str) and space_str[j].isalpha():
        j += 1
    space_type = space_str[i:j]
    dim = space_str[j:]
    return int(multiplier), space_type, int(dim)

def parse_components(arg: str, fixed_curvature: bool) -> List[Component]:
    def _create(mult: int, stype: str, dim: int, fixed: bool):
        if mult < 1:
            raise ValueError("Multiplier must be ≥1")
        if dim < 1:
            raise ValueError("Dimension must be ≥1")
        if stype not in space_creator_map:
            raise NotImplementedError(f"Unknown space type '{stype}'")
        ctor = space_creator_map[stype]
        for _ in range(mult):
            yield ctor(dim, fixed, sampling_procedure=sampling_procedure_map[ctor])

    arg = arg.lower().strip()
    if not arg:
        return []
    comps: List[Component] = []
    for spec in arg.split(","):
        mult, stype, dim = _parse_component_str(spec.strip())
        comps.extend(_create(mult, stype, dim, fixed_curvature))
    return comps

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    return torch.linspace(beta_start, beta_end, timesteps)

def _get_value_at_index(tensor: torch.Tensor, index: int):
    idx = index - 1
    if idx < 0 or idx >= tensor.numel():
        raise IndexError("Index out of bounds")
    return tensor[idx].item()

def compute_diffusion_params(timesteps: int, device: torch.device):
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_1m = torch.sqrt(1. - alphas_cumprod)
    rev_cumsum = torch.flip(sqrt_1m, dims=[0]).cumsum(dim=0)
    cum_sqrt_1m = torch.flip(rev_cumsum, dims=[0])
    return sqrt_1m, cum_sqrt_1m

class Reparametrized:
    def __init__(self, q_z, p_z, z):
        self.q_z, self.p_z, self.z = q_z, p_z, z

_LOGSTD_MAX = 2.0

class TwoLaneDiffusion(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        timesteps: int,
        components: Optional[List[Component]] = None,
        scalar_parametrization: bool = True,
        noise_type: str = "none",
        inject_lane: str = "feature",
        noise_stage: str = "encode",
        noise_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        gate_sn: bool = False,
        lambda_lip: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.timesteps = timesteps

        self.gate_sn = gate_sn
        self.lambda_lip = lambda_lip

        sqrt_1m, cum_sqrt_1m = compute_diffusion_params(timesteps, torch.device("cpu"))
        self.register_buffer("sqrt_1m_alphas_cumprod", sqrt_1m)
        self.register_buffer("cum_sqrt_1m_alphas_cumprod", cum_sqrt_1m)

        self.mu_feat_list, self.logstd_feat_list, self.appnp_feat_list = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(timesteps):
            self.mu_feat_list.append(nn.Linear(in_channels, hidden_channels))
            self.logstd_feat_list.append(nn.Linear(in_channels, hidden_channels))
            self.appnp_feat_list.append(APPNP(K=1, alpha=0.0))

        self.register_buffer("id_eye", torch.eye(num_nodes))
        self.mu_id_list, self.logstd_id_list = nn.ModuleList(), nn.ModuleList()
        for _ in range(timesteps):
            self.mu_id_list.append(nn.Linear(num_nodes, hidden_channels, bias=False))
            self.logstd_id_list.append(nn.Linear(num_nodes, hidden_channels, bias=False))

        self.components, self.expert_projs = nn.ModuleList(), nn.ModuleList()
        if components is not None:
            for comp in components:
                comp.init_layers(hidden_channels, scalar_parametrization=scalar_parametrization)
                self.components.append(comp)
                if comp.dim == hidden_channels:
                    self.expert_projs.append(nn.Identity())
                else:
                    self.expert_projs.append(nn.Linear(comp.dim, hidden_channels))

        self.num_experts = len(self.components)
        self.use_mixed = self.num_experts > 0
        if self.use_mixed and self.num_experts != 3:
            raise ValueError("MoE‑Based Geometric Fusion expects exactly 3 experts (E/H/S)")

        if self.use_mixed:
            def _sn(layer: nn.Linear):
                return nn.utils.spectral_norm(layer) if self.gate_sn else layer
            self.gate_net = nn.Sequential(
                _sn(nn.Linear(3 * hidden_channels, hidden_channels)),
                nn.ReLU(inplace=True),
                _sn(nn.Linear(hidden_channels, 3)),
            )
        self.mix_logit = nn.Parameter(torch.tensor(-4.6))

        assert noise_type in ("none", "gaussian", "struct", "learnable")
        assert inject_lane in ("feature", "id", "both")
        assert noise_stage in ("encode", "decode", "both")
        assert noise_schedule in ("linear", "cosine")
        self.noise_type, self.inject_lane, self.noise_stage, self.noise_schedule = (
            noise_type, inject_lane, noise_stage, noise_schedule
        )
        self.beta_start, self.beta_end = beta_start, beta_end
        if self.noise_type == "learnable":
            self.learnable_noise = nn.Parameter(torch.tensor(0.05))

        self.reset_parameters()

    def reset_parameters(self):
        for m in (
            list(self.mu_feat_list)
            + list(self.logstd_feat_list)
            + list(self.mu_id_list)
            + list(self.logstd_id_list)
        ):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        if self.use_mixed:
            for layer in self.gate_net:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

        for proj in self.expert_projs:
            if isinstance(proj, nn.Linear):
                nn.init.xavier_uniform_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

        for comp in self.components:
            for mod in comp.modules():
                if hasattr(mod, "reset_parameters"):
                    mod.reset_parameters()

    def _noise_scale(self, t: int) -> float:
        if self.noise_schedule == "linear":
            beta_t = self.beta_start + (self.beta_end - self.beta_start) * (t / (self.timesteps - 1))
            return float(np.sqrt(beta_t))
        s = 0.008
        t_norm = t / (self.timesteps - 1)
        alpha_bar = np.cos((t_norm + s) / (1 + s) * np.pi / 2) ** 2
        return float(np.sqrt(1 - alpha_bar))

    def _apply_noise(self, tensor: torch.Tensor, edge_index: Optional[torch.Tensor], t: int, lane: str) -> torch.Tensor:
        if self.noise_type == "none": return tensor
        scale = self._noise_scale(t)
        if self.noise_type == "gaussian":
            noisy = tensor + scale * torch.randn_like(tensor)
        elif self.noise_type == "struct":
            if edge_index is None: noisy = tensor
            else:
                noise = torch.randn_like(tensor)
                row, col = edge_index
                deg = scatter_add(torch.ones_like(row, dtype=tensor.dtype), col, dim=0, dim_size=tensor.size(0)).unsqueeze(1).clamp(min=1.)
                smooth = scatter_add(noise[row], col, dim=0, dim_size=tensor.size(0)) / deg
                noisy = tensor + scale * smooth
        elif self.noise_type == "learnable":
            noisy = tensor + self.learnable_noise * torch.randn_like(tensor)
        else: noisy = tensor
        return F.normalize(noisy, p=2, dim=1) if lane == "feature" else noisy

    @staticmethod
    def _reparam(mu: torch.Tensor, logstd: torch.Tensor, training: bool):
        return mu + torch.randn_like(mu) * torch.exp(logstd) if training else mu

    def _compute_lip_loss(self):
        if not (self.use_mixed and self.lambda_lip > 0): return torch.tensor(0., device=self.mix_logit.device)
        loss = torch.tensor(0., device=self.mix_logit.device)
        for m in self.gate_net:
            if isinstance(m, nn.Linear):
                loss += (torch.linalg.norm(m.weight, ord=2) - 1.) ** 2
        return self.lambda_lip * loss

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, *, return_weights: bool = False):
        
        zF_list, zI_list, weights_list = [], [], []
        for t in range(self.timesteps):
            muF = self.appnp_feat_list[t](self.mu_feat_list[t](x), edge_index)
            logstdF = self.appnp_feat_list[t](self.logstd_feat_list[t](x), edge_index).clamp(max=_LOGSTD_MAX)
            zF = F.normalize(muF, p=2, dim=1)
            if self.noise_stage in ("encode", "both") and self.inject_lane in ("feature", "both"):
                zF = self._apply_noise(zF, edge_index, t, "feature")

            muI = F.normalize(self.mu_id_list[t](self.id_eye), p=2, dim=1) * 0.8
            logstdI = self.logstd_id_list[t](self.id_eye).clamp(max=_LOGSTD_MAX)
            zI = self._reparam(muI, logstdI, self.training)
            if self.noise_stage in ("encode", "both") and self.inject_lane in ("id", "both"):
                zI = self._apply_noise(zI, edge_index, t, "id")

            if self.use_mixed:
                proj_outs = []
                for i, comp in enumerate(self.components):
                    q_z, _, _ = comp(zF)
                    z_c, _ = q_z.rsample_with_parts()
                    proj = F.relu(self.expert_projs[i](z_c))
                    proj_outs.append(proj)
                expert_stack = torch.stack(proj_outs, dim=1)
                psi = expert_stack.flatten(1)
                logits = self.gate_net(psi)
                weights = torch.softmax(logits, 1).unsqueeze(-1)
                mixture = F.normalize((expert_stack * weights).sum(1), p=2, dim=1)
                zF = zF + torch.sigmoid(self.mix_logit) * mixture

            else:
                pass

            zF = F.normalize(zF, p=2, dim=1)
            zF_list.append(zF); zI_list.append(zI)

        if return_weights:
            return zF_list, zI_list, weights_list
        return zF_list, zI_list

    def _maybe_noisy(self, tensor, edge_index, tau, lane):
        if self.noise_stage in ("decode", "both") and (self.inject_lane in (lane, "both")):
            return self._apply_noise(tensor, edge_index, tau - 1, lane)
        return tensor

    def decode(self, zF_list, zI_list, edge_index, start_t=1, sigmoid=True, temp=1.0):
        device = zF_list[0].device
        norm = _get_value_at_index(self.cum_sqrt_1m_alphas_cumprod, start_t)
        E = edge_index.size(1); value = torch.zeros(E, device=device)
        for tau in range(start_t, self.timesteps + 1):
            zF = self._maybe_noisy(zF_list[tau - 1], edge_index, tau, "feature")
            zI = self._maybe_noisy(zI_list[tau - 1], edge_index, tau, "id")
            sF = (zF[edge_index[0]] * zF[edge_index[1]]).sum(1)
            sI = torch.sigmoid(zI[edge_index[0], 0] + zI[edge_index[1], 0])
            det = sF.detach()
            logits = torch.stack([det, torch.zeros_like(det)], 1)
            a = F.gumbel_softmax(logits, tau=temp, hard=True)[:, 0] if self.training else F.softmax(logits, 1)[:, 0]
            value += _get_value_at_index(self.sqrt_1m_alphas_cumprod, tau) * (a * sF + (1 - a) * sI)
        value /= norm
        return torch.clamp(value, 0, 1) if sigmoid else value

    def decode_all(self, zF_list, zI_list, start_t=1, sigmoid=True, temp=1.0):
        device = zF_list[0].device
        norm = _get_value_at_index(self.cum_sqrt_1m_alphas_cumprod, start_t)
        N = self.num_nodes; adj = torch.zeros(N, N, device=device)
        for tau in range(start_t, self.timesteps + 1):
            zF = self._maybe_noisy(zF_list[tau - 1], None, tau, "feature")
            zI = self._maybe_noisy(zI_list[tau - 1], None, tau, "id")
            fv = zF @ zF.t()
            nv = torch.sigmoid(zI[:, 0].unsqueeze(1) + zI[:, 0].unsqueeze(0))
            det = fv.flatten().detach()
            logits = torch.stack([det, torch.zeros_like(det)], 1)
            a = F.gumbel_softmax(logits, tau=temp, hard=True)[:, 0] if self.training else F.softmax(logits, 1)[:, 0]
            a = a.view(N, N)
            adj += _get_value_at_index(self.sqrt_1m_alphas_cumprod, tau) * (a * fv + (1 - a) * nv)
        adj /= norm
        return torch.clamp(adj, 0, 1) if sigmoid else adj

    def recon_loss(self, zF_list, zI_list, pos_edge_index, start_t=1, temp=1.0):
        pos = self.decode(zF_list, zI_list, pos_edge_index, start_t, True, temp)
        loss_pos = -torch.log(pos + 1e-15).sum()
        ei, _ = remove_self_loops(pos_edge_index); ei, _ = add_self_loops(ei)
        neg_ei = negative_sampling(ei, num_nodes=self.num_nodes, force_undirected=False)
        neg = self.decode(zF_list, zI_list, neg_ei, start_t, True, temp)
        loss_neg = -torch.log(1 - neg + 1e-15).sum()
        return loss_pos + loss_neg

    @torch.no_grad()
    def test(self, zF_list, zI_list, pos_ei, neg_ei, start_t=1, temp=1.0):
        pos_pred = self.decode(zF_list, zI_list, pos_ei, start_t, True, temp)
        neg_pred = self.decode(zF_list, zI_list, neg_ei, start_t, True, temp)
        y = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)]).cpu().numpy()
        pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)

    def forward(self, x, edge_index, start_t=1, temp=1.0):
        zF_l, zI_l = self.encode(x, edge_index)
        recon_adj = self.decode_all(zF_l, zI_l, start_t, temp=temp)
        return recon_adj, self._compute_lip_loss()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora", choices=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=128)
    parser.add_argument("--noise_type", default="none", choices=["none", "gaussian", "struct", "learnable"])
    parser.add_argument("--inject_lane", default="id", choices=["feature", "id", "both"])
    parser.add_argument("--noise_stage", default="both", choices=["encode", "decode", "both"])
    parser.add_argument("--noise_schedule", default="linear", choices=["linear", "cosine"])
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)
    parser.add_argument("--gate_sn", action="store_true")
    parser.add_argument("--lambda_lip", type=float, default=0.)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    data_root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Planetoid")
    dataset = Planetoid(data_root, args.dataset, transform=transform)
    train_data, val_data, test_data = dataset[0]

    comps = parse_components("e2,h2,s2", fixed_curvature=True)

    model = TwoLaneDiffusion(
        num_nodes=train_data.num_nodes,
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        timesteps=args.timesteps,
        components=comps,
        noise_type=args.noise_type,
        inject_lane=args.inject_lane,
        noise_stage=args.noise_stage,
        noise_schedule=args.noise_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        gate_sn=args.gate_sn,
        lambda_lip=args.lambda_lip,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    MAX_TEMP, MIN_TEMP, decay_step = 2.0, 0.1, 150.0
    decay_w = np.log(MAX_TEMP / MIN_TEMP)
    get_temp = lambda epoch: max(MAX_TEMP * np.exp(-(epoch - 1) / decay_step * decay_w), MIN_TEMP)


    @torch.no_grad()
    def eval_auc_ap(pos_ei, neg_ei, epoch):
        model.eval()
        zF_l, zI_l, w_l = model.encode(train_data.x, train_data.edge_index, return_weights=True)
        auc, ap = model.test(zF_l, zI_l, pos_ei, neg_ei, temp=get_temp(epoch))
        return auc, ap

    for epoch in range(1, args.epochs + 1):
        t_rand = torch.randint(1, args.timesteps + 1, (1,), device=device).item()
        model.train(); optimizer.zero_grad()
        zF_l, zI_l = model.encode(train_data.x, train_data.edge_index)
        recon_loss = model.recon_loss(zF_l, zI_l, train_data.pos_edge_label_index, start_t=t_rand, temp=get_temp(epoch))
        lip_loss = model._compute_lip_loss()
        loss = recon_loss + lip_loss
        loss.backward(); optimizer.step()

        val_auc, val_ap = eval_auc_ap(val_data.pos_edge_label_index, val_data.neg_edge_label_index, epoch)
        if epoch % 2 == 0:
            print(f"Epoch {epoch:03d} | t={t_rand:03d} | Loss {recon_loss:.4f} | "
                  f"Val AUC {val_auc:.4f} | Val AP {val_ap:.4f}")

    test_auc, test_ap = eval_auc_ap(test_data.pos_edge_label_index, test_data.neg_edge_label_index, args.epochs)
    print(f"Final Test  AUC: {test_auc:.4f} | AP: {test_ap:.4f}")


if __name__ == "__main__":
    main()
