from typing import Dict, List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    UniversalSamplingProcedure,
    EuclideanConstantProcedure,
)


class MixedCurvatureMoEFusion(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        components: Optional[List[Component]] = None,
        *,
        scalar_parametrization: bool = True,
        gate_hidden: Optional[int] = None,
        gate_sn: bool = False,
        lambda_lip: float = 0.0,
        init_mix_logit: float = -4.6,
    ) -> None:
        super().__init__()
        self.lambda_lip = float(lambda_lip)
        self.gate_sn = bool(gate_sn)

        self.components = nn.ModuleList()
        self.expert_projs = nn.ModuleList()

        if components is not None:
            for comp in components:
                comp.init_layers(emb_dim, scalar_parametrization=scalar_parametrization)
                self.components.append(comp)
                if getattr(comp, "dim", emb_dim) == emb_dim:
                    self.expert_projs.append(nn.Identity())
                else:
                    self.expert_projs.append(nn.Linear(comp.dim, emb_dim))

        self.num_experts = len(self.components)
        self.use_moe = self.num_experts > 0
        if self.use_moe and self.num_experts != 3:
            raise ValueError("MoE-based geometric fusion expects exactly 3 experts (e/h/s).")

        self.mix_logit = nn.Parameter(torch.tensor(init_mix_logit))

        if self.use_moe:
            gate_hidden = gate_hidden or emb_dim

            def _sn(layer: nn.Linear):
                return nn.utils.spectral_norm(layer) if self.gate_sn else layer

            self.gate_net = nn.Sequential(
                _sn(nn.Linear(self.num_experts * emb_dim, gate_hidden)),
                nn.ReLU(inplace=True),
                _sn(nn.Linear(gate_hidden, self.num_experts)),
            )

        self.reset_parameters()

    def reset_parameters(self):
        for proj in self.expert_projs:
            if isinstance(proj, nn.Linear):
                nn.init.xavier_uniform_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
        if self.use_moe:
            for layer in self.gate_net:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        for comp in self.components:
            for mod in comp.modules():
                if hasattr(mod, "reset_parameters"):
                    mod.reset_parameters()

    def _lip_loss(self) -> torch.Tensor:
        if not (self.use_moe and self.lambda_lip > 0):
            return torch.tensor(0.0, device=self.mix_logit.device)
        loss = torch.tensor(0.0, device=self.mix_logit.device)
        for m in self.gate_net:
            if isinstance(m, nn.Linear):
                loss = loss + (torch.linalg.norm(m.weight, ord=2) - 1.0) ** 2
        return self.lambda_lip * loss

    def forward(self, z_unit: torch.Tensor, *, scale: float = 1.0):
        if not self.use_moe:
            return z_unit, None, torch.tensor(0.0, device=z_unit.device)

        proj_outs = []
        for i, comp in enumerate(self.components):
            q_z, _, _ = comp(z_unit)
            z_c, _ = q_z.rsample_with_parts()
            proj = F.relu(self.expert_projs[i](z_c))
            proj_outs.append(proj)

        expert_stack = torch.stack(proj_outs, dim=1)
        logits = self.gate_net(expert_stack.flatten(1))
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)

        mixture = (expert_stack * weights).sum(1)
        mixture = F.normalize(mixture, p=2, dim=1)

        strength = torch.sigmoid(self.mix_logit) * float(scale)
        z_fused = F.normalize(z_unit + strength * mixture, p=2, dim=1)

        return z_fused, weights.squeeze(-1), self._lip_loss()


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
