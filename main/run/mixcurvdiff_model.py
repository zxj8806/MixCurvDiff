import os
import time
import json
import random
from typing import List, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from main.run.cluster_utils import Clustering_Metrics, GraphConvSparse

from main.run.cluster_assignment import ClusterAssignment, _pairwise_squared_distance
from main.run.cluster_prototype import ClusterPrototype
from main.run.diffusion_utils import compute_diffusion_params
from main.run.moe_fusion import MixedCurvatureMoEFusion


class MixCurvDiff(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.num_neurons = kwargs["num_neurons"]
        self.num_features = kwargs["num_features"]
        self.embedding_size = kwargs["embedding_size"]
        self.nClusters = kwargs["nClusters"]
        self.T = kwargs.get("T", 30)

        act = {
            "ReLU": F.relu,
            "Sigmoid": torch.sigmoid,
            "Tanh": torch.tanh,
        }.get(kwargs.get("activation", "ReLU"), F.relu)
        self.activation = act

        init_kappa = kwargs.get("init_kappa", 10.0)
        self.align_weight = kwargs.get("align_weight", 0.1)
        self.align_alpha = kwargs.get("align_alpha", 2)
        self.align_num_neg = kwargs.get("align_num_neg", 1)
        self.align_margin = kwargs.get("align_margin", 1.0)
        self.cluster_reg_weight = kwargs.get("cluster_reg_weight", 0.1)
        self.entropy_reg_weight = kwargs.get("entropy_reg_weight", 2e-3)

        self.moe_balance_weight = kwargs.get("moe_balance_weight", 1e-2)
        self.moe_smooth_weight = kwargs.get("moe_smooth_weight", 1e-2)
        self.curv_coh_power = kwargs.get("curv_coh_power", 1.0)
        self.grad_clip = kwargs.get("grad_clip", 5.0)

        self.log_kappa = nn.Parameter(torch.tensor(float(init_kappa)))
        self.z_weight = nn.Parameter(torch.zeros(self.T))

        self.base_gcn = GraphConvSparse(self.num_features, self.num_neurons, self.activation)
        self.gcn_mean = GraphConvSparse(self.num_neurons, self.embedding_size, lambda x: x)
        self.gcn_logsigma2 = GraphConvSparse(self.num_neurons, self.embedding_size, lambda x: x)

        self.assignment = ClusterAssignment(self.nClusters, self.embedding_size, kwargs["alpha"])

        components: Optional[List] = kwargs.get("components", None)
        self.moe = MixedCurvatureMoEFusion(
            self.embedding_size,
            components=components,
            scalar_parametrization=kwargs.get("scalar_parametrization", True),
            gate_hidden=kwargs.get("gate_hidden", None),
            gate_sn=kwargs.get("gate_sn", False),
            lambda_lip=kwargs.get("lambda_lip", 0.0),
            init_mix_logit=kwargs.get("init_mix_logit", -4.6),
        )

        self.prototype_mc_lambda = kwargs.get("prototype_mc_lambda", 0.05)
        self.prototype_mc_power = kwargs.get("prototype_mc_power", 1.0)
        self.prototype_mc_dist_lambda = kwargs.get("prototype_mc_dist_lambda", 0.02)
        self.prototype_hyp_scale = kwargs.get("prototype_hyp_scale", 0.90)

    def kappa(self):
        return F.softplus(self.log_kappa)

    def aggregate_poe(self, mus, logs2):
        T = len(mus)
        w_t = F.softmax(self.z_weight[:T], 0).view(T, 1, 1)
        mu_stack = torch.stack(mus, 0)
        logvar_stack = torch.stack(logs2, 0)
        precision = torch.exp(-logvar_stack)
        prec_w = w_t * precision
        return (prec_w * mu_stack).sum(0) / prec_w.sum(0)

    @staticmethod
    def _sample_negative_pairs(adj_dense, total_neg, device):
        n = adj_dense.size(0)
        neg_r = torch.empty(0, dtype=torch.long, device=device)
        neg_c = torch.empty(0, dtype=torch.long, device=device)
        invalid = adj_dense.bool() | torch.eye(n, device=device, dtype=torch.bool)
        while neg_r.numel() < total_neg:
            r = torch.randint(0, n, (total_neg,), device=device)
            c = torch.randint(0, n, (total_neg,), device=device)
            ok = ~invalid[r, c]
            neg_r = torch.cat([neg_r, r[ok]])
            neg_c = torch.cat([neg_c, c[ok]])
        return neg_r[:total_neg], neg_c[:total_neg]

    @staticmethod
    def align_loss(z, adj, alpha=2, num_neg=1, margin=1.0):
        if adj.is_sparse:
            row, col = adj.coalesce().indices()
        else:
            row, col = adj.nonzero(as_tuple=True)
        diff_pos = z[row] - z[col]
        pos = diff_pos.norm(2, 1).pow(alpha).mean()

        total_neg = row.size(0) * num_neg
        adj_d = adj.to_dense() if adj.is_sparse else adj.clone()
        nr, nc = MixCurvDiff._sample_negative_pairs(adj_d, total_neg, z.device)
        diff_neg = z[nr] - z[nc]
        neg = F.relu(margin - diff_neg.norm(2, 1)).pow(alpha).mean()
        return pos + neg

    @staticmethod
    def align_loss_pairs(
        z: torch.Tensor,
        row_pos: torch.Tensor,
        col_pos: torch.Tensor,
        row_neg: torch.Tensor,
        col_neg: torch.Tensor,
        alpha: int = 2,
        margin: float = 1.0,
    ) -> torch.Tensor:
        dp = z[row_pos] - z[col_pos]
        pos = dp.norm(2, 1).pow(alpha).mean()

        dn = z[row_neg] - z[col_neg]
        neg = F.relu(margin - dn.norm(2, 1)).pow(alpha).mean()
        return pos + neg

    def encode(self, x, adj, T=1):
        mus, logs2, zs = [], [], []
        for _ in range(T):
            h = self.base_gcn(x, adj)
            mu = self.gcn_mean(h, adj)
            ls2 = self.gcn_logsigma2(h, adj)
            z = torch.randn_like(mu) * torch.exp(ls2 / 2) + mu
            mus.append(mu)
            logs2.append(ls2)
            zs.append(z)
        return mus, logs2, zs

    def _moe_fuse_list(self, zs, *, scale: float = 1.0):
        zs_fused, w_list = [], []
        lip = torch.tensor(0.0, device=zs[0].device)
        for z in zs:
            z_u = F.normalize(z, p=2, dim=1)
            z_f, w, lip_l = self.moe(z_u, scale=scale)
            zs_fused.append(z_f)
            w_list.append(w)
            lip = lip + lip_l
        return zs_fused, w_list, lip

    def decode_diffusion(self, zs_fused, w_list=None, start=1):
        T = len(zs_fused)
        sqrt_1m, cum = compute_diffusion_params(T, zs_fused[0].device)
        norm = cum[start - 1]
        acc = None

        for tau, z_tau in enumerate(zs_fused[start - 1 :], start=start):
            sim = z_tau @ z_tau.t()

            if (self.curv_coh_power is not None) and (self.curv_coh_power > 0) and (w_list is not None):
                w_tau = w_list[tau - 1]
                if w_tau is not None:
                    coh = (w_tau @ w_tau.t()).clamp(min=1e-6)
                    sim = sim * (coh ** float(self.curv_coh_power))

            w = sqrt_1m[tau - 1]
            acc = sim * w if acc is None else acc + sim * w

        return torch.clamp(acc / norm, 0, 1)

    def decode_diffusion_pairs_stream(
        self,
        zs_fused: List[torch.Tensor],
        w_list: Optional[List[Optional[torch.Tensor]]],
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        start: int = 1,
        micro_bs: int = 5000,
    ) -> torch.Tensor:
        T = len(zs_fused)
        device = zs_fused[0].device
        dtype = zs_fused[0].dtype

        sqrt_1m, cum = compute_diffusion_params(T, device)
        norm = cum[start - 1]

        B = int(row_idx.numel())
        out = torch.empty(B, device=device, dtype=dtype)

        for s in range(0, B, micro_bs):
            e = min(s + micro_bs, B)
            rr = row_idx[s:e]
            cc = col_idx[s:e]

            acc = None
            for tau, z_tau in enumerate(zs_fused[start - 1 :], start=start):
                sim = (z_tau[rr] * z_tau[cc]).sum(dim=1)

                if (self.curv_coh_power is not None) and (self.curv_coh_power > 0) and (w_list is not None):
                    w_tau = w_list[tau - 1]
                    if w_tau is not None:
                        coh = (w_tau[rr] * w_tau[cc]).sum(dim=1).clamp(min=1e-6)
                        sim = sim * (coh ** float(self.curv_coh_power))

                w = sqrt_1m[tau - 1]
                acc = sim * w if acc is None else acc + sim * w

            out[s:e] = torch.clamp(acc / norm, 0, 1)

        return out

    def pretrainClusterPrototype(
        self,
        features,
        adj_norm,
        adj_label,
        y,
        weight_tensor=None,
        norm=None,
        optimizer="Adam",
        epochs=200,
        lr=5e-3,
        kappa_lr=1e-3,
        save_path="./MixCurvDiff/results/",
        dataset="ogbn-arxiv",
        run_id: str = None,
        prototype_random_state: int = 0,
        moe_warmup_epochs: int = 50,
        dense_recon_max_nodes: int = 20000,
        pos_per_step: int = 4000,
        neg_ratio: float = 1.0,
        steps_per_epoch: int = 1,
        pair_micro_bs: int = 5000,
    ):
        def _cpu_byte(t):
            if isinstance(t, torch.ByteTensor) and t.device.type == "cpu":
                return t
            return torch.tensor(t, dtype=torch.uint8, device="cpu")

        def get_rng_state():
            return {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state().clone(),
                "torch_cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
            }

        def set_rng_state(state):
            try:
                random.setstate(state["python"])
                np.random.set_state(state["numpy"])
            except Exception:
                pass
            try:
                torch.set_rng_state(_cpu_byte(state["torch"]))
            except Exception:
                pass
            if torch.cuda.is_available() and state.get("torch_cuda") is not None:
                try:
                    torch.cuda.set_rng_state_all([_cpu_byte(s) for s in state["torch_cuda"]])
                except Exception:
                    pass

        os.makedirs(save_path, exist_ok=True)
        base_dir = os.path.join(save_path, dataset)
        os.makedirs(base_dir, exist_ok=True)

        if run_id is None:
            run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            need_resume = False
        else:
            need_resume = os.path.isfile(os.path.join(base_dir, run_id, "epoch0.ckpt"))

        run_dir = os.path.join(base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        ckpt0_path = os.path.join(run_dir, "epoch0.ckpt")

        base_params = [p for n, p in self.named_parameters() if n != "log_kappa"]
        optim_cls = torch.optim.Adam if optimizer == "Adam" else torch.optim.SGD
        opt = optim_cls(
            [
                {"params": base_params, "lr": lr},
                {"params": [self.log_kappa], "lr": kappa_lr},
            ],
            **({"momentum": 0.9} if optimizer == "SGD" else {}),
        )

        device = getattr(features, "device", next(self.parameters()).device)

        if need_resume:
            ckpt = torch.load(ckpt0_path, map_location=device, weights_only=False)
            self.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["optim_state"])
            if "rng_state" in ckpt:
                set_rng_state(ckpt["rng_state"])
            meta = ckpt.get("meta", {})
        else:
            rng_state = get_rng_state()
            meta = {
                "run_id": run_id,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "epochs": epochs,
                "optimizer": optimizer,
                "lr": lr,
                "kappa_lr": kappa_lr,
                "dataset": dataset,
                "model_class": self.__class__.__name__,
                "mc": {
                    "mc_lambda": self.prototype_mc_lambda,
                    "mc_power": self.prototype_mc_power,
                    "mc_dist_lambda": self.prototype_mc_dist_lambda,
                    "hyp_scale": self.prototype_hyp_scale,
                },
                "large_graph_tricks": {
                    "dense_recon_max_nodes": dense_recon_max_nodes,
                    "pos_per_step": pos_per_step,
                    "neg_ratio": neg_ratio,
                    "steps_per_epoch": steps_per_epoch,
                    "pair_micro_bs": pair_micro_bs,
                },
            }
            torch.save(
                {
                    "epoch": 0,
                    "model_state": self.state_dict(),
                    "optim_state": opt.state_dict(),
                    "rng_state": rng_state,
                    "meta": meta,
                },
                ckpt0_path,
            )
            with open(os.path.join(run_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        if adj_label.is_sparse:
            row_all, col_all = adj_label.coalesce().indices()
        else:
            row_all, col_all = adj_label.nonzero(as_tuple=True)
        mask_offdiag = row_all != col_all
        row_all = row_all[mask_offdiag]
        col_all = col_all[mask_offdiag]
        undup_mask = row_all < col_all
        pos_row_cpu = row_all[undup_mask].cpu()
        pos_col_cpu = col_all[undup_mask].cpu()
        num_pos_total = int(pos_row_cpu.numel())

        n_nodes = int(adj_norm.size(0))
        use_dense_recon = (
            (weight_tensor is not None)
            and (norm is not None)
            and (n_nodes <= int(dense_recon_max_nodes))
        )

        bce = nn.BCELoss(reduction="mean")

        bar = tqdm(range(epochs), desc="training")
        for ep in bar:
            opt.zero_grad()

            moe_scale = (
                min(1.0, float(ep + 1) / float(moe_warmup_epochs))
                if (moe_warmup_epochs and moe_warmup_epochs > 0)
                else 1.0
            )

            mus, logs2, zs = self.encode(features, adj_norm, T=self.T)
            zs_fused, w_list, lip_loss = self._moe_fuse_list(zs, scale=moe_scale)

            z_poe = self.aggregate_poe(mus, logs2)
            z_poe_u = F.normalize(z_poe, p=2, dim=1)
            z_u, w_u, lip_u = self.moe(z_poe_u, scale=moe_scale)
            lip_loss = lip_loss + lip_u

            if use_dense_recon:
                adj_out = self.decode_diffusion(zs_fused, w_list=w_list, start=1)
                loss_rec = float(norm) * F.binary_cross_entropy(
                    adj_out.reshape(-1),
                    adj_label.to_dense().reshape(-1),
                    weight=weight_tensor,
                )
                loss_aln = self.align_loss(z_u, adj_label, self.align_alpha, self.align_num_neg, self.align_margin)

                loss_smooth = torch.tensor(0.0, device=device)
                if (w_u is not None) and (self.moe_smooth_weight > 0):
                    loss_smooth = (w_u[row_all.to(device)] - w_u[col_all.to(device)]).pow(2).sum(1).mean()
            else:
                steps = max(1, int(steps_per_epoch))
                loss_rec_total = torch.tensor(0.0, device=device)
                loss_aln_total = torch.tensor(0.0, device=device)
                loss_smooth_total = torch.tensor(0.0, device=device)

                for _ in range(steps):
                    k_pos = min(int(pos_per_step), num_pos_total)
                    idx = torch.randint(0, num_pos_total, (k_pos,), device=pos_row_cpu.device)
                    pr = pos_row_cpu[idx].to(device, non_blocking=True)
                    pc = pos_col_cpu[idx].to(device, non_blocking=True)

                    num_neg = int(k_pos * float(neg_ratio))
                    nr = torch.randint(0, n_nodes, (num_neg,), device=device)
                    nc = torch.randint(0, n_nodes, (num_neg,), device=device)
                    mask = nr != nc
                    nr = nr[mask]
                    nc = nc[mask]
                    if nr.numel() > num_neg:
                        nr = nr[:num_neg]
                        nc = nc[:num_neg]

                    pos_score = self.decode_diffusion_pairs_stream(
                        zs_fused, w_list=w_list, row_idx=pr, col_idx=pc, start=1, micro_bs=int(pair_micro_bs)
                    )
                    neg_score = self.decode_diffusion_pairs_stream(
                        zs_fused, w_list=w_list, row_idx=nr, col_idx=nc, start=1, micro_bs=int(pair_micro_bs)
                    )

                    loss_rec_step = bce(pos_score, torch.ones_like(pos_score)) + bce(neg_score, torch.zeros_like(neg_score))
                    loss_rec_total = loss_rec_total + loss_rec_step

                    loss_aln_step = self.align_loss_pairs(
                        z_u,
                        pr,
                        pc,
                        nr,
                        nc,
                        alpha=self.align_alpha,
                        margin=self.align_margin,
                    )
                    loss_aln_total = loss_aln_total + loss_aln_step

                    if (w_u is not None) and (self.moe_smooth_weight > 0):
                        loss_smooth_total = loss_smooth_total + (w_u[pr] - w_u[pc]).pow(2).sum(1).mean()

                loss_rec = loss_rec_total / float(steps)
                loss_aln = loss_aln_total / float(steps)
                loss_smooth = loss_smooth_total / float(steps)

            p = self.assignment(z_u)
            centers_raw = self.assignment.cluster_centers
            dist2 = _pairwise_squared_distance(z_u, centers_raw)
            intra = (p * dist2).sum() / z_u.size(0)
            centers = F.normalize(centers_raw, 2, 1)
            inter = torch.pdist(centers, 2).mean()
            loss_clu = intra / (inter + 1e-9)

            loss_ent = (p * torch.log(p + 1e-9)).sum() / p.size(0)

            loss_balance = torch.tensor(0.0, device=device)
            if (w_u is not None) and (self.moe_balance_weight > 0):
                E = w_u.size(1)
                w_mean = w_u.mean(0).clamp(min=1e-9)
                loss_balance = (w_mean * torch.log(w_mean * E)).sum()

            loss = (
                loss_rec
                + self.align_weight * loss_aln
                + self.cluster_reg_weight * loss_clu
                + self.entropy_reg_weight * loss_ent
                + self.moe_balance_weight * loss_balance
                + self.moe_smooth_weight * loss_smooth
                + lip_loss
            )

            loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            opt.step()

            z_np = z_u.detach().cpu().numpy()
            w_np = w_u.detach().cpu().numpy() if (w_u is not None) else None

            prototype = ClusterPrototype(
                n_clusters=self.nClusters,
                n_init=10,
                max_iter=300,
                random_state=prototype_random_state,
                mc_lambda=self.prototype_mc_lambda,
                mc_power=self.prototype_mc_power,
                mc_dist_lambda=self.prototype_mc_dist_lambda,
                hyp_scale=self.prototype_hyp_scale,
            )
            prototype.fit(z_np, w_np)
            y_pred = prototype.labels_

            bar.set_description(
                f"loss={loss.item():.4f} rec={loss_rec.item():.4f} "
                f"aln={loss_aln.item():.4f} clu={loss_clu.item():.4f} "
                f"ent={loss_ent.item():.4f} "
            )

        return y_pred, y
