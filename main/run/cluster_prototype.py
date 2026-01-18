from typing import Optional

import numpy as np


class ClusterPrototype:
    def __init__(
        self,
        n_clusters: int,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        mc_lambda: float = 0.05,
        mc_power: float = 1.0,
        mc_dist_lambda: float = 0.02,
        hyp_scale: float = 0.90,
        eps: float = 1e-12,
    ):
        if n_clusters <= 0:
            raise ValueError("n_clusters must be > 0")
        if n_init <= 0:
            raise ValueError("n_init must be > 0")
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if tol < 0:
            raise ValueError("tol must be >= 0")
        if mc_lambda < 0:
            raise ValueError("mc_lambda must be >= 0")
        if mc_power <= 0:
            raise ValueError("mc_power must be > 0")
        if mc_dist_lambda < 0:
            raise ValueError("mc_dist_lambda must be >= 0")
        if not (0.0 < hyp_scale < 1.0):
            raise ValueError("hyp_scale must be in (0,1)")
        if eps <= 0:
            raise ValueError("eps must be > 0")

        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        self.mc_lambda = float(mc_lambda)
        self.mc_power = float(mc_power)

        self.mc_dist_lambda = float(mc_dist_lambda)
        self.hyp_scale = float(hyp_scale)
        self.eps = float(eps)

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

        # extra
        self.cluster_gate_centers_ = None

    @staticmethod
    def _squared_distances(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)

    @staticmethod
    def _normalize_rows(X: np.ndarray, eps: float) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n = np.maximum(n, eps)
        return X / n

    @staticmethod
    def _normalize_simplex_rows(W: np.ndarray, eps: float) -> np.ndarray:
        W = np.maximum(W, 0.0)
        s = W.sum(axis=1, keepdims=True)
        s = np.maximum(s, eps)
        return W / s

    def _spherical_proxy(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        Xn = self._normalize_rows(X, self.eps)
        Cn = self._normalize_rows(C, self.eps)
        cos = Xn @ Cn.T
        cos = np.clip(cos, -1.0, 1.0)
        return 1.0 - cos

    def _poincare_distance(self, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        U2 = np.sum(U * U, axis=1, keepdims=True)
        V2 = np.sum(V * V, axis=1, keepdims=True).T
        diff2 = ((U[:, None, :] - V[None, :, :]) ** 2).sum(axis=2)

        denom = (1.0 - U2) * (1.0 - V2)
        denom = np.maximum(denom, self.eps)

        arg = 1.0 + 2.0 * diff2 / denom
        arg = np.maximum(arg, 1.0 + 1e-9)
        return np.arccosh(arg)

    def _hyperbolic_proxy(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        Xn = self._normalize_rows(X, self.eps) * self.hyp_scale
        Cn = self._normalize_rows(C, self.eps) * self.hyp_scale
        return self._poincare_distance(Xn, Cn)

    def _coherence(self, W: np.ndarray, V: np.ndarray) -> np.ndarray:
        coh = W @ V.T
        return np.clip(coh, 0.0, 1.0)

    def _pair_mixture(self, W: np.ndarray, V: np.ndarray) -> np.ndarray:
        m = W[:, None, :] * V[None, :, :]
        s = np.sum(m, axis=2, keepdims=True)
        s = np.maximum(s, self.eps)
        return m / s

    def _mixed_distance(
        self,
        X: np.ndarray,
        C: np.ndarray,
        W: Optional[np.ndarray],
        V: Optional[np.ndarray],
    ) -> np.ndarray:
        dE = self._squared_distances(X, C)

        if W is None or V is None:
            return dE

        if self.mc_lambda > 0:
            coh = self._coherence(W, V)
            penalty = (1.0 - coh) ** self.mc_power
            d_base = dE * (1.0 + self.mc_lambda * penalty)
        else:
            d_base = dE

        if self.mc_dist_lambda <= 0:
            return d_base

        dS = self._spherical_proxy(X, C)
        dH = self._hyperbolic_proxy(X, C)

        mean_dE = float(np.mean(dE) + self.eps)
        mean_dH = float(np.mean(dH) + self.eps)
        scale_H = mean_dE / mean_dH
        dH_scaled = dH * scale_H

        m = self._pair_mixture(W, V)
        d_mix = m[:, :, 0] * dE + m[:, :, 1] * dH_scaled + m[:, :, 2] * dS

        return d_base + self.mc_dist_lambda * (d_mix - dE)

    def _prototype_plusplus_init(self, X: np.ndarray, W: Optional[np.ndarray], rs: np.random.RandomState):
        n_samples, n_features = X.shape
        k = self.n_clusters
        if k > n_samples:
            raise ValueError("n_clusters cannot be greater than number of samples")

        centers = np.empty((k, n_features), dtype=X.dtype)
        gate_centers = None

        first_idx = rs.randint(0, n_samples)
        centers[0] = X[first_idx]

        if W is not None:
            E = W.shape[1]
            gate_centers = np.empty((k, E), dtype=W.dtype)
            gate_centers[0] = W[first_idx]

        closest = None
        for c in range(1, k):
            if closest is None:
                if W is None:
                    closest = ((X - centers[0]) ** 2).sum(axis=1)
                else:
                    V0 = gate_centers[:1]
                    dist0 = self._mixed_distance(X, centers[:1], W, V0)[:, 0]
                    closest = dist0

            closest = np.maximum(closest, self.eps)
            probs = closest / closest.sum()

            idx = int(np.searchsorted(np.cumsum(probs), rs.rand(), side="right"))
            if idx >= n_samples:
                idx = n_samples - 1

            centers[c] = X[idx]
            if W is not None:
                gate_centers[c] = W[idx]

            if W is None:
                dist_new = ((X - centers[c]) ** 2).sum(axis=1)
            else:
                Vc = gate_centers[c:c + 1]
                dist_new = self._mixed_distance(X, centers[c:c + 1], W, Vc)[:, 0]

            closest = np.minimum(closest, dist_new)

        return centers, gate_centers

    @staticmethod
    def _compute_centers(X: np.ndarray, labels: np.ndarray, k: int, rs: np.random.RandomState) -> np.ndarray:
        n_samples, n_features = X.shape
        centers = np.zeros((k, n_features), dtype=X.dtype)
        counts = np.bincount(labels, minlength=k).astype(np.int64)
        for j in range(k):
            if counts[j] > 0:
                centers[j] = X[labels == j].mean(axis=0)
            else:
                centers[j] = X[rs.randint(0, n_samples)]
        return centers

    def _compute_gate_centers(self, W: np.ndarray, labels: np.ndarray, k: int, rs: np.random.RandomState) -> np.ndarray:
        n_samples, E = W.shape
        V = np.zeros((k, E), dtype=W.dtype)
        counts = np.bincount(labels, minlength=k).astype(np.int64)
        for j in range(k):
            if counts[j] > 0:
                V[j] = W[labels == j].mean(axis=0)
            else:
                V[j] = W[rs.randint(0, n_samples)]
        return self._normalize_simplex_rows(V, self.eps)

    def _lloyd(self, X: np.ndarray, W: Optional[np.ndarray], init_centers: np.ndarray, init_gate_centers: Optional[np.ndarray], rs: np.random.RandomState):
        centers = init_centers.copy()
        gate_centers = init_gate_centers.copy() if init_gate_centers is not None else None

        n_iter = 0
        for it in range(self.max_iter):
            n_iter = it + 1

            dist = self._mixed_distance(X, centers, W, gate_centers)
            labels = dist.argmin(axis=1).astype(np.int64)

            new_centers = self._compute_centers(X, labels, self.n_clusters, rs)
            new_gate_centers = None
            if W is not None:
                new_gate_centers = self._compute_gate_centers(W, labels, self.n_clusters, rs)

            shift_x = np.sqrt(((centers - new_centers) ** 2).sum(axis=1)).max()
            shift_w = 0.0
            if W is not None and gate_centers is not None:
                shift_w = np.sqrt(((gate_centers - new_gate_centers) ** 2).sum(axis=1)).max()

            centers = new_centers
            gate_centers = new_gate_centers

            if max(shift_x, shift_w) <= self.tol:
                break

        dist_final = self._mixed_distance(X, centers, W, gate_centers)
        min_dist = dist_final[np.arange(X.shape[0]), labels]
        inertia = float(min_dist.sum())

        return centers, gate_centers, labels, inertia, n_iter

    def fit(self, X: np.ndarray, W: Optional[np.ndarray] = None):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] == 0:
            raise ValueError("X must have at least one sample")
        X = np.ascontiguousarray(X)

        if W is not None:
            if not isinstance(W, np.ndarray):
                W = np.asarray(W)
            if W.ndim != 2 or W.shape[0] != X.shape[0]:
                raise ValueError("W must be a 2D array with shape (n_samples, n_experts)")
            W = np.ascontiguousarray(W)
            W = self._normalize_simplex_rows(W, self.eps)

        rs_master = np.random.RandomState(self.random_state)

        best_inertia = None
        best_centers = None
        best_gate_centers = None
        best_labels = None
        best_n_iter = None

        for _ in range(self.n_init):
            seed = rs_master.randint(0, 2**32 - 1)
            rs = np.random.RandomState(seed)

            init_centers, init_gate_centers = self._prototype_plusplus_init(X, W, rs)
            centers, gate_centers, labels, inertia, n_iter = self._lloyd(X, W, init_centers, init_gate_centers, rs)

            if (best_inertia is None) or (inertia < best_inertia):
                best_inertia = inertia
                best_centers = centers
                best_gate_centers = gate_centers
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centers
        self.cluster_gate_centers_ = best_gate_centers
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = int(best_n_iter)
        return self
