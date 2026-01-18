import torch
import torch.nn as nn


def _pairwise_squared_distance(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:

    x2 = (x * x).sum(dim=1, keepdim=True)
    c2 = (c * c).sum(dim=1).view(1, -1)
    return (x2 + c2 - 2.0 * (x @ c.t())).clamp_min_(0.0)


class EfficientClusterAssignment(nn.Module):

    def __init__(self, n_clusters: int, embedding_dimension: int, alpha: float):
        super().__init__()
        self.n_clusters = int(n_clusters)
        self.embedding_dimension = int(embedding_dimension)
        self.alpha = float(alpha)
        self.cluster_centers = nn.Parameter(torch.randn(self.n_clusters, self.embedding_dimension))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dist2 = _pairwise_squared_distance(inputs, self.cluster_centers)
        q = 1.0 / (1.0 + dist2 / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q


ClusterAssignment = EfficientClusterAssignment
