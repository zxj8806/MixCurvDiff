import os
import time
import json
import math
import random
import itertools
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from main.run.preprocessing import load_ogbn_arxiv, sparse_to_tuple, preprocess_graph
from main.run.cluster_utils import Clustering_Metrics, GraphConvSparse, ClusterAssignment as _ClusterAssignment_unused

from main.run.cluster_assignment import (
    _pairwise_squared_distance,
    EfficientClusterAssignment,
)

ClusterAssignment = EfficientClusterAssignment

from main.run.cluster_prototype import ClusterPrototype
from main.run.moe_fusion import (
    MixedCurvatureMoEFusion,
    parse_components,
)
from main.run.diffusion_utils import (
    linear_beta_schedule,
    compute_diffusion_params,
)
from main.run.mixcurvdiff_model import MixCurvDiff
from main.run.metrics_utils import (
    round_half_up,
    map_vector_to_clusters,
    plot_confusion_matrix,
)


if __name__ == "__main__":
    dataset = "ogbn-arxiv"
    print("Loading dataset:", dataset)

    adj, features, labels = load_ogbn_arxiv()
    labels = np.asarray(labels).astype(np.int64)
    nClusters = int(labels.max() + 1)

    alpha = 1.0
    gamma_1 = 1.0
    gamma_2 = 1.0
    gamma_3 = 1.0

    num_neurons = 128
    embedding_size = 128
    save_path = "./MixCurvDiff/results/"

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    adj_norm = preprocess_graph(adj)

    features_tuple = sparse_to_tuple(features.tocoo())
    num_features = int(features_tuple[2][1])

    adj_label = adj + sp.eye(adj.shape[0])
    adj_label_tuple = sparse_to_tuple(adj_label)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    adj_norm_t = torch.sparse_coo_tensor(
        torch.LongTensor(adj_norm[0].T),
        torch.FloatTensor(adj_norm[1]),
        torch.Size(adj_norm[2]),
    ).to(device)

    adj_label_t = torch.sparse_coo_tensor(
        torch.LongTensor(adj_label_tuple[0].T),
        torch.FloatTensor(adj_label_tuple[1]),
        torch.Size(adj_label_tuple[2]),
    )

    features_t = torch.sparse_coo_tensor(
        torch.LongTensor(features_tuple[0].T),
        torch.FloatTensor(features_tuple[1]),
        torch.Size(features_tuple[2]),
    ).to(device)

    comps = parse_components("e2,h2,s2", fixed_curvature=True)

    network = MixCurvDiff(
        num_neurons=num_neurons,
        num_features=num_features,
        embedding_size=embedding_size,
        nClusters=nClusters,
        activation="ReLU",
        alpha=alpha,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        gamma_3=gamma_3,
        T=10,
        components=comps,
        curv_coh_power=1.0,
        moe_balance_weight=1e-2,
        moe_smooth_weight=1e-2,
        gate_sn=False,
        lambda_lip=0.0,
        init_mix_logit=-4.6,
        grad_clip=5.0,

        prototype_mc_lambda=0.05,
        prototype_mc_power=1.0,
        prototype_mc_dist_lambda=0.02,
        prototype_hyp_scale=0.90,
    ).to(device)

    y_pred, y_true = network.pretrainClusterPrototype(
        features_t,
        adj_norm_t,
        adj_label_t,
        labels,
        weight_tensor=None,
        norm=None,
        optimizer="Adam",
        epochs=120,
        lr=1e-4,
        kappa_lr=1e-3,
        save_path=save_path,
        dataset=dataset,
        run_id="run_moe_diff_mc_distance_ClusterPrototype",
        moe_warmup_epochs=50,
        prototype_random_state=0,

        dense_recon_max_nodes=20000,
        pos_per_step=4000,
        neg_ratio=1.0,
        steps_per_epoch=1,
        pair_micro_bs=5000,
    )

    try:
        target_names = [str(i) for i in range(nClusters)]
        y_mapped = map_vector_to_clusters(y_true, y_pred)
        cm = confusion_matrix(y_true=y_mapped, y_pred=y_pred, normalize='true')
        print("Confusion matrix computed (not saved by default).")
    except Exception as e:
        print("[Warn] confusion_matrix skipped:", repr(e))
