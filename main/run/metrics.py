
import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score
f1 = f1_score

def cos_grad(grad1, grad2):
    grad1_list = []
    grad2_list = []
    for i in range(len(grad1)):
        grad1_list.append(grad1[i].flatten())
        grad2_list.append(grad2[i].flatten())
    grad1_vector = torch.cat(grad1_list, 0)
    grad2_vector = torch.cat(grad2_list, 0)

    return torch.dot(grad1_vector, grad2_vector) / ((torch.norm(grad1_vector)) * (torch.norm(grad2_vector)))

def acc(y_true, y_pred):

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size
