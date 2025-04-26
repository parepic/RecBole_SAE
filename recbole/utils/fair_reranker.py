import numpy as np
from scipy.optimize import linear_sum_assignment
import cvxpy as cp
import torch

# -------------------------------------------------
# helpers
# -------------------------------------------------
def position_bias(k):
    """e(k) for k ∈ {1,…,10}."""
    return 1.0 / np.log2(k + 1)

E = np.array([position_bias(k) for k in range(1, 11)])   # shape (10,)

def bv_decompose(X, eps=1e-9):
    """
    Deterministic Birkhoff–von Neumann decomposition.
    Returns list of (weight, permutation_vector) pairs.
    """
    X = X.copy()
    n, K = X.shape
    perms, lambdas = [], []
    while X.max() > eps:
        # Construct a permutation that follows the current positive pattern
        row_ind, col_ind = linear_sum_assignment(-X)     # greedy max
        P = np.zeros_like(X)
        P[row_ind, col_ind] = 1
        # smallest mass along that permutation
        lam = (X[P == 1]).min()
        X[P == 1] -= lam
        perms.append(col_ind)      # permutation vector length K
        lambdas.append(lam)
    lambdas = np.array(lambdas)
    lambdas /= lambdas.sum()       # normalise
    return lambdas, perms          # lists of equal length

def boost_scores(row_scores, chosen_indices):
    """
    Increase chosen_indices so they become the top-10 in the given order.
    """
    base = row_scores.max()
    K = len(chosen_indices)
    # Largest boost for rank-1, smaller for rank-10
    for rank, idx in enumerate(chosen_indices, start=1):
        row_scores[idx] = base + (K - rank + 1)
    return row_scores



def fair_rerank_exact(scores_tensor: torch.Tensor, alpha: float = 0.0):
    scores = scores_tensor.detach().cpu().numpy()
    B, N = scores.shape
    n_items = N - 1   # we ignore index 0
    K = 10
    r_true = scores[:, 1:]   # shape (B, n_items)

    # 1) Decision variable: one big matrix Pi of shape (B, n_items*K)
    Pi = cp.Variable((B, n_items * K), nonneg=True)

    # 2) Precompute weights (Merit^α) and position bias
    v = E[:, None]                      # (K,1)
    am_rel = r_true.sum(0) ** alpha     # vector of length n_items
    print("sik ", am_rel)
    # 3) Build the convex objective:
    #     sum_{d=1..n_items} am_rel[d] * log( sum_{q=1..B} r_true[q,d] * Pi[q, d*K:(d+1)*K] * v )
    obj = 0
    for d in range(n_items):
        slice_d = slice(K*d, K*(d+1))
        impact_d = r_true[:, d] @ Pi[:, slice_d] @ v  # scalar
        obj += am_rel[d] * cp.log(impact_d + 1e-12)

    # 4) Feasibility constraints (each query fills ≤1 item per slot & each item is placed ≤1 slot per query)
    one_q = np.ones((B, 1))
    constraints = []
    # 4a) For each item d: the B×K block sums ≤ 1 down each row
    for d in range(n_items):
        basis = np.zeros((n_items*K, 1))
        basis[K*d:K*(d+1)] = 1
        constraints.append(Pi @ basis <= one_q)
    # 4b) For each slot k: the B×n_items entries sum ≤ 1 down each row
    for k in range(K):
        basis = np.zeros((n_items*K, 1))
        basis[np.arange(n_items)*K + k] = 1
        constraints.append(Pi @ basis <= one_q)

    # 5) Solve the convex program: maximize ∑ am_rel[d]·log(impact_d)
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.SCS)

    # 6) Reshape back to (B, n_items, K)
    Pi_opt = Pi.value.reshape(B, n_items, K)

    # 7) For each user, BvN-decompose their slice, sample one permutation, boost scores
    topk_lists, new_scores = [], scores.copy()
    for u in range(B):
        X_opt = Pi_opt[u]                # shape (n_items, K)
        lambdas, perms = bv_decompose(X_opt)
        pick = np.random.choice(len(lambdas), p=lambdas)
        chosen = (perms[pick] + 1).tolist()   # +1 to re-index since we skipped item 0
        topk_lists.append(chosen)
        new_scores[u] = boost_scores(new_scores[u], chosen)

    return torch.from_numpy(new_scores).to(scores_tensor.device), topk_lists