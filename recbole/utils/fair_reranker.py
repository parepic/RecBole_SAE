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



def fair_rerank_exact(
        scores_tensor: torch.Tensor,
        alpha: float = 0.0,
        K: int = 10,
        top_n: int = 50,
        solver: str = "ECOS"        # ECOS is fast for tiny QPs; SCS also works
    ):
    """
    Re-rank each query independently with the NSW objective.

    * Only the `top_n` (=200) highest-scoring items of the query are kept.
    * Each query is solved separately =>   n_u × K  variables  (≤ 2 000).
    """
    scores = scores_tensor.detach().cpu().numpy()
    B, N = scores.shape
    v   = E[:, None]                          # (K, 1)
    r_all = scores[:, 1:]                     # ground‐truth relevance
    global_merit = (r_all.sum(axis=0) ** alpha)   # Merit_i^α   (length N-1)

    new_scores = scores.copy()
    topk_lists = []

    for u in range(B):
        print(u)
        # -------------------------------------------------
        # 1) keep only the top-`top_n` items of this user
        # -------------------------------------------------
        user_scores = scores[u, 1:]                 # (N-1,)
        if len(user_scores) > top_n:
            cand_idx = np.argpartition(-user_scores, top_n-1)[:top_n]
        else:
            cand_idx = np.arange(len(user_scores))
        cand_idx.sort()                             # keep ascending id order
        n_u = len(cand_idx)

        r_u = user_scores[cand_idx]                 # (n_u,)
        merit_u = global_merit[cand_idx]            # (n_u,)

        # -------------------------------------------------
        # 2) build & solve tiny CVX problem
        # -------------------------------------------------
        Pi = cp.Variable((n_u, K), nonneg=True)     # probabilities

        # NSW objective  Σ_i Merit_i^α · log( r_ui · Σ_k Pi_ik e(k) )
        impacts = cp.matmul(cp.multiply(Pi, v.T), np.ones((K, 1)))[:, 0]  # Σ_k Pi_ik e(k)
        obj = cp.sum(cp.multiply(merit_u, cp.log(cp.multiply(r_u, impacts) + 1e-12)))

        # constraints: each item ≤1 slot, each slot ≤1 item
        constraints = [cp.sum(Pi, axis=1) <= 1,    # per item
                       cp.sum(Pi, axis=0) <= 1]    # per slot

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve(solver=solver, verbose=False)

        if Pi.value is None:
            raise RuntimeError(f"Solver failed on user {u}")

        # -------------------------------------------------
        # 3) BvN → draw one permutation, boost scores
        # -------------------------------------------------
        X_opt = Pi.value                           # (n_u, K)
        lam, perms = bv_decompose(X_opt)
        perm = perms[np.random.choice(len(lam), p=lam)]   # choose permutation
        chosen_items = (cand_idx[perm] + 1).tolist()      # +1 to undo offset
        topk_lists.append(chosen_items)

        new_scores[u] = boost_scores(new_scores[u], chosen_items)

    return torch.from_numpy(new_scores).to(scores_tensor.device)