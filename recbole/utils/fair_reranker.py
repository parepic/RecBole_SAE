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



def gumbel_sample_perm(X, *, rng=np.random):
    G = rng.gumbel(size=X.shape)
    C = -(np.log(X + 1e-12) + G)
    row, col = linear_sum_assignment(C)
    perm = np.empty(X.shape[1], dtype=int)
    perm[col] = row
    return perm



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
        top_n: int = 200,
        solver: str = "SCS"
    ):
    """
    Re-rank an entire batch with ONE global NSW solve.

    • keeps only each user’s `top_n` items
    • builds one big convex program (sum of user sub-objectives)
    • after the solve, draws a single Gumbel-matching permutation per user
    """
    scores = scores_tensor.detach().cpu().numpy()      # B × N
    B, N    = scores.shape
    v       = E[:, None]                                # (K,1) position bias vector
    r_all   = scores[:, 1:]                             # drop PAD at idx 0
    global_merit = (r_all.sum(axis=0) ** alpha)         # Merit_i^α

    # -----------------------------------------------------------------------
    # build CVXPY variables, objective parts, and constraints for **all** users
    # -----------------------------------------------------------------------
    objective_terms = []
    constraints     = []
    user_infos      = []        # will keep (Pi_var, cand_idx) for post-processing

    for u in range(B):
        # ----- candidate reduction ------------------------------------------------
        user_scores = scores[u, 1:]                      # (N-1,)
        if user_scores.size > top_n:
            cand_idx = np.argpartition(-user_scores, top_n-1)[:top_n]
        else:
            cand_idx = np.arange(user_scores.size)
        cand_idx.sort()
        n_u = cand_idx.size

        r_u      = user_scores[cand_idx]                 # (n_u,)
        merit_u  = global_merit[cand_idx]                # (n_u,)


        Pi = cp.Variable((n_u, K), nonneg=True)           # (n_u × K) DSM

        # matrix–vector product :  (n_u × K)  @  (K × 1)  →  (n_u × 1)
        impacts = Pi @ v                                  # shape  (n_u, 1)
        impacts = cp.reshape(impacts, (n_u,))             # flatten to (n_u,)

        # convert numpy 1-D arrays → CVXPY Constant for broadcasting
        r_u_c     = cp.Constant(r_u)                      # (n_u,)
        merit_u_c = cp.Constant(merit_u)                  # (n_u,)

        # element-wise multiplies everywhere else
        weighted   = cp.multiply(r_u_c, impacts)          # r_ui · Imp_i
        term       = cp.sum(cp.multiply(merit_u_c,
                                        cp.log(weighted + 1e-12)))
        objective_terms.append(term)

        # ----- constraints --------------------------------------------------------
        constraints += [cp.sum(Pi, axis=1) <= 1,   # each item ≤ 1 slot
                        cp.sum(Pi, axis=0) <= 1]   # each slot ≤ 1 item

        user_infos.append((Pi, cand_idx))

    # -----------------------------------------------------------------------
    # single convex solve
    # -----------------------------------------------------------------------
    prob = cp.Problem(cp.Maximize(cp.sum(objective_terms)), constraints)
    prob.solve(
        solver="SCS",           # <- 64-bit indices
        verbose=False,
        eps=1e-4,               # relaxed tolerance speeds things up
        max_iters=2_000
    )
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"Global solver failed: {prob.status}")

    # -----------------------------------------------------------------------
    # post-process: one permutation per user + score boosting
    # -----------------------------------------------------------------------
    new_scores = scores.copy()
    topk_lists = []

    for (Pi, cand_idx), u in zip(user_infos, range(B)):
        X_opt = Pi.value
        perm  = gumbel_sample_perm(X_opt)
        chosen = (cand_idx[perm] + 1).tolist()          # +1 to undo PAD offset
        topk_lists.append(chosen)

        new_scores[u] = boost_scores(new_scores[u], chosen)

    return torch.from_numpy(new_scores).to(scores_tensor.device)