# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

from __future__ import annotations

import math
import torch
from torch import nn
from collections import defaultdict



import numpy as np
import torch
from typing import Literal, Union, Optional
Array = Union[np.ndarray, torch.Tensor]

from scipy.optimize import linprog


from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
from recbole.utils import (
    make_items_unpopular,
    make_items_popular,
    save_batch_activations,
    get_extreme_correlations,
    skew_sample,
    calculate_IPS
)

import pandas as pd

class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)
        self.to(config["device"])
        self.recommendation_count = np.zeros(self.n_items)
        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.corr_file = None
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)


    def set_dampen_hyperparam(self, corr_file=None, N=None, beta=None, gamma=None, unpopular_only=False):
        self.corr_file = corr_file
        # self.neuron_count = neuron_count
        # self.damp_percent = damp_percent
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.unpopular_only = unpopular_only
  

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

        
    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        # Convert back to PyTorch tensor        
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        
        return output  # [B H]
    
    
    def simple_reranker(self, scores):
        """
        Adjust the scores based on item popularity.
        
        Parameters:
        scores (np.ndarray): Shape [B, n_items], where B is batch size, n_items is number of items.
        csv_file (str): Path to CSV file with columns 'item_id:token' and 'pop_score'.
        
        Returns:
        adjusted_scores (np.ndarray): Adjusted scores with same shape as input.
        """
        csv_file = r"./dataset/ml-1m/item_popularity_labels_with_titles.csv"
        # Load the CSV file
        df = pd.read_csv(csv_file)
        
        # Sort by 'item_id:token' to ensure item IDs are in order (0 to n_items-1)
        df['item_id:token'] = pd.to_numeric(df['item_id:token'], errors='coerce')
        if df['item_id:token'].isna().any():
            raise ValueError("Some 'item_id:token' values could not be converted to numbers")
        
        n_items = scores.shape[1]
        expected_ids = set(range(n_items))
        present_ids = set(df['item_id:token'].astype(int))
        
        # If ID 0 is missing, add a dummy row at the start
        if 0 not in present_ids:
            df = pd.concat([pd.DataFrame({'item_id:token': [0], 'pop_score': [0.0]}), df], ignore_index=True)
        
        # Sort by 'item_id:token' to align with item indices
        df = df.sort_values('item_id:token')
        
        # Verify all expected IDs are present
        if set(df['item_id:token'].astype(int)) != expected_ids:
            raise ValueError("Item IDs in CSV do not match expected range after adding dummy")
        
        # Extract popularity scores
        pop_scores = df['pop_score'].values
        
        # Adjust scores using the formula: score * (1 / (pop_score + 1))
        adjusted_scores = scores.detach().cpu().numpy() * (1 / (pop_scores * (0.2/max(pop_scores)) + 1))
        
        return adjusted_scores
    
    
    def two_sided_calibrate_and_boost(
        self,
        scores: torch.Tensor,        # shape [B, N]
        p: float,                  # desired global niche exposure
        lambd: float,              # 0≤λ≤1 trade‑off
        K: int = 10                # size of final slate
    ) -> np.ndarray:    


        df = pd.read_csv(r"./dataset/ml-1m/item_popularity_labels_with_titles.csv")
        ids  = df["item_id:token"].astype(int).values      # e.g. [1, 2, 3, …, 3417]
        labs = df["popularity_label"].astype(int).values   # e.g. [1, 0, 1, …, 0]
        scores[:, 0] =  float("-inf")
        scores = scores.detach().cpu().numpy()

        # 2) Build a 1D BoolTensor of size (max_id+1,) so we can index by ID directly
        max_id = ids.max()
        niche_labels = np.zeros(max_id+1, dtype=bool)

        # 3) Fill it: True where label == 1 (popular)
        #    If your “popular” is actually encoded as -1, just change (labs == 1) to (labs == -1)
        niche_labels[ids] = (labs == -1)

        
        B, N = scores.shape

        # 1) Get *full* descending sort of every user’s N items
        full_rank = np.argsort(-scores, axis=1)        # shape [B, N]

        # 2) Estimate each user’s original p_u from their top‑K in that full list
        is_niche = niche_labels.astype(int)
        p_u = np.zeros((B, 2), dtype=float)
        for u in range(B):
            topK = full_rank[u, :K]
            cnt = is_niche[topK].sum()
            p_u[u,1] = cnt / K
            p_u[u,0] = 1 - p_u[u,1]

        # 3) global target
        q_hat = np.array([1.0 - p, p], dtype=float)

        # 4) compute normalized gradient g = (mean(p_u)-q_hat) / ||...||
        g_raw = p_u.mean(axis=0) - q_hat
        norm = np.linalg.norm(g_raw)
        g = g_raw / (norm + 1e-12)

        # 5) per-user max step l_u so p_u - γ_u g in [0,1]^2
        l_u = np.zeros(B, dtype=float)
        for u in range(B):
            bounds_ = []
            for i in (0,1):
                if g[i] < 0:
                    bounds_.append((p_u[u,i] - 1.0) / g[i])
                else:
                    bounds_.append(p_u[u,i] / (g[i] if g[i]>0 else 1e-12))
            l_u[u] = max(0.0, min(bounds_))

        # 6) solve LP:  min ∑γ_u  s.t.  ∑γ_u g = ∑(p_u - q_hat),  0≤γ_u≤l_u
        c = np.ones(B, dtype=float)
        A_eq = np.vstack([ np.full(B, g[0]), np.full(B, g[1]) ])  # 2×B
        b_eq = (p_u - q_hat).sum(axis=0)                           # length‑2
        bounds = [(0.0, lu) for lu in l_u]

        sol = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        γ = sol.x  # shape [B]

        # 7) personalized targets q_u_hat = p_u - γ_u * g
        q_u_hat = p_u - γ[:,None] * g[None,:]  # [B,2]

        # 8) rank‑weights for final slate positions 1…K
        r_k = 1.0 / np.log2(np.arange(K) + 2.0)

        # 9) PCT‑Reranker over the *full* list
        def pct_rerankk(R_full, qhat):
            resource = r_k.sum()
            target   = qhat * resource
            exp      = np.zeros(2, dtype=float)
            selected = set()
            slate    = [-1]*K

            # Phase 1: pilot select any “safe” items among the full list
            for pos in range(K):
                for i in R_full:
                    gi = int(niche_labels[i])
                    if i not in selected and exp[gi] + r_k[pos] <= target[gi]:
                        slate[pos] = i
                        selected.add(i)
                        exp[gi] += r_k[pos]
                        break

            # Phase 2: fill gaps via one‑step MMR over the full list
            for pos in range(K):
                if slate[pos] < 0:
                    best_score = -1e9
                    best_item  = None
                    for gi in (0,1):
                        exp2 = exp.copy()
                        exp2[gi] += r_k[pos]
                        disp = 0.5 * np.sum((exp2 - target)**2)
                        # find highest‑ranked unselected of group gi
                        for rank, i in enumerate(R_full):
                            if i not in selected and int(niche_labels[i])==gi:
                                score_mmr = lambd/(rank+1.0) - (1-lambd)*disp
                                if score_mmr > best_score:
                                    best_score = score_mmr
                                    best_item = i
                                break
                    slate[pos] = best_item
                    selected.add(best_item)
                    exp[int(niche_labels[best_item])] += r_k[pos]

            return slate

        # 10) rerank each user’s entire N to pick final K
        final_slates = np.zeros((B, K), dtype=int)
        for u in range(B):
            final_slates[u] = pct_rerankk(full_rank[u], q_u_hat[u])

        # 11) boost exactly those K so a plain top‑K on boosted reproduces them
        boosted = scores.copy()
        for u in range(B):
            row = scores[u]
            C = (row.max() - row.min()) + 1.0
            for pos, item in enumerate(final_slates[u]):
                boosted[u, item] = row[item] + C*(K-pos)

        return torch.from_numpy(boosted).to(device=self.device, dtype=torch.float32)
              
                   
     
                   
    def FAIR(self, scores):
        L=500
        scores = scores.detach().cpu()
        # 1) Load the CSV; adjust path as needed
        df = pd.read_csv(r"./dataset/ml-1m/item_popularity_labels_with_titles.csv")
        ids  = df["item_id:token"].astype(int).values      # e.g. [1, 2, 3, …, 3417]
        labs = df["popularity_label"].astype(int).values   # e.g. [1, 0, 1, …, 0]
        scores[:, 0] =  float("-inf")
        # 2) Build a 1D BoolTensor of size (max_id+1,) so we can index by ID directly
        max_id = ids.max()
        popularity_label = torch.zeros(max_id+1, dtype=torch.bool)

        # 3) Fill it: True where label == 1 (popular)
        #    If your “popular” is actually encoded as -1, just change (labs == 1) to (labs == -1)
        popularity_label[ids] = torch.from_numpy((labs != -1))

        B, N = scores.size()
        L = 500
        K = 10
        p = 0.3       # require ≥96% unpopular (“protected”)
            
        # 1) build your truncated candidate set and labels
        top500_idx    = torch.argsort(scores, dim=1, descending=True)[:, :L]  # (B,500)
        labels500     = popularity_label[top500_idx]                           # (B,500) bool
        protected500  = ~labels500                                             # True=unpopular
        
        # 2) for each batch‐row, run fair_topk
        fair_pos = []
        for b in range(B):
            row_scores     = scores[b, top500_idx[b]]      # (500,)
            row_protected  = protected500[b]               # (500,)
            sel_in_500      = self.fair_topk(row_scores, row_protected, K, p)
            # 1) map back to the *original* N‑dimensional positions
            orig_pos = top500_idx[b, sel_in_500]  # shape (K,)

            # 2) compute a “base” just above the current max score
            base = scores[b].max().item() + 1.0   # e.g. if max was 37.2, base = 38.2

            # 3) assign descending offsets so they keep FA*IR’s order
            #    [38.2 + K-1, 38.2 + K-2, …, 38.2 + 0]
            offsets = torch.arange(K-1, -1, -1, device=scores.device, dtype=scores.dtype)
            new_vals = base + offsets               # shape (K,)

            # 4) write them back into `scores[b]`
            scores[b, orig_pos] = new_vals
            fair_pos.append(sel_in_500)    
        return scores
            
            
    def fair_topk(self, scores1d, protected1d, K, p):
        idx_sorted  = np.argsort(-scores1d)   # descending
        prot_list   = [i for i in idx_sorted if protected1d[i]]
        nonprot_list= [i for i in idx_sorted if not protected1d[i]]

        sel = []
        cp = cn = pp = np_ptr = 0

        for t in range(1, K+1):
            needed = math.ceil(p * t)
            if cp < needed:
                if pp < len(prot_list):
                    choose = prot_list[pp]; pp += 1; cp += 1
                else:
                    choose = nonprot_list[np_ptr]; np_ptr += 1; cn += 1
            else:
                next_p  = prot_list[pp]    if pp < len(prot_list)    else None
                next_np = nonprot_list[np_ptr] if np_ptr < len(nonprot_list) else None

                if next_p is None and next_np is None:
                    break
                elif next_p is None:
                    choose = next_np; np_ptr += 1; cn += 1
                elif next_np is None:
                    choose = next_p; pp += 1; cp += 1
                else:
                    if scores1d[next_p] >= scores1d[next_np]:
                        choose = next_p; pp += 1; cp += 1
                    else:
                        choose = next_np; np_ptr += 1; cn += 1

            sel.append(choose)

        return np.array(sel, dtype=int)

    
    def calculate_loss(self, interaction, scores=None):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]  # shape: (batch_size,)
        
        # Forward pass
        seq_output = self.forward(item_seq, item_seq_len)  # shape: (batch_size, hidden_dim)
        test_item_emb = self.item_embedding.weight          # shape: (num_items, hidden_dim)
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # (batch_size, num_items)
        # Cross-entropy loss per sample (no reduction!)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        ce_loss = loss_fn(logits, pos_items)  # shape: (batch_size,)


        # if scores is not None:
        #     # Clamp scores to avoid exploding weights
        #     scores = torch.tensor(scores, dtype=torch.float32, device=logits.device)
        #     scores = torch.clamp(scores, min=1e-4)
        #     # Inverse propensity weighting
        #     weighted_loss = (ce_loss / scores).mean()
        # else:
        #     weighted_loss = ce_loss.mean()
        # print("ISP-weighed loss", weighted_loss)
        return ce_loss.mean()
    

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores


    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        # item_seq = make_items_popular(item_seq)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        if self.corr_file != None:
            seq_output = self.dampen_neurons_sasrec(seq_output)
        # save_batch_activations(seq_output, 64)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        # scores = torch.tensor(self.simple_reranker(scores)).to(self.device)
        # scores = self.FAIR(scores).to(self.device)
        scores = self.pct_rerank(scores=scores, user_interest=item_seq)
        top_recs = torch.argsort(scores, dim=1, descending=True)[:, :10]
        for key in top_recs.flatten():
            self.recommendation_count[key.item()] += 1
        return scores


    def dampen_neurons_sasrec(self, pre_acts):
        # If no neuron count is provided, return the original activations.
        if self.N is None:
            return pre_acts

        # Retrieve neurons from the correlations file.
        pop_neurons, unpop_neurons = get_extreme_correlations(self.corr_file, self.unpopular_only)

        # Combine both groups into one list while labeling the group type.
        # 'unpop' neurons are those with higher activations for unpopular inputs (to be reinforced),
        # while 'pop' neurons are those with lower activations (to be dampened).
        combined_neurons = [(idx, cohen, 'unpop') for idx, cohen in unpop_neurons] + \
                        [(idx, cohen, 'pop') for idx, cohen in pop_neurons]


        combined_sorted = sorted(combined_neurons, key=lambda x: abs(x[1]), reverse=True)
        top_neurons = combined_sorted[:int(self.N)]

        # Load the corresponding statistics files.
        stats_unpop = pd.read_csv(r"./dataset/ml-1m/row_stats_unpopular.csv")
        stats_pop = pd.read_csv(r"./dataset/ml-1m/row_stats_popular.csv")

        # Create tensors of the absolute Cohen's d values for the selected neurons.
        abs_cohens = torch.tensor([abs(c) for _, c, _ in top_neurons], device=pre_acts.device)

        # Define a helper normalization function.
        def normalize_to_range(x, new_min, new_max):
            min_val = torch.min(x)
            max_val = torch.max(x)
            if max_val == min_val:
                return torch.full_like(x, (new_min + new_max) / 2)
            return (x - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

        # Normalize the Cohen's d values to [0, 2.5]
        weights = normalize_to_range(abs_cohens, new_min=0, new_max=1)
        # Now update the neuron activations based on group.
        for i, (neuron_idx, cohen, group) in enumerate(top_neurons):
            weight = weights[i]
            
            if group == 'unpop':
                # For neurons to be reinforced, fetch stats from the unpopular file.
                row = stats_unpop.iloc[neuron_idx]
                mean_val = row["mean"]
                std_val = row["std"]

                # Identify positions where the neuron's activation is above its mean.
                vals = pre_acts[:, neuron_idx]
                condition = vals > mean_val + self.beta * std_val
                # Increase activations by an amount proportional to the standard deviation and effective weight.
                pre_acts[condition, neuron_idx] += weight * std_val

            else:  # group == 'pop'
                # For neurons to be dampened, use the popular statistics for impact.
                pop_mean = stats_pop.iloc[neuron_idx]["mean"]
                pop_sd = stats_pop.iloc[neuron_idx]["std"]

				# Still fetch the comparison stats from the unpopular stats file
				# (this is from your original logic; adjust if needed).
                row = stats_unpop.iloc[neuron_idx]
                mean_val = row["mean"]
                std_val = row["std"]

				# Identify positions where the neuron's activation is below its mean.
                vals = pre_acts[:, neuron_idx]
                condition = vals < pop_mean + self.gamma * pop_sd
				# Decrease activations proportionally.
                pre_acts[condition, neuron_idx] -= weight * pop_sd
            

        return pre_acts
    
    
    def _solve_personal_targets(self, p_u: np.ndarray, q_hat: np.ndarray, chunk: int = 5000) -> np.ndarray:
        B = p_u.shape[0]
        print(B, " blya ")
        gradient = p_u.mean(0) - q_hat
        if np.allclose(gradient, 0):
            return p_u.copy()
        g = gradient / np.linalg.norm(gradient)
        print(g.size)
        tile_g = np.tile(g[:, None], (1, B))
        lim = np.where(tile_g > 0, p_u / (tile_g + 1e-10), (p_u - 1) / (tile_g + 1e-10)).min(0)
        A_eq_full = tile_g[:1]
        b_eq_full = (p_u - q_hat).sum(1)[:1]
        gamma = np.empty(B)
        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            A_eq = A_eq_full[:, s:e]
            b_eq = b_eq_full - (tile_g[:1, :s] * gamma[:s]).sum(1)
            bounds = [(0, lim[i]) for i in range(s, e)]
            res = linprog(c=np.ones(e - s), A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            gamma[s:e] = res.x
        return p_u - (gamma * g).T



    def pct_rerank(
        self,
        scores: Array,
        *,
        top_k: int = 10,
        policy: Literal["Equal", "AvgEqual"] = "AvgEqual",
        personal: bool = True,
        user_interest: Optional[Array] = None,
        lambda_: float = 0.5,
    ) -> Array:
        """Post‑process *scores* so the Top‑k per user is PCT‑calibrated.

        `user_interest` options when *personal* is **True**:
        • 1‑D `(B,)` float → already the niche fraction per user.
        • 2‑D `(B,C)` int  → item‑id history, zero‑padded.  Non‑zeros are
            looked‑up in `niche_labels` to derive the fraction internally.
        """
        
        df = pd.read_csv(r"./dataset/ml-1m/item_popularity_labels_with_titles.csv")
        ids  = df["item_id:token"].astype(int).values      # e.g. [1, 2, 3, …, 3417]
        labs = df["popularity_label"].astype(int).values   # e.g. [1, 0, 1, …, 0]
        scores[:, 0] =  float("-inf")

        # 2) Build a 1D BoolTensor of size (max_id+1,) so we can index by ID directly
        max_id = ids.max()
        niche_labels = np.zeros(max_id+1, dtype=bool)

        # 3) Fill it: True where label == 1 (popular)
        #    If your “popular” is actually encoded as -1, just change (labs == 1) to (labs == -1)
        niche_labels[ids] = (labs == -1)

        # ---- Normalise inputs ---------------------------------------------------
        scores_np = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
        niche_np  = niche_labels.detach().cpu().numpy().astype(bool) if isinstance(niche_labels, torch.Tensor) else np.asarray(niche_labels, bool)
        B, N = scores_np.shape
        if niche_np.shape != (N,):
            raise ValueError("niche_labels must have shape (N,)")

        # ---- Exposure weights & system target -----------------------------------
        pos_weight = 1.0 / np.log2(np.arange(top_k) + 2)
        exp_budget = pos_weight.sum()
        if policy == "Equal":
            target_ratio = np.array([0.5, 0.5])
        elif policy == "AvgEqual":
            target_ratio = np.array([1 - niche_np.mean(), niche_np.mean()])
        else:
            raise ValueError("policy must be 'Equal' or 'AvgEqual'")

        quality_sign = niche_np.astype(int)

        # ---- Personalised targets ----------------------------------------------
        if personal:
            if user_interest is None:
                raise ValueError("personal=True requires 'user_interest'")
            ui = user_interest.detach().cpu().numpy() if isinstance(user_interest, torch.Tensor) else np.asarray(user_interest)
            if ui.ndim == 2:  # (B,C) id history
                if ui.shape[0] != B:
                    raise ValueError("user_interest first dim must match batch size B")
                frac = np.zeros(B)
                for u in range(B):
                    ids = ui[u][ui[u] != 0]
                    # print(ids, " sikim?")
                    if ids.size == 0:
                        # print("suka blya")
                        frac[u] = target_ratio[1]  # fallback to global ratio
                    else:
                        valid = ids[ids < N]  # ignore out‑of‑range
                        # print(valid, "sikim 2")
                        frac[u] = niche_np[valid].mean() if valid.size else target_ratio[1]
                        # print(frac, "sikim 3")
                # print(frac, " sikim 4")
            elif ui.ndim == 1:
                if ui.shape != (B,):
                    raise ValueError("user_interest must be shape (B,) or (B,C)")
                frac = ui.astype(float)
            else:
                raise ValueError("user_interest must be 1‑D or 2‑D tensor/array")
            
            p_u = np.column_stack([1.0 - frac, frac])
            print(p_u.size, " sikim 5")

            q_hat_u = self._solve_personal_targets(p_u, target_ratio) * exp_budget
        else:
            q_hat_u = np.tile(target_ratio * exp_budget, (B, 1))

        # ---- Reranking core ------------------------------------------------------
        reranked = scores_np.copy()
        order_idx = (-scores_np).argsort(1)
        for u in range(B):
            chosen = np.full(top_k, -1, int)
            cur_exp = np.zeros(2)
            sel = set()
            target_exp = q_hat_u[u]
            # Pass 1 – keep relevants when safe
            for pos in range(top_k):
                for j in order_idx[u]:
                    if j in sel:
                        continue
                    q = quality_sign[j]
                    if cur_exp[q] + pos_weight[pos] <= target_exp[q]:
                        sel.add(j); chosen[pos] = j; cur_exp[q] += pos_weight[pos]; break
            # Pass 2 – MMR fill
            for pos in range(top_k):
                if chosen[pos] != -1:
                    continue
                best_s, best_j = -np.inf, None
                for rnk, j in enumerate(order_idx[u]):
                    if j in sel:
                        continue
                    q = quality_sign[j]
                    assume = cur_exp.copy(); assume[q] += pos_weight[pos]
                    disp = 0.5 * ((assume - target_exp) ** 2).sum()
                    mmr = lambda_ * (1 / (rnk + 1)) - (1 - lambda_) * disp
                    if mmr > best_s:
                        best_s, best_j = mmr, j
                sel.add(best_j); chosen[pos] = best_j; cur_exp[quality_sign[best_j]] += pos_weight[pos]
            # score bump
            bump = scores_np[u].max() + 1
            for r, j in enumerate(chosen[::-1]):
                reranked[u, j] = bump + r

        return torch.as_tensor(reranked, dtype=scores.dtype, device=scores.device) if isinstance(scores, torch.Tensor) else reranked
