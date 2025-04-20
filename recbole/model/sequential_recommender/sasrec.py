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
import math
import torch
from torch import nn
import numpy as np
from collections import defaultdict

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
        p = 0.65          # require ≥96% unpopular (“protected”)
            
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
        scores = torch.tensor(self.simple_reranker(scores)).to(self.device)
        # scores = self.FAIR(scores).to(self.device)
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