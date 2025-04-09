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

        if scores is not None:
            # Clamp scores to avoid exploding weights
            scores = torch.tensor(scores, dtype=torch.float32, device=logits.device)
            scores = torch.clamp(scores, min=1e-4)
            # Inverse propensity weighting
            weighted_loss = (ce_loss / scores).mean()
        else:
            weighted_loss = ce_loss.mean()
        # print("ISP-weighed loss", weighted_loss)
        return weighted_loss
    
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
        # # item_seq = make_items_unpopular(item_seq)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        if self.corr_file != None:
            seq_output = self.dampen_neurons_sasrec(seq_output)
        # save_batch_activations(seq_output, 64)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        top_recs = torch.argsort(scores, dim=1, descending=True)[:, :10]
        for key in top_recs.flatten():
            self.recommendation_count[key.item()] += 1
        return scores


    def dampen_neurons_sasrec(self, pre_acts):
        if(self.N == None): 
            return pre_acts        
        # if self.unpopular_only:
        #     unpop_mean_sd = pd.read_csv(r"./dataset/ml-1m/row_stats_unpopular.csv")
        #     unpop_indexes, unpop_values = zip(*get_extreme_correlations(self.corr_file, self.N, self.unpopular_only))
        #     unpop_mean_sd = unpop_mean_sd.iloc[list(unpop_indexes)]
        #     print(unpop_mean_sd.columns.tolist())

        #     means = torch.tensor(unpop_mean_sd["mean"].values, device=pre_acts.device)
        #     sds = torch.tensor(unpop_mean_sd["std"].values, device=pre_acts.device)
        #     for i, neuron_idx in enumerate(list(unpop_indexes)):
        #         # Get values for this neuron across all rows
        #         vals = pre_acts[:, neuron_idx]

        #         # Condition: lower than mean and also lower than (mean - sd)
        #         condition = (vals > means[i] + 0.5)

        #         # Apply dampening only where the condition is true
        #         # pre_acts[condition, neuron_idx] += self.damp_percent

        # else:
        (lowest_corrs, highest_corrs) = get_extreme_correlations(self.corr_file, int(self.N), self.unpopular_only)
        unpop_mean_sd = pd.read_csv(r"./dataset/ml-1m/row_stats_unpopular.csv")

        unpop_indexes, unpop_values = zip(*highest_corrs)
        pop_indexes, pop_values = zip(*lowest_corrs)

        # Convert Cohen's d values to tensor and normalize them to [0, 2]
        unpop_cohens_d = torch.tensor([abs(v) for v in unpop_values], device=pre_acts.device)
        pop_cohens_d = torch.tensor([abs(v) for v in pop_values], device=pre_acts.device)

        def normalize_to_range(x, new_min=0.0, new_max=2):
            min_val = torch.min(x)
            max_val = torch.max(x)
            if max_val == min_val:
                return torch.full_like(x, (new_min + new_max) / 2)  # fallback to midpoint if all values are equal
            return (x - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

        unpop_weights = normalize_to_range(unpop_cohens_d, new_max=self.gamma)
        pop_weights = normalize_to_range(pop_cohens_d, new_max=self.gamma)

        unpop_mean_sd = unpop_mean_sd.iloc[list(unpop_indexes)]
        means_unpop = torch.tensor(unpop_mean_sd["mean"].values, device=pre_acts.device)
        sds_unpop = torch.tensor(unpop_mean_sd["std"].values, device=pre_acts.device)

        for i, neuron_idx in enumerate(list(unpop_indexes)):
            vals = pre_acts[:, neuron_idx]
            condition = (vals > (means_unpop[i] * self.beta))
            pre_acts[:, neuron_idx] += sds_unpop[i] * unpop_weights[i]

        for i, neuron_idx in enumerate(list(pop_indexes)):
            vals = pre_acts[:, neuron_idx]
            condition = (vals < (means_unpop[i] * self.beta))
            pre_acts[:, neuron_idx] -= sds_unpop[i] * pop_weights[i]
        return pre_acts	
