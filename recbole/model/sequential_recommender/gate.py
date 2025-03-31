from torch import nn
import torch
import numpy as np
class SASRecWithGating(nn.Module):    
    def __init__(self, sasrec_model, gate_indices, device='cpu', popularity_labels=None):
        super(SASRecWithGating, self).__init__()  # ✅ This line is critical
        self.to(device)        
        self.sasrec = sasrec_model
        self.recommendation_count = np.zeros(self.sasrec.n_items)
        self.loss_fct = nn.CrossEntropyLoss()

    
    def forward(self, input_seq, input_seq_len):
        # for param in self.sasrec.parameters():
        #     print('required? ', param.requires_grad)
        hidden = self.sasrec.forward(input_seq, input_seq_len)  # [batch_size, seq_len, hidden_dim]
        return hidden
    


    def calculate_loss(self, interaction):
        item_seq = interaction[self.sasrec.ITEM_SEQ]
        item_seq_len = interaction[self.sasrec.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.sasrec.POS_ITEM_ID]
        test_item_emb = self.sasrec.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        return loss

    
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.sasrec.ITEM_SEQ]
        # item_seq = make_items_unpopular(item_seq)
        item_seq_len = interaction[self.sasrec.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        # if self.corr_file:
        #     seq_output = self.dampen_neurons_sasrec(seq_output)
        # save_batch_activations(seq_output, 64)
        test_items_emb = self.sasrec.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        top_recs = torch.argsort(scores, dim=1, descending=True)[:, :10]
        for key in top_recs.flatten():
            self.recommendation_count[key.item()] += 1
        return scores
