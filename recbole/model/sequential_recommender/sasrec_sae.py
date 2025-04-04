import torch
import torch.nn as nn
from .sasrec import SASRec
from .sae import SAE
from recbole.utils import (
    make_items_unpopular,
    make_items_popular,
    save_batch_activations
)

class SASRec_SAE(SASRec):
    def __init__(self, config, dataset, sasrec_model_path=None, mode="train"):
        super(SASRec_SAE, self).__init__(config, dataset)
        self.device = config["device"]
        # Load SASRec model parameters if path is provided
        if sasrec_model_path is not None:
            self.load_sasrec(sasrec_model_path)
        self.sae_module = SAE(config, self.hidden_size)  # SAE initialization
        # Mode can be 'train', 'test', or 'inference'
        self.mode = mode
        self.to(config["device"])
        for param in self.parameters():
            param.requires_grad = False  # Freeze all parameters

        for param in self.sae_module.parameters():
            param.requires_grad = True  # Unfreeze SAE parameters


    def set_sae_mode(self, mode):
        if mode in ['train', 'test', 'dampened']:
            self.mode = mode
        else:
            raise ValueError(f"Invalid mode {mode} set for SASRec_SAE")

    def forward(self, item_seq, item_seq_len):
        # Use SASRec to process the sequence
        sasrec_output = super().forward(item_seq, item_seq_len)  # Final hidden states from SASRec
        
        # Feed the SASRec output to SAE
        if self.mode in ['train', 'test']:
            sae_output = self.sae_module(sasrec_output, train_mode=(self.mode == 'train'))
        elif self.mode == 'inference':
            sae_output = self.sae_module(sasrec_output, train_mode=False)
        else:
            raise ValueError("Invalid mode for SAE forward pass")

        return sae_output

    def calculate_loss(self, interaction):
        # Compute SASRec loss first
        # sasrec_loss = super().calculate_loss(interaction)

        # Compute SAE loss
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        sasrec_output = self.forward(item_seq, item_seq_len)
        if self.mode == 'train':
            sae_loss = self.sae_module.fvu + self.sae_module.auxk_loss / 32
        else:
            sae_loss = self.sae_module.fvu

        total_loss = sae_loss

        return total_loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = make_items_popular(item_seq)
        seq_output = self.forward(item_seq, item_seq_len)
        
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        # top_recs = torch.argsort(scores, dim=1, descending=True)[:, :10]
        if(self.mode == "test"):
            # user_ids = interaction['user_id']
            save_batch_activations(self.sae_module.last_activations, 4096) 

            # self.sae_module.update_highest_activations(item_seq, top_recs, user_ids)
        # for key in top_recs.flatten():
        #     self.recommendation_count[key.item()] += 1
        return scores



    def save_sae(self, path):
        torch.save(self.sae_module.state_dict(), path)

    def load_sae(self, path):
        self.sae_module.load_state_dict(torch.load(path))

    def load_sasrec(self, path):
        """
        Load the saved SASRec model parameters into the current model.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        a = 5
        # self.load_other_parameter(checkpoint.get("other_parameter"))

