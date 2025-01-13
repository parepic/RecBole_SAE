import torch
import torch.nn as nn
from recbole.model.sequential_recommender import SASRec

class SASRec_SAE(SASRec):
    def __init__(self, config, dataset, sasrec_model_path=None, mode="train"):
        from recbole.model.sequential_recommender.custom import SAE  # Replace with the correct path
        super(SASRec_SAE, self).__init__(config, dataset)
        self.sae_module = SAE(config, self.hidden_size)  # SAE initialization

        # Mode can be 'train', 'test', or 'inference'
        self.mode = mode

        # Load SASRec model parameters if path is provided
        if sasrec_model_path is not None:
            self.load_sasrec(sasrec_model_path)

    def set_sae_mode(self, mode):
        if mode in ['train', 'test', 'inference']:
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

    # def train_sae(self, data_loader, epoch):
    #     if(self.mode=="train"):
    #         self.sae_module.train()
    #     loss_lst = []

    #     for batch in data_loader:
    #         self.sae_optimizer.zero_grad()

    #         # Prepare the batch
    #         item_seq = batch[self.ITEM_SEQ]
    #         item_seq_len = batch[self.ITEM_SEQ_LEN]
    #         sasrec_output = super().forward(item_seq, item_seq_len).detach()

    #         # Shuffle item_ids
    #         item_ids = batch['item_id']
    #         indices = torch.argsort(torch.rand(*item_ids.shape), dim=-1)
    #         batch['item_id'] = item_ids[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices]

    #         # Compute SAE loss
    #         if epoch == 1:
    #             sae_loss = self.sae_module.fvu
    #         else:
    #             sae_loss = self.sae_module.fvu + self.sae_module.auxk_loss / 32

    #         sae_loss.backward()
    #         self.sae_optimizer.step()

    #         loss_lst.append(sae_loss.detach().cpu().numpy())

    #     mean_loss = float(torch.tensor(loss_lst).mean().item())

    #     # Debug if NaN encountered
    #     if torch.isnan(torch.tensor(mean_loss)):
    #         import ipdb; ipdb.set_trace()

    #     return mean_loss

    def save_sae(self, path):
        torch.save(self.sae_module.state_dict(), path)

    def load_sae(self, path):
        self.sae_module.load_state_dict(torch.load(path))

    def load_sasrec(self, path):
        """
        Load the saved SASRec model parameters into the current model.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint, strict=False)
        self.load_other_parameter(checkpoint.get("other_parameter"))

