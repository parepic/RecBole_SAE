import torch
import numpy as np
import json
import torch
import torch.nn as nn
import os
from time import time
from tqdm import tqdm
import logging
from recbole.utils import utils
import pandas as pd

class SAE(nn.Module):
	# @staticmethod
	# def parse_model_args(parser):
	# 	parser.add_argument('--sae_k', type=int, default=32,
	# 						help='top k activation')
	# 	parser.add_argument('--sae_scale_size', type=int, default=32,
	# 						help='scale size')
	# 	parser.add_argument('--recsae_model_path', type=str, default='',
	# 						help='Model save path.')
	# 	return parser
	
	def __init__(self,config,d_in):
		super(SAE, self).__init__()
		self.k = config["sae_k"]
		self.scale_size = config["sae_scale_size"]
		self.neuron_count = None
		self.damp_percent = None
		self.unpopular_only = False
		self.corr_file = None
		self.user_pop_scores = []
		self.device = config["device"]
		self.dtype = torch.float32
		self.to(self.device)
		self.d_in = d_in
		self.death_patience = 0
		self.hidden_dim = d_in * self.scale_size
		self.activation_count = torch.zeros(self.hidden_dim, device=config["device"])
		self.encoder = nn.Linear(self.d_in, self.hidden_dim, device=self.device,dtype = self.dtype)
		self.encoder.bias.data.zero_()
		self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
		self.set_decoder_norm_to_unit_norm()
		self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype = self.dtype, device=self.device))
		self.activate_latents = set()
		self.previous_activate_latents = None
		self.epoch_activations = {"indices": None, "values": None} 
		self.last_activations = torch.empty(0, dtype=torch.float32, device=self.device)
        # 2) highest_activations dict, each neuron j -> dict of GPU tensors
        #    (values, sequences, recommendations)
		self.highest_activations = {
            j: {
                "values": torch.empty(0, dtype=torch.float32, device=self.device),      # [<=10]
                "sequences": torch.empty((0, 50), dtype=torch.long, device=self.device),
                "recommendations": torch.empty((0, 10), dtype=torch.long, device=self.device)
            }
            for j in range(self.hidden_dim)
        }


		return

	def set_dampen_hyperparam(self, corr_file=None, neuron_count=42, damp_percent=0.1, unpopular_only=True):
		self.corr_file = corr_file
		self.neuron_count = neuron_count
		self.damp_percent = damp_percent
		self.unpopular_only = unpopular_only
  
	def get_dead_latent_ratio(self, need_update = 0):
		ans =  1 - len(self.activate_latents)/self.hidden_dim
		# only update training situation for auxk_loss
		if need_update:
			# logging.info("[SAE] update previous activated Latent here")
			self.previous_activate_latents = torch.tensor(list(self.activate_latents)).to(self.device)
		self.activate_latents = set()
		return ans


	def set_decoder_norm_to_unit_norm(self):
		assert self.W_dec is not None, "Decoder weight was not initialized."
		eps = torch.finfo(self.W_dec.dtype).eps
		norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
		self.W_dec.data /= norm + eps

	

	def topk_activation(self, x, sequences, save_result):
		topk_values, topk_indices = torch.topk(x, self.k, dim=1)
		# topk_indices: shape (B, N)
		flat_indices = topk_indices.view(-1)  # shape (B * N)

		# Count occurrences of each index
		counts = torch.bincount(flat_indices, minlength=self.hidden_dim)

		# Update activation count
		self.activation_count += counts.to(self.activation_count.device)
		self.activate_latents.update(topk_indices.cpu().numpy().flatten())

		self.last_activations = x
		# Update highest activations
		# self.update_highest_activations(x, sequences)

		if save_result:
			if self.epoch_activations["indices"] is None:
				self.epoch_activations["indices"] = topk_indices.detach().cpu().numpy()
				self.epoch_activations["values"] = topk_values.detach().cpu().numpy()
			else:
				self.epoch_activations["indices"] = np.concatenate(
					(self.epoch_activations["indices"], topk_indices.detach().cpu().numpy()), axis=0
				)
				self.epoch_activations["values"] = np.concatenate(
					(self.epoch_activations["values"], topk_values.detach().cpu().numpy()), axis=0
				)

		sparse_x = torch.zeros_like(x)
		sparse_x.scatter_(1, topk_indices, topk_values.to(self.dtype))
		return sparse_x

		

	def update_topk_recommendations(self, predictions, current_sequences, k=10):
		"""
		Update top-k recommendations for sequences in highest_activations.

		Parameters:
		- predictions: Tensor of shape [B, N], where B is batch size and N is the number of items.
		- current_sequences: List of sequences (IDs) in the current batch.
		- k: Number of top recommendations to save.
		"""
		# Convert current_sequences to a list of lists for easy comparison
		current_sequences_list = [seq.tolist() for seq in current_sequences]
  
		for neuron_idx, data in self.highest_activations.items():
			for idx, stored_sequence in enumerate(data["sequences"]):
				# Check if the stored sequence is in the current batch
				if stored_sequence in current_sequences_list:
					# Find the index of the stored sequence in the current batch
					batch_idx = current_sequences_list.index(stored_sequence)
					
					# Get predictions for this sequence
					pred_scores = predictions[batch_idx].cpu().numpy()  # Convert to numpy for sorting
					
					# Find indices of the top-k scores
					topk_indices = np.argsort(pred_scores, axis=1)[:, -k:][:, ::-1]  # Add 1 to match item IDs
     
					# Update the recommendations for this sequence
					data["recommendations"].append(topk_indices.tolist())

	def dampen_neurons(self, pre_acts):
		if self.unpopular_only:
			if(self.neuron_count == 0): 
				return pre_acts
			unpop_indexes = utils.get_top_n_neuron_indexes(self.neuron_count).to(self.device)

			# Create a scaling vector: linearly spaced from 2 (most unpop-biased) to 0 (least)
			scales = torch.linspace(2.0, 0.0, steps=self.neuron_count, device=self.device)  # shape: (neuron_count,)

			# Find which of the selected indexes are active (non-zero) in pre_acts
			nonzero_mask = pre_acts[:, unpop_indexes] > 0  # shape: (batch_size, neuron_count)

			# Apply damping only where activation > 0
			pre_acts[:, unpop_indexes] = torch.where(
				nonzero_mask,
				pre_acts[:, unpop_indexes] + scales,  # multiply by scaling factor
				pre_acts[:, unpop_indexes]            # else leave unchanged
			)

			# pre_acts[:, unpop_indexes] *= (1 + self.damp_percent)
   
			# unpop_indexes, unpop_values = zip(*utils.get_extreme_correlations(self.corr_file, self.neuron_count, self.unpopular_only))
			# print("peyser ", self.neuron_count, ' ', self.unpopular_only, ' ', self.damp_percent, ' ', self.corr_file)
			# pre_acts[:, unpop_indexes] *= (1 + self.damp_percent)
   			# differences = utils.get_difference_values(unpop_idxs)
			# # Convert to PyTorch tensors
			# unpop_idxs = torch.tensor(unpop_idxs, dtype=torch.long, device=self.device)  # Ensure correct indexing type
			# differences = torch.tensor(differences, dtype=pre_acts.dtype, device=self.device)  # Ensure correct dtype & device
			# scale = (1 - differences * self.damp_percent)
			# print(differences)  # Debugging

			# # Reshape differences to match pre_acts[:, unpop_idxs] for broadcasting
			# differences = differences.view(1, -1)  # Reshape to (1, F) where F = len(unpop_idxs)

			# # Create a mask where pre_acts is not zero
			# mask = pre_acts[:, unpop_idxs] != 0  # Boolean mask

			# Apply the operation only where pre_acts is nonzero
			# pre_acts[:, unpop_indexes] *= (1 + scale_values)  # Element-wise masking

		else:
			pop_idxs, unpop_idxs = utils.get_extreme_correlations(self.corr_file, self.neuron_count, self.unpopular_only)
			pre_acts[:, pop_idxs] *= (1 - self.damp_percent)
			pre_acts[:, unpop_idxs] *= (1 + self.damp_percent)
		return pre_acts	
  
	def forward(self, x, sequences=None, train_mode=False, save_result=False):
		sae_in = x - self.b_dec
		pre_acts = nn.functional.relu(self.encoder(sae_in))
		print(self.scale_size, ' blya olcu ')
		if self.corr_file:
			pre_acts = self.dampen_neurons(pre_acts)


		z = self.topk_activation(pre_acts, sequences, save_result=save_result)
		x_reconstructed = z @ self.W_dec + self.b_dec

		e = x_reconstructed - x
		total_variance = (x - x.mean(0)).pow(2).sum()
		self.fvu = e.pow(2).sum() / total_variance

		if train_mode:
			if self.death_patience >= 5000000:
				dead = self.get_dead_latent_ratio(need_update=1)
				print(" dead percentage: ", dead )
				self.death_patience = 0
			self.death_patience += pre_acts.shape[0]
			# First epoch, do not have dead latent info
			if self.previous_activate_latents is None:
				self.auxk_loss = 0.0
				return x_reconstructed
			num_dead = self.hidden_dim - len(self.previous_activate_latents)
			k_aux = x.shape[-1] * 4
			if num_dead == 0:
				self.auxk_loss = 0.0
				return x_reconstructed
			scale = min(num_dead / k_aux, 1.0)
			k_aux = min(k_aux, num_dead)
			dead_mask = torch.isin(
				torch.arange(pre_acts.shape[-1]).to(self.device),
				self.previous_activate_latents,
				invert=True
			)
			auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
			auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
			e_hat = torch.zeros_like(auxk_latents)
			e_hat.scatter_(1, auxk_indices, auxk_acts.to(self.dtype))
			e_hat = e_hat @ self.W_dec + self.b_dec

			auxk_loss = (e_hat - e).pow(2).sum()
			self.auxk_loss = scale * auxk_loss / total_variance
   
		return x_reconstructed


	def update_highest_activations(self, sequences, recommendations, user_ids):
		"""
		1) Find top-10 activations per neuron for this new batch on GPU.
		2) Merge them with the existing top-10 stored in self.highest_activations.
		3) Keep only the top-10 overall (no .cpu() or .tolist() here).
		
		:param sequences:       [batch_size, seq_len] GPU tensor
		:param recommendations: [batch_size, num_items] GPU tensor
								We'll extract top-10 recommended items per sample.
		"""
		# utils.save_user_popularity_score(0.9, user_ids, sequences)
		total_pop_scores, total_unpop_scores = utils.fetch_user_popularity_score(user_ids,sequences)
		utils.save_batch_user_popularities(total_pop_scores, total_unpop_scores)
		utils.save_batch_activations(self.last_activations, 4096) 

		# ------------------------
		# A) Get top-10 per neuron (column)
		# ------------------------
		# self.last_activations has shape [batch_size, hidden_dim]
		batch_top_vals, batch_top_idxs = torch.topk(self.last_activations, k=10, dim=0)
		# shapes: [10, hidden_dim]

		# ------------------------
		# C) Merge each neuron's top-10
		# ------------------------
		for j in range(self.hidden_dim):
			# 1) Gather new data for neuron j
			new_indices = batch_top_idxs[:, j]         # [10] - indices in this batch
			new_vals = batch_top_vals[:, j]           # [10]
			new_seqs = sequences[new_indices]         # [10, seq_len]
			new_recs = recommendations[new_indices]    # [10, 10]

			# 2) Retrieve old top-k from self.highest_activations[j]
			old_vals = self.highest_activations[j]["values"]           # [<=10]
			old_seqs = self.highest_activations[j]["sequences"]        # [<=10, seq_len]
			old_recs = self.highest_activations[j]["recommendations"]  # [<=10, 10]

			# 3) Concatenate old + new (-> up to 20)
			all_vals = torch.cat([old_vals, new_vals], dim=0)    # [<=20]
			all_seqs = torch.cat([old_seqs, new_seqs], dim=0)    # [<=20, seq_len]
			all_recs = torch.cat([old_recs, new_recs], dim=0)    # [<=20, 10]

			# 4) Take top-10 again
			if all_vals.numel() > 0:
				topvals, topidxs = torch.topk(all_vals, k=min(10, all_vals.size(0)), dim=0)
			else:
				# no old or new data
				continue

			# 5) Index sequences & recs
			new_values = topvals
			new_sequences = all_seqs[topidxs]
			new_recommendations = all_recs[topidxs]

			# 6) Store back as GPU tensors
			self.highest_activations[j]["values"] = new_values
			self.highest_activations[j]["sequences"] = new_sequences
			self.highest_activations[j]["recommendations"] = new_recommendations

    
	def save_highest_activations(self, filename=r"./dataset/ml-1m/popular_activatins.csv"):		
		"""
		Save the top 5 highest activations and their corresponding sequences to a file.
		"""
		df = pd.DataFrame({
			'index': np.arange(len(self.activation_count)),
			'count': self.activation_count.cpu().numpy()
		})

		# Save to CSV
		df.to_csv(filename, index=False)
  
  
		# corr_pop = utils.calculate_pearson_correlation(r"./dataset/ml-1m/user_scores_pop.h5", r"./dataset/ml-1m/correlations_pop.csv")
		# corr_unpop = utils.calculate_pearson_correlation(r"./dataset/ml-1m/user_scores_unpop.h5", r"./dataset/ml-1m/correlations_unpop.csv")
		# file_path = r'./dataset/ml-1m/ml-1m.item'
		# data_item = pd.read_csv(file_path, sep='\t', encoding='latin1')  # Try 'latin1', change to 'cp1252' if needed
		# with open(filename, "w") as f:
		# 	for neuron, data in self.highest_activations.items():
		# 		f.write(f"Neuron {neuron}:\n")
		# 		f.write(f"Popularity Correlation:{corr_pop[neuron]}\n")
		# 		f.write(f"Unpopularity Correlation:{corr_unpop[neuron]}\n")
		# 		for value, sequence_ids, sequence, recommendations_ids, recommendations in zip(data["values"],  data["sequences"], utils.get_item_titles(data["sequences"], data_item), data["recommendations"], utils.get_item_titles(data["recommendations"], data_item)):
		# 			f.write(f"  Activation: {value}\n")
		# 			f.write(f"  Last 10 Sequence titles: {sequence[-10:]}\n")
		# 			f.write(f"  Sequence ids: {sequence_ids}\n")
		# 			f.write(f"  top recommendation ids: {recommendations_ids}\n")
		# 			f.write(f"  top recommendations: {recommendations}\n")
		# 		f.write("\n")

