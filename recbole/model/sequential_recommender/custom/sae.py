import torch
import numpy as np
import json
import torch
import torch.nn as nn
import os
from time import time
from tqdm import tqdm
import logging
# from utils import utils


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
	
	def __init__(self,args,d_in):
		super(SAE, self).__init__()
		self.k = 16
		self.scale_size = 32

		self.device = args["device"]
		self.dtype = torch.float32
		self.to(self.device)
		self.d_in = d_in
		self.hidden_dim = d_in * self.scale_size

		self.encoder = nn.Linear(self.d_in, self.hidden_dim, device=self.device,dtype = self.dtype)
		self.encoder.bias.data.zero_()
		self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
		self.set_decoder_norm_to_unit_norm()
		self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype = self.dtype, device=self.device))
		self.activate_latents = set()
		self.previous_activate_latents = None
		self.epoch_activations = {"indices": None, "values": None} 
		self.last_activations = []
		self.highest_activations = {
            i: {"values": [], "sequences": [], "recommendations": []} for i in range(self.hidden_dim)
        }

		return

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
			c = 6
   
   
	def forward(self, x, sequences=None, train_mode=False, save_result=False):
		sae_in = x - self.b_dec
		pre_acts = nn.functional.relu(self.encoder(sae_in))
		z = self.topk_activation(pre_acts, sequences, save_result=save_result)
		x_reconstructed = z @ self.W_dec + self.b_dec

		e = x_reconstructed - x
		total_variance = (x - x.mean(0)).pow(2).sum()
		self.fvu = e.pow(2).sum() / total_variance

		if train_mode:
			# First epoch, do not have dead latent info
			if self.previous_activate_latents is None:
				self.auxk_loss = 0.0
				return x_reconstructed
			num_dead = self.hidden_dim - len(self.previous_activate_latents)
			k_aux = x.shape[-1] // 2
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

	def update_highest_activations(self, sequences, recommendations):
		"""
		Update the top 5 highest activations and corresponding sequences for each latent neuron.
		"""
		batch_size = self.last_activations.size(0)
		for i in range(batch_size):
			for j in range(self.hidden_dim):
				current_value = self.last_activations[i, j].item()
				current_sequence = sequences[i].tolist()
				current_recommendations = recommendations[i].tolist()
    
				# Insert the new activation if it qualifies for the top 5
				if len(self.highest_activations[j]["values"]) < 10:
					# Add to the list if it's not full
					self.highest_activations[j]["values"].append(current_value)
					self.highest_activations[j]["sequences"].append(current_sequence)
					self.highest_activations[j]["recommendations"].append(current_recommendations)
				else:
					# Replace the smallest value if the new one is higher
					min_index = self.highest_activations[j]["values"].index(
						min(self.highest_activations[j]["values"])
					)
					if current_value > self.highest_activations[j]["values"][min_index]:
						self.highest_activations[j]["values"][min_index] = current_value
						self.highest_activations[j]["sequences"][min_index] = current_sequence
						self.highest_activations[j]["recommendations"][min_index] = current_recommendations
      
				# Keep the top 5 sorted by value
				sorted_indices = sorted(
					range(len(self.highest_activations[j]["values"])),
					key=lambda idx: self.highest_activations[j]["values"][idx],
					reverse=True
				)
				self.highest_activations[j]["values"] = [
					self.highest_activations[j]["values"][idx] for idx in sorted_indices
				]
    
				self.highest_activations[j]["sequences"] = [
					self.highest_activations[j]["sequences"][idx] for idx in sorted_indices
				]
    
				self.highest_activations[j]["recommendations"] = [
					self.highest_activations[j]["recommendations"][idx] for idx in sorted_indices
				]

	# def save_highest_activations(self, filename="highest_activations.txt"):		
	# 	"""
	# 	Save the top 5 highest activations and their corresponding sequences to a file.
	# 	"""
	# 	with open(filename, "w") as f:
	# 		for neuron, data in self.highest_activations.items():
	# 			f.write(f"Neuron {neuron}:\n")
	# 			for value, sequence_ids, sequence, recommendations_ids, recommendations in zip(data["values"], data["sequences"], utils.get_titles_from_ids(data["sequences"]), data["recommendations"], utils.get_titles_from_ids(data["recommendations"])):
	# 				f.write(f"  Activation: {value}\n")
	# 				f.write(f"  Sequence titles: {sequence}\n")
	# 				f.write(f"  Sequence ids: {sequence_ids}\n")
	# 				f.write(f"  top recommendation ids: {recommendations_ids}\n")
	# 				f.write(f"  top recommendations: {recommendations}\n")
	# 			f.write("\n")
