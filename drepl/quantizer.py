import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
import ot
from torch import nn


class VectorQuantizer(nn.Module):
	def __init__(self, size_dict, dim_dict, cfgs):
		super(VectorQuantizer, self).__init__()
		self.size_dict = size_dict
		self.dim_dict = dim_dict
		self.beta = cfgs.quantization.beta
		print('Using VectorQuantizer')
	
	def forward(self, z_from_encoder, codebook, codebook_weight, flg_train):
		return self._quantize(z_from_encoder, codebook, flg_train)
	
	def _quantize(self, z, codebook, flg_train):
		z = z.permute(0, 2, 3, 1).contiguous()
		
		z_flattened = z.view(-1, self.dim_dict)
		# distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

		d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(codebook**2, dim=1) - 2 * torch.matmul(z_flattened, codebook.t())

		# find closest encodings
		min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
		min_encodings = torch.zeros(
			min_encoding_indices.shape[0], self.size_dict).to(z.device)
		min_encodings.scatter_(1, min_encoding_indices, 1)

		# get quantized latent vectors
		z_q = torch.matmul(min_encodings, codebook)
		z_q = z_q.view(z.shape)
		if flg_train:
			#loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
			loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
		else:
			loss = 0.0
		# preserve gradients
		z_q = z + (z_q - z).detach()

		# perplexity
		e_mean = torch.mean(min_encodings, dim=0)
		perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

		# reshape back to match original input shape
		z_q = z_q.permute(0, 3, 1, 2).contiguous()

		return z_q, loss, perplexity
	

	def _inference(self, z, codebook):
		
		z = z.permute(0, 2, 3, 1).contiguous()

		z_flattened = z.view(-1, self.dim_dict)
		# distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

		split = codebook.shape[1]//2
		mu = codebook[:,:split]

		cost_matrix = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
			torch.sum(mu**2, dim=1) - 2 * \
			torch.matmul(z_flattened, mu.t())

		# find closest encodings
		min_encoding_indices = torch.argmin(cost_matrix, dim=1).unsqueeze(1)
		min_encodings = torch.zeros(
			min_encoding_indices.shape[0], self.size_dict).to(z.device)
		min_encodings.scatter_(1, min_encoding_indices, 1)

		# get quantized latent vectors
		z_q = torch.matmul(min_encodings, mu)

		# perplexity
		e_mean = torch.mean(min_encodings, dim=0)
		perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

		z_q = z_q.view(z.shape)
		# reshape back to match original input shape
		z_q = z_q.permute(0, 3, 1, 2).contiguous()
		return z_q, min_encodings, min_encoding_indices, perplexity



def weights_init(m):
	classname = m.__class__.__name__
	if classname.find("EnsembleLinear") != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)
  

class Fast_WS_VectorQuantizer(VectorQuantizer):
	def __init__(self, size_dict, dim_dict, kan_net1, kan_net2, cfgs):
		super(Fast_WS_VectorQuantizer, self).__init__(size_dict, dim_dict, cfgs)

		self.kl_regularization = cfgs.quantization.kl_regularization
		self.kan_net1 = kan_net1
		self.kan_net2 = kan_net2
		self.kan_lr = cfgs.quantization.kan_lr
		self.kan_iteration = cfgs.quantization.kan_iteration
		self.softmax = nn.Softmax(dim=0)

		self.reset_kan = False
		self.optim_kan1 = torch.optim.Adam(
			self.kan_net1.parameters(),
			lr=self.kan_lr,
			weight_decay=0.1,
			amsgrad=True,
		)

		
		self.optim_kan2 = torch.optim.Adam(
			self.kan_net2.parameters(),
			lr=self.kan_lr,
			weight_decay=0.1,
			amsgrad=True,
		)

		self.epsilon = cfgs.quantization.epsilon
		self.phi_net_troff = 1.0
		self.kl_loss = nn.KLDivLoss()


	def forward(self, z_from_encoder, codebook, codebook_weight, flg_train):
		return self._quantize(z_from_encoder, codebook, codebook_weight, flg_train)
	
	
	def init_kan(self):
		self.kan_net1.apply(weights_init)
		self.kan_net2.apply(weights_init)


	def compute_OT_loss(self, ot_cost, kan_net, weight_X=None, weight_Y=None):
		phi_network = kan_net.mean(-1)
		# E_{P_y}[phi(y)]
		if weight_Y is None:
			phi_loss = torch.mean(phi_network)
		else:
			phi_loss = torch.sum(weight_Y * phi_network)

		# exp[(-d(x,y) + phi(y))/epsilon]
		exp_term = (-ot_cost + phi_network) / self.epsilon

		if weight_X is None:
			weight_X = torch.tensor(1.0 / ot_cost.shape[0])
		

		if weight_Y is None:
			OT_loss = torch.sum(weight_X*(- self.epsilon * (torch.log(torch.tensor(1.0 / exp_term.shape[1])) + torch.logsumexp(exp_term, dim=1)))) + self.phi_net_troff * phi_loss
		else:
			# using log-sum-exp trick            
			max_exp_term = exp_term.max(1)[0].clone().detach()
			sum_exp = torch.sum(weight_Y*torch.exp(exp_term-max_exp_term.unsqueeze(-1)),dim=1)
			#min_sum_exp = torch.zeros(sum_exp.size()).to(sum_exp.device)+1e-39
			#logsumexp = max_exp_term+torch.log(torch.max(sum_exp, min_sum_exp))
			logsumexp = max_exp_term+torch.log(sum_exp)
			OT_loss = torch.sum(weight_X*(- self.epsilon * (logsumexp))) + self.phi_net_troff * phi_loss

		return OT_loss

	def _quantize(self, z, codebook, codebook_weight, flg_train):
		z = z.permute(2, 3, 0, 1).contiguous()

		z_flattened = z.view(-1, self.dim_dict)
		# distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
		repeat_value = torch.ones(self.size_dict).to(z.device).int()*(z_flattened.shape[0]//self.size_dict)
		centroids = torch.repeat_interleave(codebook, repeat_value, dim=0)
		split = centroids.shape[1]//2
		mu = centroids[:,:split]
		cov = torch.exp(centroids[:,split:])


		z_sampled  = mu + cov * torch.randn(cov.shape).to(z.device)

		cost_matrix = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
			torch.sum(z_sampled**2, dim=1) - 2 * \
			torch.matmul(z_flattened, z_sampled.t())


		# find closest encodings
		min_encoding_indices = torch.argmin(cost_matrix, dim=1).unsqueeze(1)
		min_encodings = torch.zeros(
			min_encoding_indices.shape[0], z_flattened.shape[0]).to(z.device)
		min_encodings.scatter_(1, min_encoding_indices, 1)

		# get quantized latent vectors
		z_q_noise = torch.matmul(min_encodings, z_sampled)
		z_q = torch.matmul(min_encodings, mu)

		if self.reset_kan:
			self.init_kan()

		if flg_train:
			codeword_weight = torch.ones(z_flattened.shape[0]).to(z.device) / z_flattened.shape[0]
			
			for i in range(0, self.kan_iteration):
				kan2_latent_value = self.kan_net2(z_flattened.clone().detach())
				kan1_code_value = self.kan_net1(z_sampled.clone().detach())
				
				loss1 = -self.compute_OT_loss(cost_matrix.clone().detach(), kan1_code_value, weight_Y=codeword_weight)
				
				loss2 = -self.compute_OT_loss(cost_matrix.permute(1, 0).clone().detach(), kan2_latent_value, weight_X=codeword_weight)
				
				self.optim_kan1.zero_grad()
				loss1.backward() #loss1 = f(kan2)
				self.optim_kan1.step()

				self.optim_kan2.zero_grad()
				loss2.backward() #loss2 = f(kan1)
				self.optim_kan2.step()
			
			kan2_latent_value = self.kan_net2(z_flattened)
			kan1_code_value = self.kan_net1(z_sampled)
			loss1 = self.compute_OT_loss(cost_matrix, kan1_code_value, weight_Y=codeword_weight)
			loss2 = self.compute_OT_loss(cost_matrix.permute(1, 0), kan2_latent_value, weight_X=codeword_weight)
			loss = self.beta * (loss1 + loss2)
		else:
			loss = 0.0
		z_q = z_q.view(z.shape)
		# preserve gradients
		z_q = z + (z_q - z).detach()

		z_q_noise = z_q_noise.view(z.shape)
		# preserve gradients
		z_q_noise = z# + (z_q_noise - z).detach()

		#import pdb; pdb.set_trace()

		# perplexity
		e_mean = torch.mean(min_encodings, dim=0)
		perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

		# reshape back to match original input shape
		z_q = z_q.permute(2, 3, 0, 1).contiguous()
		z_q_noise = z_q_noise.permute(2, 3, 0, 1).contiguous()

		return z_q, z_q_noise, loss, perplexity


class WS_VectorQuantizer(VectorQuantizer):
	def __init__(self, size_dict, dim_dict, cfgs):
		super(WS_VectorQuantizer, self).__init__(size_dict, dim_dict, cfgs)

		self.kl_loss = nn.KLDivLoss()
		self.kl_regularization = cfgs.quantization.kl_regularization
		self.softmax = nn.Softmax(dim=0)


	def forward(self, z_from_encoder, codebook, codebook_weight, flg_train):
		return self._quantize(z_from_encoder,codebook, codebook_weight, flg_train)
	

	def _quantize(self, z, codebook, codebook_weight, flg_train):
		z = z.permute(2, 3, 0, 1).contiguous()

		z_flattened = z.view(-1, self.dim_dict)
		# distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

		split = codebook.shape[1]//2
		mu = codebook[:,:split]

		cost_matrix = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
			torch.sum(mu**2, dim=1) - 2 * \
			torch.matmul(z_flattened, mu.t())

		# find closest encodings
		min_encoding_indices = torch.argmin(cost_matrix, dim=1).unsqueeze(1)
		min_encodings = torch.zeros(
			min_encoding_indices.shape[0], self.size_dict).to(z.device)
		min_encodings.scatter_(1, min_encoding_indices, 1)

		# get quantized latent vectors
		z_q = torch.matmul(min_encodings, mu)

		if flg_train:
			loss = 0.0
			size_batch = z.shape[2]
			num_iter = z.shape[0] * z.shape[1]

			U_N = torch.ones(size_batch).to(z.device) / size_batch
			U_K = torch.ones(self.size_dict).to(z.device) / self.size_dict 
			pi = nn.Softmax(dim=1)(codebook_weight)
			
			for i in range(num_iter):
				loss += self.beta * ot.emd2(pi[i], U_N, cost_matrix[size_batch*i:size_batch*(i+1)].t())
				if self.kl_regularization > 0.0:
					loss += self.kl_regularization * self.kl_loss(U_K.log(), pi[i])
	
					
		else:
			loss = 0.0
		
		z_q = z_q.view(z.shape)
		# preserve gradients
		z_q = z + (z_q - z).detach()

		# perplexity
		e_mean = torch.mean(min_encodings, dim=0)
		perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

		# reshape back to match original input shape
		z_q = z_q.permute(2, 3, 0, 1).contiguous()

		return z_q, loss, perplexity


