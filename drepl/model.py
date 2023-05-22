import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from quantizer import VectorQuantizer, WS_VectorQuantizer, Fast_WS_VectorQuantizer
import networks.mnist as net_mnist
import networks.fashion_mnist as net_fashionmnist
import networks.cifar10 as net_cifar10
import networks.svhn as net_svhn
import networks.celeba as net_celeba
from torch.distributions.normal import Normal


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm") != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)


class EnsembleLinear(nn.Linear):
	def __init__(self, ensemble_size, in_features, out_features):
		nn.Module.__init__(self)
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
		self.bias = nn.Parameter(torch.Tensor(ensemble_size, 1, out_features))
		self.reset_parameters()

	def forward(self, x):
		x = torch.baddbmm(self.bias, x, self.weight)
		return x


class KantorovichNetwork(nn.Module):
	def __init__(self, ensemble, embeddings_size=64, output_size=1):
		super(KantorovichNetwork, self).__init__()
		if ensemble > 1:
			self.fc = EnsembleLinear(ensemble, embeddings_size, output_size)
		else:
			self.fc = nn.Linear(embeddings_size, output_size)

	def forward(self, x):
		x = self.fc(x)
		return x


class OPT_VQAE(nn.Module):
	def __init__(self, cfgs, flgs):
		super(OPT_VQAE, self).__init__()
		# Data space
		dataset = cfgs.dataset.name
		self.dim_x = cfgs.dataset.dim_x
		self.dataset = cfgs.dataset.name
		self.cfgs = cfgs
		##############################################
		# Encoder/decoder
		##############################################
		self.encoder = eval("net_{}.EncoderVq_{}".format(dataset.lower(), cfgs.network.name))(
			cfgs.quantization.dim_dict, cfgs.network, flgs.bn)
		self.decoder = eval("net_{}.DecoderVq_{}".format(dataset.lower(), cfgs.network.name))(
			cfgs.quantization.dim_dict, cfgs.network, flgs.bn)

		#self.pre_quantization_conv_m = nn.Conv2d(128, 64, kernel_size=1, stride=1)
		self.apply(weights_init)

		##############################################
		# Codebook
		##############################################
		self.size_dict = cfgs.quantization.size_dict
		self.dim_dict = cfgs.quantization.dim_dict
		
		if cfgs.quantization.name == 'OTP':
			# first half of vector is "mu" while another half of vector is log of "sigma"
			self.codebook = nn.Parameter(torch.cat([torch.randn(self.size_dict, self.dim_dict),torch.zeros(self.size_dict, self.dim_dict)],dim=-1))
		else:
			self.codebook = nn.Parameter(torch.randn(self.size_dict, self.dim_dict))
			
		self.codebook_weight = nn.Parameter(torch.ones(cfgs.quantization.partition, self.size_dict)/self.size_dict)


		##############################################
		# Quantizer
		##############################################
		if cfgs.quantization.name == "OTP":
			self.quantizer = WS_VectorQuantizer(self.size_dict, self.dim_dict, cfgs)
		
			self.kan_net1 = KantorovichNetwork(1, self.dim_dict, 1)
			self.kan_net2 = KantorovichNetwork(1, self.dim_dict, 1)
			# due to sampling, the number of points are large, we use entropic-dualform-OT for fast computation
			self.quantizer_noise = Fast_WS_VectorQuantizer(self.size_dict, self.dim_dict, self.kan_net1, self.kan_net2, cfgs)
			self.noise_weight = cfgs.quantization.noise_weight
			
			print ('noise_weight: ', self.noise_weight)
		else:
			self.quantizer = VectorQuantizer(self.size_dict, self.dim_dict, cfgs)
		

	def forward(self, real_images, flg_train=False, flg_quant_det=True):
		# Encoding
		if flg_train:
			##############################################
			# VQ-model
			##############################################
			z_from_encoder = self.encoder(real_images)
			#z_from_encoder = self.pre_quantization_conv_m(z_from_encoder)

			z_quantized_noise = None
			
			if self.cfgs.quantization.name == "OTP":
				# WS(P(z),P(c))
				z_quantized, loss_latent, perplexity = self.quantizer(z_from_encoder, self.codebook, self.codebook_weight, flg_train)
				
				# WS(P(z),P(z_tilde)
				_, z_quantized_noise, loss_latent_noise, _ = self.quantizer_noise(z_from_encoder, self.codebook, self.codebook_weight, flg_train)
				
				loss_latent = loss_latent + self.noise_weight * loss_latent_noise
			else:
				z_quantized, loss_latent, perplexity = self.quantizer(z_from_encoder, self.codebook, self.codebook_weight, flg_train)

			latents = dict(z_from_encoder=z_from_encoder, z_to_decoder=z_quantized)
			# Decoding
			#import pdb; pdb.set_trace()
			x_reconst = self.decoder(z_quantized)
			if z_quantized_noise is not None:
				x_reconst_noise = self.decoder(z_quantized_noise)
			else:
				x_reconst_noise = None

			# Loss
			loss = self._calc_loss(x_reconst, real_images, loss_latent, flg_train=flg_train, x_reconst_noise=x_reconst_noise)
			loss["perplexity"] = perplexity

			return x_reconst, latents, loss
		else:
			z_from_encoder = self.encoder(real_images)
			z_quantized, min_encodings, e_indices, perplexity = self.quantizer._inference(z_from_encoder, self.codebook)
			# Decoding
			x_reconst = self.decoder(z_quantized)
			# Loss
			loss = self._calc_loss(x_reconst, real_images, 0, flg_train=flg_train)
			loss["perplexity"] = perplexity
			return x_reconst, min_encodings, e_indices, loss
		return 0, 0, 0
		

	def _calc_loss(self, x_reconst, x, loss_latent, flg_train=False, x_reconst_noise=None):  
		bs = x.shape[0]

		if flg_train: 
			mse = F.mse_loss(x_reconst, x) 
			if x_reconst_noise is not None:
				mse_noise = F.mse_loss(x_reconst_noise, x)
				loss_all = mse + loss_latent + self.noise_weight * mse_noise
			else:
				loss_all = mse + loss_latent

		else:
			mse = torch.mean((x_reconst - x)**2)
			loss_all = mse

		loss = dict(all=loss_all, mse=mse)

		return loss

