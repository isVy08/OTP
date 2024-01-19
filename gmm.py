import torch
import torch.nn as nn
import numpy as np
from utils_sampler import Sample_Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GMMBackward(nn.Module):
    
    def __init__(self, D, H, K, tau):
        super(GMMBackward, self).__init__()  

        self.output_layer = nn.Sequential(
            nn.Linear(D, H), 
            nn.ReLU(),
            nn.Linear(H, K),
            nn.LogSoftmax(dim = -1)
        )

        self.sampler = Sample_Categorical(tau)

   
    def forward(self, x, sample_size=1):
        # [B, D]
        
        logits = self.output_layer(x)
        # z = torch.exp(logits)

        if sample_size == 1: 
            z = self.sampler(logits)
        else:
            z = 0
            for _ in range(sample_size):
                z += self.sampler(logits)
            
            z = z / sample_size
        # [B, K]
        return z, logits


class GMMForward(nn.Module):
    
    def __init__(self, K, D, a, b, var=None):
        super(GMMForward, self).__init__()
        self.mu = nn.Parameter(nn.init.uniform_(torch.empty(K, D), a=a, b=b))
        
        self.var = var # [K,1]

    def forward(self, z):
        # [B, K]
        x = z @ self.mu
        if self.var is not None:
            eps = torch.randn_like(x)
            var = x + eps * (z @ self.var)
        # [B, D]
        return x

class GMMPrior(nn.Module):
    def __init__(self, K, tau, probs):
        super(GMMPrior, self).__init__()
        
        self.sampler = Sample_Categorical(tau)
        
        if probs is None:
            self.logits = nn.Parameter(nn.init.ones_(torch.empty(1, K)))
        else:
            if probs > 0.5:
                self.logits = torch.log(torch.Tensor([[1-probs, probs]])).to(device)
            else:
                self.logits = torch.log(torch.Tensor([[probs, 1-probs]])).to(device)
    
    def forward(self, batch_size, sample_size=1):
        ones = torch.ones((batch_size, 1), device = self.logits.device).float()
        logits = torch.log_softmax(self.logits, dim = -1)
        logits_batch = torch.matmul(ones, logits)

        if sample_size == 1: 
            z = self.sampler(logits_batch)
        else:
            z = 0
            for _ in range(100):
                z += self.sampler(logits_batch)
            
            z = z / 100
        
        return z, logits_batch

class GMM(nn.Module):
    def __init__(self, D, K, tau, a, b, var=None, probs=None):
        super(GMM, self).__init__()
        self.forward_fn = GMMForward(K, D, a, b, var)
        self.prior = GMMPrior(K, tau, probs)         

    def forward(self, z):
        xhat = self.forward_fn(z)    
        return xhat


    def return_parameters(self):
        mu = self.forward_fn.mu
        mu = mu.detach().cpu().numpy().round(4)
        pi = torch.softmax(self.prior.logits, dim = -1)[0, :]
        pi = pi.detach().cpu().numpy().round(4)
        mu = np.sort(mu)
        pi = np.sort(pi)
        return mu, pi
        