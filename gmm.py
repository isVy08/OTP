import torch
import torch.nn as nn
import numpy as np
from utils_sampler import Sample_Categorical


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
    
    def __init__(self, K, D, a, b):
        super(GMMForward, self).__init__()
        self.mu = nn.Parameter(nn.init.uniform_(torch.empty(K, D), a=a, b=b))

    def forward(self, z):
        # [B, K]
        x = z @ self.mu
        # [B, D]
        return x

class GMMPrior(nn.Module):
    def __init__(self, K, tau):
        super(GMMPrior, self).__init__()
        
        self.sampler = Sample_Categorical(tau)
        
        self.logits = nn.Parameter(nn.init.ones_(torch.empty(1, K)))
    
    def forward(self, batch_size, sample_size=1):
        ones = torch.ones((batch_size, 1), device = self.logits.device).float()
        logits = torch.log_softmax(self.logits, dim = -1)
        logits_batch = torch.matmul(ones, logits)
        # z = self.sampler(logits_batch)

        if sample_size == 1: 
            z = self.sampler(logits_batch)
        else:
            z = 0
            for _ in range(100):
                z += self.sampler(logits_batch)
            
            z = z / 100
        
        return z, logits_batch

class GMM(nn.Module):
    def __init__(self, D, K, tau, a=0, b=1):
        super(GMM, self).__init__()
        self.forward_fn = GMMForward(K, D, a, b)
        self.prior = GMMPrior(K, tau)         

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
        
        
def rbf_kernel(X, Y):
    batch_size, h_dim = X.shape

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x
    
    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y
    
    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd
    
    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 / scale
        res1 = torch.exp(-C * dists_x) + torch.exp(-C * dists_y)

        res1 = (1 - torch.eye(batch_size).to(X.device)) * res1
        
        res1 = res1.sum() / (batch_size - 1)
        res2 = torch.exp(-C * dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats / batch_size