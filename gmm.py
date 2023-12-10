import torch
import torch.nn as nn
from utils_sampler import Sample_Categorical


class GMMBackward(nn.Module):
    
    def __init__(self, D, H, K, tau):
        super(GMMBackward, self).__init__()  

        self.output_layer = nn.Sequential(
            nn.Linear(D, H), 
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, K),
            nn.LogSoftmax(dim = -1)
        )

        self.sampler = Sample_Categorical(tau)

   
    def forward(self, x):
        # [B, D]
        
        logits = self.output_layer(x)
        z = self.sampler(logits)
        
       
        # [B, K]
        return z, logits


class GMMForward(nn.Module):
    
    def __init__(self, K, D):
        super(GMMForward, self).__init__()
        self.mu = nn.Parameter(nn.init.normal_(torch.empty(K, D)))

    def forward(self, z):
        # [B, K]
        x = z @ self.mu
        x = x + torch.randn_like(x)
        # [B, D]
        return x

class GMMPrior(nn.Module):
    def __init__(self, K, tau):
        super(GMMPrior, self).__init__()
        
        self.sampler = Sample_Categorical(tau)
        
        self.logits = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, K)))
    
    def forward(self, z):
        batch_size  = z.size(0)
        ones = torch.ones((batch_size, 1), device = z.device).float()
        logits = torch.log_softmax(self.logits, dim = -1)
        logits_batch = torch.matmul(ones, logits)
        z = self.sampler(logits_batch)
        return z

class GMM(nn.Module):
    def __init__(self, D, K, H, tau):
        super(GMM, self).__init__()
        self.backward_fn = GMMBackward(D, H, K, tau)
        self.forward_fn = GMMForward(K, D)
        self.prior = GMMPrior(K, tau)
         

    def forward(self, x):
        zhat, logits = self.backward_fn(x)
        xhat = self.forward_fn(zhat)
        z = self.prior(zhat)      
        return xhat, z, zhat

    def return_parameters(self):
        mu = self.forward_fn.mu
        mu = mu.detach().cpu().numpy().round(4)
        pi = torch.softmax(self.prior.logits, dim = -1)[0, :]
        pi = pi.detach().cpu().numpy().round(4)
        return mu, pi
        
        
