import torch
import torch.nn as nn
from utils_sampler import Sample_Categorical


class PoissonBackward(nn.Module):
    
    def __init__(self, D, L, K, tau):
        super(PoissonBackward, self).__init__()  

        
        self.linear = nn.Sequential(
            nn.Linear(1, D),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(D, K)
        )

        self.sampler = Sample_Categorical(tau)
       
    def forward(self, x, sample_size=1):
        # [B, L, 1]
        
        h = self.linear(x)
        logits = torch.log_softmax(h, dim = -1)

        if sample_size == 1:
            z = self.sampler(logits) 
        else: 
            logits_ = torch.repeat_interleave(logits.unsqueeze(0), repeats=sample_size, dim=0)
            z = self.sampler(logits_)
            z = z.mean(dim=0)

        # [B, L, K]
        return z, logits

class PoissonForward(nn.Module):
    
    def __init__(self, K, m, s):
        super(PoissonForward, self).__init__()  
        
        self.rate = nn.Parameter(nn.init.uniform_(torch.empty(K, 1), s, m))
        
    
    def forward(self, z):
        '''
        [B, L, K]
        '''
        rate = torch.exp(self.rate)
        rate = torch.sort(rate)[0]
        mu = torch.matmul(z, rate) # [B, L, 1]
        sigma = torch.sqrt(mu)
        if self.training:
            eps = torch.randn_like(sigma, device = mu.device)
            x = eps * sigma + mu
        else:
            x = torch.normal(mean=mu, std=sigma)

        x = x.squeeze(dim=-1)        
        return x

class PoissonPrior(nn.Module):
    def __init__(self, K, tau, device):
        super(PoissonPrior, self).__init__()
        
        self.K = K
        self.device = device

        self.logits = nn.Parameter(nn.init.normal_(torch.empty(1, device=device)))
        self.I = torch.eye(K, device = device)
        self.sampler = Sample_Categorical(tau)
    
    def get_transition_matrix(self):
        ####################################
        p = torch.sigmoid(0.1 * self.logits)
        np = (1 - p)/(self.K-1)
        A = (1 - self.I) * np + self.I * p
        return A
    
    def forward(self, z): 
        '''
        return target transition distribution for z
        '''
         # [K, K]
            
        z = z.to(self.device)
        A = self.get_transition_matrix()
        target = torch.matmul(z, A)   
        # import pdb; pdb.set_trace() 
        # target = torch.matmul(one_hot, A)    
        target = target[:, :-1, :]
        
        z0 = torch.ones((z.shape[0], 1, self.K), device = self.device) / self.K
        target = torch.cat((z0, target), dim = 1)
        target = torch.log(target)
        z = self.sampler(target)
        return z



class PoissonModel(nn.Module):
    def __init__(self, K, tau, m, s, device):
        super(PoissonModel, self).__init__() 
        self.forward_fn = PoissonForward(K, m, s)
        self.prior = PoissonPrior(K, tau, device)

        
    def forward(self, zhat):
        xhat = self.forward_fn(zhat)
        z = self.prior(zhat)
        return xhat, z

