import torch
import torch.nn as nn
from utils_sampler import Sample_Dirichlet, Sample_Categorical, Sample_Bernoulli

class TopicBackward(nn.Module):
    
    def __init__(self, V, K, H, D, tau):
        super(TopicBackward, self).__init__()  

        self.embed_layer = nn.Sequential(
            nn.Linear(1, H), 
            nn.ReLU(),
            nn.Linear(H, H),
        )

        self.lstm = nn.LSTM(V, D, batch_first = True)
        self.topic_layer = nn.Sequential(
            nn.Linear(D * H, H),
            nn.ReLU(),
            nn.Linear(H, K),
            nn.LogSoftmax(dim = -1)
        )

        self.sampler = Sample_Categorical(tau)
   
    def forward(self, x):
        # [B, V]
        x = x.unsqueeze(-1)
        e = self.embed_layer(x)
        e = torch.swapaxes(e, 1, 2)

        h, _ = self.lstm(e)
        h = torch.flatten(h, 1, 2)
        
        logits = self.topic_layer(h) 
        z = self.sampler(logits)
        
       
        # [B, K]
        return z, logits

class TopicForward(nn.Module):
    
    def __init__(self, V, K, tau):
        super(TopicForward, self).__init__()
        self.beta = nn.Parameter(nn.init.xavier_uniform_(torch.empty(K, V)))
        self.prior = Sample_Dirichlet()
        self.tau = tau

    def forward(self, z):
        # [B, K]
        beta = torch.exp(self.beta)
        gamma = self.prior(beta)
        x = torch.matmul(z, gamma)
        # [B, V]
        return x

class TopicPrior(nn.Module):
    def __init__(self, K, learnable = True):
        super(TopicPrior, self).__init__()
        
        self.learnable = learnable
        if learnable:
            self.alpha = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, K)))
            self.sampler = Sample_Dirichlet()
        else: 
            alpha = torch.ones(K) / K
            self.sampler = torch.distributions.dirichlet.Dirichlet(alpha)
    
    def forward(self, x):
        batch_size  = x.size(0)
        if self.learnable:
            ones = torch.ones((batch_size, 1), device = x.device).float()
            alpha = torch.exp(self.alpha)
            alpha_batch = torch.matmul(ones, alpha)
            theta = self.sampler(alpha_batch)
        else:
            theta = self.sampler.sample((batch_size, ))
            theta = theta.to(x.device)
            
        return theta

class TopicModel(nn.Module):
    def __init__(self, V, K, H, D, tau, learnable = True):
        super(TopicModel, self).__init__()
        self.backward_fn = TopicBackward(V, K, H, D, tau)
        self.forward_fn = TopicForward(V, K, tau)
        self.prior = TopicPrior(K, learnable)
         

    def forward(self, x):
        zhat, logits = self.backward_fn(x)
        xhat = self.forward_fn(zhat)
        theta = self.prior(zhat)      
        return xhat, theta, logits
