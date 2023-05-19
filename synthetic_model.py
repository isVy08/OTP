import torch
import torch.nn as nn
from utils_sampler import Sample_Dirichlet, Sample_Categorical

class TopicBackward(nn.Module):
    
    def __init__(self, L, V, K, H, D, tau):
        super(TopicBackward, self).__init__()  

    
        self.lstm = nn.LSTM(V, D, batch_first = True)
        self.topic_layer = nn.Sequential(
            nn.Linear(L * D, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, K),
            nn.LogSoftmax(dim = -1)
        )


        self.sampler = Sample_Categorical(tau)
   
    def forward(self, x):
        # [B, L, V]
  

        h, _ = self.lstm(x) # [B, L, D]        
        h = torch.flatten(h, 1, 2)
        logits = self.topic_layer(h) 
        
        L = x.shape[1]
        B = x.shape[0]
        K = logits.shape[-1]
        
        logits = logits.unsqueeze(1)
        ones = torch.ones((B, L, K), device = x.device)
        logits = torch.mul(logits, ones)  
        z = self.sampler(logits)

        
       
        # [B, L, K]
        return z, logits

class TopicForward(nn.Module):
    
    def __init__(self, V, K, tau):
        super(TopicForward, self).__init__()
        self.gamma = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, K, V)))
        self.sampler = Sample_Categorical(tau)

    def forward(self, z):
        # [B, L, K]
        
        gamma = torch.matmul(z, self.gamma)
        x = torch.softmax(gamma, dim = -1)
        
        # [B, L, V]
        return x

class TopicPrior(nn.Module):
    def __init__(self, K, learnable = True):
        super(TopicPrior, self).__init__()
        
        self.learnable = learnable
        
        self.alpha = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, 1, K)))
        self.sampler = Sample_Dirichlet()
         
    def forward(self, x):
        batch_size  = x.size(0)
        L = x.size(1)
        
        ones = torch.ones((batch_size, L, 1), device = x.device).float()
        alpha = torch.softmax(self.alpha, dim = -1)
        alpha_batch = torch.matmul(ones, alpha)
        theta = self.sampler(alpha_batch)
            
        return theta

class TopicModel(nn.Module):
    def __init__(self, L, V, K, H, D, tau, learnable = True):
        super(TopicModel, self).__init__()
        self.backward_fn = TopicBackward(L, V, K, H, D, tau)
        self.forward_fn = TopicForward(V, K, tau)
        self.prior = TopicPrior(K, learnable)
         

    def forward(self, x):
        
        zhat, logits = self.backward_fn(x)
        
        xhat = self.forward_fn(zhat)
        
        theta = self.prior(zhat) 
        probs = torch.exp(logits) 
        return xhat, theta, probs
