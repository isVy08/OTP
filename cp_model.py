import torch
import torch.nn as nn
from utils_sampler import Sample_Categorical

class PoissonBackward(nn.Module):
    
    def __init__(self, D, K, L, tau):
        super(PoissonBackward, self).__init__()  

        
        # self.selector = nn.RNN(1, K, L, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(1, L),
            nn.ReLU(),
            nn.Linear(L, L),
            nn.ReLU(),
            nn.Linear(L, K),
            nn.LogSoftmax(-1)
        )


        self.sampler = Sample_Categorical(tau)
    
    def forward(self, x):
        '''
        [L, 1, 1]
        '''
        
        # h, _ = self.selector(x)
        # h = h.squeeze(1)
        h = self.linear(x.squeeze(-1))
        z = self.sampler(h)

      
        return h, z

class PoissonForward(nn.Module):
    
    def __init__(self, K, m, s):
        super(PoissonForward, self).__init__()  

 
        self.rate = nn.Parameter(nn.init.normal_(torch.empty(K, 1), mean=m, std=s))
    
    def forward(self, z):
        '''
        [L, K]
        '''
        rate = torch.exp(self.rate)
        mu = torch.matmul(z, rate) # [L, 1]
        sigma = torch.sqrt(mu)
        if self.training:
            eps = torch.randn_like(sigma, device = mu.device)
            x = eps * sigma + mu
        else:
            x = torch.normal(mean=mu, std=sigma)

        x = x.squeeze()        
        return x

class PoissonPrior(nn.Module):
    def __init__(self, K, p):
        super(PoissonPrior, self).__init__()
        self.z0 = torch.ones((1, K)) * 1/K
        A = []
        for i in range(K):
            arr = [(1-p)/(K-1) for _ in range(K)] 
            arr[i] = p 
            A.append(arr)
        self.A = torch.Tensor(A)
        self.K = K
    
    def forward(self, z): 
        '''
        return target transition distribution for z
        '''
        selection = z.argmax(-1)
        one_hot = torch.eye(self.K)[selection]
        target = torch.matmul(one_hot, self.A)
        target = target[:-1, :]
        target = torch.cat((self.z0, target), dim = 0)
        target = target.to(z.device)
        return target



class PoissonModel(nn.Module):
    def __init__(self, D, K, L, tau, p, m, s):
        super(PoissonModel, self).__init__() 
        self.backward_fn = PoissonBackward(D, K, L, tau)
        self.forward_fn = PoissonForward(K, m, s)
        self.prior = PoissonPrior(K, p)

        
    def forward(self, x):
        logits, z = self.backward_fn(x)
        x = self.forward_fn(z)
        target = self.prior(logits)
        return x, logits, target


# K = 4
# m = PoissonModel(K, 3, 0.2, 0.95, 2)
# x = torch.rand((70, 1, 1))
# x, logits, target = m(x)

