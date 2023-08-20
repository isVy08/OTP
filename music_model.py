import torch, ot
import torch.nn as nn
from utils_sampler import Sample_Categorical, Sample_Bernoulli

class MusicBackward(nn.Module):
    
    def __init__(self, K, H, D, tau):
        super(MusicBackward, self).__init__()  

        self.linear = nn.Sequential(
            nn.Linear(D, H),
            nn.ReLU(),
            nn.Linear(H, H), 
            nn.ReLU(),
            nn.Linear(H, K),
            nn.LogSoftmax(-1)
        )
        self.sampler = Sample_Categorical(tau)
    
    def forward(self, x):
        '''
        [B, L, D]
        returns [B, L, K]
        '''
        h = self.linear(x)
        z = self.sampler(h)
        return h, z

class MusicForward(nn.Module):
    
    def __init__(self, K, D, tau):
        super(MusicForward, self).__init__()  

 
        self.logits = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, K, D)))
        self.K = K
        self.D = D
        self.sampler = Sample_Bernoulli(1.0)
    
    def forward(self, z):
        '''
        [B, L, K]
        '''
        logits = torch.matmul(z, self.logits)
        x = torch.sigmoid(logits) 
        # [B, L, D] 
        return x

class DirichletPrior(nn.Module):
    def __init__(self, A):
        super(DirichletPrior, self).__init__()
        self.K = A.shape[0]
        self.A = A
    
    def forward(self, z): 
        '''
        [B, L, K] : logits
        '''
        selection = z.argmax(-1).to('cpu')
        one_hot = torch.eye(self.K)[selection].to(z.device)
        target = torch.matmul(one_hot, self.A)
        
        target = target[:, :-1, :]
        z0 = torch.ones((z.size(0), self.K)) * 1/self.K
        z0 = z0.unsqueeze(1).to(z.device)
        target = torch.cat((z0, target), dim = 1)
        return target


class MusicModel(nn.Module):
    def __init__(self, K, H, D, tau, TM):
        super(MusicModel, self).__init__() 
        self.backward_fn = MusicBackward(K, H, D, tau)
        self.forward_fn = MusicForward(K, D, tau)
        self.prior = DirichletPrior(TM)

        
    def forward(self, x):
        logits, z = self.backward_fn(x)
        x = self.forward_fn(z)
        target = self.prior(logits)
        return x, logits, target

class Criterion(nn.Module):
    def __init__(self, weight):
        super(Criterion, self).__init__() 
        self.weight = weight
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.bce = nn.BCELoss()
    
    def forward(self, x, output):
        x_tilde, logits, target = output # probs
        loss = self.bce(x_tilde, x)
        B = x.size(0)
        a = torch.ones((B,), device = x.device) / B 
        M = torch.zeros((B, B), device = x.device)
        for i in range(B):
            for j in range(B):
                M[i, j] = self.kl(logits[i, :], target[j])
        
        ws  = ot.emd2(a, a, M)
      
        loss += self.weight * ws
        return loss

