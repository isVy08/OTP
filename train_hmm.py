import torch, os
import torch.nn as nn
import random, ot
from tqdm import tqdm
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from hmm import PoissonModel, PoissonBackward
from torch.utils.data import DataLoader
from utils_io import load_model, write_pickle, load_pickle
from utils_model import free_params, frozen_params

from geomloss import SamplesLoss

import sys
action = sys.argv[1]
ver = sys.argv[2]
batch = sys.argv[3]


def plot_result(predicted, source, path, title = None):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(predicted, 'o', c='tab:green', lw=1, alpha = 0.8, label='inferred rate')
    ax.plot(source, c='black', alpha=0.3, label='observed counts')
    ax.set_ylabel("latent rate")
    ax.set_xlabel("time")
    if title:
        ax.set_title(f"Inferred latent rate over time: {title}")
    else:
        ax.set_title("Inferred latent rate over time")
    ax.legend(loc=4)
    plt.show()
    # plt.savefig('segment.pdf')
    plt.savefig(path)



# Generating data 
data_path = f'data/V{ver}.pkl'
force = False
if os.path.isfile(data_path) and not force:
    print('Loading data ...')
    # observed_counts = np.load(data_path)
    observed_counts, true_rates, true_p = load_pickle(data_path)
else:
    print('Generating data ...')
    
    # Each rate is sampled from a Uniform distribution [a,b] with a, b > 1
    true_uniform = [(10,20), (80,90), (50,60), (30,40)]

    # Sample rate here
    K = len(true_uniform)
    
    # true_rates = [12, 87, 60, 33]
    true_rates = []
    for a,b in true_uniform:
        rate = np.random.randint(a,b)
        true_rates.append(rate)

    # Generate true transition probability following a HMM
    p = np.random.uniform(0.80, 0.90, 1)[0].round(2)
    
    def sample_state(previous_state, p, K):
        # if it is state 0
        if previous_state is None: 
            probs = torch.Tensor([1/K] * K)
            dist = torch.distributions.categorical.Categorical(probs=probs)
        else: 
            probs = [(1-p)/(K-1)] * K
            probs[previous_state] = p
            probs = torch.Tensor(probs)
            dist = torch.distributions.categorical.Categorical(probs=probs)

        return dist.sample().item()
    
    random_state = random.choice(range(100))
    observed_counts = []
    states = []
    for _ in range(200):
        if len(states) == 0: 
            s = sample_state(None, p, K)
        else: 
            s = sample_state(states[-1], p, K)
        states.append(s)
        rate = true_rates[s]
        counts = stats.poisson(rate).rvs(50000, random_state=random_state)
        observed_counts.append(counts)
    
    
    observed_counts = np.stack(observed_counts, axis=1)
    true_p = p
    write_pickle((observed_counts, true_rates, true_p), data_path)



X = torch.from_numpy(observed_counts).unsqueeze(-1).float()
true_rates = sorted(true_rates)
print(true_rates, true_p, ver)

K, tau = 4, 0.1
m, s = 4, 2
D = 64
L = X.shape[1]

weight = 0.01


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = f'model_{batch}/hmm_v{ver}.pt'

phi = PoissonBackward(D, L, K, tau)
model = PoissonModel(K, tau, m, s, device)

lr = 0.005

num_epochs = 1000
fopt = torch.optim.Adam(model.parameters(), lr=lr)
bopt = torch.optim.Adam(phi.parameters(), lr=lr)

train_indices = list(range(X.size(0)))
train_loader = DataLoader(train_indices, batch_size=500, shuffle=True)


if os.path.isfile(model_path):
    prev_loss = load_model(model, fopt, None, model_path, device)
else:
    prev_loss = 30.

print('Previous loss', prev_loss)

def compute_loss(x, xhat, z, zhat, eta, metric):
      
      recons = nn.SmoothL1Loss()(xhat, x)

      if len(z.shape) == 3:
        z = z.flatten(1,2)
        zhat = zhat.flatten(1,2)


      if metric == 'l2': 
        B = x.size(0)
        a = torch.ones((B,), device = device) / B 
        M = ot.dist(zhat, z)
        dist = ot.emd2(a, a, M)
      
      elif metric == 'sk':
        sk = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=0.9, backend="tensorized")
        dist = sk(zhat, z)
      
      elif metric == 'kl':
        B = x.size(0)
        a = torch.ones((B,), device = device) / B 
        M = torch.zeros((B, B), device = device)
        kl = nn.KLDivLoss(reduction="batchmean")
        lhat = torch.log(zhat + 1e-8)
        l = torch.log(z + 1e-8)
        for i in range(B):
          for j in range(B):
            M[i, j] = 0.5 * (kl(lhat[i, :], z[j, :]) + kl(l[i, :], zhat[j, :]))
        dist = ot.emd2(a, a, M)

      if eta is None:
          return recons 
      elif eta < 0:
          return dist 
      else: 
         return recons + eta * dist




X = X.to(device)

if action == 'train':
    # model = nn.DataParallel(model, device_ids)
   
        
    model.to(device)
    phi.to(device)
    print('Training begins ...')
    model.train()
    phi.train()
    
    for epoch in range(num_epochs):

      
        Loss = 0
        for idx in train_loader:
            x = X[idx, ]
        
            free_params(phi)
            frozen_params(model)
            zhat, _ = phi(x)
            xhat, z = model(zhat)   
            loss = compute_loss(x.squeeze(-1), xhat, z, zhat, eta=0.01, metric='l2')
                
            bopt.zero_grad()
            loss.backward()
            bopt.step()

            frozen_params(phi)
            free_params(model)
            zhat, _ = phi(x)
            xhat, z = model(zhat)   
            loss = compute_loss(x.squeeze(-1), xhat, z, zhat, eta=0.01, metric='l2')
            
            fopt.zero_grad()
            loss.backward()
            fopt.step()

            Loss += loss.item()
        
        Loss /= len(train_loader)
    
        if epoch % 100 == 0 or Loss < prev_loss:
            p = model.prior.get_transition_matrix()
            rates = torch.exp(model.forward_fn.rate)
            a = rates[:, 0].detach().cpu().round().tolist()
            a = sorted(a)
            print('Epoch:', epoch)
            print('True rate:', true_rates)
            print('Rate:', a)
            print('Loss:', Loss)
            print(p.detach().cpu())
            print('========================================================')

        if Loss < prev_loss: 
            prev_loss = Loss
            print('Saving model ...')
            ckpt = {'model_state_dict': model.state_dict() ,
                        'optimizer_state_dict': fopt.state_dict(),
                        'prev_loss': prev_loss,
                        }
            torch.save(ckpt, model_path)

        
else:
    if os.path.isfile(model_path):
        print('Evaluation begins ...')

        model.to(device)
        model.eval()
        load_model(model, None, None, model_path, device)
        rate = torch.exp(model.forward_fn.rate)
        
        rate = rate[:, 0].detach().cpu().tolist()
        rate = sorted(rate)
        print(rate)
        p = model.prior.get_transition_matrix()
        p = p[0,0].item()
        print(p)
        file = open(f'result/hmm_result_{batch}.txt', 'a+')
        file.write(f"V{ver};{str(true_rates)};{str(rate)};{true_p};{p}\n")
        file.close()
        
        