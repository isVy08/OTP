import torch, os
import numpy as np
import torch.nn as nn
import sys, random, ot
from scipy import stats
from hmm import PoissonModel
from geomloss import SamplesLoss
from torch.utils.data import DataLoader
from utils_io import load_model, write_pickle, load_pickle


action = sys.argv[1]

torch.manual_seed(8)


# Generating data 
data_path = f'data/hmm.pkl'
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
print(true_rates, true_p)

K, tau = 4, 0.1
m, s = 4, 2
D = 64
L = X.shape[1]

weight = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = f'cp/hmm.pt'

model = PoissonModel(D, L, K, tau, m, s, device)

lr = 0.005
num_epochs = 50


optimizer = torch.optim.Adam(model.parameters(), lr=lr)



if os.path.isfile(model_path):
    prev_loss = load_model(model, optimizer, None, model_path, device)
else:
    prev_loss = 10.

train_indices = list(range(X.size(0)))
train_loader = DataLoader(train_indices, batch_size=500, shuffle=True)

X = X.to(device)
kl = nn.KLDivLoss(reduction="batchmean")
sk = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=0.9, backend="tensorized")
ground_cost = 'euclidean'

if action == 'train':
    model.to(device)
    print('Training begins ...')
    model.train()
    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        
        losses = 0
        for idx in train_loader:
            x = X[idx, ]
            x_tilde, logits, theta = model(x)
 
            loss = nn.SmoothL1Loss()(x_tilde, x.squeeze(-1))
            probs = torch.exp(logits)

            if ground_cost == 'kl':
                B = probs.size(0)
                a = torch.ones((B,), device = device) / B 
                M = torch.zeros((B, B), device = device)
                for i in range(B):
                    for j in range(B):                        
                        M[i, j] = kl(logits[i, :], theta[j, :]) 
            

                dist  = ot.emd2(a, a, M)
            elif ground_cost == 'sk':
                dist = sk(probs, theta).mean()
            else:
                B = probs.size(0)
                a = torch.ones((B,), device = device) / B 
                
                probs = probs.flatten(1,2)
                theta = theta.flatten(1,2)
                M = ot.dist(probs, theta)
                dist  = ot.emd2(a, a, M)
            
            
            total_loss = loss + weight * dist
            losses += loss 

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
        
        
        losses /= len(train_loader)


        if losses < prev_loss: 
            prev_loss = losses
            print('Saving model ...')
            torch.save({'model_state_dict': model.state_dict() ,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'prev_loss': prev_loss,
                        }, model_path)
        
else:
    print('Evaluation begins ...')
    

    model.to(device)
    model.eval()
    load_model(model, None, None, model_path, device)
    X_tilde, logits, target = model(X)
    loss = nn.SmoothL1Loss()(X_tilde, X.squeeze())
    print('Reconstruction loss:', loss)
    rate = torch.exp(model.forward_fn.rate)
    
    rate = rate[:, 0].detach().cpu().tolist()
    rate = sorted(rate)
    print('True rate:', true_rates)
    print('Est. rate:', rate)
    p = model.prior.get_transition_matrix()
    print('True Trans Prob:', true_p)
    print('Est. Trans Prob:', p[0,0].item())