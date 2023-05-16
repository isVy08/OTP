import torch, os
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from cp_model import PoissonModel
from utils_io import load_model

import sys
action = sys.argv[1]


device_ids = [2,3,1,0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', device_ids[0])

def plot_result(predicted, source):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(predicted, 'o', c='tab:green', lw=1, alpha = 0.8, label='inferred rate')
    ax.plot(source, c='black', alpha=0.3, label='observed counts')
    ax.set_ylabel("latent rate")
    ax.set_xlabel("time")
    ax.set_title("Inferred latent rate over time")
    ax.legend(loc=4)
    plt.show()


# Generating data 
data_path = f'data/cp.npy'
force = False
if os.path.isfile(data_path) and not force:
    print('Loading data ...')
    observed_counts = np.load(data_path)
else:
    print('Generating data ...')
    
    true_rates = [12, 87, 60, 33]
    true_durations = [40, 60, 55, 45]
    
    random_state = 8
    
    observed_counts = [
    stats.poisson(rate).rvs(num_steps, random_state=random_state)
      for (rate, num_steps) in zip(true_rates, true_durations)
  ]

    observed_counts = np.concatenate(observed_counts)
    np.save(data_path, observed_counts)
    plt.figure()
    plt.plot(observed_counts)
    # plt.savefig('cp.png')


arr = observed_counts.reshape(-1, 1)
X = torch.from_numpy(arr).unsqueeze(-1).float()


# [12, 87, 60, 33]
K, tau = 4, 0.1
m, s = 4, 1
L = 64
lr = 1e-3
p = 0.95

model_path = sys.argv[2]

model = PoissonModel(K, L, tau, p, m, s)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
num_epochs = 20000

if os.path.isfile(model_path):
    prev_loss = load_model(model, optimizer, None, model_path, device)
    print('Previous loss:', prev_loss)
else:
    prev_loss = 100.



X = X.to(device)
kl = nn.KLDivLoss(reduction="batchmean")
weight = 0.1

if action == 'train':
    print('Training begins ...')
    model.train()
    pbar = tqdm(range(num_epochs))
    for epoch in pbar :
        X_tilde, logits, target = model(X)

        loss = nn.SmoothL1Loss()(X_tilde, X.squeeze())
        loss += weight * kl(logits, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch: {epoch} - Train Loss: {loss:.5f}")
      
        if loss < prev_loss: 
            prev_loss = loss
            print('Saving model ...')
            torch.save({'model_state_dict': model.state_dict() ,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'prev_loss': prev_loss,
                        }, model_path)


else:
    print('Evaluation begins ...')
    # [12, 87, 60, 33]

    
    model.eval()
    load_model(model, None, None, model_path, device)
    logits, _ = model.backward_fn(X)
    Z = logits.argmax(-1)
    rate = torch.exp(model.forward_fn.rate)
    print(rate, Z)