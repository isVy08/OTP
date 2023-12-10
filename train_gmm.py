import torch, os
import numpy as np
import torch.nn as nn
import sys, ot
from gmm import GMM
from utils_io import load_model, write_pickle, load_pickle

torch.manual_seed(8)
D = 5 
N = 500
# Generating data 
data_path = 'data/gmm.pkl'
if os.path.isfile(data_path):
    print('Loading data ...')
    X, true_pi, true_mu, labels = load_pickle(data_path)
else: 
    print('Generating data ...')

    # Gaussian mixtures
    mu1 = torch.exp(torch.randn(D).abs())
    mu2 = torch.exp(torch.randn(D).abs())
    p = 0.60
    true_pi = torch.Tensor([p, 1-p])
    m1 = torch.distributions.multivariate_normal.MultivariateNormal(mu1, torch.eye(D))
    m2 = torch.distributions.multivariate_normal.MultivariateNormal(mu2, torch.eye(D))

    cat = torch.distributions.categorical.Categorical(probs=true_pi)

    X = []
    labels = []
    for _ in range(N): 
        c = cat.sample().item()
        if c == 0: 
            x = m1.sample()
        else:
            x = m2.sample()
        X.append(x)
        labels.append(c)
        
    X = torch.stack(X, dim = 0)
    true_mu = torch.stack([mu1, mu2], dim = 0)
    labels = torch.Tensor(labels)
    data = (X, true_pi, true_mu, labels)
    write_pickle(data, data_path)


print('Data size:', X.shape)
true_mu = true_mu.numpy().round(4)
true_pi = true_pi.numpy().round(4)

# Train configurations
H, K = 10, 2
eta = 1
tau = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X = X.to(device)
model_path = 'model/gmm.pt'
model = GMM(D, K, H, tau)
lr = 0.01 
num_iters = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if os.path.isfile(model_path):
    prev_loss = load_model(model, optimizer, None, model_path, device)
else:
    prev_loss = 10.


model.to(device)
print('Training begins ...')
model.train()
losses = 0
for iter in range(num_iters):
    
    
    Xhat, Z, Zhat = model(X)
    M = ot.dist(Zhat, Z)
    a = torch.ones((N,), device = device) / N
    dist  = ot.emd2(a, a, M)

    loss = nn.MSELoss()(Xhat, X) + eta * dist

    losses += loss 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    est_mu, est_pi = model.return_parameters() 
    print('Iteration:', iter)
    print('- True mu1:', true_mu[0, :])
    print('  Est. mu1:', est_mu[0, :])
    print('- True mu2:', true_mu[1, :])
    print('  Est. mu2:', est_mu[1, :])
    print('- True pi :', true_pi)
    print('  Est. pi :', est_pi)
    print('=====================================')



Y = model.backward_fn.output_layer(X)
pred = Y.argmax(-1).cpu()
acc = (pred == labels).sum() / N
print('Accuracy:', acc.item())

