import torch, os
import numpy as np
import torch.nn as nn
import sys, ot
from gmm import GMM, GMMBackward, rbf_kernel
from utils_io import write_pickle, load_pickle

no = sys.argv[1]
D = 5 
N = 500
# Generating data 
data_path = f'gmm/data{no}.pkl'
if os.path.isfile(data_path):
    print('Loading data ...')
    X, true_pi, true_mu, labels = load_pickle(data_path)
else: 
    print('Generating data ...')

    # Gaussian mixtures
    mu1 = torch.distributions.uniform.Uniform(0,2).sample((D,))
    mu2 = torch.distributions.uniform.Uniform(4,6).sample((D,))
    p = torch.rand(1).item()
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
true_mu = np.sort(true_mu)
true_pi = np.sort(true_pi)

# Train configurations
H, K = 5, 2
tau = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X = X.to(device)

model = GMM(D, K, tau, a=X.min().item(), b=X.max().item())
phi = GMMBackward(D, H, K, tau)

lr = 0.1
num_iters = 1000
fopt = torch.optim.Adam(model.parameters(), lr=lr)
bopt = torch.optim.Adam(phi.parameters(), lr=lr)


model.to(device)
phi.to(device)
print('Training begins ...')
model.train()
mus, pis = [], []

def compute_loss(Xhat, Z, Zhat, eta, metric='ws'):
    if metric == 'ws':
        M = ot.dist(Z, Zhat)
        a = torch.ones((N,), device = device) / N
        dist  = ot.emd2(a, a, M)
    elif metric == 'mmd':
        dist = rbf_kernel(Z, Zhat)
    
    if eta is None:
        return nn.MSELoss()(Xhat, X) 
    elif eta < 0:
        return dist
    else: 
        return nn.MSELoss()(Xhat, X) + eta * dist

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

metric = 'mmd'
for iter in range(num_iters):
    
    
    free_params(phi)
    frozen_params(model)
    Z = model.prior(X.shape[0])
    for _ in range(1):
        Zhat, _ = phi(X)        
        Xhat = model(Zhat)
        loss = compute_loss(Xhat, Z, Zhat, eta=10, metric=metric)
        loss.backward()
        bopt.step()
        bopt.zero_grad()

    
    fopt.zero_grad()
    frozen_params(phi)
    free_params(model)
    
    Z = model.prior(X.shape[0])
    Zhat, _ = phi(X)
    Xhat = model(Zhat)
    loss = compute_loss(Xhat, Z, Zhat, eta=0.1, metric=metric)
    loss.backward()
    fopt.step()


    est_mu, est_pi = model.return_parameters()
    mus.append(est_mu)
    pis.append(est_pi) 
    print('Iteration:', iter)
    print('Loss:', loss.item())
    print('Samples', Zhat.argmax(-1).sum().item(), Z.argmax(-1).sum().item())
    print('- True mu1:', true_mu[0, :])
    print('  Est. mu1:', est_mu[0, :])
    print('- True mu2:', true_mu[1, :])
    print('  Est. mu2:', est_mu[1, :])
    print('- True pi :', true_pi)
    print('  Est. pi :', est_pi)
    print('=====================================')



# Y = model.backward_fn.output_layer(X)
# pred = Y.argmax(-1).cpu()
# acc = (pred == labels).sum() / N
model_path = f'gmm/{metric}{no}.pkl'
output = (mus, pis)
write_pickle(output, model_path)

