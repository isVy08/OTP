import torch
import numpy as np
import torch.nn as nn
import sys, ot
from gmm import GMM, GMMBackward
from utils_model import free_params, frozen_params, kl_matrix
from geomloss import SamplesLoss


setting = sys.argv[1]
lr = 0.1
D = 2
N = 500
# Generating data 



print('Generating data ...')

# High overlapping Gaussian mixtures
mu1 = torch.distributions.uniform.Uniform(0,2).sample((D,))
mu2 = torch.distributions.uniform.Uniform(1,3).sample((D,))
m1 = torch.distributions.multivariate_normal.MultivariateNormal(mu1, torch.eye(D))
m2 = torch.distributions.multivariate_normal.MultivariateNormal(mu2, torch.eye(D))

p = np.random.uniform(0.50, 0.70)
true_pi = torch.Tensor([p, 1-p]) 

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
    
def compute_loss(x, xhat, z, zhat, eta, metric, logz, logzhat):
    recons = nn.MSELoss()(xhat, x)
    if metric == 'l2':
        M = ot.dist(z, zhat)
        a = torch.ones((N,), device = device) / N
        dist  = ot.emd2(a, a, M)
    elif metric == 'kl':
        M = kl_matrix(zhat, z, logzhat, logz)
        a = torch.ones((N,), device = device) / N
        dist  = ot.emd2(a, a, M)
    elif metric == 'sk':
        sk = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=0.9, backend="tensorized")
        dist = sk(Zhat, Z)
    
    if eta is None:
        return recons
    elif eta < 0:
        return dist
    else: 
        return recons + eta * dist

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

if setting == 'gmm':
    # Correctly specified model
    var = torch.Tensor([[1.0], [1.0]]).to(device)
    probs = None
    
elif setting == 'pgmm':
    # Mis-specified mixture weights
    probs = np.random.uniform(0, 1)
    var = torch.Tensor([[1.0], [1.0]]).to(device)
   
elif setting == 'vgmm':
    # Mis-specified mixture variances
    v1 = np.random.uniform(0, 2)
    v2 = np.random.uniform(0, 2)
    var = torch.Tensor([[v1], [v2]]).to(device)
    probs = true_pi[1]
    
elif setting == 'vpgmm': 
    # Mis-specified mixture weights and variances
    v1 = np.random.uniform(0, 2)
    v2 = np.random.uniform(0, 2)
    probs = np.random.uniform(0, 1)
    var = torch.Tensor([[v1], [v2]]).to(device)

else:
    raise ValueError

model = GMM(D, K, tau, a=X[0, ].min().item(), b=X[0, ].max().item(), var=var, probs=probs)
phi = GMMBackward(D, H, K, tau)
num_iters = 1000
fopt = torch.optim.Adam(model.parameters(), lr=lr)
bopt = torch.optim.Adam(phi.parameters(), lr=lr)

metric = 'kl'
model.to(device)
phi.to(device)
print('Training begins ...')
model.train()
phi.train()
mus, pis = [], []


eta = 0.1
print(f'Running {setting} model ...')
for iter in range(num_iters):

    free_params(phi)
    frozen_params(model)
    Z, logZ = model.prior(X.shape[0])
    Zhat, logZhat  = phi(X, 100)

    z, _  = phi(X, 1)        
    Xhat = model(z)
    loss = compute_loss(X, Xhat, Z, Zhat, eta, metric, logZ, logZhat)
    bopt.zero_grad()
    loss.backward()
    bopt.step()

    frozen_params(phi)
    free_params(model)
    
    Z, logZ = model.prior(X.shape[0])
    Zhat, logZhat = phi(X, 1)

    z, _  = phi(X, 1)        
    Xhat = model(z)
    obj = compute_loss(X, Xhat, Z, Zhat, eta, metric, logZ, logZhat)
    fopt.zero_grad()
    obj.backward()
    fopt.step()

    est_mu, est_pi = model.return_parameters()
    mus.append(est_mu)
    pis.append(est_pi) 
    print('Iteration:', iter)
    print('Loss:', loss.item())
    print('- True mu1:', true_mu[0, :])
    print('  Est. mu1:', est_mu[0, :])
    print('- True mu2:', true_mu[1, :])
    print('  Est. mu2:', est_mu[1, :])
    print('- True pi :', true_pi)
    print('  Est. pi :', est_pi)
    print('=====================================')


