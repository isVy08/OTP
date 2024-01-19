import matplotlib.pyplot as plt
from utils_io import load_pickle 
import numpy as np 
import sys

steps = 1000
L = 30
dims = 2
root = sys.argv[1]


colors = {'ws': 'red', 'em': 'black'}
methods = list(colors.keys())
mu_errs = {}
pi_errs = {}

def match_mu(a, b):
    dims = []
    for i in range(a.shape[0]):
        errs = np.abs(a[i, ] - b).mean(axis=1)
        while errs.argmin() in dims:
            errs[errs.argmin()] = errs[errs.argmin()] + 1 
        dims.append(errs.argmin())
    a = a.reshape(-1, )
    b = np.concatenate([b[i] for i in dims])
    return a, b
    

for method in methods:
    mu_errs[method] = {i: [] for i in range(steps)}
    pi_errs[method] = {i: [] for i in range(steps)}
for no in range(1, L): 
    data_path = f'data/gmm/data{no}.pkl'
    X, true_pi, true_mu, labels = load_pickle(data_path)
    true_mu = true_mu.numpy().round(4)
    true_pi = np.sort(true_pi.numpy().round(4))
    true_mu = np.sort(true_mu)
    for method in methods:
        mus, pis = load_pickle(f'{root}/{method}{no}.pkl')
        pis = np.sort(pis)
        for i in range(steps):
            if method == 'em':
                est_mu = np.sort(np.stack(mus[i], axis=0))
            else:
                est_mu = np.sort(mus[i])
            est_mu, true_mu_ = match_mu(est_mu, true_mu)
            mu_errs[method][i].append(np.abs(true_mu_ - est_mu))
            pi_errs[method][i].append(np.abs(true_pi - pis[i]))

# D x 2 by flattening mus
mu_data = {}
pi_data = {}
for method in methods:
    mu_data[method] = np.stack([np.stack(v, axis=0).mean(axis=0) for v in mu_errs[method].values()],axis=0)
    pi_data[method] = np.stack([np.stack(v, axis=0).mean(axis=0) for v in pi_errs[method].values()],axis=0)


fig, axs = plt.subplots(2,dims, figsize=(12,4), sharex=True)
d = 0
for c in range(dims):
    for r in range(2):
        for method in methods:
            axs[r,c].plot(mu_data[method][:, d], '-', c=colors[method], lw=1, label=method)
        d += 1
plt.legend()
plt.tight_layout()
plt.savefig(f'image/{root}_mu.png')

fig, axs = plt.subplots(1,2, figsize=(12,4), sharex=True)

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1)
for method in methods:
    arr = pi_data[method]
    ax.plot(arr[:, 0], '-', c=colors[method], lw=1, label=method)

plt.legend()
plt.tight_layout()
plt.savefig(f'image/{root}_pi.png')
