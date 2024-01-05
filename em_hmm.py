import sys 
import pickle
import time
import numpy as np
from hmmlearn import hmm
from utils_io import load_pickle

def get_transition_matrix(p, K):
        pp = (1 - p)/(K-1)
        I = np.eye(K)
        A = (1 - I) * pp + I * p
        return A




ver = sys.argv[1]
batch = sys.argv[2]



print('Loading data ...')
data_path = f'data/V{ver}.pkl'
observed_counts, true_rates, true_p = load_pickle(data_path)

K = len(true_rates)
# Fixed transition matrix
model = hmm.PoissonHMM(n_components=K, n_iter=3, params="sl", init_params="sl")
model.transmat_ = get_transition_matrix(true_p, K)

start = time.time()
X = observed_counts.reshape(-1, 1)
lengths = [observed_counts.shape[1]] * observed_counts.shape[0]
print('Training begins ...')
remodel = model.fit(X, lengths)

end = time.time()
print('Training time:', end - start)
true_rates = sorted(true_rates)

print('True rate:', true_rates)
print('Rate:', remodel.lambdas_)

print('Saving model ...')
model_path = f'model_{batch}/em_v{ver}.pkl'
with open(model_path, "wb") as file: pickle.dump(remodel, file)


