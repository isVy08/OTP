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

action = sys.argv[1]
ver = sys.argv[2]
batch = sys.argv[3]
code = sys.argv[4]

print('Loading data ...')
ver = ver + code
data_path = f'subdata/V{ver}.pkl'
observed_counts, true_rates, true_p = load_pickle(data_path)
model_path = f'submodel_{batch}/em_v{ver}.pkl'


true_rates = sorted(true_rates)

if action == 'train':
        K = len(true_rates)
        # Fixed transition matrix
        model = hmm.PoissonHMM(n_components=K, n_iter=5000, params="sl", init_params="sl")
        model.transmat_ = get_transition_matrix(true_p, K)

        start = time.time()
        X = observed_counts.reshape(-1, 1)
        lengths = [observed_counts.shape[1]] * observed_counts.shape[0]
        print('Training begins ...')
        remodel = model.fit(X, lengths)

        end = time.time()

        file = open('subresult/em_runtime', 'a+')
        runtime = end - start
        file.write(str(runtime)+'\n')

        file.close()

        
        rates = sorted([item[0] for item in remodel.lambdas_])

        print('True rate:', true_rates)
        print('Rate:', rates)

        # print(remodel.transmat_)

        print('Saving model ...')
        with open(model_path, "wb") as file: pickle.dump(remodel, file)

else: 
        model = load_pickle(model_path)
        rates = sorted([item[0] for item in model.lambdas_])
        file = open(f'subresult/em_result_{batch}.txt', 'a+')
        file.write(f"V{ver};{str(true_rates)};{str(rates)};{true_p};{true_p}\n")
        file.close()

