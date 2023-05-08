from utils_io import load_pickle
import numpy as np
import copy, time

class MultinomialDirichlet(object):
    """
    This class implements a Multinomial component with a symmetric Dirichlet
    prior on top. The component contains no data points.

    class attributes:
    dd: (1x1) dimensionality
    aa: (1x1) prior Dirichlet concentration parameter
    pi: (dx1) Multinomial probability vector

    xx: (dx1) data
    mm: (1x1) total count in xx
    mi: (dx1) individual counts
    nn: (1x1) number of data points

    model:
    pi|aa ~ Dirichlet(aa/dd,...,aa/dd)
    xx|mm, mi, pi ~ Multinomial(mm, mi, pi)

    """
    def __init__(self, hh=None):
        if hh == None:
            self.dd = 0
            self.aa = 0
            self.pi = 0
            self.xx = []
            self.mm = 0
            self.mi = []
            self.nn = 0
            self.Z0 = []

        else:
            try:
                self.dd = hh['dd']
            except KeyError:
                self.dd = 0
            try:
                self.aa = hh['aa'] / hh['dd']
            except KeyError:
                self.aa = 0
            try:
                self.mm = hh['mm']
            except KeyError:
                self.mm = 0
            try:
                self.mi = copy.deepcopy(hh['mi'])
            except KeyError:
                self.mi = np.zeros(self.dd, dtype=int)
            self.nn = 0
            self.pi = 0
            self.xx = []
            self.Z0 = []

    def add_item(self, xx):
        """
        adds a data point xx to the component.
        """
        self.mi[xx] += 1
        self.mm += 1
        self.nn += 1

    def del_item(self, xx):
        """
        deletes a data point xx from the component.
        """
        self.mi[xx] -= 1
        self.mm -= 1
        self.nn -= 1

    def log_predictive(self, xx):
        """
        returns the log-predictive probability of xx given other data items in the component
        """
        ll = np.log((self.aa + self.mi[xx])/(self.aa * self.dd + self.mm))
        # ll = log_predictive_jit(self.aa, self.mi, self.dd, self.mm, xx)
        return ll

    def mean(self):
        return (self.aa + self.mi) / (self.aa * self.dd + self.mm)

    def view_attributes(self):
        print(self.__dict__)



if __name__ == "__main__":

    import sys
    action = sys.argv[1]

    from utils_model import load_topic_config
    config_index = int(sys.argv[2])
    # config_index = 0
    config = load_topic_config(config_index)

    true_K = config[0]                # true number of component
    N = n_groups = config[1] 
    L = n_tokens = config[2]
    
  
    data_path = f'data/lda{config_index}.pickle'
    true_beta, true_topics, word_matrix  = load_pickle(data_path)
    K, V = true_topics.shape
    KK = int(np.sqrt(V))

    from utils_io import load_pickle, write_pickle

    if action == 'train':

        # constructing the canonical Bayesian component
        hh = {}
        hh['dd'] = V      # cardinality of MultinomialDirichlet
        hh['aa'] = 0.1    # concentration parameter for prior Dirichlet

        q0 = MultinomialDirichlet(hh)
        # initializing the LDA object
        # concentration parameter for the Dirichlet prior on components
        lda_aa = 1/K
        lda_KK = K                 # numbe rof LDA topics
        from LDA import LDA
        lda = LDA(q0, lda_aa, lda_KK, word_matrix)

        lda.initialize_z()
        start = time.time()
        lda.gibbs_sampler(n_burnins=0, n_samples=300, n_lags=0)
        end = time.time()
        print('========== TRAINING TIME ==========', end - start)
        write_pickle(lda, f'model/gibbs{config_index}.pickle')

    else:
        lda = load_pickle(f'model/gibbs{config_index}.pickle')
        atoms, _ = lda.posterior_sampler()
        #%%
        inferred_topics = np.zeros((K, V))
        for kk in range(K):
            inferred_topics[kk] = atoms[kk].mean()
        
        
        from utils_model import match_topics, visualize_topics
        truth = {k : sorted(np.where(true_topics[k, :])[0].tolist()) for k in range(K)}
        
        estimated = match_topics(inferred_topics, truth)
        topics = list(range(K)) 
        visualize_topics(topics, V, truth, estimated, f'gibbs{config_index}')

        # Compute estimates
        import torch
        P = torch.Tensor([lda.qq[i].mean() for i in range(K) ])
        Q = torch.Tensor(true_topics)

        from utils_model import compute_topic_estimates
        compute_topic_estimates(P, Q)

        
