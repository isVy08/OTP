import torch, os
import torch.nn as nn
import sys, time, ot
import numpy as np
from torch.utils.data import DataLoader
from utils_model import load_topic_config, free_params, frozen_params
from amortized_model import TopicModel, TopicBackward
from utils_io import load_model, write_pickle, load_pickle
from geomloss import SamplesLoss

def compute_loss(x, xhat, z, zhat, eta, metric):
      
      bce = nn.BCELoss()
      recons = bce(xhat, x)

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
            M[i, j] = kl(lhat[i, :], z[j, :]) + kl(l[i, :], zhat[j, :])
        dist = ot.emd2(a, a, M)

      if eta is None:
          return recons 
      elif eta < 0:
          return dist 
      else: 
         return recons + eta * dist
        
            
def train_epoch(model, phi, fopt, bopt, X, train_loader, device):
    
    
    Loss = 0
    
    for idx in train_loader:
      x = X[idx, :].to(device)
      

      free_params(phi)
      frozen_params(model)
      zhat, _ = phi(x)
      xhat, z = model(zhat)   
      loss = compute_loss(x, xhat, z, zhat, eta=1e-4, metric='sk')
      phi.lstm.flatten_parameters()
      
      bopt.zero_grad()
      loss.backward()
      bopt.step()

      frozen_params(phi)
      free_params(model)
      zhat, _ = phi(x)
      xhat, z = model(zhat)   
      loss = compute_loss(x, xhat, z, zhat, eta=1e-4, metric='sk')
      
      fopt.zero_grad()
      loss.backward()
      fopt.step()

    
      Loss += loss.item()
   
    return Loss / len(train_loader)

def sample_index(p):
    """
    draws a asample from a Multinomial distribution with probability vector p.
    It draws a saomple between K elemnts and returns the index of selected element.
    """
    i = np.random.multinomial(1, p, 1)
    return i.argmax()

def generate_bartopics(n_topics):
    """
    generates the bartopics of the ggraphical example
    """

    KK = int(n_topics / 2)
    vocab_size = KK * KK
    true_topics = np.zeros((n_topics, vocab_size))

    # horizontal topics
    for kk in range(KK):
        b = np.zeros((KK, KK))
        b[kk, :] = np.ones(KK)
        b /= b.sum()
        true_topics[kk, :] = b.reshape((1, vocab_size))

    # vertical topics
    for kk in range(KK, n_topics):
        b = np.zeros((KK, KK))
        b[:, kk - KK] = np.ones(KK)
        b /= b.sum()
        true_topics[kk, :] = b.reshape((1, vocab_size))

    return true_topics

def generate_topics(true_beta, n_topics):
    """
    generates topics distribution following a Dirichlet parameterized by Beta (list)
    """
    vocab_size = len(true_beta)
    true_topics = np.zeros((n_topics, vocab_size))
    for i in range(n_topics):
      dist = np.random.mtrand.dirichlet(true_beta)
      true_topics[i, :] = dist
    return true_topics


if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    action = sys.argv[1]
    config_index = int(sys.argv[2])
    config = load_topic_config(config_index)

    '''
    Generate synthetic data ...
    '''

    true_K = config[0]                # true number of component
    N = n_groups = config[1] 
    L = n_tokens = config[2]
    
  
    data_path = f'data/lda{config_index}.pickle'
    if os.path.isfile(data_path):
      print('Loading data ...')
      true_beta, true_topics, word_matrix  = load_pickle(data_path)
    
    else:
      print('Generating synthetic data ...')
      

      # topic-vocab distribution gamma: K x V
      true_topics = generate_bartopics(true_K)
      true_beta = None

      n_group_j = [n_tokens] * n_groups

      # generating the corpus
      xx = [[] for i in range(n_groups)]
      alpha = 1 / true_K

      for jj in range(n_groups):
          theta = np.random.mtrand.dirichlet([alpha] * true_K)
          for ii in range(n_group_j[jj]):

              kk = sample_index(theta)
              xx[jj].append(sample_index(true_topics[kk, :]))

      word_matrix = np.array(xx)
      write_pickle((true_beta, true_topics, word_matrix), data_path)

    K, V = true_topics.shape
    KK = int(np.sqrt(V))


    from utils_model import one_hot_sequential_vectorizer
    word2id = {i: i for i in range(V)}

    corpus = word_matrix.tolist()
    X = one_hot_sequential_vectorizer(corpus, word2id, L)
  
    print(f'Training {X.shape[0]} documents ...')
    print('Number of topics:', K)


    
    H, D = 50, 50
    lr = 1e-3
    B = 50
    
    tau = 0.20
    model_path = f'model/topic_synth_{config_index}.pt'
    
    train_indices = list(range(X.size(0)))
    train_loader = DataLoader(train_indices, batch_size=B, shuffle=True)

    
    phi = TopicBackward(L, V, K, H, D, tau)
    model = TopicModel(V, K, tau, True)
    phi.to(device)
    model.to(device)



    if action == "train":
          
      fopt = torch.optim.Adam(model.parameters(), lr=lr)
      bopt = torch.optim.Adam(phi.parameters(), lr=lr)
      num_epochs = 300
      model.train()
      phi.train()

      start = time.time()      
      for epoch in range(num_epochs + 1):
          loss = train_epoch(model, phi, fopt, bopt, X, train_loader, device)
          print(f"Epoch: {epoch} - Loss: {loss:.5f}")
          torch.save({'model_state_dict': model.state_dict() ,
                      'optimizer_state_dict': fopt.state_dict(),
                      'prev_loss': loss,
                      }, model_path)
      end = time.time()
      print('========== TRAINING TIME ==========', end - start)


    elif action == 'eval': 
      load_model(model, None, None, model_path, device)
      gamma = model.forward_fn.gamma
      gamma = gamma[0, :, :]
      alpha = torch.softmax(model.prior.alpha, dim = -1)
      alpha = alpha[0, 0, :]
      

      print('Computing estimates ....')
      Q = torch.Tensor(true_topics).to(device)
      from utils_model import compute_topic_estimates
      compute_topic_estimates(gamma, Q)
      probs = torch.softmax(model.forward_fn.gamma, dim = -1)
      print(probs.mean(), Q.mean())
      print(alpha.mean())

      from utils_model import visualize_topics, match_topics
      print('Visualizing ...')
      truth = {k : sorted(np.where(true_topics[k, :])[0].tolist()) for k in range(K)}
      estimated = match_topics(gamma, truth)
      topics = list(range(K))
      visualize_topics(topics, V, truth, estimated, f'topic{config_index}')
    
