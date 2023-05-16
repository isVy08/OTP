import torch, os
import torch.nn as nn
import sys, time
from tqdm import tqdm
from torch.utils.data import DataLoader
from topic_model import TopicModel
from utils_model import load_topic_config
from utils_io import load_model, write_pickle, load_pickle
import numpy as np



def train_epoch(model, optimizer, scheduler,
                X, train_loader, weight, device):
    
    model.train()
    bce = nn.BCELoss()
    kl = nn.KLDivLoss(reduction="batchmean")

    Loss = 0
    
    for idx in tqdm(train_loader):
      x = X[idx, :].to(device)
      xhat, z, zhat = model(x)   
      loss = bce(xhat, x)

      loss += weight * kl(nn.functional.log_softmax(zhat, dim=-1), z)
      model.module.backward_fn.lstm.flatten_parameters()
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if scheduler:
        scheduler.step()
      
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
    
    device_ids = [0,1,2,3]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', device_ids[0])
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


    if config_index == 0:
      H, D = 50, 50
    elif config_index == 1:
      H, D = 150, 50
    else:
      H, D = 150, 50 
    lr = 1e-3
    B = 50
    
    tau = 1.0
    model_path = f'model/topic_synth{config_index}.pt'
    
    train_indices = list(range(X.size(0)))
    train_loader = DataLoader(train_indices, batch_size=B, shuffle=True)

    from synthetic_model import TopicModel
    model = TopicModel(L, V, K, H, D, tau, True)
    model.to(device)
    if action == "train":
      optimizer = torch.optim.Adam(model.parameters(), lr=lr)
      scheduler = None
      num_epochs = 300
      weight = 1e-4


      if os.path.isfile(model_path):
          prev_loss = load_model(model, optimizer, scheduler, model_path, device)
      else:
          prev_loss = 10.0

      model = nn.DataParallel(model, device_ids)
      start = time.time()
      for epoch in range(num_epochs + 1):
          loss = train_epoch(model, optimizer, scheduler, X, train_loader, weight, device)
          print(f"Epoch: {epoch} - Loss: {loss:.5f}")
          if loss < prev_loss:
            prev_loss = loss
            print('Saving model ...')
            torch.save({'model_state_dict': model.module.state_dict() ,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'prev_loss': prev_loss,
                        }, model_path)
            

      end = time.time()
      print('========== TRAINING TIME ==========', end - start)


    elif action == 'eval': 
      load_model(model, None, None, model_path, device)
      gamma = torch.softmax(model.forward_fn.gamma, dim = -1)
      gamma = gamma[0, :, :] # .cpu().detach().numpy()
      alpha = torch.softmax(model.prior.alpha, dim = -1)
      alpha = alpha[0, 0, :]
      

      print('Computing estimates ....')
      Q = torch.Tensor(true_topics).to(device)
      from utils_model import compute_topic_estimates
      compute_topic_estimates(gamma, Q)
      print(gamma.mean(), Q.mean())
      print(alpha.mean())

      from utils_model import visualize_topics, match_topics
      print('Visualizing ...')
      truth = {k : sorted(np.where(true_topics[k, :])[0].tolist()) for k in range(K)}
      estimated = match_topics(gamma, truth)
      topics = list(range(K))
      visualize_topics(topics, V, truth, estimated, f'topic{config_index}')
    