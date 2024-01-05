import torch
import numpy as np
import matplotlib.pyplot as plt
import ot
from utils_io import load_celeba, load_mnist
from tqdm import tqdm
from geomloss import SamplesLoss

def one_hot_sequential_vectorizer(corpus, word2id, L):
    corpus = [lt for lt in corpus if len(lt) >= L]
    B = len(corpus)
    V = len(word2id)
    inp = torch.zeros((B, L, V)) 
    for i, doc in enumerate(corpus):
        for j in range(L):
            # find id of word j
            if j < len(doc):
                word = doc[j]
                word_id = word2id[word]
                inp[i, j, word_id] = 1 
    return inp


def vectorizer(corpus, word2id, _type = "one-hot", vocab = None):
    if _type == "tf-idf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        assert vocab is not None, 'Vocabulary is requried!'
        vectorizer = TfidfVectorizer(vocabulary=vocab)
        corpus = [' '.join(lt) for lt in corpus]
        inp = vectorizer.fit_transform(corpus)
        inp = torch.from_numpy(inp)
        return inp

    B = len(corpus)
    V = len(word2id)
    inp = torch.zeros((B, V)) 
    for i, doc in enumerate(corpus):
        for word in doc:
            # find id of word j
            word_id = word2id[word]
            if _type == 'one-hot':
                inp[i, word_id] = 1
            elif _type == 'count':
                inp[i, word_id] += 1
            else:
                raise ValueError
    return inp
    

def neural_vectorizer(corpus, word2id, L):
    B = len(corpus)
    L = max([len(doc) for doc in corpus]) if L is None else L
    inp = torch.zeros((B, L)) 
    for i, doc in enumerate(corpus):
        if len(doc) > L: doc = doc[:L]
        for j, word in enumerate(doc):
            # find id of word j
            word_id = word2id[word]
            inp[i, j] = word_id 
    return inp.long()

def generate_embedding_matrix(id2word):
    import gensim.downloader
    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
    
    embedding_matrix = []
    V = len(id2word)
    for i in range(V):
        w = id2word[i]
        try:
            vec = glove_vectors[w]
        except KeyError:
            vec = np.zeros((50, ))
        embedding_matrix.append(vec)
    embedding_matrix = np.stack(embedding_matrix, axis = 0)
    embedding_ts = torch.from_numpy(embedding_matrix)
    embedding_ts = embedding_ts.unsqueeze(0).float()
    return embedding_ts


def pick_top_k(probs, k = 10):
        _, topk = torch.topk(probs, k = k, dim = -1)
        masked = torch.zeros_like(probs, device = probs.device)
        for b in range(probs.size(0)):
            idx = topk[b, :]
            for i in idx: 
                masked[b,i] = 1.0
        probs = torch.mul(probs, masked)
        return probs

def visualize_topics(topics, V, truth, estimated, name):
    K = len(topics)
    fig, axs = plt.subplots(2, K)  
    KK = int(np.sqrt(V))
    for r in range(2): 
        for loc, k in enumerate(topics):
        # choose which row 
            if r == 0: 
                words = truth[k]
            else: 
                words = estimated[k]
            m = np.zeros((V))
            for w in words:
                m[w] = 0.20
            img = m.reshape((KK, KK))
            if r == 0:
                axs[r, loc].imshow(img, cmap='cividis')
            else:
                axs[r, loc].imshow(img, cmap='nipy_spectral')
            axs[r, loc].set_axis_off()
    fig.tight_layout()
    plt.savefig(f'image/{name}.png', bbox_inches='tight')

def match_topics(beta, truth):
    K = len(truth)
    estimated = {k: None for k in range(K)}
    remains = []
    for k in range(K):
        top = len(truth[k])
        if isinstance(beta, torch.Tensor):
            _, topics = torch.topk(beta[k, :], k = top)
        else:
            topics = np.argpartition(beta[k,], -top)[-top:]
        topics = sorted(topics.tolist())
        minv, mink = 0, None
        for key, value in truth.items():
            overlap = len(set(value) & set(topics))
            if overlap > minv:
                minv = overlap
                mink = key
            
        if estimated[mink] is None:
            estimated[mink] = topics
        else: 
            remains.append(topics) 
    
    for k in estimated: 
        if estimated[k] is None:
            top = remains.pop()
            estimated[k] = top
                
    return estimated

def load_topic_config(config_index):
      # no. topics, num samples, sequence length, vocab size = K / 2 * K / 2
  config = { 
    0: [10, 1000, 100],
    1: [20, 5000, 200],
    2: [30, 10000, 300],
  }
  return config[config_index]

def compute_topic_estimates(estimated, true, method = None):
    '''
    true, estimate: torch.Tensor, K x V
    '''
    kl = torch.nn.KLDivLoss(reduction="batchmean")
    ce = torch.nn.CrossEntropyLoss()

    L1 = torch.nn.L1Loss()
    L2 = torch.nn.MSELoss()

    if method in ('em', 'svi'):
        prob_estimated = estimated
    else:
        prob_estimated = torch.softmax(estimated, dim = 1)
    log_estimated = torch.log_softmax(estimated, dim = 1)

    log_true = torch.log_softmax(true, dim=1)

    B = estimated.shape[0]
    M = torch.zeros((B, B))
        
    for i in range(B):
        for j in range(B):
            M[i, j] = 0.5 * (kl(log_estimated[i, :], true[j, :]) + kl(log_true[i, :], prob_estimated[j, :]))
    
    unif = torch.ones((B,)) / B
    ws = ot.emd2(unif, unif, M)
    
    l1 = L1(prob_estimated, true)
    l2 = L2(prob_estimated, true)
    
    kldiv = kl(log_estimated, true)
    denom = 0.5 * (true + prob_estimated)
    klpm = kl(log_estimated, denom)

    log_true = torch.log_softmax(true, dim=-1)
    klqm = kl(log_true, denom)
    js = 0.5 * (klpm + klqm)
    

    # hellinger distance
    _SQRT2 = np.sqrt(2)
    hl = torch.sqrt(torch.sum((torch.sqrt(prob_estimated) - torch.sqrt(true)) ** 2)) / _SQRT2
    print(l1.item())
    print(l2.item())
    print(js.item())
    print(hl.item())
    print(kldiv.item())    
    print(ws.item())

def free_params(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def kl_matrix(a, b):
    assert len(a.shape) == 2, 'This works on 2D tensor only!'
    cost_matrix_h1 = ((b * b.log()).sum(dim = 1) - torch.einsum('ik, jk -> ij', a.log(), b)) / a.shape[1]
    cost_matrix_h2 = ((a * a.log()).sum(dim = 1) - torch.einsum('ik, jk -> ij', a.log(), a)) / a.shape[1]
    cost_matrix_h = (cost_matrix_h1 + cost_matrix_h2)/2
    return cost_matrix_h   