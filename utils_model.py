import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def compute_topic_estimates(true, estimated):
    '''
    true, estimate: torch.Tensor, K x V
    '''
    kl = torch.nn.KLDivLoss(reduction="batchmean")
    L1 = torch.nn.L1Loss()
    L2 = torch.nn.MSELoss()

    l1 = L1(estimated, true)
    l2 = L2(estimated, true)
    kldiv = kl(torch.log_softmax(estimated, dim = -1), true)
    M = 0.5 * (true + estimated)
    klpm = kl(torch.log_softmax(estimated, dim = -1), M)
    klqm = kl(torch.log_softmax(true, dim = -1), M)
    js = 0.5 * (klpm + klqm)
    # hellinger distance
    _SQRT2 = np.sqrt(2)
    hl = torch.sqrt(torch.sum((torch.sqrt(estimated) - torch.sqrt(true)) ** 2)) / _SQRT2
    print("L1:", l1.item())
    print("L2:", l2.item())
    print("KL:", kldiv.item())
    print("JS:", js.item())
    print("HL:", hl.item())

