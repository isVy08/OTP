import time, sys
import numpy as np
import gensim.corpora as corpora
from octis.dataset.dataset import Dataset
from utils_model import load_topic_config
from utils_io import load_pickle
from sklearn.decomposition import LatentDirichletAllocation
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

def evaluate(model, corpus, K, method, print_output = False):
    topics = []
    if method in ('batch_em', 'online_em'):
        topic_matrix = model.components_
        for k in range(K):
            top = topic_matrix[k, ].argsort()[::-1].tolist()
            top = [id2word[word] for word in top]
            topics.append(top)
            if print_output: print(top[:10])
        model_output = {}
        model_output["topics"] = topics
    elif method == 'prod_lda': 
        topics = model.model.get_topics(10)
        model_output = {'topics': topics}
    else:
        raise ValueError

    nmpi = Coherence(texts=corpus, topk=10, measure='c_npmi')
    diversity = TopicDiversity(topk=10)
    TD = diversity.score(model_output)
    TC = nmpi.score(model_output)
    print(f'Coherence: {TC:.5f} - Diversity: {TD:.5f}')


dataset_name = sys.argv[1]
method = sys.argv[2] # ("batch_em", "online_em", "prod_lda")

if dataset_name in ("20NewsGroup","BBC_News","DBLP"):
    dataset = Dataset()
    dataset.fetch_dataset(dataset_name)

    corpus = dataset.get_corpus()
    vocab = dataset.get_vocabulary()
    id2word = corpora.Dictionary(dataset.get_corpus())
    word2id = {v: k for k, v in id2word.items()}
    from utils_model import vectorizer
    X = vectorizer(corpus, word2id, "one-hot")
    X = X.numpy()
    K = 10

else: 
    
    config_index = int(sys.argv[3])
    config = load_topic_config(config_index)

    K = config[0]    
    N = n_groups = config[1] 
    L = n_tokens = config[2]
    
    data_path = f'data/lda{config_index}.pickle'
    true_beta, true_topics, word_matrix  = load_pickle(data_path)
    K, V = true_topics.shape
    word2id = {i: i for i in range(V)}

    corpus = word_matrix.tolist()

    from utils_model import vectorizer
    X = vectorizer(corpus, word2id, "count")
    X = X.numpy()

print(f"Running {method} on data size {X.shape} ...")

if method == 'batch_em':

    model = LatentDirichletAllocation(
        n_components=K,
        max_iter=1000,
        learning_method="batch",
        random_state=7,
        batch_size=50
    )

    start = time.time()
    model.fit(X)

elif method == 'online_em':
    model = LatentDirichletAllocation(
        n_components=K,
        max_iter=1000,
        learning_method="online",
        random_state=7,
        batch_size=50, 
    )

    start = time.time()
    model.fit(X)
    

elif method == 'prod_lda':
    from octis.models.ProdLDA import ProdLDA
    if dataset_name in ("20NewsGroup","BBC_News","DBLP"):
        model = ProdLDA(num_topics=K, num_epochs=1000, use_partitions=True, batch_size=50, lr=1e-3)
        

    if dataset_name == 'lda':
        corpus = [[str(item) for item in doc] for doc in corpus]
        vocab = [str(i) for i in range(V)]

        dataset = Dataset(corpus = corpus, vocabulary = vocab)
        model = ProdLDA(num_topics=K, num_epochs=300, use_partitions=False, 
                        batch_size=50, lr=1e-3, learn_priors=True, prior_mean=1/K)
    
    start = time.time()
    model_output = model.train_model(dataset)
    
    


end = time.time()
print('========== TRAINING TIME ==========', end - start)


if dataset_name in ("20NewsGroup","BBC_News","DBLP"):
    evaluate(model, corpus, K, method, print_output = False)

else: 

    import torch
    from utils_model import compute_topic_estimates, visualize_topics, match_topics
    
    if method in ('batch_em', 'online_em'):

        topic_matrix = model.exp_dirichlet_component_
    
    else: 
        alpha = model.model.model.prior_mean.data
        alpha = torch.softmax(alpha, dim = -1)
        topic_matrix = model.model.model.beta.detach().cpu().numpy()
    
    
    P = torch.Tensor(topic_matrix)
    Q = torch.Tensor(true_topics)    
    compute_topic_estimates(P, Q, method)

    truth = {k : sorted(np.where(true_topics[k, :])[0].tolist()) for k in range(K)}
    estimated = match_topics(topic_matrix, truth)
    
    topics = list(range(K))
    visualize_topics(topics, V, truth, estimated, f'{method}{config_index}')

