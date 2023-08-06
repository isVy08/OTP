from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
import time
import gensim.corpora as corpora
from utils_io import write_pickle, load_pickle
from utils_model import load_topic_config

import sys 
dataset_name = sys.argv[1]
from octis.models.NeuralLDA import NeuralLDA
from octis.models.ProdLDA import ProdLDA


if dataset_name != 'lda':
    arch = sys.argv[2]
    dataset = Dataset()
    # dataset.fetch_dataset("20NewsGroup")
    # dataset.fetch_dataset("BBC_News")
    # dataset.fetch_dataset("DBLP")
    
    dataset.fetch_dataset(dataset_name)
    K = 10
    corpus = dataset.get_corpus()
    vocab = dataset.get_vocabulary()
    id2word = corpora.Dictionary(dataset.get_corpus())
    word2id = {v: k for k, v in id2word.items()}

    print('Training begins ...')
    if arch == 'neural':
        model = NeuralLDA(num_topics=K, num_epochs=1000, use_partitions=True, batch_size=50, lr=1e-3)
    else: 
        model = ProdLDA(num_topics=K, num_epochs=1000, use_partitions=True, batch_size=50, lr=1e-3)
   
    model_output = model.train_model(dataset)
    topic_matrix = model_output['topic-word-matrix']
    
    
    file = open('result.txt', 'a+')
    topics = model.model.get_topics(10)
    nmpi = Coherence(texts=corpus, topk=10, measure='c_npmi')
    diversity = TopicDiversity(topk=10)
    
    output = {'topics': topics}
    TD = diversity.score(output)
    TC = nmpi.score(output)
    file.write(f'{dataset_name}-{arch} / Coherence: {TC:.5f} - Diversity: {TD:.5f}\n')
    file.close()


elif dataset_name == 'lda': 
    
    config_index = int(sys.argv[2])
    arch = sys.argv[3]
    config = load_topic_config(config_index)

    K = config[0]    
    N = n_groups = config[1] 
    L = n_tokens = config[2]
    
    data_path = f'data/lda{config_index}.pickle'
    true_beta, true_topics, word_matrix  = load_pickle(data_path)
    K, V = true_topics.shape
    word2id = {i: i for i in range(V)}

    corpus = word_matrix.tolist()
    corpus = [[str(item) for item in doc] for doc in corpus]
    vocab = [str(i) for i in range(V)]

    dataset = Dataset(corpus = corpus, vocabulary = vocab)

    if arch == 'neural':
        model = NeuralLDA(num_topics=K, num_epochs=300, use_partitions=False, batch_size=50, lr=1e-3)
    else:
        model = ProdLDA(num_topics=K, num_epochs=300, use_partitions=False, batch_size=50, lr=1e-3)
    print('Training begins ...')

    model_output = model.train_model(dataset)
    topic_matrix = model_output['topic-word-matrix']

    import torch, scipy
    from utils_model import compute_topic_estimates
    
    if arch == 'prod':
        topic_matrix = scipy.special.softmax(topic_matrix, axis = -1)     
    P = torch.Tensor(topic_matrix)
    Q = torch.Tensor(true_topics)    
    compute_topic_estimates(P, Q)

    print('Visualizing ...')
    from utils_model import visualize_topics, match_topics
    import numpy as np
    truth = {k : sorted(np.where(true_topics[k, :])[0].tolist()) for k in range(K)}
    estimated = match_topics(topic_matrix, truth)
    k = min(K, 15)
    topics = list(range(k))
    visualize_topics(topics, V, truth, estimated, f'{arch}{config_index}')
    
