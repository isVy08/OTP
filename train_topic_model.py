import torch
import time, ot
import torch.nn as nn
import gensim.corpora as corpora
from octis.dataset.dataset import Dataset
from torch.utils.data import DataLoader
from utils_io import load_model
from train_synth_topic import train_epoch

from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence


def evaluate(model, corpus, K, print_output = False):
    topic_matrix = torch.exp(model.forward_fn.beta)
    topic_matrix = topic_matrix.sort(descending=True)[1]
    topics = []
    for k in range(K):
        top = topic_matrix[k, ].detach().tolist()
        top = [id2word[word] for word in top]
        topics.append(top)
        if print_output: print(top[:10])
    model_output = {}
    model_output["topics"] = topics

    nmpi = Coherence(texts=corpus, topk=10, measure='c_npmi')
    diversity = TopicDiversity(topk=10)
    TD = diversity.score(model_output)
    TC = nmpi.score(model_output)
    print(f'Coherence: {TC:.5f} - Diversity: {TD:.5f}')



if __name__ == "__main__":
    
    import sys 
    action = sys.argv[1]
    dataset_name = sys.argv[2] # '20NewsGroup', 'BBC_News', 'DBLP'
    ver = sys.argv[3]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Dataset()
    dataset.fetch_dataset(dataset_name)

    corpus = dataset.get_corpus()
    vocab = dataset.get_vocabulary()
    id2word = corpora.Dictionary(dataset.get_corpus())
    word2id = {v: k for k, v in id2word.items()}
    V = len(vocab)

    from utils_model import vectorizer
    X = vectorizer(corpus, word2id, "one-hot")
    
    print(f'Training {X.shape[0]} documents ...')


    K = 10
    H = 50
    lr = 1e-3
    B = 50
    D = 50
    tau = 2.0
    
    model_path = f'model/topic_{dataset_name}_{ver}.pt'
    
    train_indices = list(range(X.size(0)))
    train_loader = DataLoader(train_indices, batch_size=B, shuffle=True)

    from topic_model import TopicModel, TopicBackward

    phi = TopicBackward(V, K, H, D, tau)
    model = TopicModel(V, K, tau, True)
    phi.to(device)
    model.to(device)
        
    if action == 'train':
      fopt = torch.optim.Adam(model.parameters(), lr=lr)
      bopt = torch.optim.Adam(phi.parameters(), lr=lr)
      num_epochs = 1000
      weight = 0.1
      model.train()
      phi.train()
      
      start = time.time()

      grc = 'kl' # 'l2' if dataset_name == 'DBLP' else 'kl'
      
      for epoch in range(num_epochs):
          
          loss = train_epoch(model, phi, fopt, bopt, X, train_loader, weight, 
                             device, grc, torch.nn.BCELoss())
          print(f"Epoch: {epoch} - Loss: {loss:.5f}")
          torch.save({'model_state_dict': model.state_dict() ,
                      'optimizer_state_dict': fopt.state_dict(),
                      'prev_loss': loss,
                      }, model_path)

      end = time.time()
      print('========== TRAINING TIME ==========', end - start)
      
  
    else:
      
      load_model(model, None, None, model_path, device)
      model.eval()
      evaluate(model, corpus, K, True)
                
        
