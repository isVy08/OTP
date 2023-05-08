import torch, os
import torch.nn as nn
import time

from tqdm import tqdm
import gensim.corpora as corpora
from octis.dataset.dataset import Dataset
from torch.utils.data import DataLoader
from topic_model import TopicModel
from utils_io import load_model


from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
    


def train_epoch(model, optimizer, scheduler,
                X, train_loader, weight, device, embedding_ts = None):
    
    model.train()
    bce = nn.BCELoss()
    l2 = nn.MSELoss()
    kl = nn.KLDivLoss(reduction="batchmean")

    Loss = 0
    
    for idx in tqdm(train_loader):
      x = X[idx, :].to(device)
      xhat, z, zhat = model(x)   

      if embedding_ts is not None:
        e = torch.repeat_interleave(embedding_ts, x.size(0), dim = 0)
        y = torch.matmul(x, e)
        yhat = torch.matmul(xhat, e)
        loss = l2(yhat, y)
      else: 

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
    
    device_ids = [0,1,2,3]
    import sys 
    dataset_name = sys.argv[2] # 20NewsGroup, BBC_News, DBLP
    action = sys.argv[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', device_ids[0])

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
    tau = 2.5
    model_path = f'model/topic_{dataset_name}.pt'
    
    train_indices = list(range(X.size(0)))
    train_loader = DataLoader(train_indices, batch_size=B, shuffle=True)

    model = TopicModel(V, K, H, D, tau, True)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None 
    num_epochs = 1000
    weight = 0.1

    if os.path.isfile(model_path):
        prev_loss = load_model(model, optimizer, scheduler, model_path, device)
    else:
        prev_loss = 1.0

    if action == 'train':
      model = nn.DataParallel(model, device_ids)
      start = time.time()
      
      for epoch in range(num_epochs):
          loss = train_epoch(model, optimizer, scheduler, X, train_loader, weight, device)
          print(f"Epoch: {epoch} - Loss: {loss:.5f}")
          if loss < prev_loss:
            prev_loss = loss
            print('Saving model ...')
            torch.save({'model_state_dict': model.module.state_dict() ,
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'scheduler_state_dict': scheduler.state_dict(),
                        'prev_loss': prev_loss,
                        }, model_path)

      end = time.time()
      print('========== TRAINING TIME ==========', end - start)
      
  
    else:
      model.eval()
      evaluate(model, corpus, K, True)
                
        
        

    
      

          


