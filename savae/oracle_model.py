import os
import torch
import torch.nn as nn
import torch.nn.init as init

class Sample_Categorical(nn.Module):
    
  def __init__(self, tau):
    super(Sample_Categorical, self).__init__()
    self.tau = tau
  
  def forward(self, logits):
    # logits : [B, K, 1], K categories
    logits = logits.squeeze(-1)
    sample = nn.functional.gumbel_softmax(logits, hard=False, tau=self.tau, dim=-1)
    return sample
  
class MLP(nn.Module):
    def __init__(self, D, H, vocab_size, embed_size, mode):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(D, H), 
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(), 
            nn.Linear(H, vocab_size), 
            nn.LogSoftmax(dim = -1) 
        )
        self.vocab_size = vocab_size
        self.embed_layer = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.mode = mode
        self.sampler = Sample_Categorical(0.2)
    def forward(self, h, c, z):
        '''
        input: h_(t), dim of hidden_size + 2
        returns: x_(t+1) length of vocab_size
        '''
        if z is None:
            z = torch.randn((h.size(0), 2), device = h.device)
        z = torch.cat((h, c, z), dim = 1)
        logits = self.layer(z)
        if self.mode == 'train':
            x = self.sampler(logits) # soft x
            e = x @ self.embed_layer.weight 
        else:
            m = torch.distributions.categorical.Categorical(logits=logits)
            x = m.sample()
            e = self.embed_layer(x)
        return x, e

class Oracle(nn.Module):
    def __init__(self, hidden_size, vocab_size, embed_size, seq_length, mode):
        super(Oracle, self).__init__()
        self.lstm = nn.LSTMCell(embed_size, hidden_size) 
        self.mlp = MLP(2 * hidden_size + 2, hidden_size, vocab_size, embed_size, mode)
        self._initialize()
        self.seq_length = seq_length

    def forward(self, hx, cx, z = None):
        X = []
        
        for t in range(self.seq_length):
            if t == 0:
                x, e = self.mlp(hx, cx, z)
            else: 
                hx, cx = self.lstm(e, (hx, cx))
                x, e = self.mlp(hx, cx, z)
            X.append(x)

        input = torch.stack(X, dim=1)
        return input
        

    def _initialize(self):
        for param in self.lstm.parameters():
            init.uniform_(param.data, a = -1.0, b = 1.0)

        for param in self.mlp.layer[0].parameters(): 
            init.uniform_(param.data, a = -5.0, b = 5.0)       
        
        for layer in self.mlp.layer[1:]:
            if isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    init.uniform_(param.data, a = -1.0, b = 1.0)

class DataGenerator:
    def __init__(self, vocab_size = 1000, seq_length = 5, hidden_size = 100, 
                 embed_size = 100, sample_size = 10000, device = 'cuda',
                 model_path = 'oracle.pt', data_path = 'input.pt'):
        '''
        Semi-amortized VAE: https://arxiv.org/pdf/1802.02550.pdf
        '''
        self.oracle = Oracle(hidden_size, vocab_size, embed_size, seq_length, mode = 'gen')

        if os.path.isfile(model_path):
            print('Loading params for Oracle ...')
            ckpt = torch.load(model_path, map_location=device)
            self.oracle.load_state_dict(ckpt['model_state_dict'])
         
        if os.path.isfile(data_path):
            self.input = torch.load(data_path)
        else: 
            print('Generating synthetic data ...')

            hx = torch.zeros((sample_size, hidden_size)) # batch, hidden size
            cx = torch.zeros((sample_size, hidden_size))
            self.input = self.oracle(hx, cx)
            torch.save(self.input, data_path)
            torch.save({'model_state_dict': self.oracle.state_dict()}, model_path)
        
        train_size = int(0.9 * sample_size)
        self.train_sents = self.input[:train_size, ]
        self.val_sents = self.input[train_size:, ]
        self.device = device


DataGenerator()



# # Generative process 
# vocab_size = 1000
# seq_length = 5
# hidden_size = 100
# embed_size = 100
# batch_size = 40


