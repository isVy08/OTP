import torch, os
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt
from music_model import MusicModel, Criterion
from utils_io import load_model, load_music_data
from trainer import train_epoch, val_epoch

torch.manual_seed(8)
import sys 
action = sys.argv[1]
device_ids = [2,3,1,0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', device_ids[0])



trainX, valX = load_music_data("train")
trainX = torch.from_numpy(trainX).float().to(device)
valX = torch.from_numpy(valX).float().to(device)



# Config
K, tau = 24, 0.1
H = 128
L, D = trainX.shape[1], trainX.shape[2]
batch_size = 50
lr = 1e-3



if os.path.isfile('data/tm.npy'):
    TM = np.load('data/tm.npy')
    TM = torch.from_numpy(TM)
else:
    print('Initialize transition matrix ...')
    alpha = 0.9 * torch.eye(K) + 0.1
    sampler = torch.distributions.dirichlet.Dirichlet(alpha)
    TM = sampler.sample()
    TM_arr = TM.cpu().numpy()
    np.save('data/tm.npy', TM_arr)

TM = TM.to(device)
model_path = f'model/music.pt'

model = MusicModel(K, H, D, tau, TM)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = None 
num_epochs = 4000
weight = 1e-4
loss_fn = Criterion(weight)

if os.path.isfile(model_path):
    
    prev_loss = load_model(model, optimizer, scheduler, model_path, device)
    print('Previous loss:', prev_loss)
else:
    prev_loss = 10.



train_indices = range(trainX.size(0))
val_indices = range(valX.size(0))
train_loader = torch.utils.data.DataLoader(train_indices , batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_indices, batch_size=batch_size, shuffle=False)


model.train()

for epoch in range(num_epochs):
train_loss = train_epoch(model, optimizer, scheduler, loss_fn, trainX, train_loader, device)
val_loss = val_epoch(model, loss_fn, valX, val_loader, device)

if math.isnan(train_loss):
    print('NaN values!')
    break

print(f"Epoch: {epoch} - Train Loss: {train_loss:.5f} / Val Loss: {val_loss:.5f}")
if val_loss < prev_loss:
    prev_loss = val_loss
    print('Saving model ...')
    torch.save({'model_state_dict': model.state_dict() ,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'prev_loss': prev_loss,
                }, model_path)
