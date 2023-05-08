from tqdm import tqdm
def train_epoch(model, optimizer, scheduler, loss_fn, X, loader, device):
    
    L = 0
   
    for idx in tqdm(loader):
      x = X[idx, ].to(device)    
      output = model(x)
    
      loss = loss_fn(x, output)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if scheduler:
        scheduler.step()
    
      L += loss.item()
  
    return L / len(loader)

def val_epoch(model, loss_fn, X, loader, device):
    
    L = 0
    
    for idx in tqdm(loader):
      x = X[idx, :].to(device) 
      output = model(x)
      loss = loss_fn(x, output)
      L += loss.item()
  
    return L / len(loader)