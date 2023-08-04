#!/usr/bin/env python3

import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn.functional as F
import numpy as np

import time
from optim_n2n import OptimN2N
from models_text import RNNVAE
import utils

from oracle_model import DataGenerator



# Model options
import sys
latent_dim = 2 
enc_word_dim = enc_h_dim = dec_word_dim = dec_h_dim = 100 
enc_num_layers = dec_num_layers = 1
dec_dropout = None
model_name = sys.argv[1] #'savae'
train_n2n = 1 
train_kl = 1
vocab_size = 1000
length = 5
global_batch_size = 50

# Optimization options
checkpoint_path = f'{model_name}.pt'
warmup = 10
num_epochs = 100
min_epochs = 15
start_epoch = 0
svi_steps = 20

svi_lr1 = 1. 
svi_lr2 = 1.
eps = 1e-5

max_grad_norm = 5
svi_max_grad_norm = 5
seed = 8
print_every = 100 
momentum = 0.5

def main():
  np.random.seed(seed)
  torch.manual_seed(seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dg = DataGenerator(model_path = 'oracle.pt', data_path='input.pt', device = device)


  train_indices = range(dg.train_sents.size(0))
  val_indices = range(dg.val_sents.size(0))
  train_loader = torch.utils.data.DataLoader(train_indices , batch_size=global_batch_size, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_indices, batch_size=global_batch_size, shuffle=False)
  model = RNNVAE()
  if not os.path.isfile(checkpoint_path):
    for param in model.parameters():    
      param.data.uniform_(-0.1, 0.1)      
  else:
    print('loading model from ' + checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']

  model.to(device)
      
  lr = 1. # LEARNING RATE HERE
  decay = 0.5
  optimizer = torch.optim.SGD(model.parameters(), lr=lr)

  if warmup == 0:
    beta = 1.
  else:
    beta = 0.1
    
  criterion = nn.NLLLoss()
  model.train()

  def variational_loss(input, sents, model, z = None):
    mean, logvar = input
    z_samples = model._reparameterize(mean, logvar, z)
    # preds = model._dec_forward(sents, z_samples)
    preds = model._decode(z_samples)
    nll = sum([criterion(preds[:, l], sents[:, l]) for l in range(preds.size(1))])
    kl = utils.kl_loss_diag(mean, logvar)
    return nll + beta*kl

  update_params = list(model.dec.parameters())
  meta_optimizer = OptimN2N(variational_loss, model, update_params, eps = eps, 
                            lr = [svi_lr1, svi_lr2],
                            iters = svi_steps, momentum = momentum,
                            acc_param_grads= train_n2n == 1,  
                            max_grad_norm = svi_max_grad_norm)

  t = 0
  best_val_nll = 1e5
  best_epoch = 0
  val_stats = []
  epoch = 0
  while epoch < num_epochs:
    start_time = time.time()
    epoch += 1  
    print('Starting epoch %d' % epoch)
    train_nll_vae = 0.
    train_nll_autoreg = 0.
    train_kl_vae = 0.
    train_nll_svi = 0.
    train_kl_svi = 0.
    train_kl_init_final = 0.
    num_sents = 0
    num_words = 0
    b = 0
    
    for idx in train_loader:
      if warmup > 0:
        beta = min(1, beta + 1./(warmup*len(train_loader)))
      
      sents = dg.train_sents[idx].to(device)
      batch_size = len(idx)
      
      b += 1
      
      optimizer.zero_grad()
      if model_name == 'autoreg':
        preds = model._dec_forward(sents, None, True)
        nll_autoreg = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
        train_nll_autoreg += nll_autoreg.item()*batch_size
        nll_autoreg.backward()
      elif model_name == 'svi':        
        mean_svi = Variable(0.1*torch.zeros(batch_size, latent_dim).cuda(), requires_grad = True)
        logvar_svi = Variable(0.1*torch.zeros(batch_size, latent_dim).cuda(), requires_grad = True)
        var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], sents,
                                                  b % print_every == 0)
        mean_svi_final, logvar_svi_final = var_params_svi
        z_samples = model._reparameterize(mean_svi_final.detach(), logvar_svi_final.detach())
        # preds = model._dec_forward(sents, z_samples)
        preds = model._decode(z_samples)
        nll_svi = sum([criterion(preds[:, l], sents[:, l]) for l in range(length)])
        train_nll_svi += nll_svi.item()*batch_size
        kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
        train_kl_svi += kl_svi.item()*batch_size      
        var_loss = nll_svi + beta*kl_svi          
        var_loss.backward(retain_graph = True)
      else:
        mean, logvar = model._enc_forward(sents)
        z_samples = model._reparameterize(mean, logvar)

        # preds = model._dec_forward(sents, z_samples)
        preds = model._decode(z_samples)
        nll_vae = sum([criterion(preds[:, l], sents[:, l]) for l in range(length)])

        train_nll_vae += nll_vae.item()*batch_size
        kl_vae = utils.kl_loss_diag(mean, logvar)
        train_kl_vae += kl_vae.item()*batch_size        



        if model_name == 'vae':
          vae_loss = nll_vae + beta*kl_vae          
          vae_loss.backward(retain_graph = True)
        
        if model_name == 'savae':
          var_params = torch.cat([mean, logvar], 1)        
          mean_svi = Variable(mean.data, requires_grad = True)
          logvar_svi = Variable(logvar.data, requires_grad = True)
          var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], sents, b % print_every == 0)
          mean_svi_final, logvar_svi_final = var_params_svi
          z_samples = model._reparameterize(mean_svi_final, logvar_svi_final)
          # preds = model._dec_forward(sents, z_samples)
          preds = model._decode(z_samples)
          nll_svi = sum([criterion(preds[:, l], sents[:, l]) for l in range(length)])
          train_nll_svi += nll_svi.item()*batch_size
          kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
          train_kl_svi += kl_svi.item()*batch_size      
          var_loss = nll_svi + beta*kl_svi          
          var_loss.backward(retain_graph = True)
          if train_n2n == 0:
            if train_kl == 1:
              mean_final = mean_svi_final.detach()
              logvar_final = logvar_svi_final.detach()            
              kl_init_final = utils.kl_loss(mean, logvar, mean_final, logvar_final)
              train_kl_init_final += kl_init_final.item()*batch_size
              kl_init_final.backward(retain_graph = True)              
            else:
              vae_loss = nll_vae + beta*kl_vae
              var_param_grads = torch.autograd.grad(vae_loss, [mean, logvar], retain_graph=True)
              var_param_grads = torch.cat(var_param_grads, 1)
              var_params.backward(var_param_grads, retain_graph=True)              
          else:
            var_param_grads = meta_optimizer.backward([mean_svi_final.grad, logvar_svi_final.grad],
                                                      b % print_every == 0)
            var_param_grads = torch.cat(var_param_grads, 1)
            var_params.backward(var_param_grads)
      if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)        
      optimizer.step()
      num_sents += batch_size
      num_words += batch_size * length
      
      if b % print_every == 0:
        param_norm = sum([p.norm()**2 for p in model.parameters()]).item()**0.5
        print('Iters: %d, Epoch: %d, Batch: %d/%d, LR: %.4f, TrainARPPL: %.2f, TrainVAE_PPL: %.2f, TrainVAE_KL: %.4f, TrainVAE_PPLBnd: %.2f, TrainSVI_PPL: %.2f, TrainSVI_KL: %.4f, TrainSVI_PPLBnd: %.2f, KLInitFinal: %.2f, |Param|: %.4f, BestValPerf: %.2f, BestEpoch: %d, Beta: %.4f, Throughput: %.2f examples/sec' % 
              (t, epoch, b+1, len(train_loader), lr, np.exp(train_nll_autoreg / num_words), 
               np.exp(train_nll_vae/num_words), train_kl_vae / num_sents,
               np.exp((train_nll_vae + train_kl_vae)/num_words),
               np.exp(train_nll_svi/num_words), train_kl_svi/ num_sents,
               np.exp((train_nll_svi + train_kl_svi)/num_words), train_kl_init_final / num_sents,
               param_norm, best_val_nll, best_epoch, beta,
               num_sents / (time.time() - start_time)))
          
    print('--------------------------------')
    print('Checking validation perf...')    
    val_nll = eval(val_loader, model_name, model, meta_optimizer, dg)
    val_stats.append(val_nll)
    if val_nll < best_val_nll:
      best_val_nll = val_nll
      best_epoch = epoch
      checkpoint = {
        'model': model,
        'val_stats': val_stats
      }
      print('Saving checkpoint to %s' % checkpoint_path)      
      torch.save(checkpoint, checkpoint_path)
    else:
      
      if epoch >= min_epochs:
        decay = 1
    if decay == 1:
      lr = lr * 0.5      
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
      if lr < 0.03:
        break
      
def eval(val_loader, model_name, model, meta_optimizer, dg):
    
#   model.eval()
  criterion = nn.NLLLoss()
  num_sents = 0
  num_words = 0
  total_nll_autoreg = 0.
  total_nll_vae = 0.
  total_kl_vae = 0.
  total_nll_svi = 0.
  total_kl_svi = 0.
  for idx in val_loader:
    sents = dg.val_sents[idx].to(dg.device)
    batch_size = len(idx)
    num_words += batch_size*length
    num_sents += batch_size
    if model_name == 'autoreg':
      preds = model._dec_forward(sents, None, True)
      nll_autoreg = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length)])
      total_nll_autoreg += nll_autoreg.item()*batch_size
    elif model_name == 'svi':
      mean_svi = Variable(0.1*torch.randn(batch_size, latent_dim).cuda(), requires_grad = True)
      logvar_svi = Variable(0.1*torch.randn(batch_size, latent_dim).cuda(), requires_grad = True)
      var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], sents)
      mean_svi_final, logvar_svi_final = var_params_svi
      z_samples = model._reparameterize(mean_svi_final.detach(), logvar_svi_final.detach())
      # preds = model._dec_forward(sents, z_samples)
      preds = model._decode(z_samples)
      nll_svi = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length-1)])
      total_nll_svi += nll_svi.item()*batch_size
      kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
      total_kl_svi += kl_svi.item()*batch_size
      mean, logvar = mean_svi_final, logvar_svi_final
    else:
      mean, logvar = model._enc_forward(sents)
      z_samples = model._reparameterize(mean, logvar)
      # preds = model._dec_forward(sents, z_samples)
      preds = model._decode(z_samples)
      nll_vae = sum([criterion(preds[:, l], sents[:, l+1]) for l in range(length-1)])
      total_nll_vae += nll_vae.item()*batch_size
      kl_vae = utils.kl_loss_diag(mean, logvar)
      total_kl_vae += kl_vae.item()*batch_size        
      if model_name == 'savae':
        mean_svi = Variable(mean.data, requires_grad = True)
        logvar_svi = Variable(logvar.data, requires_grad = True)
        var_params_svi = meta_optimizer.forward([mean_svi, logvar_svi], sents)
        mean_svi_final, logvar_svi_final = var_params_svi
        z_samples = model._reparameterize(mean_svi_final, logvar_svi_final)
        # preds = model._dec_forward(sents, z_samples)
        preds = model._decode(z_samples)
        nll_svi = sum([criterion(preds[:, l], sents[:, l]) for l in range(length)])
        total_nll_svi += nll_svi.item()*batch_size
        kl_svi = utils.kl_loss_diag(mean_svi_final, logvar_svi_final)
        total_kl_svi += kl_svi.item()*batch_size      
        mean, logvar = mean_svi_final, logvar_svi_final

  ppl_autoreg = np.exp(total_nll_autoreg / num_words)
  ppl_vae = np.exp(total_nll_vae/ num_words)
  kl_vae = total_kl_vae / num_sents
  ppl_bound_vae = np.exp((total_nll_vae + total_kl_vae)/num_words)
  ppl_svi = np.exp(total_nll_svi/num_words)
  kl_svi = total_kl_svi/num_sents
  ppl_bound_svi = np.exp((total_nll_svi + total_kl_svi)/num_words)
  print('AR PPL: %.4f, VAE PPL: %.4f, VAE KL: %.4f, VAE PPL BOUND: %.4f, SVI PPL: %.4f, SVI KL: %.4f, SVI PPL BOUND: %.4f' %
        (ppl_autoreg, ppl_vae, kl_vae, ppl_bound_vae, ppl_svi, kl_svi, ppl_bound_svi))
#   model.train()
  if model_name == 'autoreg':
    return ppl_autoreg
  elif model_name == 'vae':
    return ppl_bound_vae
  elif model_name == 'savae' or model_name == 'svi':
    return ppl_bound_svi

if __name__ == '__main__':
  main()