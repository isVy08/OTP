import pickle
import torch, os
from tqdm import tqdm
import torchvision.transforms as transforms
import PIL
import random
import numpy as np

def load_txt(datadir):
    with open(datadir, encoding='utf-8') as f:
        data = f.read().splitlines()
    return data

def load_pickle(datadir):      
  file = open(datadir, 'rb')
  data = pickle.load(file)
  return data

def write_pickle(data, savedir):
  file = open(savedir, 'wb')
  pickle.dump(data, file)
  file.close()

def load_model(model, optimizer, scheduler, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['prev_loss']



def _pad_sequence(sequences):
    # like torch.nn.utils.rnn.pad_sequence with batch_first=True
    max_length = max(x.shape[0] for x in sequences)
    padded_sequences = []
    for x in sequences:
        pad = [(0, 0)] * np.ndim(x)
        pad[0] = (0, max_length - x.shape[0])
        padded_sequences.append(np.pad(x, pad))
    return np.stack(padded_sequences)

def load_jsb_chorales(PATH):
    data = load_pickle(PATH)


    # XXX: we might expose those in `load_dataset` keywords
    min_note = 21
    note_range = 88
    processed_dataset = {}
    for split, data_split in data.items():
        processed_dataset[split] = {}
        n_seqs = len(data_split)
        processed_dataset[split]["sequence_lengths"] = np.zeros(n_seqs, dtype=int)
        processed_dataset[split]["sequences"] = []
        for seq in range(n_seqs):
            seq_length = len(data_split[seq])
            processed_dataset[split]["sequence_lengths"][seq] = seq_length
            processed_sequence = np.zeros((seq_length, note_range))
            for t in range(seq_length):
                note_slice = np.array(list(data_split[seq][t])) - min_note
                slice_length = len(note_slice)
                if slice_length > 0:
                    processed_sequence[t, note_slice] = np.ones(slice_length)
            processed_dataset[split]["sequences"].append(processed_sequence)

    for k, v in processed_dataset.items():
        lengths = v["sequence_lengths"]
        sequences = v["sequences"]
        processed_dataset[k] = (lengths, _pad_sequence(sequences).astype("int32"))
    return processed_dataset

def load_music_data(split, root='data/jsb_chorales.pickle'):
    data = load_jsb_chorales(root)
    _, trainX = data["train"]
    present_notes = (trainX == 1).sum(0).sum(0) > 0
    trainX = trainX[..., present_notes]
    L = trainX.shape[1]
    if split == "train": split = 'valid'

    _, valX = data[split]
    valX = valX[..., present_notes][:, :L, :]
    return trainX, valX