import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader

import time
import os
import imageio
import requests
from tqdm import tqdm
from datasets.block import BlockDataset, LatentBlockDataset


def load_latent_block(data_file_path, is_label):
    # data_folder_path = os.getcwd()
    # data_file_path = data_folder_path + \
    #     '/data/'+name+'.npz'
    train = LatentBlockDataset(data_file_path, is_label=is_label, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, is_label=is_label, train=False,
                       transform=None)
    return train, val


def data_loaders(train_data, val_data, batch_size):
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False)
    return train_loader, val_loader


def load_data_and_data_loaders_LATENT_BLOCK(dataset, data_file_path, batch_size, is_label):
    training_data, validation_data = load_latent_block(data_file_path, is_label)
    training_loader, validation_loader = data_loaders(
    training_data, validation_data, batch_size)
    x_train_var = np.var(training_data.data)
    return training_data, validation_data, training_loader, validation_loader, x_train_var
    

def set_seeds(seed=0, fully_deterministic=True):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if fully_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def myprint(statement, noflg=False):
    if not noflg:
        print(statement)


def get_loader(dataset, path_dataset, bs=64, n_work=2):
    if dataset == "MNIST" or  dataset == "FashionMNIST":
        preproc_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        all_size = 60000
        train_size = 50000
        subset1_indices = list(range(0, train_size))
        subset2_indices = list(range(train_size, all_size))
        trainval_dataset = eval("datasets."+dataset)(
                os.path.join(path_dataset, "{}/".format(dataset)),
                train=True, download=True, transform=preproc_transform
        )
        train_dataset = Subset(trainval_dataset, subset1_indices)
        val_dataset   = Subset(trainval_dataset, subset2_indices)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=bs, shuffle=True,
            num_workers=n_work, pin_memory=False, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=bs, shuffle=False,
            num_workers=n_work, pin_memory=False
        )
        test_loader = torch.utils.data.DataLoader(
            eval("datasets."+dataset)(
                os.path.join(path_dataset, "{}/".format(dataset)),
                train=False, download=True, transform=preproc_transform
            ), batch_size=bs, shuffle=False,
            num_workers=n_work, pin_memory=False
        )
    elif dataset == 'SVHN':
        transform = transforms.Compose([transforms.ToTensor(), 
            #transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        #import pdb; pdb.set_trace()
        trainval_dataset = datasets.SVHN(os.path.join(path_dataset, "SVHN/"), split='train', download=False, transform=transform)
        all_size = len(trainval_dataset)
        train_size = 63257
        subset1_indices = list(range(0, train_size))
        subset2_indices = list(range(train_size, all_size))
        
        train_dataset = Subset(trainval_dataset, subset1_indices)
        val_dataset   = Subset(trainval_dataset, subset2_indices)
        test_dataset = datasets.SVHN(os.path.join(path_dataset, "SVHN/"), split='test', download=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=n_work, pin_memory=False, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=n_work, pin_memory=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=n_work, pin_memory=False, drop_last=False)

    elif dataset == "CelebA":
        preproc_transform = transforms.Compose([
            transforms.CenterCrop(140),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        dset_train = datasets.CelebA(os.path.join(path_dataset, "CelebA/"), split="train", target_type="attr",
            transform=preproc_transform, target_transform=None, download=True)
        dset_valid = datasets.CelebA(os.path.join(path_dataset, "CelebA/"), split="valid", target_type="attr",
            transform=preproc_transform, target_transform=None, download=True)
        dset_test = datasets.CelebA(os.path.join(path_dataset, "CelebA/"), split="test", target_type="attr",
            transform=preproc_transform, target_transform=None, download=True)
        train_loader = torch.utils.data.DataLoader(dset_train,
            batch_size=bs, shuffle=True, num_workers=n_work, pin_memory=False, drop_last=True)
        val_loader = torch.utils.data.DataLoader(dset_valid,
            batch_size=bs, shuffle=False, num_workers=n_work, pin_memory=False
        )
        test_loader = torch.utils.data.DataLoader(dset_test,
            batch_size=bs, shuffle=False, num_workers=n_work, pin_memory=False
        )
    elif dataset == "CIFAR10":
        preproc_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        all_size = 50000
        train_size = 40000
        subset1_indices = list(range(0, train_size))
        subset2_indices = list(range(train_size, all_size))
        trainval_dataset = datasets.CIFAR10(
                os.path.join(path_dataset, "{}/".format(dataset)), train=True, download=False,
                transform=preproc_transform
        )
        train_dataset = Subset(trainval_dataset, subset1_indices)
        val_dataset   = Subset(trainval_dataset, subset2_indices)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=bs, shuffle=True,
            num_workers=n_work, pin_memory=False, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=bs, shuffle=False,
            num_workers=n_work, pin_memory=False
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                os.path.join(path_dataset, "{}/".format(dataset)), train=False, download=False,
                transform=preproc_transform
            ), batch_size=bs, shuffle=False,
            num_workers=n_work, pin_memory=False
        )

    return train_loader, val_loader, test_loader


def logits_to_onehot(prob, n_class=19):
    size = prob.size()
    label_idx = torch.argmin(prob, dim=1, keepdim=True)
    oneHot_size = (size[0], n_class, size[2], size[3])
    label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    label = label.scatter_(1, label_idx.data.long().cuda(), 1.0)

    return label



## Generate images
def plot_images(images, filename, nrows=4, ncols=8, flg_norm=False):
    if images.shape[1] == 1:
        images = np.repeat(images, 3, axis=1)
    fig = plt.figure(figsize=(nrows * 2, ncols))
    gs = gridspec.GridSpec(nrows * 2, ncols)
    gs.update(wspace=0.05, hspace=0.05)
    for i, image in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        if flg_norm:
            plt.imshow(image.transpose((1,2,0)) * 0.5 + 0.5)
        else:
            plt.imshow(image.transpose((1,2,0)))

    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_images_paper(images, filename, nrows=4, ncols=8, flg_norm=False):
    if images.shape[1] == 1:
        images = np.repeat(images, 3, axis=1)
    fig = plt.figure(figsize=(nrows, ncols))
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.05, hspace=0.05)
    for i, image in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_aspect("equal")
        if flg_norm:
            plt.imshow(image.transpose((1,2,0)) * 0.5 + 0.5)
        else:
            plt.imshow(image.transpose((1,2,0)))

    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)



## CelebA Mask
def idx_to_onehot(idx, n_class=19):
    size = idx.size()
    oneHot_size = (size[0], n_class, size[2], size[3])
    label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    label = label.scatter_(1, idx.data.long().cuda(), 1.0)

    return label


def logits_to_onehot(prob, n_class=19):
    size = prob.size()
    label_idx = torch.argmin(prob, dim=1, keepdim=True)
    oneHot_size = (size[0], n_class, size[2], size[3])
    label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    label = label.scatter_(1, label_idx.data.long().cuda(), 1.0)

    return label


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def labelcolormap(N):
    if N == 19: # CelebAMask-HQ
        cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0),
                     (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                     (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                     (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153), 
                     (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)], 
                     dtype=np.uint8) 
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def tensor2label(label_tensor, n_label, imtype=np.uint8):
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0

    return label_numpy


def generate_label(inputs, imsize):
    pred_batch = []
    for input in inputs:
        input = input.view(1, 19, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
            
    label_batch = []
    for p in pred_batch:
        p = p.view(1, imsize, imsize)
        label_batch.append(tensor2label(p, 19))
                
    label_batch = np.array(label_batch)
    label_batch = torch.from_numpy(label_batch)	

    return label_batch


def label_to_segment(label, nclass=19):
    color_list = [[0., 0., 0],
        [204., 0., 0],
        [76., 153., 0],
        [204., 204., 0],
        [51., 51., 255.],
        [204., 0., 204],
        [0., 255., 255.],
        [51., 255., 255.],
        [102., 51., 0],
        [255., 0., 0],
        [102., 204., 0],
        [255., 255., 0],
        [0., 0., 153],
        [0., 0., 204],
        [255., 51., 153],
        [0., 204., 204],
        [0., 51., 0],
        [255., 153., 51],
        [0., 204., 0]
    ]
    color_torch = torch.tensor(color_list).cuda()
    shape = label.shape
    label_permuted = label.permute(0, 2, 3, 1).contiguous()
    label_reshaped = label_permuted.view(-1, nclass)
    label_colored = label_reshaped @ color_torch
    label_colored_reshaped = label_colored.view(shape[0], shape[2], shape[3], 3)
    label_colored_permuted = label_colored_reshaped.permute(0, 3, 1, 2).contiguous()

    return label_colored_permuted

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        if image_tensor.dim() == 4:
            #image_numpy = ((image_tensor[0]+1.0)/2.0).clamp(0,1).cpu().float().numpy()
            image_numpy = (image_tensor[0]).clamp(0,1).cpu().float().numpy()
        else:
            #image_numpy = ((image_tensor+1.0)/2.0).clamp(0,1).cpu().float().numpy() # convert it into a numpy array
            image_numpy = (image_tensor).clamp(0,1).cpu().float().numpy() # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    imageio.imwrite(image_path, image_numpy)
