# OPT-VQAE
## Training
The training of a model can be done by calling main.py with the corresponding yaml file. The list of yaml files can be found below.
Please refer to main.py (or execute 'python main.py --help') for the usage of extra arguments.

### Setup steps before training of a model
* Set the checkpoint path "_C.path" (/configs/defaults.py:4) 
* Set the dataset path, "_c.path_dataset" (/configs/defaults.py:5).


### Train a model
```
python main.py -c "cifar10_K512.yaml" --save
```

### Test a trained model
```
python main.py -c "cifar10_K512.yaml" --save -timestamp resnet_seed0_0916_0610
```

### Where to find the checkpoints
If the trainning is successful, checkpoint folders will be generated under the folder (cfgs represents the yaml file specified when calling main.py):
```
configs.defaults._C.path + '/' + cfgs.path_spcific
```

**Evaluation:** goto WQVAE/evaluations/ and run:

Fid score:
```
python3 fid_score.py folder_groudtruth_images  folder_recontructed_images --batch-size 192 --gpu 1
```

Lips, PSNR, SSIM score:
```
python evaluation.py --gt_path folder_groudtruth_images  --g_path folder_recontructed_images 
```

**Note:** Other hyper-parameters can be found in Arguments.py


### List of yaml files: models work on continuous/discrete data distributions
| Config file | Description |
|---|---|
| cifar10_K512.yaml | OTP_VQAE on CIFAR10 with codebook size of 512 |
| mnist10_K512.yaml | OTP_VQAE on MNIST with codebook size of 512 |
| svhn10_K512.yaml | OTP_VQAE on SVHN with codebook size of 512  |




## Experiments
"[checkpoint_foldername_with_timestep]" means the folder names under the path "[configs.defaults._C.path + '/' + cfgs.path_spcific]".
These folder names are consist of the model names, the seed indices and the timestamps.

## Dependencies
numpy
scipy
torch
torchvision
PIL
ot

## Acknowledgements
Codes are adapted from https://github.com/sony/sqvae/tree/main/vision. We thank them for their excellent projects.

