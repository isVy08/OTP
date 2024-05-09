# Parameter Estimation in DAGs from Incomplete Data via Optimal Transport

This repo contains codes for reproducing the experiments in the paper [Parameter Estimation in DAGs from Incomplete Data via Optimal Transport](https://arxiv.org/abs/2305.15927) accepted at ICML 2024.

## Dependencies 
```
pip install -r requirements.txt
```

## Experiments
The scripts for each experiment are as follows:
* Gaussian Mixture Model (Figure 1): `gmm.py`, `train_gmm.py`, `em_gmm.py`
* Latent Dirichlet Allocation (Figure 4, Tables 1 & 2): `topic_model.py`, `train_synth_topic.py`, `train_topic_model.py`, `train_topic_baselines.py`
* Hidden Markov Model (Table 3): `hmm.py`, `train_hmm.py`, `em_hmm.py`
* Discrete Representation Learning (Table 4): `drepl/`

## Citation 
If you use the codes or datasets in this repository, please cite our paper.
