# OTP

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


```
@InProceedings{pmlr-v235-vo24a,
  title = 	 {Parameter Estimation in {DAG}s from Incomplete Data via Optimal Transport},
  author =       {Vo, Vy and Le, Trung and Vuong, Long Tung and Zhao, He and Bonilla, Edwin V. and Phung, Dinh},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {49580--49604},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/vo24a/vo24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/vo24a.html},
  abstract = 	 {Estimating the parameters of a probabilistic directed graphical model from incomplete data is a long-standing challenge. This is because, in the presence of latent variables, both the likelihood function and posterior distribution are intractable without assumptions about structural dependencies or model classes. While existing learning methods are fundamentally based on likelihood maximization, here we offer a new view of the parameter learning problem through the lens of optimal transport. This perspective licenses a general framework that operates on any directed graphs without making unrealistic assumptions on the posterior over the latent variables or resorting to variational approximations. We develop a theoretical framework and support it with extensive empirical evidence demonstrating the versatility and robustness of our approach. Across experiments, we show that not only can our method effectively recover the ground-truth parameters but it also performs comparably or better than competing baselines on downstream applications.}
}
```
