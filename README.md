# Online Conformal Probabilistic Numerics via Adaptive Edge-Cloud Offloading
This repository contains code for "[Online Conformal Probabilistic Numerics via Adaptive Edge-Cloud Offloading](https://arxiv.org/pdf/2503.14453)" -- Qiushuo Hou, Sangwoo Park, Matteo Zecchin, Yunlong Cai, Guanding Yu, and Osvaldo Simeone

<div align="center">
  <img src="https://github.com/qiushuo0913/PN_with_CP_code/blob/main/figure_1.png" alt="system_model" width="300">
</div>
Figure 1: At each round t, a user submits a linear system defined by the pair $(A_t, b_t)$ to an edge device. Given the available computing budget $I_t$, the edge device employs a probabilistic linear solver (PLS), obtaining a highest-probability-density (HPD) set $\mathcal{C}_t$ for the true solution $x^\*_t = A_t^{-1}b_t$. The set $\mathcal{C}_t$ is returned in a timely fashion to the user. The proposed method, OCP-PLS, ensures long-term coverage guarantees for the HPD sets $\mathcal{C}_t$, addressing model misspecification in PLS. To this end, OCP-PLS allows for sporadic communication between the cloud and the edge.

## Dependencies
Python 3.9.19  

Probnum 0.1.27 

Numpy 1.26.0

Scipy 1.11.3

Matplotlib 3.7.1 (used for figure plotting, optional) 

## How to use
**naive_4_seq.py** --- for the conventional PLS (prior on solution $x$ is the standard multivariate Gaussian $x\sim\mathcal{N}(\mu, \Sigma)$, with $\mu=0$, $\Sigma = I$, the search directions are selected via BayesCG.) -- *python naive_4_seq.py* 

**online_CP_4_seq.py** --- for the OCP/I-OCP methods-- *python online_CP_4_seq.py* 

**run.sh** --- for the overall running 
