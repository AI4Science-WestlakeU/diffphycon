# DiffPhyCon: A Generative Approach to Control Complex Physical Systems (NeurIPS 2024)

[Paper](https://openreview.net/forum?id=MbZuh8L0Xg) | [arXiv](https://web3.arxiv.org/abs/2407.06494) 
<!-- | [Poster](https://github.com/AI4Science-WestlakeU/cindm/blob/main/assets/CinDM_poster.pdf)  -->
<!-- | [Tweet](https://twitter.com/tailin_wu/status/1747259448635367756)  -->

Official repo for the paper [DiffPhyCon: A Generative Approach to Control Complex Physical Systems](https://openreview.net/forum?id=MbZuh8L0Xg).<br />
[Long Wei*](https://longweizju.github.io/), [Peiyan Hu*](https://peiyannn.github.io/), [Ruiqi Feng*](https://weenming.github.io/), [Haodong Feng](https://scholar.google.com/citations?user=0GOKl_gAAAAJ&hl=en), [Yixuan Du](https://openreview.net/profile?id=~Yixuan_Du1), [Tao Zhang](https://zhangtao167.github.io), [Rui Wang](https://openreview.net/profile?id=~Rui_Wang56), [Yue Wang](https://www.microsoft.com/en-us/research/people/yuwang5/), [Zhi-Ming Ma](http://homepage.amss.ac.cn/research/homePage/8eb59241e2e74d828fb84eec0efadba5/myHomePage.html), [Tailin Wu](https://tailin.org/)<br />
NeurIPS 2024. 

We introduce a novel method, DiffPhyCon, for controlling complex physical systems using generative models, by minimizing the learned generative energy function and specified control objectives. Further, we enhance DiffPhyCon with prior reweighting, enabling the discovery of control sequences that significantly deviate from the training distribution.

Framework of DiffPhyCon:

<a href="url"><img src="https://github.com/AI4Science-WestlakeU/diffphycon/blob/main/assets/figure1.png" align="center" width="800" ></a>

Movement and fluid field visualization on the jellyfish controlled by DiffPhyCon:

<a href="url"><img src="https://github.com/AI4Science-WestlakeU/diffphycon/blob/main/assets/jellyfish.jpg" align="center" width="800" ></a>

Density of smoke (left) and control signals (middle and right) controlled by DiffPhyCon:

<a href="url"><img src="https://github.com/AI4Science-WestlakeU/diffphycon/blob/main/assets/smoke.gif" align="center" width="800" ></a> 

<!-- Example generated trajectories and the airfoil boundary:

<a href="url"><img src="https://github.com/AI4Science-WestlakeU/pde_gen_control/tree/main/assets/generated_examples.gif" align="center" width="600" ></a> -->

# Installation

Run the following commonds to install dependencies. In particular, when run the smoke control task, the python version must be 3.8 due to the requirement of the Phiflow software.

```code
conda env create -f environment.yml
conda activate DiffPhyCon
```

# Dataset and checkpoints
## Dataset
The checkpoints of both our DiffPhyCon and baselines on the three tasks (1D Burgers', 2D jellyfish, 2D smoke), and our released Jellyfish datasets can be downloaded in [link](https://drive.google.com/drive/folders/1_ECmMZ77Lm02znhQ72MvwKe1MqGQquzU). To run the following training and inference scripts locally, replace the path names in filepath.py by your local paths.
<!-- Because the training dataset in the 2D experiment is over 100GB, it is not contained in this link. -->


# Training:
## 1D Burgers' Equation Control:

### Surrogate model
```code
python /model/pde_1d_surrogate_model/burgers_operator.py --date_time "2024-01-09_1d_surrogate_partial_ob_partial_ctr_autoregress-3" --epochs 500 --gpu 0 --train_batch_size 151 --autoregress_steps 3 --dataset_path "/dataset_control_burgers/free_u_f_1e5_front_rear_quarter" --is_partially_observable 1 --is_partially_controllable 1 --lr 0.001
```

### DiffPhyCon

In the diffusion_1d folder, run
```code
bash script/train_pw_XOXC.sh
bash script/train_XOXC.sh
```
where XOXC stands for one of the setting names: FOPC, POFC, and POPC.


## 2D Jellyfish Control:

### DiffPhyCon

In the scrpits/ folder, for the full observation scenario, run the following script to train a diffusion model for joint distribution of the state trajectory and control signals:
```
bash jellyfish_train_joint_full.sh
```

run the following script to train an additional diffusion model for the control signels prior:
```
bash jellyfish_train_w_full.sh
```
Similarly, for the partial observation scenario, run the following two scripts, respectively:
```
bash jellyfish_train_joint_partial.sh
bash jellyfish_train_w_partial.sh
```


## 2D Smoke Control:

### DiffPhyCon

In the scrpits/ folder, run the following script to train a diffusion model for joint distribution of the state trajectory and control signals:
```
bash smoke_train_joint.sh
```

run the following script to train an additional diffusion model for the control signels prior:
```
bash smoke_train_w.sh
```

# Inference:
## 1D Burgers' Equation Control:


### DiffPhyCon

In the diffusion_1d folder, run
```code
bash script/eval_XOXC.sh
```
where XOXC stands for one of the setting names: FOPC, POFC, and POPC.


## 2D Jellyfish Control:
### DiffPhyCon
In the scripts/ folder, for the full/partial observation scenarios, run the following two scripts, repectively:
```
bash jellyfish_inference_full.sh
bash jellyfish_inference_partiall.sh
```


## 2D Smoke Control:
### DiffPhyCon
In the scripts/ folder, run the following script:
```
bash smoke_inference.sh
```


## Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{
wei2024diffphycon,
title={DiffPhyCon: A Generative Approach to Control Complex Physical Systems},
author={Wei, Long and Hu, Peiyan and Feng, Ruiqi and Feng, Haodong and Du, Yixuan and Zhang, Tao and Wang, Rui and Wang, Yue and Ma, Zhi-Ming and Wu, Tailin},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=MbZuh8L0Xg}
}
```
