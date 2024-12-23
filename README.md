# Refine, Discriminate and Align (RDA)
This repository houses the code for ["Refine, Discriminate and Align: Stealing Encoders via Sample-Wise Prototypes and Multi-Relational Extraction"](https://arxiv.org/pdf/2312.00855) (🎉 accepted to ECCV 2024), a novel model-stealing approach targeting pre-trained image encoders. The datasets (query & downstream) and models (target & surrogate encoders) employed in this work are accessible at [datasets and models](https://drive.google.com/drive/folders/1VV97lBVwt5rPlKSHtKQ8PjCuH7d1-fK-?usp=sharing).

![](https://github.com/ShuchiWu/SDA/blob/master/Pipeline.jpg)

## Pre-train image encoders
We pre-train image encoders (target encoders) using [SimCLR](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf), with our implementation detailed in SimCLR.py, which references the implementation of [BadEncoder](https://arxiv.org/pdf/2108.00352). Our default model architecture is ResNet18, but you can specify a different model by modifying the relevant files in the `/model` folder. Ensure the training data is downloaded first, then execute the following script to pre-train the image encoders:
```scrpit
python SimCLR.py
```

## Steal image encoders
The query data we utilize remains consistent across all the stealing methods we compare. Our method for generating sample-wise prototypes references the official code of [EMP-SSL](https://arxiv.org/pdf/2304.03977). We have compared RDA with three existing image encoder-stealing methods, i.e., [Conventional (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Sha_Cant_Steal_Cont-Steal_Contrastive_Stealing_Attacks_Against_Image_Encoders_CVPR_2023_paper.pdf), [StolenEncoder (CCS 2022)](https://dl.acm.org/doi/pdf/10.1145/3548606.3560586), and [Cont-Steal (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Sha_Cant_Steal_Cont-Steal_Contrastive_Stealing_Attacks_Against_Image_Encoders_CVPR_2023_paper.pdf). The code for each compared method and our RDA is provided as follows:
```scrpit
# Conventional (proposed as the baseline by Cont-Steal)
python Conventional_Attack.py

# StolenEncoder
# Since the original paper hasn't released the code, we have implemented it ourselves based on the paper.
python StolenEncoder.py  

# Cont-Steal
python Con-Steal.py  

# RDA (ours)
python RDA.py  
```

## Downstream classifiers
At first, we utilize (target or surrogate) image encoders to extract features from all training samples, storing them in a feature bank, which is subsequently used to train downstream classifiers. Configure the downstream task (i.e., dataset) you wish to use, then execute the following script for training:
```scrpit
python downstream_tasks.py
```

## Robustness evaluation
In our paper, we have explored four defense mechanisms: three perturbation-based techniques (namely, noising, rounding, and top-k) and one watermark-based approach (backdoor). The code for perturbation-based defenses is provided as follows:
```scrpit
# Nosing
python noise_defense.py

# Rounding
python Rounding_defense.py

# Top-k
python top-k.py
```
Regarding the watermark-based approach, we adopt BadEncoder. Specifically, we utilize the ResNet18 model pretrained and backdoored by [BadEncoder](https://arxiv.org/pdf/2108.00352), which can be downloaded from the link provided above or [the official github repository of BadEncoder](https://github.com/jinyuan-jia/BadEncoder). Our next step involves evaluating the backdoor attack success rate for both the target and surrogate models to assess the robustness of RDA. For evaluation, we select the surrogate model that demonstrates the highest classification accuracy for downstream tasks.

## Citation
If you use this code, please cite the following paper:
```script
@inproceedings{wu2025refine,
  title={Refine, Discriminate and Align: Stealing Encoders via Sample-Wise Prototypes and Multi-Relational Extraction},
  author={Wu, Shuchi and Ma, Chuan and Wei, Kang and Xu, Xiaogang and Ding, Ming and Qian, Yuwen and Xiao, Di and Xiang, Tao},
  booktitle={European Conference on Computer Vision},
  pages={186--203},
  year={2025},
  organization={Springer}
}
```
