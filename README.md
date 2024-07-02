# Refine, Discriminate and Align: Stealing Encoders via Sample-Wise Prototypes and Multi-Relational Extraction

This repository contains the code of ["Refine, Discriminate and Align: Stealing Encoders via Sample-Wise Prototypes and Multi-Relational Extraction"](https://arxiv.org/abs/2312.00855) (🎉 accepted to ECCV 2024), a novel model-stealing method against pre-trained image encoders. Datasets (surrogate datasets & downstream datasets) and models (target encoders & surrogate encoders we trained) we use can be found at [dataset and models](https://drive.google.com/drive/folders/1VV97lBVwt5rPlKSHtKQ8PjCuH7d1-fK-?usp=sharing).

![](https://github.com/ShuchiWu/SDA/blob/master/Pipeline.jpg)

## Pre-train image encoders
We pre-train image encoders (target encoders) using [SimCLR](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf), and the file SimCLR.py is our implementation which we refer to the implementation of [BadEncoder](https://arxiv.org/pdf/2108.00352). Our default model architecture is ResNet18, and you can speicify the model you want by making corresponding modifications to the files in the `/model` folder. The [training data] should be downloaded firsr, and then you could run the following script to pre-train image encoders:

```scrpit
python SimCLR.py
```

## Steal image encoders
The query data we use keeps identical across all the stealing methods we compare. Our implementation of generating sample-wise prototypes has refferred the official code of [EMP-SSL](https://arxiv.org/pdf/2304.03977). We have compared RDA with three existing image encoder-stealing methods, i.e., [Conventional (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Sha_Cant_Steal_Cont-Steal_Contrastive_Stealing_Attacks_Against_Image_Encoders_CVPR_2023_paper.pdf), [StolenEncoder (CCS 2022)](https://dl.acm.org/doi/pdf/10.1145/3548606.3560586), and [Cont-Steal (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Sha_Cant_Steal_Cont-Steal_Contrastive_Stealing_Attacks_Against_Image_Encoders_CVPR_2023_paper.pdf). The code for each compared method and our RDA is as follows:
```scrpit
python Conventional_Attack.py  # Conventional (proposed and choosen as the baseline by Cont-Steal)
python StolenEncoder.py  # StolenEncoder (The original paper hasn't released the code so we implement it ourselves based on the paper.)
python Con-Steal.py  # Cont-Steal 
python RDA.py  # RDA (ours)
```

## Downstream classification
We use (target or surrogate) image encoders to extract features of all training samples and store them to a feature bank first, which is utlized to train downstream classifiers next. You should configure the downstream task (i.e., dataset) you want first and run the following script for training:
```scrpit
python downstream_tasks.py
```
