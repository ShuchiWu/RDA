# Refine, Discriminate and Align: Stealing Encoders via Sample-Wise Prototypes and Multi-Relational Extraction

This repository contains the code of ["Refine, Discriminate and Align: Stealing Encoders via Sample-Wise Prototypes and Multi-Relational Extraction"](https://arxiv.org/abs/2312.00855), a novel model-stealing method against pre-trained image encoders. Datasets (surrogate datasets & downstream datasets) and models (target encoders & surrogate encoders we trained) we use can be found at [dataset and models](https://drive.google.com/drive/folders/1VV97lBVwt5rPlKSHtKQ8PjCuH7d1-fK-?usp=sharing).

![](https://github.com/ShuchiWu/SDA/blob/master/Pipeline.jpg)

## Pre-train an image encoder
We pre-train image encoders (target encoders) using [SimCLR](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf), and the file SimCLR.py is our implementation which we refer to the implementation of [BadEncoder](https://arxiv.org/pdf/2108.00352). Our default model architecture is ResNet18, and you can speicify the model you want by making corresponding modifications to the files in the `/model` folder. The [training data] should be downloaded firsr, and then you could run the following script to pre-train image encoders:

```python
python SimCLR.py
```

## Steal image encoders
The query data we use keeps identical across all the stealing methods we compare. Our implementation of generating sample-wise prototypes has refferred the official code of [EMP-SSL](https://arxiv.org/pdf/2304.03977). The mapping of each method to its code is as follows:
```python
[Conventional](https://openaccess.thecvf.com/content/CVPR2023/papers/Sha_Cant_Steal_Cont-Steal_Contrastive_Stealing_Attacks_Against_Image_Encoders_CVPR_2023_paper.pdf) --> Conventional_Attack.py
[StolenEncoder](https://dl.acm.org/doi/pdf/10.1145/3548606.3560586) --> StolenEncoder.py
[Cont-Steal](https://openaccess.thecvf.com/content/CVPR2023/papers/Sha_Cant_Steal_Cont-Steal_Contrastive_Stealing_Attacks_Against_Image_Encoders_CVPR_2023_paper.pdf) --> Con-Steal.py
```
