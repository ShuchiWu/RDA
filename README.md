# Refine, Discriminate and Align: Stealing Encoders via Sample-Wise Prototypes and Multi-Relational Extraction

This repository contains the code of ["Refine, Discriminate and Align: Stealing Encoders via Sample-Wise Prototypes and Multi-Relational Extraction"](https://arxiv.org/abs/2312.00855), a novel model-stealing method against pre-trained image encoders. Datasets (surrogate datasets & downstream datasets) and models (target encoders & surrogate encoders we trained) we use can be found at [dataset and code](https://drive.google.com/drive/folders/1VV97lBVwt5rPlKSHtKQ8PjCuH7d1-fK-?usp=sharing).

![](https://github.com/ShuchiWu/SDA/blob/master/RDA.png)

## Pre-train an image encoder
We pre-train image encoders (target encoders) using [SimCLR](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf), and the file SimCLR.py is our implementation which we refer to the implementation of [BadEncoder](https://arxiv.org/pdf/2108.00352). Our default model architecture is ResNet18, and you speicify the model you want by making corresponding modifications to the files in the <font color="gray">model</font> folder. 

