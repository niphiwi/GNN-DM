# GNN-DM

## Introduction
This repository contains the code of **GNN-DM**, a graph neural network for gas distribution mapping (GDM). GDM describes the process of mapping the spatial and temporal distributions of gases in a given area.

## Gas Distribution Dataset
The synthetic gas distribution dataset is based on the dataset that was previously made available in the repository of [Super-Resolution for Gas Distribution Mapping](https://github.com/BAMresearch/SRGDM/). In the linked repo, training and validation datasets are available through Git LFS (see `data/30x25.zip`). Unzip the files to this directory: `data/30x25/raw`.

## Citation
If you find this code useful, please cite our paper:
```
@ARTICLE{11197193,
  author={Winkler, Nicolas P. and Neumann, Patrick P. and Albizu, Natalia and Schaffernicht, Erik and Lilienthal, Achim J.},
  journal={IEEE Sensors Journal}, 
  title={GNN-DM: A Graph Neural Network Framework for Real-World Gas Distribution Mapping}, 
  year={2025},
  volume={25},
  number={22},
  pages={42171-42179},
  keywords={Sensors;Sensor phenomena and characterization;Training;Layout;Gas detectors;Mathematical models;Data models;Automobiles;Training data;Synthetic data;Environmental monitoring;gas distribution mapping (GDM);graph neural networks (GNNs);sensor networks;transfer learning},
  doi={10.1109/JSEN.2025.3617158}}
