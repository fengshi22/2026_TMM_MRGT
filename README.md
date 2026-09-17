<h1 align="center">Multi-Relational Graph Transformer for Multi-View Semi-Supervised Classification</h1>

## Abstract

Multi-view learning constructs robust joint representations by effectively integrating consistent and complementary information from heterogeneous sources.
 Graph neural networks have achieved substantial progress in multi-view learning.
However, most existing approaches primarily rely on localized message-passing mechanisms to capture local topological structures, while insufficiently modeling heterogeneous node relationships and their varying influence.
In this paper, we propose a multi-relational graph Transformer for multi-view learning. 
This architecture integrates multi-view information to model complex node associations by constructing multi-relational graphs, where samples are explicitly regarded as nodes to encode heterogeneous relationships.
A relational graph Transformer is designed to learn the impact of heterogeneous relationships on feature propagation, yielding more discriminative node representations. 
Additionally, this method introduces learnable multi-relation fusion to integrate relation propagated messages with original node representations and combine relation-specific features.

## Model Architecture

<p align="center">
  <img src="assets/MRGT_framework.png" width="100%" alt="MRGT architecture">
</p>

## Experiment

The datasets used can be downloaded from [here](https://drive.google.com/drive/folders/1kZRnt6RKthwGXGePmQYIduErOgNK_UCU?usp=drive_link), please download them and put them in datasets to `data`.

| Dataset | #Views | # Samples | # Classes |
|:---:|:---:|:---:|:---:|
| 20newsgroups | 2,000/2,000/2,000          | 500    |  5 |
| ALOI         | 64/64/77/13                | 1,079  | 10 |
| Hdigit       | 784/256                    | 10,000 | 10 |
| HW           | 153/596/301/481/157/27     | 2,000  | 10 |
| Mfeat        | 216/76/64/6/240/47         | 2,000  | 10 |
| NoisyMNIST   | 784/784                    | 15,000 | 10 |
| Prokaryotic  | 438/3/393                  | 551    |  4 | 
| UCI          | 240/76/6                   | 2,000  | 10 |

## Requirements

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt  
```

Main dependencies include PyTorch, PyTorch Lightning, PyTorch Geometric, NumPy, SciPy, scikit-learn, hdf5storage, and PyYAML.

## Reference

If you find this work useful in your research, please consider citing:

```bibtex
@article{zhao2026mrgt,
  title={Multi-Relational Graph Transformer for Multi-View Semi-Supervised Classification},
  author={Zhao, Wendi and Guo, Canyang and Du, Shide and Shi, Yongquan and Chen, Dewang and Wang, Shiping},
  journal={IEEE Transactions on Multimedia},
  year={2026}
}
```
