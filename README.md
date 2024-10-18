# Recommendation Task as a Link Prediction Problem with Graph Auto-Encoder

## Overview

This project formulates **matrix completion** as a **link prediction** problem on a bipartite graph using a **Graph Auto-Encoder (GAE)**. We model user-item interactions through **graph convolutional networks (GCNs)** to predict missing ratings efficiently.

## Problem Definition

Given a bipartite graph \( G = (U, V, E, R) \), where:
- \( U \): Users
- \( V \): Items
- \( E \): Edges representing observed user-item interactions (ratings)
- \( R \): Set of possible rating values

The task is to predict unobserved entries in the user-item matrix \( M \) by learning an embedding for users and items.

### Model Architecture

1. **Graph Encoder**: Uses a multi-layer **GCN** to learn embeddings $U$ (for users) and $V$ (for items) from the input data.

   $$
   H_i^{(l+1)} = \sigma \left( \sum_{j \in N(i)} \frac{1}{\sqrt{d_i d_j}} H_j^{(l)} W^{(l)} \right)
   $$

2. **Pairwise Bilinear Decoder**: Reconstructs the interaction matrix by predicting ratings based on user-item embeddings.

   $$
   \tilde{M}_{ij} = u_i^\top Q v_j
   $$

   where $Q$ is a learnable parameter matrix.

## Results

The best performing model on the MovieLens 100k dataset achieved:
- **RMSE**: 0.758
- **Training Time**: ~15.5 seconds per epoch (with SAGEConv and bilinear decoder)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo
   
2. Install the dependencies
   ```bash
   pip install torch pyyaml tensorboard

## How to run

1. Train the model:
   ```bash
   python main.py

2. Monitor training (optional):
   ```bash
   tensorboard --logdir=runs

## Configuration
Hyperparameters, such as learning rate, epochs, and dataset paths, can be adjusted in the `config.yml` file:
   ```
   lr: 0.01
   epochs: 100
   train_path: 'data/train.csv'
   test_path: 'data/test.csv'
   round_output: true

**Note** : This `README.md` file has been generatd by an LLM.
