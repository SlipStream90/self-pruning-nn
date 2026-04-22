# Self-Pruning Neural Network — CIFAR-10

A feed-forward neural network that learns to prune its own weights during training using learnable gate parameters and sparsity regularization.

## How It Works

Each weight in the network is paired with a learnable gate score. During the forward pass, gate scores are passed through a sigmoid to produce a value between 0 and 1, which is multiplied element-wise with the weights. A sparsity regularization term in the loss function penalizes open gates, forcing the network to decide which weights are worth keeping.

## Project Structure

```
├── submission.py        # Full training script
├── report.md            # Analysis and results
├── gate_distribution.png
└── requirements.txt
```

## Requirements

```
torch
torchvision
matplotlib
numpy
tqdm
```

Install with:
```bash
pip install -r requirements.txt
```

## Running the Code

```bash
python submission.py
```

CIFAR-10 will download automatically to `./data/`. GPU is used automatically if available.

## Results Summary

| Lambda | Test Accuracy | Sparsity |
|--------|--------------|----------|
| 0.1    | 54.86%       | 0.73%    |
| 1.0    | 54.11%       | 1.35%    |
| 5.0    | 55.52%       | 15.53%   |
| 10.0   | 54.98%       | 36.52%   |

At λ=10.0, the network prunes 36% of its weights while retaining full accuracy — demonstrating that a large fraction of connections are redundant.
