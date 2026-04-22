# Self-Pruning Neural Network — CIFAR-10

A feed-forward neural network that learns to prune its own weights during training using learnable gate parameters and L1 sparsity regularization.

## How It Works

Each weight is paired with a learnable gate score. During the forward pass, gate scores are passed through a sigmoid to produce a value between 0 and 1, multiplied element-wise with the weights. An L1 sparsity term penalizes open gates, forcing the network to decide which weights are worth keeping.

## Project Structure

```
├── submission.py
├── submission.ipynb
├── report.md
├── gate_distribution.png
├── requirements.txt
└── .gitignore
```

## Requirements

```
torch
torchvision
matplotlib
numpy
tqdm
```

```bash
pip install -r requirements.txt
```

## Running

```bash
python submission.py
```

CIFAR-10 downloads automatically to `./data/`. GPU used automatically if available.

## Results

| Lambda | Test Accuracy | Sparsity (%) |
|--------|--------------|--------------|
| 1e-5   | 55.75%       | 54.67%       |
| 1e-4   | 55.67%       | 97.93%       |
| 1e-3   | 53.48%       | 99.94%       |
| 1e-2   | 46.31%       | 100.00%      |

At λ=1e-4, the network prunes 97.93% of its weights while retaining 55.67% accuracy.
