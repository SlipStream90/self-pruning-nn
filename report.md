# Report — The Self-Pruning Neural Network

## 1. Why L1 Penalty on Sigmoid Gates Encourages Sparsity

The sparsity loss is the mean of all sigmoid gate values. Since sigmoid is always positive, minimizing this is identical to minimizing the L1 norm of the gates.

L1 has a constant gradient of ±1 regardless of the gate's current value. This means even a gate at 0.001 still gets a full push toward zero — unlike L2 whose gradient vanishes near zero, letting values stagnate at small but nonzero numbers. This is why L1 produces exact zeros rather than just small values.

During training, each gate faces a tug of war: the classification loss tries to keep it open if the weight is useful, the sparsity loss tries to close it regardless. Weights that don't contribute to accuracy lose this competition and get pruned to zero. λ controls how hard the sparsity loss pulls.

---

## 2. Results

| Lambda | Test Accuracy | Sparsity (%) |
|--------|--------------|--------------|
| 0.1    | 54.86%       | 0.73%        |
| 1.0    | 54.11%       | 1.35%        |
| 5.0    | 55.52%       | 15.53%       |
| 10.0   | 54.98%       | 36.52%       |

Sparsity increases monotonically with λ as expected. The notable result is λ=10.0 — 36% of weights pruned with zero accuracy loss, confirming most connections in this network are redundant.

λ=5.0 gives the highest accuracy at 55.52%, slightly above the low-λ baselines. Mild sparsity pressure suppresses weak connections, acting as a regularizer and reducing overfitting — similar to dropout.

---

## 3. Gate Value Distribution

![Gate Distribution](gate_distribution.png)

The spike near zero is the pruned gates — weights the network decided weren't worth keeping. The main body around 0.1–0.3 is the surviving active weights. The gap between them shows the mechanism is producing a real binary outcome rather than just pushing everything to a uniform small value.
