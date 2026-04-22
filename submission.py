import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False)

class PrunableLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
    
    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return f.linear(x, pruned_weights, self.bias)

class pruned_nn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_net = nn.Sequential(
            nn.Flatten(),
            PrunableLayer(32*32*3, 512),
            nn.ReLU(),
            PrunableLayer(512, 256),
            nn.ReLU(),
            PrunableLayer(256, 128),
            nn.ReLU(),
            PrunableLayer(128, 10)
        )

    def forward(self, x):
        return self.neural_net(x)

    def sparse_error(self):
        total= 0
        for m in self.modules():
            if isinstance(m, PrunableLayer):
                total += torch.sigmoid(m.gate_scores).sum()
        return total

def train(model, lambda_val, epochs=15):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            cls_loss = criterion(outputs, labels)
            spar_loss = model.sparse_error()
            total_loss = cls_loss + lambda_val * spar_loss

            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} (λ={lambda_val}) loss={total_loss.item():.4f}")

def evaluate(model, threshold=0.1):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total

    total_gates, pruned_gates = 0, 0
    for module in model.modules():
        if isinstance(module, PrunableLayer):
            gates = torch.sigmoid(module.gate_scores).detach()
            pruned_gates += (gates < threshold).sum().item()
            total_gates += gates.numel()
    sparsity = 100 * pruned_gates / total_gates

    return accuracy, sparsity

lambdas = [1e-5, 1e-4, 1e-3, 1e-2]

results = []
best_model = None
best_acc = -1

for lam in lambdas:
    model = pruned_nn_model()
    train(model, lambda_val=lam, epochs=15)

    acc, sparsity = evaluate(model)
    results.append((lam, acc, sparsity))

    if acc > best_acc:
        best_acc = acc
        best_model = model

    print(f"Results for λ={lam}: Accuracy={acc:.2f}%, Sparsity={sparsity:.2f}%")

all_gates = []
for module in best_model.modules():
    if isinstance(module, PrunableLayer):
        gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy().flatten()
        all_gates.extend(gates)

plt.figure(figsize=(8, 4))
plt.hist(all_gates, bins=100, color='steelblue', edgecolor='black')
plt.title('Distribution of Gate Values (Best Model)')
plt.xlabel('Gate Value')
plt.ylabel('Count')
plt.savefig('gate_distribution.png')
plt.show()