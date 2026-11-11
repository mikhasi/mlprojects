import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

# 3. Train a Simple Neural Network on the Moons Dataset Using Custom Autograd and PyTorch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from value import Value

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def value_sigmoid(self):
    val = 1.0 / (1.0 + math.exp(-self.data))
    out = Value(val, (self,), 'sigmoid')
    def _backward():
        self.grad += (val * (1.0 - val)) * out.grad
    out._backward = _backward
    return out

def value_log(self):
    val = math.log(self.data + 1e-15)
    out = Value(val, (self,), 'log')
    def _backward():
        self.grad += (1.0 / (self.data + 1e-15)) * out.grad
    out._backward = _backward
    return out

import math
Value.sigmoid = value_sigmoid
Value.log = value_log

class CustomMLP:
    def __init__(self):
        self.W1 = [[Value(random.gauss(0, 0.1)) for _ in range(2)] for _ in range(8)]
        self.b1 = [Value(0.0) for _ in range(8)]
        
        self.W2 = [Value(random.gauss(0, 0.1)) for _ in range(8)]
        self.b2 = Value(0.0)

    def parameters(self):
        params = []
        for row in self.W1:
            params.extend(row)
        params.extend(self.b1)
        params.extend(self.W2)
        params.append(self.b2)
        return params

    def forward(self, x_row):
        h = []
        for j in range(8):
            act = sum((x_row[i] * self.W1[j][i] for i in range(2)), self.b1[j])
            h.append(act.sigmoid())
        
        out_act = sum((h[j] * self.W2[j] for j in range(8)), self.b2)
        return out_act.sigmoid()

def train_custom(X, y, epochs=20, lr=0.1):
    set_seeds(42)
    model = CustomMLP()
    params = model.parameters()

    for epoch in range(epochs):
        total_loss = Value(0.0)
        
        for i in range(len(X)):
            x_i = [Value(X[i][0]), Value(X[i][1])]
            pred = model.forward(x_i)
            target = y[i]
            
            if target == 1:
                loss_i = -pred.log()
            else:
                loss_i = -(Value(1.0) - pred).log()
            
            total_loss = total_loss + loss_i
            
        loss = total_loss * (1.0 / len(X))

        for p in params:
            p.grad = 0.0

        # Backward pass
        loss.backward()

        # SGD step
        for p in params:
            p.data -= lr * p.grad

class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Sigmoid(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_pytorch(X, y, epochs=20, lr=0.1):
    set_seeds(42)
    model = TorchMLP()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

def benchmark():
    set_seeds(42)
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    
    epochs = 20
    repeats = 5

    print(f"Dataset: make_moons (100 samples) | Training for {epochs} epochs\n")

    train_custom(X, y, epochs=epochs)
    
    custom_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        train_custom(X, y, epochs=epochs)
        custom_times.append(time.perf_counter() - start)
        
    custom_median = np.median(custom_times)

    train_pytorch(X, y, epochs=epochs)
    
    torch_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        train_pytorch(X, y, epochs=epochs)
        torch_times.append(time.perf_counter() - start)
        
    torch_median = np.median(torch_times)

    print("--- Benchmark Results (Median of 5 runs) ---")
    print(f"Custom Scalar Engine: {custom_median * 1000:.2f} ms")
    print(f"PyTorch Vectorized:   {torch_median * 1000:.2f} ms")
    print(f"Speedup Ratio:        PyTorch is {custom_median / torch_median:.1f}x faster")

if __name__ == "__main__":
    benchmark()