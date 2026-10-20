import os
import random
import time
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from dataset import load_or_download_digits
from implementation import ScratchMLP

def benchmark_passes():
    X_train, _, y_train, _ = load_or_download_digits()
    X_batch = X_train[:32]
    y_batch = y_train[:32]
    
    W = 64
    mlp = ScratchMLP(W=W)
    
    num_runs = 100
    
    mlp.forward(X_batch)
    mlp.backward(X_batch, y_batch)
    
    forward_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        mlp.forward(X_batch)
        end = time.perf_counter()
        forward_times.append((end - start) * 1000.0)
        
    backward_times = []
    for _ in range(num_runs):
        mlp.forward(X_batch)
        start = time.perf_counter()
        mlp.backward(X_batch, y_batch)
        end = time.perf_counter()
        backward_times.append((end - start) * 1000.0)
        
    median_forward = np.median(forward_times)
    median_backward = np.median(backward_times)
    
    print(f"\n================ Benchmark Results (W={W}, Batch Size=32) ================")
    print(f"Forward Pass  - Median Time over {num_runs} runs: {median_forward:.4f} ms")
    print(f"Backward Pass - Median Time over {num_runs} runs: {median_backward:.4f} ms")
    print(f"Ratio (Backward / Forward): {median_backward / median_forward:.2f}x")

if __name__ == "__main__":
    benchmark_passes()