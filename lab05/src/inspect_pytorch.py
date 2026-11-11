import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.fx as fx

# 4. Inspect PyTorch Computation Graphs, Benchmark Execution, and Analyze Tensor Properties

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

def inspect_graph():
    set_seeds(42)
    model = SmallModel()
    
    traced = fx.symbolic_trace(model)
    print("==================================================")
    print("1. FX COMPUTATION GRAPH INSPECTION")
    print("==================================================")
    print(traced.code)
    print("-" * 50)
    for node in traced.graph.nodes:
        print(f"Node: {node.name:10} | Op: {node.op:8} | Target: {node.target}")

def benchmark_execution():
    set_seeds(42)
    model = SmallModel().eval()
    x = torch.randn(1000, 4)

    mode_name = "torch.compile"
    try:
        compiled_model = torch.compile(model)
        _ = compiled_model(x)
    except Exception as e:
        print(f"\n[Notice] torch.compile unavailable/failed ({type(e).__name__}); falling back to torch.jit.trace.")
        compiled_model = torch.jit.trace(model, (x,))
        mode_name = "torch.jit.trace"

    repeats = 5

    _ = model(x)
    eager_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(100):
            _ = model(x)
        eager_times.append(time.perf_counter() - start)
    eager_median = np.median(eager_times)

    _ = compiled_model(x)
    compiled_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(100):
            _ = compiled_model(x)
        compiled_times.append(time.perf_counter() - start)
    compiled_median = np.median(compiled_times)

    print("\n==================================================")
    print(f"2. EAGER VS. {mode_name.upper()} EXECUTION (Median of 5 runs)")
    print("==================================================")
    print(f"Eager Mode Time:         {eager_median * 1000:.3f} ms")
    print(f"{mode_name} Time: {compiled_median * 1000:.3f} ms")
    print(f"Speedup:                 {eager_median / compiled_median:.2f}x")

def inspect_tensor_properties():
    set_seeds(42)
    t = torch.randn(3, 3, dtype=torch.float32)
    
    print("\n==================================================")
    print("3. TENSOR PROPERTIES INSPECTION")
    print("==================================================")
    print(f"Original Tensor Shape:      {t.shape}")
    print(f"Original Tensor Dtype:      {t.dtype}")
    print(f"Original Is Contiguous:     {t.is_contiguous()}")
    print(f"Original Stride:            {t.stride()}")

    t_transposed = t.transpose(0, 1)
    print("\n[After transpose(0, 1)]")
    print(f"Transposed Shape:           {t_transposed.shape}")
    print(f"Transposed Is Contiguous:   {t_transposed.is_contiguous()}")
    print(f"Transposed Stride:          {t_transposed.stride()}")

    t_contiguous_again = t_transposed.contiguous()
    print("\n[After .contiguous()]")
    print(f"Contiguous Again Stride:    {t_contiguous_again.stride()}")
    print(f"Is Contiguous Now:          {t_contiguous_again.is_contiguous()}")

def benchmark_contiguity():
    set_seeds(42)
    size = 4000
    tensor_large = torch.randn(size, size)

    t_contig = tensor_large
    t_non_contig = tensor_large.t()

    repeats = 5

    _ = torch.sum(t_contig)
    contig_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(50):
            _ = torch.sum(t_contig)
        contig_times.append(time.perf_counter() - start)
    contig_median = np.median(contig_times)

    _ = torch.sum(t_non_contig)
    non_contig_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(50):
            _ = torch.sum(t_non_contig)
        non_contig_times.append(time.perf_counter() - start)
    non_contig_median = np.median(non_contig_times)

    print("\n==================================================")
    print("4. CONTIGUOUS VS NON-CONTIGUOUS SPEED (Median of 5 runs)")
    print("==================================================")
    print(f"Contiguous Sum Time:     {contig_median * 1000:.3f} ms")
    print(f"Non-Contiguous Sum Time: {non_contig_median * 1000:.3f} ms")
    print(f"Overhead Ratio:          {non_contig_median / contig_median:.2f}x slower for non-contiguous")

if __name__ == "__main__":
    inspect_graph()
    benchmark_execution()
    inspect_tensor_properties()
    benchmark_contiguity()