import os
import sys
import time
import statistics
import torch
import torch.nn as nn
from torchinfo import summary
from thop import profile

# 3

from dataset import load_or_download_fashion_mnist, set_seed
from models import FashionMLP, FashionCNN, FashionAttention

def measure_batch_latency(model, sample_batch, device, runs=100):
    model.eval()
    sample_batch = sample_batch.to(device)

    with torch.no_grad():
        _ = model(sample_batch)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    latencies = []
    for _ in range(runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(sample_batch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)  # ms

    return statistics.median(latencies)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running compute cost evaluation on: {device}\n")

    models = {
        "MLP": FashionMLP().to(device),
        "CNN": FashionCNN().to(device),
        "Attention": FashionAttention().to(device)
    }

    dummy_b1 = torch.randn(1, 1, 28, 28).to(device)
    dummy_b128 = torch.randn(128, 1, 28, 28).to(device)

    print("=" * 85)
    print(f"{'Model':<12} | {'Params (torchinfo)':<20} | {'FLOPs (thop)':<15} | {'Batch 1 Latency':<18} | {'Batch 128 Latency':<18}")
    print(" " * 12 + " | " + " " * 20 + " | " + " " * 15 + " | " + "(100 runs, ms)" + " " * 4 + " | " + "(20 runs, ms)")
    print("=" * 85)

    for name, model in models.items():
        info_summary = summary(model, input_size=(1, 1, 28, 28), verbose=0)
        total_params = info_summary.total_params

        macs, _ = profile(model, inputs=(dummy_b1,), verbose=False)
        thop_flops = macs * 2

        lat_b1 = measure_batch_latency(model, dummy_b1, device, runs=100)
        lat_b128 = measure_batch_latency(model, dummy_b128, device, runs=20)

        print(f"{name:<12} | {total_params:<20,d} | {thop_flops:<15,.0f} | {lat_b1:<18.4f} | {lat_b128:<18.4f}")

    print("=" * 85)


if __name__ == "__main__":
    main()