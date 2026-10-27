import os
import time
import statistics
import torch
import matplotlib.pyplot as plt

from dataset import load_or_download_fashion_mnist, set_seed
from models import FashionMLP, FashionCNN, FashionAttention, count_parameters
from train import evaluate_model, train_model

# 5

def get_latency_ms(model, sample_tensor, device, runs=100):
    model.eval()
    sample_tensor = sample_tensor.to(device)
    
    with torch.no_grad():
        _ = model(sample_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
    latencies = []
    for _ in range(runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(sample_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0) # ms
        
    return statistics.median(latencies)

def save_plot_safely(fig, filename="accuracy_vs_params_and_latency.png"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lab03_dir = os.path.abspath(os.path.join(script_dir, "../"))
    results_dir = os.path.join(lab03_dir, "results")
    
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, filename)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Results successfully saved to: {output_path}")
    return output_path

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    train_loader, _, test_loader = load_or_download_fashion_mnist(batch_size=128)
    
    models = {
        "MLP": FashionMLP().to(device),
        "CNN": FashionCNN().to(device),
        "Attention": FashionAttention().to(device)
    }
    
    names = []
    params = []
    accuracies = []
    latencies = []
    
    dummy_b1 = torch.randn(1, 1, 28, 28).to(device)
    
    for name, model in models.items():
        print(f"Training {name} for metrics...")
        train_model(model, train_loader, device, epochs=5, lr=1e-3)
        acc = evaluate_model(model, test_loader, device)
        num_params = count_parameters(model)
        latency = get_latency_ms(model, dummy_b1, device, runs=100)
        
        names.append(name)
        params.append(num_params)
        accuracies.append(acc)
        latencies.append(latency)
        print(f"[{name}] Params: {num_params:,} | Acc: {acc:.2f}% | Latency (b=1): {latency:.4f} ms\n")
        
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    axes[0].scatter(params, accuracies, color=colors, s=120, zorder=5)
    for i, txt in enumerate(names):
        axes[0].annotate(
            f" {txt}\n ({params[i]:,} params, {accuracies[i]:.1f}%)", 
            (params[i], accuracies[i]), 
            fontsize=10, 
            xytext=(5, -10), 
            textcoords='offset points'
        )
    axes[0].set_xlabel('Parameter Count', fontsize=12)
    axes[0].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[0].set_title('Accuracy vs Parameter Count', fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].scatter(latencies, accuracies, color=colors, s=120, zorder=5)
    for i, txt in enumerate(names):
        axes[1].annotate(
            f" {txt}\n ({latencies[i]:.2f} ms, {accuracies[i]:.1f}%)", 
            (latencies[i], accuracies[i]), 
            fontsize=10, 
            xytext=(5, -10), 
            textcoords='offset points'
        )
    axes[1].set_xlabel('Single-Sample Latency (ms)', fontsize=12)
    axes[1].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy vs Latency (Batch Size 1)', fontsize=14, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    save_plot_safely(fig, "accuracy_vs_params_and_latency.png")
    plt.close()

if __name__ == "__main__":
    main()