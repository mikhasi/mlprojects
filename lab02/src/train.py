import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from dataset import load_or_download_digits
from implementation import ScratchMLP

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_or_download_digits()
    
    learning_rates = [0.001, 0.01, 0.1]
    widths = [16, 64]
    batch_size = 32
    epochs = 50
    
    results = {}
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "../results")
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    for W in widths:
        for lr in learning_rates:
            print(f"\n--- Training MLP: Width = {W}, Learning Rate = {lr} ---")
            mlp = ScratchMLP(W=W)
            train_losses = []
            
            num_samples = X_train.shape[0]
            
            for epoch in range(epochs):
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
                
                epoch_loss = 0.0
                num_batches = int(np.ceil(num_samples / batch_size))
                
                for b in range(num_batches):
                    start_idx = b * batch_size
                    end_idx = min(start_idx + batch_size, num_samples)
                    
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]
                    
                    mlp.forward(X_batch)
                    batch_loss = mlp.compute_loss(y_batch)
                    epoch_loss += batch_loss * (end_idx - start_idx)
                    
                    mlp.backward(X_batch, y_batch)
                    
                    mlp.W1 -= lr * mlp.dW1
                    mlp.b1 -= lr * mlp.db1
                    mlp.W2 -= lr * mlp.dW2
                    mlp.b2 -= lr * mlp.db2
                    mlp.W3 -= lr * mlp.dW3
                    mlp.b3 -= lr * mlp.db3
                
                avg_epoch_loss = epoch_loss / num_samples
                train_losses.append(avg_epoch_loss)
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")
            
            test_probs = mlp.forward(X_test)
            test_preds = np.argmax(test_probs, axis=1)
            true_labels = np.argmax(y_test, axis=1)
            test_accuracy = np.mean(test_preds == true_labels) * 100.0
            print(f"Final Test Accuracy (W={W}, lr={lr}): {test_accuracy:.2f}%")
            
            results[(W, lr)] = {
                "train_losses": train_losses,
                "test_accuracy": test_accuracy
            }
            
            label_str = f"W={W}, lr={lr} (Acc: {test_accuracy:.1f}%)"
            plt.plot(range(1, epochs + 1), train_losses, label=label_str)

    plt.title("Training Loss Curves across Widths and Learning Rates")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, "loss_curves.png")
    plt.savefig(plot_path)
    print(f"\nLoss curves saved successfully to {plot_path}")
    plt.close()

if __name__ == "__main__":
    train_and_evaluate()