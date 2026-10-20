import os
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from dataset import load_or_download_digits

class ScratchMLP:
    def __init__(self, W=16):
        self.W = W
        self.W1 = np.random.normal(0, 0.01, (64, W))
        self.b1 = np.zeros(W)
        
        self.W2 = np.random.normal(0, 0.01, (W, W))
        self.b2 = np.zeros(W)
        
        self.W3 = np.random.normal(0, 0.01, (W, 10))
        self.b3 = np.zeros(10)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.maximum(0, self.Z2)
        
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        
        exp_scores = np.exp(self.Z3 - np.max(self.Z3, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return self.probs

    def compute_loss(self, y_true):
        N = y_true.shape[0]
        clipped_probs = np.clip(self.probs, 1e-15, 1.0 - 1e-15)
        loss = -np.sum(y_true * np.log(clipped_probs)) / N
        return loss

    def backward(self, X, y_true):
        N = X.shape[0]
        
        dZ3 = (self.probs - y_true) / N
        
        self.dW3 = np.dot(self.A2.T, dZ3)
        self.db3 = np.sum(dZ3, axis=0)
        
        dA2 = np.dot(dZ3, self.W3.T)
        
        dZ2 = dA2 * (self.Z2 > 0)
        
        self.dW2 = np.dot(self.A1.T, dZ2)
        self.db2 = np.sum(dZ2, axis=0)
        
        dA1 = np.dot(dZ2, self.W2.T)
        
        dZ1 = dA1 * (self.Z1 > 0)
        
        self.dW1 = np.dot(X.T, dZ1)
        self.db1 = np.sum(dZ1, axis=0)

def gradient_check(model, X, y, num_checks=10, eps=1e-5):
    model.forward(X)
    model.backward(X, y)
    
    params = [
        ('W1', model.W1, model.dW1),
        ('b1', model.b1, model.db1),
        ('W2', model.W2, model.dW2),
        ('b2', model.b2, model.db2),
        ('W3', model.W3, model.dW3),
        ('b3', model.b3, model.db3)
    ]
    
    all_coords = []
    for name, param, grad in params:
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            all_coords.append((name, param, grad, idx))
            it.iternext()
            
    selected_coords = random.sample(all_coords, min(num_checks, len(all_coords)))
    
    success_count = 0
    print("\n--- Running Numerical Gradient Check ---")
    for i, (name, param, grad, idx) in enumerate(selected_coords):
        analytic_grad = grad[idx]
        
        orig_val = param[idx]
        param[idx] = orig_val + eps
        model.forward(X)
        loss_high = model.compute_loss(y)
        
        param[idx] = orig_val - eps
        model.forward(X)
        loss_low = model.compute_loss(y)
        
        param[idx] = orig_val
        
        numerical_grad = (loss_high - loss_low) / (2 * eps)
        
        rel_error = np.abs(analytic_grad - numerical_grad) / max(np.abs(analytic_grad), np.abs(numerical_grad), 1e-8)
        
        passed = rel_error < 1e-6
        if passed:
            success_count += 1
            
        print(f"[{i+1}] Param: {name} at {idx} | Analytic: {analytic_grad:.6e} | Numerical: {numerical_grad:.6e} | Rel Error: {rel_error:.2e} | Passed: {passed}")

    print(f"\nGradient Check Result: {success_count}/{len(selected_coords)} weights passed (Requirement: >= 9/10).")
    return success_count >= 9

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_or_download_digits()
    X_sample = X_train[:32]
    y_sample = y_train[:32]
    
    for W in [16, 64]:
        print(f"\n==================== Testing with W = {W} ====================")
        mlp = ScratchMLP(W=W)
        
        probs = mlp.forward(X_sample)
        print(f"Output shape: {probs.shape} (Expected: (32, 10))")
        row_sums = np.sum(probs, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Row sums do not equal 1 within 1e-6!"
        print("✓ All output rows sum to 1 within 1e-6.")
        
        loss = mlp.compute_loss(y_sample)
        print(f"Untrained network loss: {loss:.4f} (Expected ~ ln(10) = 2.3026)")
        
        gradient_check(mlp, X_sample, y_sample, num_checks=10)