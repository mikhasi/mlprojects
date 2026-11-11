import sys
import os
import random
import numpy as np
import torch

# 2. Verify Custom Autograd Implementation Against PyTorch and Numerical Gradients

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from value import Value

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_numerical_gradient(f_func, inputs, eps=1e-6):
    grads = []
    for i in range(len(inputs)):
        inputs_plus = list(inputs)
        inputs_minus = list(inputs)
        
        inputs_plus[i] += eps
        inputs_minus[i] -= eps
        
        f_plus = f_func(*inputs_plus)
        f_minus = f_func(*inputs_minus)
        
        num_grad = (f_plus - f_minus) / (2 * eps)
        grads.append(num_grad)
    return grads

def verify():
    set_seeds(42)
    
    a_val, b_val, c_val = 1.5, -2.5, 4.0
    
    a_custom = Value(a_val, label='a')
    b_custom = Value(b_val, label='b')
    c_custom = Value(c_val, label='c')
    
    out_custom = ((a_custom * b_custom + c_custom) ** 2).relu()
    out_custom.backward()
    
    custom_grads = np.array([a_custom.grad, b_custom.grad, c_custom.grad])
    
    a_torch = torch.tensor(a_val, requires_grad=True, dtype=torch.float64)
    b_torch = torch.tensor(b_val, requires_grad=True, dtype=torch.float64)
    c_torch = torch.tensor(c_val, requires_grad=True, dtype=torch.float64)
    
    out_torch = torch.relu((a_torch * b_torch + c_torch) ** 2).sum()
    out_torch.backward()
    
    torch_grads = np.array([a_torch.grad.item(), b_torch.grad.item(), c_torch.grad.item()])
    
    def f_eval(a, b, c):
        res = (a * b + c) ** 2
        return max(0.0, res)
    
    num_grads = np.array(compute_numerical_gradient(f_eval, [a_val, b_val, c_val]))
    
    diff_custom_vs_torch = np.max(np.abs(custom_grads - torch_grads))
    diff_custom_vs_num = np.max(np.abs(custom_grads - num_grads))
    
    print("--- Gradient Results ---")
    print(f"Custom Gradients:    [da={custom_grads[0]:.6f}, db={custom_grads[1]:.6f}, dc={custom_grads[2]:.6f}]")
    print(f"PyTorch Gradients:   [da={torch_grads[0]:.6f}, db={torch_grads[1]:.6f}, dc={torch_grads[2]:.6f}]")
    print(f"Numerical Gradients: [da={num_grads[0]:.6f}, db={num_grads[1]:.6f}, dc={num_grads[2]:.6f}]")
    
    print("\n--- Difference Checks ---")
    print(f"Max Abs Diff (Custom vs PyTorch):   {diff_custom_vs_torch:.2e}")
    print(f"Max Abs Diff (Custom vs Numerical): {diff_custom_vs_num:.2e}")
    
    assert diff_custom_vs_torch < 1e-6, f"PyTorch diff too high: {diff_custom_vs_torch}"
    assert diff_custom_vs_num < 1e-6, f"Numerical diff too high: {diff_custom_vs_num}"
    
    print("\nSUCCESS: All gradient absolute differences are < 1e-6!")

if __name__ == "__main__":
    verify()