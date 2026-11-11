# Lab 5: Frameworks and automatic differentiation

## 1. Goal
The goal of this lab was to implement a scalar automatic-differentiation engine with graph construction, perform topological backpropagation with correct gradient accumulation, validate gradients against PyTorch autograd and numerical estimates, and assess eager versus compiled or traced execution and the cost of non-contiguous tensors.

## 2. Method
* **Custom Autograd Engine (`value.py`)**:
  * Each `Value` node stores its scalar data `.data`, gredient `.grad`, parent nodes `_prev`, and local derivative rule `_backward`.
  * Implemented operations: addition, multiplication, powers, and ReLU.
  * Backpropagation `backward()` topologically sorts the graph, sets the output gradient to `1.0`, and visits nodes in reverse order to accumulate gradients without overwriting.
* **Gradient Verification (`verify.py`)**: 
  * Verified expression `((a*b + c)**2).relu().sum()` across custom Autograd, PyTorch Autograd and numerical gradients with max absolute
difference `<1e-6`.
* **Model Training and Benchmarking (`train.py`)**: 
  * Trained a 2 $\rightarrow$ 8 $\rightarrow$ 1 MLP on 100 samples of `make_moons` dataset for 20 epochs using BCE loss for custom autograd and a SGD loss for PyTorch, for benchmarking.
  * 1 warm-up run followed by 5 timed repetitions, reporting the median time.
* **PyTorch Inspection and Analysis (`inspect_pytorch.py`)**: 
  * Used `torch.fx` to trace model operations and inspect nodes.
  * Measured eager and traced execution with a fallback to `torch.jit.trace` if compilation is unsupported.
  * Measured operation speed of `.sum()` on contiguous and non-contiguous tensors.

## 3. Results Table and plot

### A. Gradient Verification (`((a*b + c)**2).relu()`)
| Method | da | db | dc | Max Absolute Difference vs. Custom |
| :--- | :---: | :---: | :---: | :---: |
| **Custom Autograd** | -1.250000 | 0.750000 | 0.500000 | 0 |
| **PyTorch Autograd** | -1.250000 | 0.750000 | 0.500000 | $< 10^{-6}$ |
| **Numerical gradients** | -1.250000 | 0.750000 | 0.500000 | $< 10^{-6}$ |

### B. Training Benchmark of `make_moons` dataset (median of five runs)
| Framework | Median Training Time |
| :--- | :---: |
| **Custom Engine** | $\sim 1186.04 \text{ ms}$ |
| **PyTorch** | $\sim 14.22 \text{ ms}$ |

### C. Eager vs torch.jit.trace execution time (Median of 5 runs)
| Execution type | Median Execution Time |
| :--- | :---: |
| **Eager Mode** | $\sim 7.504 \text{ ms}$ |
| **torch.jit.trace** | $\sim 8.901 \text{ ms}$ |

### D. Tensor Memory Contiguity Performance ($4000 \times 4000$ Tensor Sum)
| Layout State | Median Execution Time |
| :--- | :---: |
| **Contiguous Sum** | $\sim 189.664 \text{ ms}$ |
| **Non-Contiguous Sum** | $\sim 188.566 \text{ ms}$ |

## 4. Three Conclusions
1. **Building the Autograd shows neural networks clearly**: Building an autograd engine from scratch shows how many processes in netural networks really work step-by-step. 
2. **Python libraries are way faster at some computations**: A scalar autograd engine can be compared in some computations to the vectorized frameworks like PyTorch, but in other benchmarks it is orders-of-magnitude slower becauase Pytorch analyzes models ahead of time rather than running operations line-by-line.
3. **Many benchmarks show almost no diffrence**: Many of the benchmarks give very similar results for both sides because, as I understand, models and datasets used are pretty small, so math takes only a few millisaconds either way.