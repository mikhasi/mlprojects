# Lab 02 Report: Scratch Multilayer Perceptron on Digits Dataset

## 1. Goal

The primary goal of this project is to implement a Multilayer Perceptron (MLP) entirely from scratch using Python and NumPy (without leveraging autograd or deep learning frameworks like PyTorch for model training). Specifically, the pipeline aims to:

* Load, scale, and preprocess the scikit-learn `load_digits` dataset with a stratified 70/30 train-test split and one-hot encoded labels.

* Design and initialize a two-hidden-layer neural network with ReLU activations and a softmax output layer.

* Derive analytical gradients using the chain rule and verify correctness using numerical gradient checking.

* Train the model using mini-batch gradient descent across multiple hyperparameter configurations ($W \in \{16, 64\}$, $\text{learning rates} \in \{0.001, 0.1, 0.1\}$).

* Benchmark forward and backward execution timings to analyze computational complexity.

## 2. Method

* **Data Pipeline**: The raw 64-dimensional digit features are scaled by dividing by $16.0$ to map inputs to $[0, 1]$. Labels are one-hot encoded into 10 classes, and split using a stratified 70/30 ratio ($\text{random\_state}=42$). Data storage is managed locally in `root/data/` and ignored via `.gitignore`.

* **Architecture & Initialization**:

  * Input: 64 features $\rightarrow$ Hidden 1 ($W$ units) $\rightarrow$ ReLU $\rightarrow$ Hidden 2 ($W$ units) $\rightarrow$ ReLU $\rightarrow$ Output (10 units) $\rightarrow$ Softmax.

  * Tested hidden widths: $W = 16$ and $W = 64$.

  * Weights initialized from $\mathcal{N}(0, 0.01^2)$ and biases set to zero.

* **Optimization & Training**:

  * Trained for 50 epochs with a mini-batch size of 32 using standard Stochastic Gradient Descent (SGD).

  * Evaluated loss curves and recorded final test set accuracies for all 6 hyperparameter combinations.

* **Benchmarking & Validation**:

  * Analytical gradients derived layer-by-layer and verified against numerical approximations ($\epsilon = 10^{-5}$).

  * Forward and backward pass latencies measured over 100 runs following a warm-up phase (reporting medians).

## 3. Results Table or Plot

### Summary Table of Final Test Accuracies

| Hidden Width ($W$) | Learning Rate ($\eta$) | Final Test Accuracy (%) | 
| ----- | ----- | ----- | 
| 16 | 0.001 | 82.41% | 
| 16 | 0.01 | 93.15% | 
| 16 | 0.1 | 96.30% | 
| 64 | 0.001 | 85.19% | 
| 64 | 0.01 | 95.37% | 
| 64 | 0.1 | 97.59% | 

### Loss Curves Visualization

The training loss curves across all tested configurations have been compiled and saved to:
`results/loss_curves.png`

## 4. Three Conclusions

1. **Learning Rate Matters More Than Network Size**: Increasing the learning rate from $0.001$ to $0.1$ made the model train much faster and reach higher test accuracy. Picking a good learning rate had a bigger impact on performance than making the hidden layer bigger (16 vs. 64 units).

2. **Backward Pass Is Slower Than Forward Pass**: The backward pass takes about 2 to 3 times longer than the forward pass. This is because the forward pass only does one matrix multiplication per layer, while the backward pass has to do two: one to update the weights ($dW$) and one to pass error gradients back to the previous layer ($dA$).

3. **Building from Scratch Works**: Writing the code manually in NumPy shows that standard cross-entropy loss, ReLU, and Softmax work as expected without deep learning libraries. Getting over 97% test accuracy proves that the manual math and update logic are correct.