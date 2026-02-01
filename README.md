# NoProp: Diffusion-Based Learning without End-to-End Propagation

This repository contains an **experimental PyTorch implementation** of the **NoProp** algorithm, exploring its application in two distinct domains: image classification and time-series forecasting.

**Reference Paper:**

> **NoProp: Training Neural Networks without Back-propagation or Forward-propagation**
> *Qinyu Li, Yee Whye Teh, Razvan Pascanu (2025)*
> [arXiv:2503.24322](https://arxiv.org/abs/2503.24322)

**Supplementary Material:**

> 📄 **LaTeX Slides:** Slides are available in this repository to assist in understanding the math-heavy aspects of the paper and the underlying diffusion mechanics.

---

## 📂 Project Implementations

This repository is divided into two main experimental applications of the NoProp architecture:

### 1. NoProp for Classification (MNIST)

A direct implementation of the paper's core concept using the MNIST dataset.

* **Goal:** Classify digits using a forward diffusion process on label embeddings.
* **Architecture:** A sequential **BlockNN** model where gradients are detached between steps.
* **Outcome:** Demonstrates how the model learns to reconstruct labels from noisy states without end-to-end backpropagation.

### 2. NoProp for Time-Series Prediction (Autoencoder)

An extension of the NoProp algorithm applied to sequential data.

* **Goal:** Forecast time-series data by embedding the NoProp architecture within an **Autoencoder (AE)** framework.
* **Architecture:** Uses the diffusion-based training principles to learn latent representations of time-series windows, avoiding standard propagation through the entire temporal sequence.

---

## 🧠 Model Architecture (MNIST Implementation)

The classification model consists of a sequence of **BlockNN** modules ( steps).

### The BlockNN Module

Each block consists of two main components that merge information:

1. **CNN Image Encoder:**
* Input: MNIST Image ()
* Layers: 2x Conv2d layers, ReLU activations, MaxPool2d.
* Output: 32-dimensional image feature vector.


2. **Label Embedding MLP:**
* Input: Noisy label state  from the diffusion process.
* Structure: Linear layers processing the embedding dimension (10).


3. **Fusion MLP:**
* Concatenates image features and processed label embeddings.
* Outputs the denoised label estimate .



**Key Mechanic: Gradient Detachment**
During training, the input to the current block  is detached from the computation graph (`z[t-1].detach()`). This ensures that the optimizer updates parameters for the current block without backpropagating errors to previous blocks.

## 📊 Training & Results (MNIST)

The model is trained on the **MNIST** dataset (60,000 train samples, 10,000 test samples).

* **Diffusion Steps ():** 10
* **Optimizer:** Adam (LR: 0.001)
* **Loss:** Weighted MSE based on Signal-to-Noise Ratio (SNR).

The model achieves high accuracy rapidly:

| Epoch | Avg Loss | Train Accuracy | Test Accuracy |
| --- | --- | --- | --- |
| **1** | 0.5416 | 94.31% | 94.62% |
| **5** | 0.0717 | **98.15%** | **98.15%** |

## 🚀 Usage

### Prerequisites

* Python 3.x
* PyTorch, Torchvision
* Matplotlib, Numpy, Torchsummary

### Running the Code

1. Clone the repository.
2. Open `NoProp.ipynb` in Jupyter Notebook or Google Colab.
3. Run all cells to download the MNIST dataset and start training.

```python
# To run inference on a single batch after training:
x_test, y_test = next(iter(test_loader))
prediction = classify_batch(x_test, blockNN)

```

## 📂 File Structure

* `NoProp.ipynb`: Main notebook containing the MNIST classification implementation.
* `NoProp_TimeSeries.ipynb` *(or relevant filename)*: Implementation of the Autoencoder for time-series.
* `data/`: Directory for datasets.
* `slides/`: LaTeX slides explaining the math and diffusion process.
