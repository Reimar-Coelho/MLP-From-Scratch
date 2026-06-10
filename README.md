# MLP-From-Scratch

Multilayer Perceptron (MLP) made from scratch. A MLP is a type of neural network that uses layers of connected nodes to learn patterns. It gets its name from having multiple layers — typically an input layer, one or more middle (hidden) layers, and an output layer. [More about MLP here](https://medium.com/data-science/multilayer-perceptron-explained-a-visual-guide-with-mini-2d-dataset-0ae8100c5d1c).

## Getting Started

These instructions will give you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

Requirements for the software to run

- [Git](https://help.github.com/en/articles/set-up-git)
- [Python](https://www.python.org/downloads/)
- [Jupyter](https://jupyter.org/install)

### Installing

A step by step series of examples that tell you how to get a development environment running

    pip install -r requirements.txt

## Walkthrough

This is my step by step journey to complete this project.

### Roadmap

| Step | What was done                                                                 |
|------|-------------------------------------------------------------------------------|
| 1    | Setting up the environment (venv, requirements.txt, __init__.py)              |
| 2    | activations.py — ReLU, sigmoid, tanh, softmax + derivatives                   |
| 3    | cross-entropy + gradient combined with softmax                                |
| 4    | network.py — forward pass (arbitrary number of layers, weight initialization) |
| 5    | network.py — backpropagation (chain rule)                                     |
| 6    | optimizers.py — SGD (and optionally: Momentum/Adam)                           |
| 7    | Validation — XOR + numeric gradient check                                      |
| 8    | MNIST — data loading, preprocessing and mini-batch SGD training                |
| 9    | Results — training curves, experiment comparison and confusion matrix          |

### 1. Repo setup

In the first step, i created the repository structure with the necessary files and folders, installed all the libraries, and established a stable base to work from.

### 2. Activation Functions (activations.py)

In the second step, I implemented the functions that introduce nonlinearity into the network (without them, stacking layers would be pointless—it would simply become a single linear transformation) and their derivatives (which are necessary for backpropagation).

### 3. Loss function (losses.py)

In the third step, I will create the loss functions to measure how far off the network is (cross-entropy) and calculate the gradient of the loss with respect to the logits—the starting point for backpropagation.

### 4. Forward pass (network.py)

In the fourth step, I built the MLP class, which constructs a network with an arbitrary number of layers and performs the forward pass—passing the input through each layer until it produces the output logits. We also resolved the issue of weight initialization (which is where the infamous “all zeros” bug lies).

### 5. Backpropagation (network.py)

In the fifth step, I calculated the gradient of the loss with respect to each weight and bias in the network by backpropagating the output error to the input (backpropagation rule). This is what allows the network to learn.

### 6. Optimizer (optimizers.py)

In the sixth step, I implemented the optimizers, which update the weights based on the backpropagation gradients. The essential algorithm is SGD (W ← W − lr · dW). I implemented SGD with optional momentum and an Adam optimizer.

### 7. Validation: XOR + Gradient Check (notebooks/experimentos.ipynb)

In the seventh step, before scaling up to MNIST, I validated the whole network on the smallest non-linear problem: XOR. XOR is not linearly separable, so only a network with hidden layers can solve it — a good minimal test for the forward/backward/optimizer working together.

I also implemented a numeric gradient check, which is the part that actually proves backpropagation is correct. The idea is to approximate the gradient of each weight directly from the definition of a derivative, `(L(p + e) - L(p - e)) / (2e)`, and compare it against the analytic gradient produced by `backward`. If they match (difference below `1e-5`), the chain rule was implemented correctly.

The result was a maximum difference of `~6e-12` between the analytic and numeric gradients (well below the `1e-5` threshold), and the network learned XOR perfectly (final loss `~0.0003`, predictions `[0 1 1 0]`). With the math validated, I had the confidence to scale the same network to MNIST.

### 8. MNIST — Data & Training (notebooks/experimentos.ipynb)

In the eighth step, I scaled the validated network to the real task: classifying the 10 handwritten digits of MNIST. I load the dataset from the same `.npz` file keras uses (downloaded once and cached locally, ignored by git), then preprocess it: each 28x28 image is flattened into a 784-vector, pixels are normalized to `[0, 1]`, and the labels are one-hot encoded to match the softmax + cross-entropy output. Normalizing the inputs matters — feeding raw `0–255` values makes the pre-activations large and destabilizes training.

For training I wrote a mini-batch loop: every epoch the data is shuffled and split into batches of 64, and for each batch I run forward -> backward -> optimizer step. Shuffling each epoch is what makes the gradient descent *stochastic*, which helps the network escape bad local patterns and generalize better.

With the architecture `784 -> 128 -> 64 -> 10` (two ReLU hidden layers), plain SGD at `lr = 0.1`, batch size 64 and 15 epochs, the network reached **97.8% test accuracy** in about 8 seconds — comfortably above the 92% target.

### 9. Results (notebooks/experimentos.ipynb)

In the ninth step, I generated the plots and analysis. The training curves (`results/training_curves.png`) show the loss falling steadily while train and test accuracy rise together and stay close, which means the network is learning without heavy overfitting.

**Experiment comparison.** I compared five configurations, changing one factor at a time (10 epochs each, same seed):

| Config | Architecture | Activation | Optimizer | Test acc |
|--------|--------------|------------|-----------|----------|
| A (baseline) | 784-128-64-10 | ReLU | SGD lr=0.1 | 0.9764 |
| B | 784-128-64-10 | ReLU | SGD lr=0.01 | 0.9531 |
| C | 784-128-64-10 | tanh | SGD lr=0.1 | 0.9755 |
| D | 784-128-64-10 | ReLU | Adam lr=1e-3 | 0.9769 |
| E | 784-256-10 | ReLU | SGD lr=0.1 | 0.9772 |

The clearest takeaway is the **learning rate**: dropping it from 0.1 to 0.01 (config B) cost over 2 points of accuracy in the same number of epochs, because the smaller steps simply hadn't converged yet. ReLU and tanh performed almost the same here; Adam reached a similar accuracy to SGD but converged faster in the early epochs. A single wider hidden layer (E) was competitive with two narrower ones.

**Confusion matrix.** On the test set, the most confused pairs were `5 -> 3` (22 times), `7 -> 2`, `8 -> 3` and `9 -> 4`. These are all visually similar digits, so the mistakes are intuitive rather than random — a good sign that the network learned meaningful features. The matrix is saved at `results/confusion_matrix.png`.