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