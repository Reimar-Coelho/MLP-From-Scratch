"""Activation functions and their derivatives.

Each activation function has two sides:
- forward: applies the nonlinearity during the forward pass.
- derivative: used in backpropagation. It takes `z` (the pre-activation value, before the function) and returns dphi/dz, which is used in the chain rule.
"""

import numpy as np

# ---------------------------------------------------------------------------
# ReLU: max(0, z). 
# Derivative: 1 if z>0, else 0.
# ---------------------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    # z > 0->True->1.0, z < 0->False->0.0
    return (z > 0).astype(z.dtype)


# ---------------------------------------------------------------------------
# Sigmoid: (0, 1). Good for comparison, but saturate at the ends
# ---------------------------------------------------------------------------
def sigmoid(z):
    # np.clip prevents overflow in exp() for extremely negative or positive values
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)


# ---------------------------------------------------------------------------
# Tanh: (-1, 1).  Derivative: 1 - tanh(z)^2.
# ---------------------------------------------------------------------------
def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2


# ---------------------------------------------------------------------------
# Softmax: transforms the logits from the final layer into a probability distribution (summing to 1). Used at the OUTPUT layer, along with cross-entropy.
# ---------------------------------------------------------------------------
def softmax(z):
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z_shift)
    return exp / np.sum(exp, axis=1, keepdims=True)


ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
}