"""MLP: architecture, forward pass and backpropagation.

X : (batch, n_features)   -> each ROW is one example.
W : (fan_in, fan_out)     -> weights of a layer.
b : (1, fan_out)          -> bias (broadcast over the batch).
"""
import numpy as np
from .activations import ACTIVATIONS, softmax
from .losses import softmax_cross_entropy_grad


class MLP:
    def __init__(self, layer_sizes, activations, seed=42):
        """Build the network.

        layer_sizes : neurons per layer, including input and output.
        activations : activation per HIDDEN layer. Output uses softmax (in loss/predict).
        """
        assert len(activations) == len(layer_sizes) - 2, (
            "Need one activation per hidden layer (layers - input - output)."
        )
        self.layer_sizes = layer_sizes
        self.activation_names = activations

        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []

        # One weight matrix per consecutive pair of layers.
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            std = np.sqrt(2.0 / fan_in)
            W = rng.normal(0, std, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))  # bias can be zero
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """Run X through the network and return the LOGITS (no softmax).

        Caches the intermediate z and a of each layer in self.cache, since the backpropagation needs them for the chain rule.
        """
        self.cache = {"a": [X], "z": []}  # a[0] is the input itself
        a = X
        n_layers = len(self.weights)

        for i in range(n_layers):
            z = a @ self.weights[i] + self.biases[i]  # linear combination
            self.cache["z"].append(z)

            if i < n_layers - 1:
                # Hidden layer: apply the configured activation.
                act_fn, _ = ACTIVATIONS[self.activation_names[i]]
                a = act_fn(z)
            else:
                # Last layer: return raw logits; softmax happens in the
                # loss (training) or in predict.
                a = z

            self.cache["a"].append(a)

        return a  # logits of the last layer
    
    def backward(self, y_true):
        """Backpropagation: compute the gradients of every W and b.
 
        Requires forward(X) to have run already (uses self.cache).
        Returns (grads_W, grads_b), same order as self.weights.
        """
        n_layers = len(self.weights)
        grads_W = [None] * n_layers
        grads_b = [None] * n_layers
 
        logits = self.cache["a"][-1]
        # Last-layer delta: combined softmax+CE gradient = (probs - y)/N.
        # Starting point for the whole backward pass.
        delta = softmax_cross_entropy_grad(logits, y_true)  # (batch, n_classes)
 
        # Walk the layers back to front.
        for i in reversed(range(n_layers)):
            a_prev = self.cache["a"][i]  # input this layer received
            # Weight grad: (input) x (output error).
            grads_W[i] = a_prev.T @ delta
            # Bias grad: sum the error over the batch.
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
 
            if i > 0:
                # Propagate the error to the previous layer: push it back
                # through the weights (delta @ W.T), then multiply by the
                # previous activation's derivative at z (chain rule).
                _, act_deriv = ACTIVATIONS[self.activation_names[i - 1]]
                z_prev = self.cache["z"][i - 1]
                delta = (delta @ self.weights[i].T) * act_deriv(z_prev)
 
        return grads_W, grads_b



    def predict(self, X):
        """Return the predicted class index for each example in X."""
        logits = self.forward(X)
        probs = softmax(logits)
        return np.argmax(probs, axis=1)