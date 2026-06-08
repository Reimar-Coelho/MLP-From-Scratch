"""Cross-entropy loss for classification using softmax.

Cross-entropy measures the “distance” between the predicted distribution (softmax probabilities) and the actual distribution (one-hot labels). The more confident and correct the network is, the lower the loss; the more confident and WRONG it is, the higher the loss (strong penalty).
"""

import numpy as np

from .activations import softmax


def cross_entropy(probs, y_true):
    """
    -1/N * sum( y_true * log(probs) ).
    """
    n = probs.shape[0]
    log_likelihood = -np.log(probs + 1e-12)
    loss = np.sum(y_true * log_likelihood) / n
    return loss


def softmax_cross_entropy_grad(logits, y_true):
    """Gradient of (softmax + cross-entropy) with respect to the logits.

    dL/dlogits = (probs - y_true) / N
    """
    n = logits.shape[0]
    probs = softmax(logits)
    return (probs - y_true) / n
