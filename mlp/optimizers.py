"""Optimizers: weight update rules from gradients.
"""
import numpy as np


class SGD:
    """Stochastic Gradient Descent, with optional momentum.

    Base rule: param <- param - lr * grad.
    """

    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.v_W = None  # velocities (only used if momentum > 0)
        self.v_b = None

    def step(self, weights, biases, grads_W, grads_b):
        # First call: init velocities to zero.
        if self.v_W is None:
            self.v_W = [np.zeros_like(W) for W in weights]
            self.v_b = [np.zeros_like(b) for b in biases]

        for i in range(len(weights)):
            # v = momentum * v - lr * grad
            self.v_W[i] = self.momentum * self.v_W[i] - self.lr * grads_W[i]
            self.v_b[i] = self.momentum * self.v_b[i] - self.lr * grads_b[i]
            # param += v
            weights[i] += self.v_W[i]
            biases[i] += self.v_b[i]


class Adam:
    """Adam: per-parameter learning rate from estimates of the 1st moment (mean of gradients) and 2nd moment (mean of squares). Usually converges faster and is less sensitive to the initial lr.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_W = self.v_W = None
        self.m_b = self.v_b = None
        self.t = 0  # step counter (for bias correction)

    def step(self, weights, biases, grads_W, grads_b):
        if self.m_W is None:
            self.m_W = [np.zeros_like(W) for W in weights]
            self.v_W = [np.zeros_like(W) for W in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]

        self.t += 1
        for i in range(len(weights)):
            # 1st moment (moving average of the gradient)
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * grads_W[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
            # 2nd moment (moving average of the squared gradient)
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * grads_W[i] ** 2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grads_b[i] ** 2
            # Bias correction (moments start at 0, underestimated early on).
            m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            weights[i] -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
            biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)