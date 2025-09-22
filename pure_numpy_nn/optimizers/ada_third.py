import numpy as np
from .base import Optimizer

class AdaThird(Optimizer):
    """
    An experimental third-order optimizer.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, beta3=0.9, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.c = None
        self.prev_grads = None
        self.t = 0

    def step(self, params, grads):
        """
        Performs a single optimization step.
        """
        # Lazy initialization of moments
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            self.c = [np.zeros_like(p) for p in params]
            self.prev_grads = [np.zeros_like(p) for p in params]

        self.t += 1
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Estimate of the change in gradient
            grad_diff = grad - self.prev_grads[i]
            self.prev_grads[i] = grad

            # Update biased third moment estimate
            self.c[i] = self.beta3 * self.c[i] + (1 - self.beta3) * (grad_diff ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # Compute bias-corrected third raw moment estimate
            c_hat = self.c[i] / (1 - self.beta3 ** self.t)

            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + np.sqrt(c_hat) + self.epsilon)
