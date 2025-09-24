import numpy as np
from .base import Optimizer

class Athena(Optimizer):
    """
    Athena optimizer.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, beta3=0.9999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.s = None
        self.t = 0

    def step(self, params, closure):
        """
        Performs a single optimization step.
        """
        loss, grads = closure()

        # Lazy initialization of moments
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            self.s = [np.zeros_like(p) for p in params]

        self.t += 1
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate (short-term)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            # Update biased second raw moment estimate (long-term)
            self.s[i] = self.beta3 * self.s[i] + (1 - self.beta3) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate (short-term)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # Compute bias-corrected second raw moment estimate (long-term)
            s_hat = self.s[i] / (1 - self.beta3 ** self.t)

            # Use the max of the two second moment estimates
            v_final = np.maximum(v_hat, s_hat)

            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_final) + self.epsilon)

        return loss
