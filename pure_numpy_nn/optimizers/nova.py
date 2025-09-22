import numpy as np
from .base import Optimizer

class Nova(Optimizer):
    """
    A novel optimizer that combines momentum, RMSProp, and a trust factor.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, beta_s=0.9, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta_s = beta_s
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.s = None
        self.t = 0

    def step(self, params, grads):
        """
        Performs a single optimization step.
        """
        # Lazy initialization of moments
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            self.s = [np.zeros_like(p) for p in params]

        self.t += 1
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Update sign consistency
            self.s[i] = self.beta_s * self.s[i] + (1 - self.beta_s) * np.sign(grad)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Compute bias-corrected sign consistency
            s_hat = self.s[i] / (1 - self.beta_s ** self.t)

            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon) * (1 + np.abs(s_hat))
