import numpy as np
from .base import Optimizer

class CognitiveDissonanceOptimizer(Optimizer):
    """
    Cognitive Dissonance Optimizer.
    This optimizer uses the dissonance between short-term and long-term momentum
    to modulate the learning rate.
    """
    def __init__(self, lr=0.001, beta1_short=0.9, beta1_long=0.99, beta2=0.999, epsilon=1e-8, k=2):
        super().__init__(lr=lr)
        self.beta1_short = beta1_short
        self.beta1_long = beta1_long
        self.beta2 = beta2
        self.epsilon = epsilon
        self.k = k
        self.m_short = None
        self.m_long = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        """
        Performs a single optimization step.
        """
        if self.m_short is None:
            self.m_short = [np.zeros_like(p) for p in params]
            self.m_long = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        for i, (param, grad) in enumerate(zip(params, grads)):

            # Update biased momentum estimates
            self.m_short[i] = self.beta1_short * self.m_short[i] + (1 - self.beta1_short) * grad
            self.m_long[i] = self.beta1_long * self.m_long[i] + (1 - self.beta1_long) * grad

            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected momentum estimates
            m_short_hat = self.m_short[i] / (1 - self.beta1_short ** self.t)
            m_long_hat = self.m_long[i] / (1 - self.beta1_long ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Calculate cognitive dissonance
            m_short_flat = m_short_hat.flatten()
            m_long_flat = m_long_hat.flatten()

            dot_product = np.dot(m_short_flat, m_long_flat)
            norm_short = np.linalg.norm(m_short_flat)
            norm_long = np.linalg.norm(m_long_flat)

            if norm_short == 0 or norm_long == 0:
                cosine_similarity = 0.0
            else:
                cosine_similarity = dot_product / (norm_short * norm_long)

            dissonance = (1.0 - cosine_similarity) / 2.0

            # Adapt learning rate based on dissonance
            lr_factor = (1.0 - dissonance) ** self.k

            # Update parameters
            param -= self.lr * lr_factor * m_short_hat / (np.sqrt(v_hat) + self.epsilon)
