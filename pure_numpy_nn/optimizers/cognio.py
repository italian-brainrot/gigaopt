import numpy as np
from .base import Optimizer
from collections import deque

class CogniO(Optimizer):
    """
    Cognitive Optimizer (CogniO).
    This optimizer uses gradient agreement to modulate the learning rate.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, history_len=5):
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.history_len = history_len
        self.m = None
        self.v = None
        self.grad_history = None
        self.t = 0

    def step(self, params, grads):
        """
        Performs a single optimization step.
        """
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            self.grad_history = [deque(maxlen=self.history_len) for _ in params]

        self.t += 1
        for i, (param, grad) in enumerate(zip(params, grads)):

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Calculate gradient agreement
            if len(self.grad_history[i]) > 0:
                history = np.array(self.grad_history[i])
                avg_past_grad = np.mean(history, axis=0)

                current_grad_flat = grad.flatten()
                avg_past_grad_flat = avg_past_grad.flatten()

                dot_product = np.dot(current_grad_flat, avg_past_grad_flat)
                norm_current = np.linalg.norm(current_grad_flat)
                norm_avg = np.linalg.norm(avg_past_grad_flat)

                if norm_current == 0 or norm_avg == 0:
                    cosine_similarity = 0.0
                else:
                    cosine_similarity = dot_product / (norm_current * norm_avg)

                agreement = (1.0 + cosine_similarity) / 2.0
            else:
                agreement = 1.0

            # Add current grad to history
            self.grad_history[i].append(grad.copy())

            # Update parameters
            param -= self.lr * agreement * m_hat / (np.sqrt(v_hat) + self.epsilon)
