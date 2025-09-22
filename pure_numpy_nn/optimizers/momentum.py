import numpy as np
from .base import Optimizer

class Momentum(Optimizer):
    """
    Momentum optimizer.
    """
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocities = None

    def step(self, params, grads):
        """
        Performs a single optimization step.
        """
        # Lazy initialization of velocities
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update velocities
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad

            # Update parameters in-place
            param += self.velocities[i]
