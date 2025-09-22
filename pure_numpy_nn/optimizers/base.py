import numpy as np

class Optimizer:
    """
    Base class for optimizers.
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        """
        Updates the parameters.
        """
        raise NotImplementedError
