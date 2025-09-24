import numpy as np

class Optimizer:
    """
    Base class for optimizers.
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, closure):
        """
        Performs a single optimization step.
        Arguments:
            params (list of ndarray): parameters to be optimized.
            closure (callable): A closure with a `backward` argument
            that reevaluates the model and returns the loss and gradients.
            If `backward=False`, Gradients are not computed and returned as None.
        """
        raise NotImplementedError
