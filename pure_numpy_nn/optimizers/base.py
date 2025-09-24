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
            closure (callable): 
                A closure with a `backward` argument that reevaluates the model and returns (loss, gradients).
                If `backward=False`, gradients are not computed and returned as None.

        Closure example:
        ```py
        def closure(backward=True):
            y_pred = net.forward(x_batch)
            loss = mse_loss(y_batch, y_pred)
            grads = None
        
            if backward:
                loss_grad = mse_loss_derivative(y_batch, y_pred)
                grads = net.backward(loss_grad)
        
            return loss, grads
        ```
        """
        raise NotImplementedError
