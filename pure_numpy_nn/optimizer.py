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

class Adam(Optimizer):
    """
    Adam optimizer.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        """
        Performs a single optimization step.
        """
        # Lazy initialization of moments
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

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

            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
