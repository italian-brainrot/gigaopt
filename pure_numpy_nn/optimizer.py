import numpy as np

class Optimizer:
    """
    Base class for optimizers.
    """
    def __init__(self, net, lr=0.01):
        self.net = net
        self.lr = lr

    def step(self, grads):
        """
        Updates the network's parameters.
        """
        raise NotImplementedError

    def zero_grad(self):
        """
        In a more complex scenario, we might zero out gradients here.
        For this simple numpy implementation, gradients are recalculated in each backprop pass.
        """
        pass

class Momentum(Optimizer):
    """
    Momentum optimizer.
    """
    def __init__(self, net, lr=0.01, momentum=0.9):
        super().__init__(net, lr)
        self.momentum = momentum
        # Initialize velocities for each parameter
        self.velocities = {f'W{i}': np.zeros_like(w) for i, w in enumerate(net.weights)}
        self.velocities.update({f'b{i}': np.zeros_like(b) for i, b in enumerate(net.biases)})

    def step(self, grads):
        """
        Performs a single optimization step.
        """
        for i in range(len(self.net.weights)):
            w_key = f'W{i}'
            b_key = f'b{i}'

            # Update velocities
            self.velocities[w_key] = self.momentum * self.velocities[w_key] - self.lr * grads[w_key]
            self.velocities[b_key] = self.momentum * self.velocities[b_key] - self.lr * grads[b_key]

            # Update weights and biases
            self.net.weights[i] += self.velocities[w_key]
            self.net.biases[i] += self.velocities[b_key]
