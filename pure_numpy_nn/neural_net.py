import numpy as np

class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, d_out):
        return d_out * (self.x > 0)

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # Initialize weights with small random values
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1)
            # Initialize biases to zeros
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
            # Add ReLU activation for all but the last layer
            if i < len(layer_sizes) - 2:
                self.layers.append(ReLU())

    def forward(self, x):
        self.activations = [x]
        current_activation = x
        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]

            # Apply activation function
            if i < len(self.layers):
                current_activation = self.layers[i].forward(z)
            else:
                # No activation on the output layer
                current_activation = z
            self.activations.append(current_activation)
        return current_activation

    def backward(self, d_output):
        grads = {}
        d_current = d_output

        for i in reversed(range(len(self.weights))):
            # Gradient of the activation function
            if i < len(self.layers):
                d_current = self.layers[i].backward(d_current)

            # Gradients of weights and biases
            grads[f'W{i}'] = np.dot(self.activations[i].T, d_current)
            grads[f'b{i}'] = np.sum(d_current, axis=0, keepdims=True)

            # Gradient to propagate to the previous layer
            d_current = np.dot(d_current, self.weights[i].T)

        return grads

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]
