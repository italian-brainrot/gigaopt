import numpy as np
from .neural_net import NeuralNetwork, mse_loss, mse_loss_derivative
from .optimizers import Adam, AdaThird, Nova
from .dataset import generate_data, get_mini_batches

# Hyperparameters
N_SAMPLES = 1000
N_FEATURES = 32
# 5 layers: 1 input, 3 hidden, 1 output
LAYER_SIZES = [N_FEATURES, 64, 128, 64, N_FEATURES]
EPOCHS = 101 # Run for 100 epochs, print every 10
BATCH_SIZE = 32

def run_experiment(optimizer_class, optimizer_params, n_samples, n_features, layer_sizes, epochs, batch_size):
    """
    Runs a training experiment for a given optimizer.
    """
    # For reproducibility
    np.random.seed(42)

    # Generate data
    X, y = generate_data(n_samples, n_features)

    # Initialize network and optimizer
    net = NeuralNetwork(layer_sizes)
    optimizer = optimizer_class(**optimizer_params)

    # Training loop
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for x_batch, y_batch in get_mini_batches(X, y, batch_size):
            # Forward pass
            y_pred = net.forward(x_batch)

            # Compute loss
            loss = mse_loss(y_batch, y_pred)
            epoch_loss += loss
            num_batches += 1

            # Backward pass
            loss_grad = mse_loss_derivative(y_batch, y_pred)
            grads = net.backward(loss_grad)

            # Update weights
            params = net.get_params()
            optimizer.step(params, grads)

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
    return losses

def main():
    """
    Main comparison function.
    """
    # Adam experiment
    adam_params = {'lr': 0.001}
    adam_losses = run_experiment(Adam, adam_params, N_SAMPLES, N_FEATURES, LAYER_SIZES, EPOCHS, BATCH_SIZE)

    # AdaThird experiment
    adathird_params = {'lr': 0.003, 'beta3': 0.99}
    adathird_losses = run_experiment(AdaThird, adathird_params, N_SAMPLES, N_FEATURES, LAYER_SIZES, EPOCHS, BATCH_SIZE)

    # Nova experiment
    nova_params = {'lr': 0.001}
    nova_losses = run_experiment(Nova, nova_params, N_SAMPLES, N_FEATURES, LAYER_SIZES, EPOCHS, BATCH_SIZE)

    # Print comparison table
    print("Optimizer Performance Comparison")
    print("-" * 70)
    print(f"{'Epoch':<10}{'Adam Loss':<20}{'AdaThird Loss':<20}{'Nova Loss':<20}")
    print("-" * 70)
    for epoch in range(0, EPOCHS, 10):
        print(f"{epoch:<10}{adam_losses[epoch]:<20.4f}{adathird_losses[epoch]:<20.4f}{nova_losses[epoch]:<20.4f}")
    print("-" * 70)
    print(f"{'Final Loss':<10}{adam_losses[-1]:<20.4f}{adathird_losses[-1]:<20.4f}{nova_losses[-1]:<20.4f}")
    print("-" * 70)

if __name__ == "__main__":
    main()
