import numpy as np
from neural_net import NeuralNetwork, mse_loss, mse_loss_derivative
from optimizer import Momentum
from dataset import generate_data, get_mini_batches

# Hyperparameters
N_SAMPLES = 1000
N_FEATURES = 1
# 5 layers: 1 input, 3 hidden, 1 output
LAYER_SIZES = [N_FEATURES, 64, 128, 64, N_FEATURES]
LR = 0.0001 # Adjusted learning rate for stability
MOMENTUM = 0.9
EPOCHS = 101 # Run for 100 epochs, print every 10
BATCH_SIZE = 32

def main():
    """
    Main training function.
    """
    # For reproducibility
    np.random.seed(42)

    # Generate data
    X, y = generate_data(N_SAMPLES, N_FEATURES)

    # Initialize network and optimizer
    net = NeuralNetwork(LAYER_SIZES)
    optimizer = Momentum(lr=LR, momentum=MOMENTUM)

    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        num_batches = 0
        for x_batch, y_batch in get_mini_batches(X, y, BATCH_SIZE):
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
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    main()
