import argparse
import math
import time

import numpy as np

from .dataset import generate_data, get_mini_batches
from .mbs import mbs_minimize
from .neural_net import NeuralNetwork, mse_loss, mse_loss_derivative
from .optimizers import Adam, SGD

# Hyperparameters
N_SAMPLES = 1000
N_FEATURES = 32
# 5 layers: 1 input, 3 hidden, 1 output
LAYER_SIZES = (N_FEATURES, 64, 128, 64, N_FEATURES)
EPOCHS = 100
BATCH_SIZE = 32

def run_experiment(optimizer_class, optimizer_params, lr_low=1e-5, lr_high=1, n_samples=N_SAMPLES, n_features=N_FEATURES, layer_sizes=LAYER_SIZES, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Runs a training experiment for a given optimizer, with learning rate tuning.
    """
    # For reproducibility
    np.random.seed(0)

    # Generate data
    X, y = generate_data(n_samples, n_features)
    num_trials = 0

    def objective(lr):
        nonlocal num_trials
        num_trials += 1
        start_sec = time.perf_counter()

        # Create a copy of params and update lr
        current_params = optimizer_params.copy()
        current_params['lr'] = lr

        # Initialize network and optimizer
        net = NeuralNetwork(layer_sizes)
        optimizer = optimizer_class(**current_params)

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            for i, (x_batch, y_batch) in enumerate(get_mini_batches(X, y, batch_size)):
                params = net.get_params()

                def closure():
                    y_pred = net.forward(x_batch)
                    loss = mse_loss(y_batch, y_pred)
                    loss_grad = mse_loss_derivative(y_batch, y_pred)
                    grads = net.backward(loss_grad)
                    return loss, grads

                loss = optimizer.step(params, closure)
                epoch_loss += loss
                num_batches += 1

                if num_trials == 1 and epoch == 0 and i == 0:
                    print(f"first batch took {(time.perf_counter() - start_sec):.2f} seconds")

            if num_trials == 1 and epoch == 0:
                print(f"first epoch took {(time.perf_counter() - start_sec):.2f} seconds")

            avg_loss = epoch_loss / num_batches

        if num_trials == 1:
            print(f"first trial took {(time.perf_counter() - start_sec):.2f} seconds")

        return avg_loss


    grid = np.linspace(math.log10(lr_low), math.log10(lr_high), 6)
    start_sec = time.perf_counter()
    ret = mbs_minimize(objective, grid=grid, num_binary=6, step=1, log_scale=True)
    print(f"lr search took {(time.perf_counter() - start_sec):.2f} seconds")

    trials = sorted([(lr, loss[0]) for lr,loss in ret.items()], key=lambda x: x[1])

    best_lr, best_value = trials[0]
    print(f"Best LR for {optimizer_class.__name__}: {best_lr:.5f}")
    print(f"Best value: {best_value:.5f}")
    print("---")

    return best_value

def main():
    """
    By default, this script will print the hardcoded results of the last run.
    To run an experiment, add `run_experiment` with your optimizer to hardcoded ``losses`` dictionary,
    after testing it make sure to replace it with the obtained result.

    To run all experiments, use the --run-all flag.
    Example: python -m pure_numpy_nn.comparison --run-all.
    Note: running all experiments takes a while and should only be used if training code changed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-all', action='store_true', help='Run all experiments')
    args = parser.parse_args()

    if args.run_all:
        losses = {
            "SGD": run_experiment(SGD, {}),
            "Momentum": run_experiment(SGD, {"momentum": 0.95}),
            "Adam": run_experiment(Adam, {}),
        }
    else:
        losses = {
            "SGD": 0.20991,
            "Momentum": 0.19919,
            "Adam": 0.16420,
        }

    # Print comparison table
    print("Optimizer Performance Comparison")
    for optimizer, final_loss in losses.items():
        print(f'{optimizer:<30} {final_loss:.5f}')

if __name__ == "__main__":
    main()
