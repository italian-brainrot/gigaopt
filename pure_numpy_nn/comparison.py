import time

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from .neural_net import NeuralNetwork, mse_loss, mse_loss_derivative
from .optimizers import Adam, AdaThird, Nova, AdaThirdV2, CogniO
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
    Runs a training experiment for a given optimizer, with learning rate tuning.
    """
    # For reproducibility
    np.random.seed(0)

    # Generate data
    X, y = generate_data(n_samples, n_features)
    num_trials = 0

    def objective(trial):
        nonlocal num_trials
        num_trials += 1
        start_sec = time.perf_counter()

        # Suggest learning rate
        lr = trial.suggest_float("lr", 1e-5, 1, log=True)

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
                y_pred = net.forward(x_batch)
                loss = mse_loss(y_batch, y_pred)
                epoch_loss += loss
                num_batches += 1
                loss_grad = mse_loss_derivative(y_batch, y_pred)
                grads = net.backward(loss_grad)
                params = net.get_params()
                optimizer.step(params, grads)
                if num_trials == 1 and epoch == 0 and i == 0:
                    print(f"first batch took {(time.perf_counter() - start_sec):.2f} seconds")

            if num_trials == 1 and epoch == 0:
                print(f"first epoch took {(time.perf_counter() - start_sec):.2f} seconds")

            avg_loss = epoch_loss / num_batches
            trial.report(avg_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if num_trials == 1:
            print(f"first trial took {(time.perf_counter() - start_sec):.2f} seconds")

        return avg_loss

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=100)

    best_lr = study.best_trial.params['lr']
    print(f"Best LR for {optimizer_class.__name__}: {best_lr}")
    print(f"Best value: {study.best_value}")

    return study.best_value

def main():
    """
    Main comparison function.

    IMPORTANT: After an experiment is ran, replace it with the final loss value so that we don't re-run every optimizer each time.
    """

    losses = {
        # adam example:
        # "Adam": run_experiment(Adam, {}, N_SAMPLES, N_FEATURES, LAYER_SIZES, EPOCHS, BATCH_SIZE)

        "Adam": 0.17723168193770694,
        "AdaThird": 0.1769168308425637,
        "Nova": 0.18443070685599255,
        "AdaThirdV2": 0.1719002045491398,
        "CogniO": 0.16145324453875395,
    }

    # Print comparison table
    print("Optimizer Performance Comparison")
    for optimizer, final_loss in losses.items():
        print(f'{optimizer:<30} {final_loss}')

if __name__ == "__main__":
    main()
