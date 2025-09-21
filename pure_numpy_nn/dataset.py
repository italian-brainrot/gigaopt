import numpy as np

def generate_data(n_samples=1000, n_features=1):
    """
    Generates a simple dataset for regression.
    y = sin(x) + noise
    """
    X = np.random.rand(n_samples, n_features) * 10 - 5 # Data points between -5 and 5
    y = np.sin(X) + np.random.randn(n_samples, n_features) * 0.1
    return X, y

def get_mini_batches(X, y, batch_size):
    """
    A generator that yields mini-batches for training.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        excerpt = indices[start_idx:end_idx]
        yield X[excerpt], y[excerpt]
