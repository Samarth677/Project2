import numpy as np

def generate_synthetic_data(n_samples=100, noise=0.1, n_classes=2):
    X = np.random.randn(n_samples, 2)
    if n_classes == 2:
        y = (X[:, 0] + X[:, 1] + noise * np.random.randn(n_samples) > 0).astype(int)
    else:
        y = np.digitize(X[:, 0] + X[:, 1] + noise * np.random.randn(n_samples), bins=np.linspace(-2, 2, n_classes))
    return X, y
