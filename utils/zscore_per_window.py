import numpy as np

def zscore_per_window(X, epsilon=1e-6):
    # Hitung mu dan sigma sepanjang dimensi T (axis=1)
    mu    = np.mean(X, axis=1, keepdims=True)   # (N, 1, C)
    sigma = np.std(X,  axis=1, keepdims=True)   # (N, 1, C)
 
    X_norm = (X - mu) / (sigma + epsilon)
 
    return X_norm, mu, sigma
 