import numpy as np


def gradient_descent(X, y, grad, initial_theta, alpha, num_iters):

    m = len(y)
    theta = np.copy(initial_theta)
    for _ in range(num_iters):
        theta = theta - alpha * (1 / m) * grad(theta, X, y, m)

    return theta


def feature_normalize(X):
    mu = np.mean(X)
    sigma = np.std(X)

    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma
