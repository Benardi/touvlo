import numpy as np

# sigmoid function
g = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))


def feature_normalize(X):
    mu = np.mean(X)
    sigma = np.std(X)

    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def h(X, theta):
    return g(X.dot(theta))


def grad(theta, X, y, m):

    grad = (1 / m) * (X.T).dot(h(X, theta) - y)
    return grad


def cost_function(theta, X, y):

    m = len(y)
    J = (1 / m) * ((-y.T).dot(np.log(h(X, theta)))
                   - ((1 - y).T).dot(np.log(1 - h(X, theta))))

    return J, grad


def gradient_descent(X, y, initial_theta, alpha, num_iters):
    m = len(y)
    theta = np.copy(initial_theta)
    for i in range(num_iters):
        theta = theta - alpha * grad(theta, X, y, m)

    return theta
