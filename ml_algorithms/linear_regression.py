import numpy as np


def compute_cost(X, y, theta):

    m = len(X)  # number of training examples
    J = (1 / (2 * m)) * ((X.dot(theta) - y).T).dot(X.dot(theta) - y)
    return J


# model hypothesis
def h(X, theta):
    return X.dot(theta)


def gradient_descent(X, y, initial_theta, alpha, num_iters):

    m = len(y)
    theta = np.copy(initial_theta)
    for i in range(num_iters):
        theta = theta - alpha * (1 / m) * (X.T).dot(h(X, theta) - y)

    return theta
