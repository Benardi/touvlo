import numpy as np


def compute_cost(X, y, theta):

    m = len(X)  # number of training examples

    return (1 / (2 * m)) * ((X.dot(theta) - y).T).dot(X.dot(theta) - y)


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    opt_theta = np.copy(theta)
    for i in range(num_iters):
        opt_theta = opt_theta - alpha * \
            (1 / m) * (X.T).dot(X.dot(opt_theta) - y)

    return opt_theta
