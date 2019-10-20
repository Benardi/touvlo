"""
.. module:: utils
    :synopsis: Provides routines of interest to different ML models.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from numpy import zeros, copy, std, mean, float64, exp, seterr, where


# sigmoid gradient function
def g(x):
    """This function applies the sigmoid function on a given value.

    Args:
        x (obj): Input value or object containing value .

    Returns:
        obj: Sigmoid function at value.
    """
    return 1 / (1 + exp(-x))


# sigmoid gradient function
def g_grad(x):
    """This function calculates the sigmoid gradient at a given value.

    Args:
        x (obj): Input value or object containing value .

    Returns:
        obj: Sigmoid gradient at value.
    """
    return g(x) * (1 - g(x))


def BGD(X, y, grad, initial_theta,
        alpha, num_iters, **kwargs):
    """Performs parameter optimization via Batch Gradient Descent.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        y (numpy.array): Column vector of expected values.
        grad (numpy.array): Routine that generates the partial derivatives
            given theta.
        initial_theta (numpy.array): Initial value for parameters to be
            optimized.
        alpha (float): Learning rate or _step size of the optimization.
        num_iters (int): Number of times the optimization will be performed.

    Returns:
        numpy.array: Optimized model parameters.
    """
    theta = copy(initial_theta)
    for _ in range(num_iters):
        theta = theta - alpha * grad(X, y, theta, **kwargs)

    return theta


def SGD(X, y, grad, initial_theta,
        alpha, num_iters, **kwargs):
    """Performs parameter optimization via Stochastic Gradient Descent.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        y (numpy.array): Column vector of expected values.
        grad (numpy.array): Routine that generates the partial derivatives
            given theta.
        initial_theta (numpy.array): Initial value for parameters to be
            optimized.
        alpha (float): Learning rate or _step size of the optimization.
        num_iters (int): Number of times the optimization will be performed.

    Returns:
        numpy.array: Optimized model parameters.
    """
    m = len(y)
    theta = copy(initial_theta)

    for _ in range(num_iters):
        for i in range(m):
            theta = theta - alpha * grad(X[[i], :], y[[i], :], theta, **kwargs)

    return theta


def MBGD(X, y, grad, initial_theta,
         alpha, num_iters, b, **kwargs):
    """Performs parameter optimization via Mini-Batch Gradient Descent.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        y (numpy.array): Column vector of expected values.
        grad (numpy.array): Routine that generates the partial derivatives
            given theta.
        initial_theta (numpy.array): Initial value for parameters to be
            optimized.
        alpha (float): Learning rate or _step size of the optimization.
        num_iters (int): Number of times the optimization will be performed.
        b (int): Number of examples in mini batch.

    Returns:
        numpy.array: Optimized model parameters.
    """
    m = len(y)
    theta = copy(initial_theta)
    _steps = [el for el in range(0, m, b)]

    for _ in range(num_iters):
        for _step in _steps[:-1]:
            theta = theta - alpha * grad(X[_step:(_step + b), :],
                                         y[_step:(_step + b), :],
                                         theta, **kwargs)

        theta = theta - alpha * grad(X[_steps[-1]:, :],
                                     y[_steps[-1]:, :],
                                     theta, **kwargs)

    return theta


def numerical_grad(J, theta, err):
    """Numerically calculates the gradient of a given cost function.

    Args:
        J (Callable): Function handle that computes cost given theta.
        theta (numpy.array): Model parameters.
        err (float): distance between points where J is evaluated.

    Returns:
        numpy.array: Computed numeric gradient.
    """
    num_grad = zeros(theta.shape, dtype=float64)
    perturb = zeros(theta.shape, dtype=float64)

    for i in range(len(theta)):
        perturb[i] = err
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        num_grad[i] = (loss2 - loss1) / (2 * err)
        perturb[i] = 0

    return num_grad


def feature_normalize(X):
    """Performs Z score normalization in a numeric dataset.

    Args:
        X (numpy.array): Features' dataset plus bias column.

    Returns:
        (numpy.array, numpy.array, numpy.array): A 3-tuple of X_norm,
            normalized features' dataset, mu, mean of each feature, and sigma,
            standard deviation of each feature.
    """
    seterr(divide='ignore', invalid='ignore')
    mu = mean(X, axis=0)
    sigma = std(X, axis=0, ddof=1)

    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def mean_normlztn(Y, R):
    """Performs mean normalization in a numeric dataset.

    :param Y: Scores' dataset.
    :type Y: numpy.array

    :param R: Dataset of 0s and 1s (whether there's a rating).
    :type R: numpy.array

    :returns:
        - Y_norm - Normalized scores' dataset (row wise).
        - Y_mean - Column vector of calculated means.

    :rtype:
        - Y_norm (:py:class: numpy.array)
        - Y_mean (:py:class: numpy.array)
    """
    m, n = Y.shape
    Y_mean = zeros((m, 1))
    Y_norm = zeros((m, n))

    for i in range(len(R)):
        idx = where(R[i, :] == 1)[0]
        Y_mean[i] = mean(Y[i, idx])
        Y_norm[i, idx] = Y[i, idx] - Y_mean[i]

    return Y_norm, Y_mean
