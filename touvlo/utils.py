"""
.. module:: utils
    :synopsis: Provides routines of interest to different ML models.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from numpy import zeros, copy, std, mean, float64, exp, seterr


# sigmoid gradient function
def g(x):
    """This function applies the sigmoid function on a given value.

    :param x: Input value or object containing value .
    :type x: obj

    :returns: Sigmoid function at value.
    :rtype: obj
    """
    return 1 / (1 + exp(-x))


# sigmoid gradient function
def g_grad(x):
    """This function calculates the sigmoid gradient at a given value.

    :param x: Input value or object containing value .
    :type x: obj

    :returns: Sigmoid gradient at value.
    :rtype: obj
    """
    return g(x) * (1 - g(x))


def BGD(X, y, grad, initial_theta,
        alpha, num_iters, **kwargs):
    """Performs parameter optimization via Batch Gradient Descent.

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param y: Column vector of expected values.
    :type y: numpy.array

    :param grad: Routine that generates the partial derivatives given theta.
    :type grad: numpy.array

    :param initial_theta: Initial value for parameters to be optimized.
    :type initial_theta: numpy.array

    :param alpha: Learning rate or _step size of the optimization.
    :type alpha: float

    :param num_iters: Number of times the optimization will be performed.
    :type num_iters: int

    :returns: Optimized model parameters.
    :rtype: numpy.array
    """
    theta = copy(initial_theta)
    for _ in range(num_iters):
        theta = theta - alpha * grad(X, y, theta, **kwargs)

    return theta


def SGD(X, y, grad, initial_theta,
        alpha, num_iters, **kwargs):
    """Performs parameter optimization via Stochastic Gradient Descent.

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param y: Column vector of expected values.
    :type y: numpy.array

    :param grad: Routine that generates the partial derivatives given theta.
    :type grad: numpy.array

    :param initial_theta: Initial value for parameters to be optimized.
    :type initial_theta: numpy.array

    :param alpha: Learning rate or _step size of the optimization.
    :type alpha: float

    :param num_iters: Number of times the optimization will be performed.
    :type num_iters: int

    :returns: Optimized model parameters.
    :rtype: numpy.array
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

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param y: Column vector of expected values.
    :type y: numpy.array

    :param grad: Routine that generates the partial derivatives given theta.
    :type grad: numpy.array

    :param initial_theta: Initial value for parameters to be optimized.
    :type initial_theta: numpy.array

    :param alpha: Learning rate or _step size of the optimization.
    :type alpha: float

    :param num_iters: Number of times the optimization will be performed.
    :type num_iters: int

    :param b: Number of examples in mini batch.
    :type b: int

    :returns: Optimized model parameters.
    :rtype: numpy.array
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

    :param J: Function handle that computes cost given theta.
    :type J: function

    :param theta: Model parameters.
    :type theta: numpy.array

    :param err: distance between points where J is evaluated.
    :type err: float

    :returns: Computed numeric gradient.
    :rtype: numpy.array
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

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :returns:
        - X_norm - Normalized features' dataset.
        - mu - Mean of each feature
        - sigma - Standard deviation of each feature.

    :rtype:
        - X_norm (:py:class: numpy.array)
        - mu (:py:class: numpy.array)
        - sigma (:py:class: numpy.array)
    """
    seterr(divide='ignore', invalid='ignore')
    mu = mean(X, axis=0)
    sigma = std(X, axis=0, ddof=1)

    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma
