"""
.. module:: lin_rg
    :synopsis: Provides routines to construct a Linear Regression.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from numpy import zeros, float64
from numpy.linalg import inv, LinAlgError


# model hypothesis
def h(X, theta):
    """Linear regression hypothesis.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        theta (numpy.array): Column vector of model's parameters.

    Returns:
        numpy.array: The projected value for each line of the dataset.
    """
    return X.dot(theta)


def cost_func(X, y, theta):
    """Computes the cost function J for Linear Regression.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        y (numpy.array): Column vector of expected values.
        theta (numpy.array): Column vector of model's parameters.

    Returns:
        float: Computed cost.
    """
    m = len(y)  # number of training examples
    J = (1 / (2 * m)) * ((h(X, theta) - y).T).dot(h(X, theta) - y)
    return J


def reg_cost_func(X, y, theta, _lambda):
    """Computes the regularized cost function J for Linear Regression.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        y (numpy.array): Column vector of expected values.
        theta (numpy.array): Column vector of model's parameters.
        _lambda (float): The regularization hyperparameter.

    Returns:
        float: Computed cost with regularization.
    """
    m = len(y)  # number of training examples
    J = (1 / (2 * m)) * ((h(X, theta) - y).T).dot(h(X, theta) - y)
    J = J + (_lambda / (2 * m)) * (theta[1:, :].T).dot(theta[1:, :])
    return J


def grad(X, y, theta):
    """Computes the gradient for Linear Regression.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        y (numpy.array): Column vector of expected values.
        theta (numpy.array): Column vector of model's parameters.

    Returns:
        numpy.array: Gradient column vector.
    """
    m = len(y)
    grad = (1 / m) * (X.T).dot(h(X, theta) - y)
    return grad


def reg_grad(X, y, theta, _lambda):
    """Computes the regularized gradient for Linear Regression.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        y (numpy.array): Column vector of expected values.
        theta (numpy.array): Column vector of model's parameters.
        _lambda (float): The regularization hyperparameter.

    Returns:
        numpy.array: Regularized gradient column vector.
    """
    m = len(y)
    grad = (1 / m) * (X.T).dot(h(X, theta) - y)
    grad[1:, :] = grad[1:, :] + (_lambda / m) * theta[1:, :]
    return grad


def predict(X, theta):
    """Computes prediction vector.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        theta (numpy.array): Column vector of model's parameters.

    Returns:
        numpy.array: vector with predictions for each input line.
    """
    return X.dot(theta)


def normal_eqn(X, y):
    """Produces optimal theta via normal equation.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        y (numpy.array): Column vector of expected values.

    Raises:
        LinAlgError

    Returns:
        numpy.array: Optimized model parameters theta.
    """
    n = X.shape[1]  # number of columns
    theta = zeros((n + 1, 1), dtype=float64)

    try:
        X_T = X.T
        theta = inv(X_T.dot(X)).dot(X_T).dot(y)

    except LinAlgError:
        pass

    return theta
