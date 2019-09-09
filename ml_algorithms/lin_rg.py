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

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param theta: Column vector of model's parameters.
    :type theta: numpy.array

    :returns: The projected value for each line of the dataset.
    :rtype: numpy.array
    """
    return X.dot(theta)


def cost_func(X, y, theta):
    """Computes the cost function J for Linear Regression.

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param y: Column vector of expected values.
    :type y: numpy.array

    :param theta: Column vector of model's parameters.
    :type theta: numpy.array

    :returns: Computed cost.
    :rtype: float
    """
    m = len(y)  # number of training examples
    J = (1 / (2 * m)) * ((h(X, theta) - y).T).dot(h(X, theta) - y)
    return J


def reg_cost_func(X, y, theta, _lambda):
    """Computes the regularized cost function J for Linear Regression.

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param y: Column vector of expected values.
    :type y: numpy.array

    :param theta: Column vector of model's parameters.
    :type theta: numpy.array

    :param _lambda: The regularization hyperparameter.
    :type _lambda: float

    :returns: Computed cost with regularization.
    :rtype: float
    """
    m = len(y)  # number of training examples
    J = (1 / (2 * m)) * ((h(X, theta) - y).T).dot(h(X, theta) - y)
    J = J + (_lambda / (2 * m)) * (theta[1:, :].T).dot(theta[1:, :])
    return J


def grad(X, y, theta):
    """Computes the gradient for Linear Regression.

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param y: Column vector of expected values.
    :type y: numpy.array

    :param theta: Column vector of model's parameters.
    :type theta: numpy.array

    :returns: Gradient column vector.
    :rtype: numpy.array
    """
    m = len(y)
    grad = (1 / m) * (X.T).dot(h(X, theta) - y)
    return grad


def reg_grad(X, y, theta, _lambda):
    """Computes the regularized gradient for Linear Regression.

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param y: Column vector of expected values.
    :type y: numpy.array

    :param theta: Column vector of model's parameters.
    :type theta: numpy.array

    :param _lambda: The regularization hyperparameter.
    :type _lambda: float

    :returns: Regularized gradient column vector.
    :rtype: numpy.array
    """
    m = len(y)
    grad = (1 / m) * (X.T).dot(h(X, theta) - y)
    grad[1:, :] = grad[1:, :] + (_lambda / m) * theta[1:, :]
    return grad


def predict(X, theta):
    """Computes prediction vector.

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param theta: Column vector of model's parameters.
    :type theta: numpy.array

    :returns: vector with predictions for each input line.
    :rtype: numpy.array
    """
    return X.dot(theta)


def normal_eqn(X, y):
    """Produces optimal theta via normal equation.

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param y: Column vector of expected values.
    :type y: numpy.array

    :raises: LinAlgError

    :returns: Optimized model parameters theta.
    :rtype: numpy.array
    """
    n = X.shape[1]  # number of columns
    theta = zeros((n + 1, 1), dtype=float64)

    try:
        X_T = X.T
        theta = inv(X_T.dot(X)).dot(X_T).dot(y)

    except LinAlgError:
        pass

    return theta
