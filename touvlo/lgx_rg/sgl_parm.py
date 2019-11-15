"""
.. module:: sgl_parm
    :synopsis: Provides routines to construct a Logistic Regression of
        single parameter theta.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from numpy import log, zeros

from touvlo.utils import g


# predict function
def p(x, threshold=0.5):
    """Predicts whether a probability falls into class 1.

    Args:
        x (obj): Probability that example belongs to class 1.
        threshold (float): point above which a probability is deemed of class
            1.

    Returns:
        int: Binary value to denote class 1 or 0
    """
    prediction = None
    if x >= threshold:
        prediction = 1
    else:
        prediction = 0

    return prediction


def h(X, theta):
    """Logistic regression hypothesis.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        theta (numpy.array): Column vector of model's parameters.

    Raises:
        ValueError

    Returns:
        numpy.array: The probability that each entry belong to class 1.
    """
    return g(X.dot(theta))


def cost_func(X, Y, theta):
    """Computes the cost function J for Logistic Regression.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        Y (numpy.array): Column vector of expected values.
        theta (numpy.array): Column vector of model's parameters.

    Returns:
        float: Computed cost.
    """
    m = len(Y)
    J = (1 / m) * ((-Y.T).dot(log(h(X, theta)))
                   - ((1 - Y).T).dot(log(1 - h(X, theta))))
    return J


def reg_cost_func(X, Y, theta, _lambda):
    """Computes the regularized cost function J for Logistic Regression.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        Y (numpy.array): Column vector of expected values.
        theta (numpy.array): Column vector of model's parameters.
        _lambda (float): The regularization hyperparameter.

    Returns:
        float: Computed cost with regularization.
    """
    m = len(Y)
    J = - (1 / m) * ((Y.T).dot(log(h(X, theta)))
                     + (1 - Y.T).dot(log(1 - h(X, theta))))
    J = J + (_lambda / (2 * m)) * ((theta[1:, :]).T).dot(theta[1:, :])
    return J


def grad(X, Y, theta):
    """Computes the gradient for the parameters theta.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        Y (numpy.array): Column vector of expected values.
        theta (numpy.array): Column vector of model's parameters.

    Returns:
        numpy.array: Gradient column vector.
    """
    m = len(Y)
    grad = (1 / m) * (X.T).dot(h(X, theta) - Y)
    return grad


def reg_grad(X, Y, theta, _lambda):
    """Computes the regularized gradient for Logistic Regression.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        Y (numpy.array): Column vector of expected values.
        theta (numpy.array): Column vector of model's parameters.
        _lambda (float): The regularization hyperparameter.

    Returns:
        numpy.array: Regularized gradient column vector.
    """
    m = len(Y)
    grad = zeros(theta.shape)

    grad = (1 / m) * (X.T).dot(h(X, theta) - Y)
    grad[1:, :] = grad[1:, :] + (_lambda / m) * theta[1:, :]

    return grad


def predict_prob(X, theta):
    """Produces the probability that the entries belong to class 1.

    Returns:
        X (numpy.array): Features' dataset plus bias column.
        theta (numpy.array): Column vector of model's parameters.

    Raises:
        ValueError

    Returns:
        numpy.array: The probability that each entry belong to class 1.
    """
    return g(X.dot(theta))


def predict(X, theta):
    """Classifies each entry as class 1 or class 0.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        theta (numpy.array): Column vector of model's parameters.

    Returns:
        numpy.array: Column vector with each entry classification.
    """
    return p(predict_prob(X, theta))
