"""
.. module:: lgx_rg
    :synopsis: Provides routines to construct a Logistic Regression.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from numpy import log, zeros, vectorize, exp

# sigmoid function
g = vectorize(lambda x: 1 / (1 + exp(-x)))
# predict function
p = vectorize(lambda x: 1 if x >= 0.5 else 0)


def h(X, theta):
    """Logistic regression hypothesis.

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param theta: Column vector of model's parameters.
    :type theta: numpy.array

    :raises: ValueError

    :returns: The probability that each entry belong to class 1.
    :rtype: numpy.array
    """
    return g(X.dot(theta))


def cost_func(X, y, theta):
    """Computes the cost function J for Logistic Regression.

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param y: Column vector of expected values.
    :type y: numpy.array

    :param theta: Column vector of model's parameters.
    :type theta: numpy.array

    :returns: Computed cost.
    :rtype: float
    """
    m = len(y)
    J = (1 / m) * ((-y.T).dot(log(h(X, theta)))
                   - ((1 - y).T).dot(log(1 - h(X, theta))))
    return J


def reg_cost_func(X, y, theta, _lambda):
    """Computes the regularized cost function J for Logistic Regression.

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
    m = len(y)
    J = - (1 / m) * ((y.T).dot(log(h(X, theta)))
                     + (1 - y.T).dot(log(1 - h(X, theta))))
    J = J + (_lambda / (2 * m)) * ((theta[1:, :]).T).dot(theta[1:, :])
    return J


def grad(X, y, theta):
    """Computes the gradient for the parameters theta.

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
    """Computes the regularized gradient for Logistic Regression.

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
    grad = zeros(theta.shape)

    grad = (1 / m) * (X.T).dot(h(X, theta) - y)
    grad[1:, :] = grad[1:, :] + (_lambda / m) * theta[1:, :]

    return grad


def predict_prob(X, theta):
    """ Produces the probability that the entries belong to class 1.

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param theta: Column vector of model's parameters.
    :type theta: numpy.array

    :raises: ValueError

    :returns: The probability that each entry belong to class 1.
    :rtype: numpy.array
    """
    return g(X.dot(theta))


def predict(X, theta):
    """ Classifies each entry as class 1 or class 0.

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param theta: Column vector of model's parameters.
    :type theta: numpy.array

    :returns: Column vector with each entry classification.
    :rtype: numpy.array
    """
    return p(predict_prob(X, theta))
