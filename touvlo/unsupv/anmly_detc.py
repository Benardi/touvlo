"""
.. module:: anmly_detc
    :synopsis: Provides routines to perform Anomaly Detection.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from functools import reduce
from math import pi

from numpy import array, mean, var, sqrt, exp, power, sum, multiply
from numpy.linalg import det, inv


# predict function
def is_anomaly(p, threshold=0.5):
    """Predicts whether a probability falls into class 1 (anomaly).

    :param p: Probability that example belongs to class 1 (is anomaly).
    :type x: numpy.array

    :param threshold: point below which an example is considered of class 1.
    :type threshold: float

    :returns: Binary value to denote class 1 or 0
    :rtype: int
    """
    prediction = array([[1] if el < threshold else [0] for el in p])
    return prediction


def cov_matrix(X, mu):
    """Calculates the covariance matrix for matrix X (m x n).

    :param X: Features' dataset.
    :type X: numpy.array

    :param mu: Mean of each feature/column of.
    :type mu: numpy.array

    :returns: Covariance matrix (n x n)
    :rtype: int
    """
    m, n = X.shape
    X_minus_mu = X - mu
    sigma = (1 / m) * (X_minus_mu.T).dot(X_minus_mu)

    return sigma


def estimate_uni_gaussian(X):
    """Estimates parameters for Univariate Gaussian distribution.

    :param X: Features' dataset.
    :type X: numpy.array

    :returns:
        - mu - Mean of each feature/column of X.
        - sigma2 - Variance of each feature/column of X.

    :rtype:
        - mu (:py:class: numpy.array)
        - sigma2 (:py:class: numpy.array)
    """
    mu = mean(X, axis=0)
    sigma2 = var(X, axis=0)
    return mu, sigma2


def estimate_multi_gaussian(X):
    """Estimates parameters for Multivariate Gaussian distribution.

    :param X: Features' dataset.
    :type X: numpy.array

    :returns:
        - mu - Mean of each feature/column of X.
        - sigma - Covariance matrix for X.

    :rtype:
        - mu (:py:class: numpy.array)
        - sigma (:py:class: numpy.array)
    """
    m, n = X.shape
    mu = mean(X, axis=0)
    sigma = cov_matrix(X, mu)

    return mu, sigma


def uni_gaussian(X, mu, sigma2):
    """Estimates probability that examples belong to Univariate Gaussian.

    :param X: Features' dataset.
    :type X: numpy.array

    :param mu: Mean of each feature/column of X.
    :type mu: numpy.array

    :param sigma2: Variance of each feature/column of X.
    :type sigma2: numpy.array

    :returns: Probability density function for each example
    :rtype: numpy.array
    """
    p = (1 / sqrt(2 * pi * sigma2))
    p = p * exp(-power(X - mu, 2) / (2 * sigma2))

    def prod(x, y):
        return x * y
    p = array([[reduce(prod, el)] for el in p])

    return p


def multi_gaussian(X, mu, sigma):
    """Estimates probability that examples belong to Multivariate Gaussian.

    :param X: Features' dataset.
    :type X: numpy.array

    :param mu: Mean of each feature/column of X.
    :type mu: numpy.array

    :param sigma: Covariance matrix for X.
    :type sigma: numpy.array

    :returns: Probability density function for each example
    :rtype: numpy.array
    """
    m, n = X.shape
    X = X - mu

    factor = X.dot(inv(sigma))
    factor = multiply(factor, X)
    factor = - (1 / 2) * sum(factor, axis=1, keepdims=True)

    p = 1 / (power(2 * pi, n / 2) * sqrt(det(sigma)))
    p = p * exp(factor)

    return p


def predict(X, epsilon, gaussian, **kwargs):
    """Predicts whether examples are anomalies.

    :param X: Features' dataset.
    :type X: numpy.array

    :param epsilon: point below which an example is considered of class 1.
    :type epsilon: float

    :param gaussian: Function that estimates pertinency probability.
    :type gaussian: numpy.array

    :returns: Column vector of classification
    :rtype: numpy.array
    """
    p = gaussian(X=X, **kwargs)
    return is_anomaly(p, threshold=epsilon)
