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

    Args:
        p (numpy.array): Probability that example belongs to class 1 (is
            anomaly).
        threshold (float): point below which an example is considered of class
            1.

    Returns:
        int: Binary value to denote class 1 or 0
    """
    prediction = array([[1] if el < threshold else [0] for el in p])
    return prediction


def cov_matrix(X, mu):
    """Calculates the covariance matrix for matrix X (m x n).

    Args:
        X (numpy.array): Features' dataset.
        mu (numpy.array): Mean of each feature/column of.

    Returns:
        int: Covariance matrix (n x n)
    """
    m, n = X.shape
    X_minus_mu = X - mu
    sigma = (1 / m) * (X_minus_mu.T).dot(X_minus_mu)

    return sigma


def estimate_uni_gaussian(X):
    """Estimates parameters for Univariate Gaussian distribution.

    Args:
        X (numpy.array): Features' dataset.

    Returns:
        (numpy.array, numpy.array): A 2-tuple of mu, the mean of each
            feature/column of X, and sigma2, the variance of each
            feature/column of X.
    """
    mu = mean(X, axis=0)
    sigma2 = var(X, axis=0)
    return mu, sigma2


def estimate_multi_gaussian(X):
    """Estimates parameters for Multivariate Gaussian distribution.

    Args:
        X (numpy.array): Features' dataset.

    Returns:
        (numpy.array, numpy.array): A 2-tuple of mu, the mean of each
            feature/column of X, and sigma, the covariance matrix for X.
    """
    m, n = X.shape
    mu = mean(X, axis=0)
    sigma = cov_matrix(X, mu)

    return mu, sigma


def uni_gaussian(X, mu, sigma2):
    """Estimates probability that examples belong to Univariate Gaussian.

    Args:
        X (numpy.array): Features' dataset.
        mu (numpy.array): Mean of each feature/column of X.
        sigma2 (numpy.array): Variance of each feature/column of X.

    Returns:
        numpy.array: Probability density function for each example
    """
    p = (1 / sqrt(2 * pi * sigma2))
    p = p * exp(-power(X - mu, 2) / (2 * sigma2))

    def prod(x, y):
        return x * y
    p = array([[reduce(prod, el)] for el in p])

    return p


def multi_gaussian(X, mu, sigma):
    """Estimates probability that examples belong to Multivariate Gaussian.

    Args:
        X (numpy.array): Features' dataset.
        mu (numpy.array): Mean of each feature/column of X.
        sigma (numpy.array): Covariance matrix for X.

    Returns:
        numpy.array: Probability density function for each example
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

    Args:
        X (numpy.array): Features' dataset.
        epsilon (float): point below which an example is considered of class 1.
        gaussian (numpy.array): Function that estimates pertinency probability.

    Returns:
        numpy.array: Column vector of classification
    """
    p = gaussian(X=X, **kwargs)
    return is_anomaly(p, threshold=epsilon)
