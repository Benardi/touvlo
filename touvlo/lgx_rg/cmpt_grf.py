"""
.. module:: cmpt_grf
    :synopsis: Provides routines to construct a Logistic Regression of
        parameters w and b via computation graph.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from numpy import log, zeros, dot
from numpy import sum as add

from touvlo.utils import g


def h(X, w, b):
    """Logistic regression hypothesis.

    Args:
        X (numpy.array): Transposed features' dataset.
        w (numpy.array): Column vector of model's parameters.
        b (float): Model's intercept parameter.

    Returns:
        numpy.array: The probability that each entry belong to class 1.
    """
    return g(dot(w.T, X) + b)


def cost_func(X, Y, hyp=None, **kwargs):
    """Computes the cost function J for Logistic Regression.

    Args:
        X (numpy.array): Features' dataset.
        Y (numpy.array): Row vector of expected values.
        hyp (numpy.array): The calculated model hypothesis, if not provided
        the named parameters to calculate it should be provided instead.

    Returns:
        float: Computed cost.
    """
    if hyp is None:
        hyp = h(X, **kwargs)

    _, m = Y.shape
    J = -(1 / m) * add((Y * log(hyp) + (1 - Y) * log(1 - hyp)))

    return J


def grad(X, Y, w, b):
    """Computes the gradient for the parameters w and b.

    Args:
        X (numpy.array): Transpose features' dataset.
        Y (numpy.array): Row vector of expected values.
        w (numpy.array): Column vector of model's parameters.
        b (float): Model's intercept parameter.

    Returns:
        (numpy.array, float): A 2-tuple consisting of a gradient column
            vector and a gradient value.
    """
    _, m = X.shape
    A = h(X, w, b)  # compute activation

    dz = A - Y
    dw = (1 / m) * dot(X, dz.T)
    db = (1 / m) * add(dz)

    return dw, db


def predict(X, w, b, threshold=0.5):
    """Predicts whether the given probabilities fall into class 1.

    Args:
        X (numpy.array): Transpose features' dataset.
        threshold (float): Point above which a probability is assigned
        to class 1.

    Returns:
        numpy.array: Binary value to denote class 1 or 0 for each example.
    """
    n, m = X.shape
    Y_hat = zeros((1, m))
    w = w.reshape(n, 1)

    A = h(X, w, b)

    for i in range(A.shape[1]):
        if A[:, i] >= threshold:
            Y_hat[:, i] = 1
        else:
            Y_hat[:, i] = 0

    return Y_hat
