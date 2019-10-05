"""
.. module:: pca
    :synopsis: Provides routines to perform Principal Component Analysis.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from numpy.linalg import svd
from numpy import diag


def pca(X):
    """Runs Principal Component Analysis on dataset

    :param X: Features' dataset
    :type X: numpy.array

    :returns:
        - U - eigenvectors of covariance matrix
        - S - eigenvalues (on diagonal) of covariance matrix

    :rtype:
        - U (:py:class: numpy.array)
        - S (:py:class: numpy.array)
    """
    m, n = X.shape
    Sigma = (1 / m) * X.T.dot(X)
    U, S, V = svd(Sigma)
    S = diag(S)

    return U, S


def project_data(X, U, k):
    """Computes reduced data representation (projected data)

    :param X: Normalized features' dataset
    :type X: numpy.array

    :param U: eigenvectors of covariance matrix
    :type U: numpy.array

    :param k: Number of features in reduced data representation

    :returns: Reduced data representation (projection)
    :rtype: numpy.array
    """
    U_reduce = U[:, 0:k]
    Z = X.dot(U_reduce)
    return Z


def recover_data(Z, U, k):
    """Recovers an approximation of original data using the projected data

    :param Z: Reduced data representation (projection)
    :type Z: numpy.array

    :param U: eigenvectors of covariance matrix
    :type U: numpy.array

    :param k: Number of features in reduced data representation

    :returns: Approximated features' dataset
    :rtype: numpy.array
    """
    X_rec = Z.dot(U[:, 0:k].T)
    return X_rec
