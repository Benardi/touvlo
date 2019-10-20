"""
.. module:: pca
    :synopsis: Provides routines to perform Principal Component Analysis.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from numpy.linalg import svd
from numpy import diag


def pca(X):
    """Runs Principal Component Analysis on dataset

    Args:
        X (numpy.array): Features' dataset

    Returns:
        (numpy.array, numpy.array): A 2-tuple of U, eigenvectors of covariance
            matrix, and S, eigenvalues (on diagonal) of covariance matrix.
    """
    m, n = X.shape
    Sigma = (1 / m) * X.T.dot(X)
    U, S, V = svd(Sigma)
    S = diag(S)

    return U, S


def project_data(X, U, k):
    """Computes reduced data representation (projected data)

    Args:
        X (numpy.array): Normalized features' dataset
        U (numpy.array): eigenvectors of covariance matrix
        k (int): Number of features in reduced data representation

    Returns:
        numpy.array: Reduced data representation (projection)
    """
    U_reduce = U[:, 0:k]
    Z = X.dot(U_reduce)
    return Z


def recover_data(Z, U, k):
    """Recovers an approximation of original data using the projected data

    Args:
        Z (numpy.array): Reduced data representation (projection)
        U (numpy.array): eigenvectors of covariance matrix
        k (int): Number of features in reduced data representation

    Returns:
        numpy.array: Approximated features' dataset
    """
    X_rec = Z.dot(U[:, 0:k].T)
    return X_rec
