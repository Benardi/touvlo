"""
.. module:: cf
    :synopsis: Provides routines to apply Collaborative Filtering.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from numpy import power, multiply, where, zeros, reshape, append
from numpy import sum as add


def unravel_params(params, num_users, num_products, num_features):
    """Unravels flattened array into features' matrices

    Args:
        params (numpy.array): Row vector of coefficients.
        num_users (int): Number of users in this instance.
        num_products (int): Number of products in this instance.
        num_features (int): Number of features in this instance.

    Returns:
        (numpy.array, numpy.array): A 2-tuple consisting of a matrix of
        product features and a matrix of user features.
    """
    X = params[0:(num_products * num_features)]
    X = reshape(X, (num_products, num_features))
    theta = params[(num_products * num_features):]
    theta = reshape(theta, (num_users, num_features))
    return X, theta


def cost_function(X, Y, R, theta, _lambda):
    """Computes the cost function J for Collaborative Filtering.

    Args:
        X (numpy.array): Matrix of product features.
        Y (numpy.array): Scores' matrix.
        R (numpy.array): Matrix of 0s and 1s (whether there's a rating).
        theta (numpy.array): Matrix of user features.
        _lambda (float): The regularization hyperparameter.

    Returns:
        float: Computed cost.
    """
    J = power(X.dot(theta.T) - Y, 2)
    J = (1 / 2) * add(multiply(J, R))
    J = J + (_lambda / 2) * add(power(theta, 2))
    J = J + (_lambda / 2) * add(power(X, 2))
    return J


def grad(params, Y, R, num_users, num_products, num_features, _lambda):
    """Calculates gradient of Collaborative Filtering's parameters

    Args:
        params (numpy.array): flattened product and user features..
        Y (numpy.array): Scores' matrix.
        R (numpy.array): Matrix of 0s and 1s (whether there's a rating).
        num_users (int): Number of users in this instance.
        num_products (int): Number of products in this instance.
        num_features (int): Number of features in this instance.
        _lambda (float): The regularization hyperparameter.

    Returns:
        numpy.array: Flattened gradient of product and user parameters.
    """

    X, theta = unravel_params(params, num_users, num_products, num_features)
    X_grad = zeros(X.shape)
    theta_grad = zeros(theta.shape)

    for i in range(num_products):
        idx = where(R[i, :] == 1)  # users that have rated product i
        X_grad[i, :] = (X[i, :].dot(theta[idx].T)
                        - Y[i, idx[0]]).dot(theta[idx])
        X_grad[i, :] = X_grad[i, :] + _lambda * X[i, :]

    for j in range(num_users):
        idx = where(R[:, j] == 1)  # products that have been rated by user j
        theta_grad[j, :] = (theta[j, :].dot(
            X[idx].T) - Y[idx[0], j]).dot(X[idx])
        theta_grad[j, :] = theta_grad[j, :] + _lambda * theta[j, :]

    flat_params = append(X_grad.flatten(), theta_grad.flatten())
    return flat_params
