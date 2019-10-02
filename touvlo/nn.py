"""
.. module:: nn
    :synopsis: Provides routines to construct a Neural Network.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from math import sqrt

from numpy.random import uniform
from numpy import (float64, ones, append, sum, dot, log,
                   power, zeros, reshape, empty)

from touvlo.utils import g, g_grad


def cost_function(X, y, theta, _lambda, num_labels, n_hidden_layers=1):
    """Computes the cost function J for Neural Network.

    :param X: Features' dataset.
    :type X: numpy.array

    :param y: Column vector of expected values.
    :type y: numpy.array

    :param theta: Column vector of model's parameters.
    :type theta: numpy.array

    :param _lambda: The regularization hyperparameter.
    :type _lambda: float

    :param num_labels: Number of classes in multiclass classification.
    :type num_labels: int

    :param n_hidden_layers: Number of hidden layers in network.
    :type n_hidden_layers: int

    :returns: Computed cost.
    :rtype: float
    """
    m, n = X.shape
    intercept = ones((m, 1), dtype=float64)
    X = append(intercept, X, axis=1)

    z, a = feed_forward(X, theta, n_hidden_layers)
    L = n_hidden_layers + 1  # last layer

    J = 0
    for c in range(num_labels):
        _J = dot(1 - (y == c).T, log(1 - a[L][:, c]))
        _J = _J + dot((y == c).T, log(a[L][:, c]))
        J = J - (1 / m) * sum(_J)

    theta_squared_term = 0
    for j in range(len(theta)):
        theta_squared_term += sum(power(theta[j][:, 1:], 2))

    J = J + (_lambda / (2 * m)) * theta_squared_term

    return J


def unravel_params(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, n_hidden_layers=1):
    """Unravels flattened array into list of weight matrices

    :param nn_params: Row vector of model's parameters.
    :type nn_params: numpy.array

    :param input_layer_size: Number of units in the input layer.
    :type input_layer_size: int

    :param hidden_layer_size: Number of units in a hidden layer.
    :type input_layer_size: int

    :param num_labels: Number of classes in multiclass classification.
    :type num_labels: int

    :param n_hidden_layers: Number of hidden layers in network.
    :type n_hidden_layers: int

    :returns: array with model's weight matrices.
    :rtype: numpy.array(numpy.array)
    """
    input_layer_n_units = hidden_layer_size * (input_layer_size + 1)
    hidden_layer_n_units = hidden_layer_size * (hidden_layer_size + 1)

    theta = empty((n_hidden_layers + 1), dtype=object)

    # input layer to hidden layer
    theta[0] = nn_params[0:input_layer_n_units]
    theta[0] = reshape(theta[0], (hidden_layer_size, (input_layer_size + 1)))

    # hidden layer to hidden layer
    for i in range(1, n_hidden_layers):

        start = input_layer_n_units + (i - 1) * hidden_layer_n_units
        end = input_layer_n_units + i * hidden_layer_n_units

        theta[i] = nn_params[start:end]
        theta[i] = reshape(
            theta[i], (hidden_layer_size, (hidden_layer_size + 1)))

    # hidden layer to output layer
    start = input_layer_n_units + (n_hidden_layers - 1) * hidden_layer_n_units

    theta[n_hidden_layers] = nn_params[start:]
    theta[n_hidden_layers] = reshape(theta[n_hidden_layers],
                                     (num_labels, (hidden_layer_size + 1)))

    return theta


def feed_forward(X, theta, n_hidden_layers=1):
    """Applies forward propagation to calculate model's hypothesis.

    :param X: Features' dataset.
    :type X: numpy.array

    :param theta: Column vector of model's parameters.
    :type theta: numpy.array

    :param n_hidden_layers: Number of hidden layers in network.
    :type n_hidden_layers: int

    :returns:
        - z - array of parameters prior to activation by layer.
        - a - array of activation matrices by layer.

    :rtype:
        - z (:py:class: numpy.array(numpy.array))
        - a (:py:class: numpy.array(numpy.array))
    """
    z = empty((n_hidden_layers + 2), dtype=object)
    a = empty((n_hidden_layers + 2), dtype=object)

    # Input layer
    a[0] = X

    # Hidden unit layers
    for l in range(1, (len(a) - 1)):
        z[l] = a[l - 1].dot(theta[l - 1].T)
        a[l] = g(z[l])
        a[l] = append(ones((len(a[l]), 1), float64),  # add intercept
                      a[l], axis=1)

    # Output layer
    z[len(a) - 1] = a[(len(a) - 2)].dot(theta[(len(a) - 2)].T)
    a[len(a) - 1] = g(z[len(a) - 1])  # hypothesis

    return z, a


def back_propagation(y, theta, a, z, num_labels, n_hidden_layers=1):
    """Applies back propagation to minimize model's loss.

    :param y: Column vector of expected values.
    :type y: numpy.array

    :param theta: array of model's weight matrices by layer.
    :type theta: numpy.array(numpy.array)

    :param a: array of activation matrices by layer.
    :type a: numpy.array(numpy.array)

    :param z: array of parameters prior to sigmoid by layer.
    :type z: numpy.array(numpy.array)

    :param num_labels: Number of classes in multiclass classification.
    :type num_labels: int

    :param n_hidden_layers: Number of hidden layers in network.
    :type n_hidden_layers: int

    :returns: array of matrices of 'error values' by layer.
    :rtype: numpy.array(numpy.array)
    """
    delta = empty((n_hidden_layers + 2), dtype=object)
    L = n_hidden_layers + 1  # last layer
    delta[L] = zeros(shape=a[L].shape, dtype=float64)

    for c in range(num_labels):
        delta[L][:, c] = a[L][:, c] - (y == c)

    for l in range(L, 1, -1):
        delta[l - 1] = delta[l].dot(theta[l - 1])[:, 1:] * g_grad(z[l - 1])

    return delta


def grad(X, y, nn_params, _lambda, input_layer_size,
         hidden_layer_size, num_labels, n_hidden_layers=1):
    """Calculates gradient of neural network's parameters.

    :param X: Features' dataset.
    :type X: numpy.array

    :param y: Column vector of expected values.
    :type y: numpy.array

    :param nn_params: Column vector of model's parameters.
    :type theta: numpy.array

    :param _lambda: The regularization hyperparameter.
    :type _lambda: float

    :param input_layer_size: Number of units in the input layer.
    :type input_layer_size: int

    :param hidden_layer_size: Number of units in a hidden layer.
    :type input_layer_size: int

    :param num_labels: Number of classes in multiclass classification.
    :type num_labels: int

    :param n_hidden_layers: Number of hidden layers in network.
    :type n_hidden_layers: int

    :returns: array of gradient values by weight matrix.
    :rtype: numpy.array(numpy.array)
    """
    theta = unravel_params(nn_params, input_layer_size, hidden_layer_size,
                           num_labels, n_hidden_layers)

    # Initi gradient with zeros
    theta_grad = empty((n_hidden_layers + 1), dtype=object)
    for i in range(len(theta)):
        theta_grad[i] = zeros(shape=theta[i].shape, dtype=float64)

    m, n = X.shape
    intercept = ones((m, 1), dtype=float64)
    X = append(intercept, X, axis=1)

    for t in range(m):

        z, a = feed_forward(X[[t], :], theta, n_hidden_layers)
        delta = back_propagation(y[t, :], theta, a, z,
                                 num_labels, n_hidden_layers)

        for l in range(len(theta_grad)):
            theta_grad[l] = theta_grad[l] + dot(delta[l + 1].T, a[l])

    for i in range(len(theta_grad)):
        theta_grad[i] = (1 / m) * theta_grad[i]

    # regularization
    for i in range(len(theta_grad)):
        theta_grad[i][:, 1:] = theta_grad[i][:, 1:] + \
            (_lambda / m) * theta[i][:, 1:]

    flat_theta_grad = append(theta_grad[0].flatten(), theta_grad[1].flatten())
    for i in range(2, len(theta_grad)):
        flat_theta_grad = append(flat_theta_grad, theta_grad[i].flatten())

    return flat_theta_grad


def rand_init_weights(L_in, L_out):
    """Initializes weight matrix with random values.

    :param X: Features' dataset.
    :type X: numpy.array

    :param L_in: Number of units in previous layer.
    :type L_in: int

    :param n_hidden_layers: Number of units in next layer.
    :type n_hidden_layers: int

    :returns: Random values' matrix of conforming dimensions.
    :rtype: numpy.array
    """
    W = zeros((L_out, 1 + L_in), float64)  # plus 1 for bias term
    epsilon_init = sqrt(6) / sqrt((L_in + 1) + L_out)

    W = uniform(size=(L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init
    return W


def init_nn_weights(input_layer_size, hidden_layer_size,
                    num_labels, n_hidden_layers=1):
    """Initialize the weight matrices of a network with random values.

    :param hidden_layer_size: Number of units in a hidden layer.
    :type input_layer_size: int

    :param input_layer_size: Number of units in the input layer.
    :type input_layer_size: int

    :param num_labels: Number of classes in multiclass classification.
    :type num_labels: int

    :param n_hidden_layers: Number of hidden layers in network.
    :type n_hidden_layers: int

    :returns: array of weight matrices of random values.
    :rtype: numpy.array(numpy.array)
    """
    theta = empty((n_hidden_layers + 1), dtype=object)
    theta[0] = rand_init_weights(input_layer_size, hidden_layer_size)

    for l in range(1, n_hidden_layers):
        theta[l] = rand_init_weights(hidden_layer_size, hidden_layer_size)

    theta[n_hidden_layers] = rand_init_weights(hidden_layer_size, num_labels)

    return theta
