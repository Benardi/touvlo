from math import sqrt

from numpy.random import uniform
from numpy import (genfromtxt, float64, ones, append, sum,
                   vectorize, exp, dot, delete, log, power,
                   zeros, reshape, copy, seterr, mean, std, isnan)

from utils import gradient_descent

# sigmoid gradient function
g = lambda x: 1 / (1 + exp(-x))
# sigmoid gradient function
g_grad = lambda x: g(x) * (1 - g(x))


def cost_function(X, theta1, theta2, _lambda):
    m, n = X.shape
    intercept = ones((m, 1), dtype=float64)

    X = append(intercept, X, axis=1)

    # Input layer
    a1 = X

    # First hidden layer
    z2 = a1.dot(theta1.T)
    a2 = g(z2)
    a2 = append(ones((len(a2), 1), float64), a2, axis=1)  # Add intercept

    # Output layer
    z3 = a2.dot(theta2.T)
    h = g(z3)

    J = 0

    for c in range(num_labels):
        _J = dot(1 - (y == c).T, log(1 - h[:, c]))
        _J = dot((y == c).T, log(h[:, c])) + _J
        J = J - (1 / m) * sum(_J)

    J = J + (_lambda / (2 * m)) * (sum(power(theta1[:, 1:], 2)) +
                                   sum(power(theta2[:, 1:], 2)))
    return J


def unravel_params(nn_params, hidden_layer_size,
                   input_layer_size, num_labels):

    theta1 = nn_params[0:(hidden_layer_size * (input_layer_size + 1))]
    theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):]

    theta1 = reshape(theta1,
                     (hidden_layer_size, (input_layer_size + 1)))
    theta2 = reshape(theta2,
                     (num_labels, (hidden_layer_size + 1)))

    return theta1, theta2

def feed_foward(X, theta1, theta2):
    # Input layer
    a1 = X

    # First hidden layer
    z2 = a1.dot(theta1.T)
    a2 = g(z2)
    a2 = append(ones((len(a2), 1), float64),
                a2, axis=1)  # Add intercept

    # Output layer
    z3 = a2.dot(theta2.T)
    a3 = g(z3)

    return a1, a2, a3, z2  

def back_propagation(theta2, h, z2, y_j):

    delta3 = zeros(shape=h.shape, dtype=float64)
    for c in range(num_labels):
        delta3[:, c] = h[:, c] - (y_j == c)

    delta2 = (delta3.dot(theta2))[:, 1:] * g_grad(z2)

    return delta2, delta3

def grad(nn_params, X, y, _lambda):

    theta1, theta2 = unravel_params(nn_params, hidden_layer_size,
                                    input_layer_size, num_labels)
    theta1_grad = zeros(shape=theta1.shape, dtype=float64)
    theta2_grad = zeros(shape=theta2.shape, dtype=float64)

    m, n = X.shape
    intercept = ones((m, 1), dtype=float64)
    X = append(intercept, X, axis=1)

    for t in range(m):

        a1, a2, h, z2 = feed_foward(X[[t],:], theta1, theta2)
        delta2, delta3 = back_propagation(theta2, h, z2, y[t, :])

        theta1_grad = theta1_grad + dot(delta2.T, a1)
        theta2_grad = theta2_grad + dot(delta3.T, a2)

    theta1_grad = (1 / m) * theta1_grad
    theta2_grad = (1 / m) * theta2_grad

    theta1_grad[:, 1:] = theta1_grad[:, 1:] + (_lambda / m) * theta1[:, 1:]
    theta2_grad[:, 1:] = theta2_grad[:, 1:] + (_lambda / m) * theta2[:, 1:]

    theta_grad = append(theta1_grad.flatten(),
                        theta2_grad.flatten())

    return theta_grad


def rand_init_weights(L_in, L_out):
    W = zeros((L_out, 1 + L_in), float64)  # plus 1 for bias term
    epsilon_init = sqrt(6) / sqrt((L_in + 1) + L_out)

    W = uniform(size=(L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init
    return W


#data = genfromtxt("mimic_100lines.csv", delimiter=",",
#                  dtype=float64, skip_header=1)
#X = data[:,:-1]
#y = data[:,[-1]]
X = genfromtxt("X.csv", delimiter=",", dtype=float64)
y = genfromtxt("y.csv", delimiter=",", dtype=float64)
y = y[...,None]
y[y == 10] = 0
_, n = X.shape

input_layer_size = n
hidden_layer_size = 25
num_labels = 10
num_iters = 100
_lambda = 1
alpha = 0.1

