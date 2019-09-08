from numpy import zeros, float64
from numpy.linalg import inv, LinAlgError


# model hypothesis
def h(X, theta):
    return X.dot(theta)


def cost_func(X, y, theta):

    m = len(y)  # number of training examples
    J = (1 / (2 * m)) * ((h(X, theta) - y).T).dot(h(X, theta) - y)
    return J


def reg_cost_func(X, y, theta, _lambda):

    m = len(y)  # number of training examples
    J = (1 / (2 * m)) * ((h(X, theta) - y).T).dot(h(X, theta) - y)
    J = J + (_lambda / (2 * m)) * (theta[1:, :].T).dot(theta[1:, :])
    return J


def grad(X, y, theta):

    m = len(y)
    grad = (1 / m) * (X.T).dot(h(X, theta) - y)
    return grad


def reg_grad(X, y, theta, _lambda):

    m = len(y)
    grad = (1 / m) * (X.T).dot(h(X, theta) - y)
    grad[1:, :] = grad[1:, :] + (_lambda / m) * theta[1:, :]
    return grad


def predict(X, theta):
    return X.dot(theta)


def normal_eqn(X, y):
    n = X.shape[1]  # number of columns
    theta = zeros((n + 1, 1), dtype=float64)

    try:
        X_T = X.T
        theta = inv(X_T.dot(X)).dot(X_T).dot(y)

    except LinAlgError:
        pass

    return theta
