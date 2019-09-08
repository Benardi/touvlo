from numpy import log, zeros, vectorize, exp

# sigmoid function
g = vectorize(lambda x: 1 / (1 + exp(-x)))
# predict function
p = vectorize(lambda x: 1 if x >= 0.5 else 0)


def h(X, theta):
    return g(X.dot(theta))


def cost_func(X, y, theta):

    m = len(y)
    J = (1 / m) * ((-y.T).dot(log(h(X, theta)))
                   - ((1 - y).T).dot(log(1 - h(X, theta))))
    return J


def reg_cost_func(X, y, theta, _lambda):

    m = len(y)
    J = - (1 / m) * ((y.T).dot(log(h(X, theta)))
                     + (1 - y.T).dot(log(1 - h(X, theta))))
    J = J + (_lambda / (2 * m)) * ((theta[1:, :]).T).dot(theta[1:, :])
    return J


def grad(X, y, theta):
    m = len(y)
    grad = (1 / m) * (X.T).dot(h(X, theta) - y)
    return grad


def reg_grad(X, y, theta, _lambda):
    m = len(y)
    grad = zeros(theta.shape)

    grad = (1 / m) * (X.T).dot(h(X, theta) - y)
    grad[1:, :] = grad[1:, :] + (_lambda / m) * theta[1:, :]

    return grad


def predict_prob(X, theta):
    return g(X.dot(theta))


def predict(X, theta):
    return p(predict_prob(X, theta))
