import numpy as np

# sigmoid function
g = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
# predict function
p = np.vectorize(lambda x: 1 if x >= 0.5 else 0)


def h(X, theta):
    return g(X.dot(theta))


def grad(theta, X, y, m):

    grad = (1 / m) * (X.T).dot(h(X, theta) - y)
    return grad


def cost_function(theta, X, y):

    m = len(y)
    J = (1 / m) * ((-y.T).dot(np.log(h(X, theta)))
                   - ((1 - y).T).dot(np.log(1 - h(X, theta))))

    return J


def predict_prob(theta, X):
    return g(X.dot(theta))


def predict(theta, X):
    return p(predict_prob(theta, X))
