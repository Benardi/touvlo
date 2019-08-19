def cost_function(X, y, theta):

    m = len(y)  # number of training examples
    J = (1 / (2 * m)) * ((X.dot(theta) - y).T).dot(X.dot(theta) - y)
    return J


# model hypothesis
def h(X, theta):
    return X.dot(theta)


def grad(theta, X, y, m):

    grad = (1 / m) * (X.T).dot(h(X, theta) - y)
    return grad
