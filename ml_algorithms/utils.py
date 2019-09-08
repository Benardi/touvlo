from numpy import zeros, copy, std, mean, float64


def gradient_descent(X, y, grad, initial_theta, alpha, num_iters):

    m = len(y)
    theta = copy(initial_theta)
    for _ in range(num_iters):
        theta = theta - alpha * (1 / m) * grad(theta, X, y, m)

    return theta


def numerical_grad(J, theta, err):
    num_grad = zeros(theta.shape, dtype=float64)
    perturb = zeros(theta.shape, dtype=float64)

    for i in range(len(theta)):
        perturb[i] = err
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        num_grad[i] = (loss2 - loss1) / (2 * err)
        perturb[i] = 0

    return num_grad


def feature_normalize(X):
    mu = mean(X)
    sigma = std(X)

    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma
