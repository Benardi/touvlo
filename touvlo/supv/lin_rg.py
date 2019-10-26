"""
.. module:: lin_rg
    :synopsis: Provides routines to construct a Linear Regression.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from abc import ABC, abstractmethod

from numpy import zeros, float64, ones, append, identity
from numpy.linalg import inv

from touvlo.utils import BGD, SGD, MBGD


# model hypothesis
def h(X, theta):
    """Linear regression hypothesis.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        theta (numpy.array): Column vector of model's parameters.

    Returns:
        numpy.array: The projected value for each line of the dataset.
    """
    return X.dot(theta)


def cost_func(X, y, theta):
    """Computes the cost function J for Linear Regression.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        y (numpy.array): Column vector of expected values.
        theta (numpy.array): Column vector of model's parameters.

    Returns:
        float: Computed cost.
    """
    m = len(y)  # number of training examples
    J = (1 / (2 * m)) * ((h(X, theta) - y).T).dot(h(X, theta) - y)
    return J


def reg_cost_func(X, y, theta, _lambda):
    """Computes the regularized cost function J for Linear Regression.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        y (numpy.array): Column vector of expected values.
        theta (numpy.array): Column vector of model's parameters.
        _lambda (float): The regularization hyperparameter.

    Returns:
        float: Computed cost with regularization.
    """
    m = len(y)  # number of training examples
    J = (1 / (2 * m)) * ((h(X, theta) - y).T).dot(h(X, theta) - y)
    J = J + (_lambda / (2 * m)) * (theta[1:, :].T).dot(theta[1:, :])
    return J


def grad(X, y, theta):
    """Computes the gradient for Linear Regression.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        y (numpy.array): Column vector of expected values.
        theta (numpy.array): Column vector of model's parameters.

    Returns:
        numpy.array: Gradient column vector.
    """
    m = len(y)
    grad = (1 / m) * (X.T).dot(h(X, theta) - y)
    return grad


def reg_grad(X, y, theta, _lambda):
    """Computes the regularized gradient for Linear Regression.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        y (numpy.array): Column vector of expected values.
        theta (numpy.array): Column vector of model's parameters.
        _lambda (float): The regularization hyperparameter.

    Returns:
        numpy.array: Regularized gradient column vector.
    """
    m = len(y)
    grad = (1 / m) * (X.T).dot(h(X, theta) - y)
    grad[1:, :] = grad[1:, :] + (_lambda / m) * theta[1:, :]
    return grad


def predict(X, theta):
    """Computes prediction vector.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        theta (numpy.array): Column vector of model's parameters.

    Returns:
        numpy.array: vector with predictions for each input line.
    """
    return X.dot(theta)


def normal_eqn(X, y):
    """Produces optimal theta via normal equation.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        y (numpy.array): Column vector of expected values.

    Raises:
        LinAlgError

    Returns:
        numpy.array: Optimized model parameters theta.
    """
    n = X.shape[1]  # number of columns
    theta = zeros((n, 1), dtype=float64)

    X_T = X.T
    theta = inv(X_T.dot(X)).dot(X_T).dot(y)

    return theta


def reg_normal_eqn(X, y, _lambda):
    """Produces optimal theta via normal equation.

    Args:
        X (numpy.array): Features' dataset plus bias column.
        y (numpy.array): Column vector of expected values.
        _lambda (float): The regularization hyperparameter.

    Returns:
        numpy.array: Optimized model parameters theta.
    """
    n = X.shape[1]  # number of columns, already has bias
    theta = zeros((n, 1), dtype=float64)
    L = identity(n)
    L[0, 0] = 0
    X_T = X.T

    theta = inv(X_T.dot(X) + _lambda * L).dot(X_T).dot(y)

    return theta


class AbstractLinearRegression(ABC):
    """Represents the definitons common to all Linear Regressions.

    Args:
        theta (numpy.array): Column vector of model's parameters.
            Defaults to None

    Attributes:
        theta (numpy.array): Column vector of model's parameters.
    """

    def __init__(self, theta=None):
        self.theta = theta

    def _add_bias_term(self, X):
        m = len(X)
        intercept = ones((m, 1), dtype=int)
        X = append(intercept, X, axis=1)
        return X

    def predict(self, X):
        """Computes prediction vector.

        Args:
            X (numpy.array): Features' dataset.
        Returns:
            numpy.array: vector with a prediction for each example.
        """
        X = self._add_bias_term(X)
        return predict(X, self.theta)

    @abstractmethod
    def _normal_fit(self, X, y):
        pass

    @abstractmethod
    def _gradient_fit(self, grad_descent, X, y, alpha, num_iters):
        pass

    def fit(self, X, y, strategy='BGD', alpha=0.1,
            num_iters=100, **kwargs):
        """Adjusts model parameters to training data.

        Args:
            X (numpy.array): Features' dataset.
            y (numpy.array): Column vector of expected values.
            strategy (str) : Which optimization strategy should be employed:

             * 'BGD': Performs Batch Gradient Descent
             * 'SGD': Performs Stochastic Gradient Descent
             * 'MBGD': Performs Mini-Batch Gradient Descent
             * 'normal_equation': Employs Normal Equation
            alpha (float): Learning rate or _step size of the optimization.
            num_iters (int): Number of times the optimization will be
                             performed.

        Returns:
            None
        """
        if isinstance(strategy, str) and strategy == 'BGD':
            self._gradient_fit(BGD, X, y, alpha, num_iters, **kwargs)

        elif isinstance(strategy, str) and strategy == 'SGD':
            self._gradient_fit(SGD, X, y, alpha, num_iters, **kwargs)

        elif isinstance(strategy, str) and strategy == 'MBGD':
            self._gradient_fit(MBGD, X, y, alpha, num_iters, **kwargs)

        elif isinstance(strategy, str) and strategy == 'normal_equation':
            self._normal_fit(X, y)
        else:
            raise ValueError(
                ("'%s' (type '%s') was passed. " % (strategy, type(strategy)),
                 "The strategy parameter for the fit function should ",
                 "be 'BGD' or 'SGD' or 'MBGD' or 'normal_equation'."))


class LinearRegression(AbstractLinearRegression):
    """Represents an Unregularized Linear Regression model

    Args:
        theta (numpy.array): Column vector of model's parameters.
            Defaults to None

    Attributes:
        theta (numpy.array): Column vector of model's parameters.
    """

    def __init__(self, theta=None):
        super().__init__(theta)

    def _normal_fit(self, X, y):
        X = self._add_bias_term(X)
        self.theta = normal_eqn(X, y)

    def _gradient_fit(self, grad_descent, X, y, alpha,
                      num_iters, **kwargs):
        if self.theta is None:
            _, n = X.shape
            self.theta = zeros((n + 1, 1), dtype=float64)

        X = self._add_bias_term(X)
        self.theta = grad_descent(X, y, grad, self.theta, alpha,
                                  num_iters, **kwargs)

    def cost(self, X, y):
        """Computes the cost function J for Linear Regression.

        Args:
            X (numpy.array): Features' dataset plus bias column.
            y (numpy.array): Column vector of expected values.

        Returns:
            float: Computed cost.
        """
        X = self._add_bias_term(X)
        return cost_func(X, y, self.theta)


class RidgeLinearRegression(AbstractLinearRegression):
    """Represents a Ridge Linear Regression model

    Args:
        theta (numpy.array): Column vector of model's parameters.
            Defaults to None
        _lambda (float): The regularization hyperparameter.

    Attributes:
        theta (numpy.array): Column vector of model's parameters.
        _lambda (float): The regularization hyperparameter.
    """

    def __init__(self, theta=None, _lambda=0):
        super().__init__(theta)
        self._lambda = _lambda

    def _normal_fit(self, X, y):
        X = self._add_bias_term(X)
        self.theta = reg_normal_eqn(X, y, self._lambda)

    def _gradient_fit(self, grad_descent, X, y, alpha,
                      num_iters, **kwargs):
        if self.theta is None:
            _, n = X.shape
            self.theta = zeros((n + 1, 1), dtype=float64)

        X = self._add_bias_term(X)
        self.theta = grad_descent(X, y, reg_grad, self.theta, alpha,
                                  num_iters, _lambda=self._lambda, **kwargs)

    def cost(self, X, y, **kwargs):
        """Computes the regularized cost function J for
            Ridge Linear Regression.

        Args:
            X (numpy.array): Features' dataset plus bias column.
            y (numpy.array): Column vector of expected values.

        Returns:
            float: Computed cost.
        """
        X = self._add_bias_term(X)
        return reg_cost_func(X, y, self.theta, self._lambda, **kwargs)
