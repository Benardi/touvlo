from math import radians

import pytest
from numpy import array, cos, sin, exp
from numpy.testing import assert_allclose

from touvlo.utils import numerical_grad, g_grad, gradient_descent


class TestLogisticRegression:

    @pytest.fixture(scope="module")
    def err(self):
        return 0.0001

    def test_numeric_grad_1(self, err):
        def J(x):
            return sum(3 * (x ** 2))

        theta = array([[0], [4], [10]])

        assert_allclose(array([[0], [24], [60]]),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_numeric_grad_2(self, err):
        def J(x):
            return sum(1 / x)

        theta = array([[5], [8], [20]])

        assert_allclose(array([[-0.04], [-0.015625], [-0.0025]]),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_numeric_grad_3(self, err):
        def J(x):
            return sum(cos(x))

        theta = array([[radians(30)],
                       [radians(45)],
                       [radians(60)],
                       [radians(90)]])

        assert_allclose(array([[-sin(radians(30))],
                               [-sin(radians(45))],
                               [-sin(radians(60))],
                               [-sin(radians(90))]]),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_numeric_grad_4(self, err):
        def J(x):
            return sum(exp(x))

        theta = array([[-10], [-1], [0], [1], [10]])

        assert_allclose(array([[exp(-10)],
                               [exp(-1)],
                               [exp(0)],
                               [exp(1)],
                               [exp(10)]]),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_numeric_grad_5(self, err):
        def J(x):
            return sum(7 * x)

        theta = array([[-10], [-1], [0], [1], [10]])

        assert_allclose(array([[7],
                               [7],
                               [7],
                               [7],
                               [7]]),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_sigmoid_gradient(self):
        z = array([-1, -0.5, 0, 0.5, 1])
        assert_allclose(g_grad(z),
                        [0.196612, 0.235004, 0.25, 0.235004, 0.196612],
                        rtol=0, atol=0.001, equal_nan=False)

    def test_gradient_descent1(self, err):
        def grad(X, y, theta):
            m = len(y)
            grad = (1 / m) * (X.T).dot(X.dot(theta) - y)
            return grad

        X = array([[0, 1, 2], [-1, 5, 3], [2, 0, 1]])
        initial_theta = array([[0], [0], [0]])
        y = array([[0.3], [1.2], [0.5]])
        num_iters = 3
        alpha = 1

        assert_allclose(array([[-46.415], [276.248], [192.204]]),
                        gradient_descent(X, y, grad, initial_theta,
                                         alpha, num_iters),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_gradient_descent2(self, err):
        def grad(X, y, theta, schleem, plumbus, wubba, lubba):
            m = len(y)
            grad = (schleem / (m * wubba))
            grad = grad * (X.T).dot(X.dot(theta) - y)
            grad = grad + plumbus / (2 * lubba)
            return grad

        X = array([[0, 1, 2], [-1, 5, 3], [2, 0, 1]])
        initial_theta = array([[0], [0], [0]])
        y = array([[0.3], [1.2], [0.5]])
        num_iters = 5
        plumbus = 0.8
        schleem = 0.6
        wubba = 3.4
        lubba = 2.7
        alpha = 0.01

        assert_allclose(array([[-0.0078777], [0.0106179], [0.0060865]]),
                        gradient_descent(X, y, grad, initial_theta,
                                         alpha, num_iters, lubba=lubba,
                                         schleem=schleem, wubba=wubba,
                                         plumbus=plumbus),
                        rtol=0, atol=0.001, equal_nan=False)
