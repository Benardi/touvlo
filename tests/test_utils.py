from math import radians

import pytest
from numpy import array, cos, sin, exp
from numpy.testing import assert_allclose

from touvlo.utils import (numerical_grad, g_grad, BGD, SGD,
                          MBGD, mean_normlztn, feature_normalize)


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

    def test_BGD1(self, err):
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
                        BGD(X, y, grad, initial_theta,
                            alpha, num_iters),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_BGD2(self, err):
        def grad(X, y, theta):
            m = len(y)
            grad = (1 / m) * (X.T).dot(X.dot(theta) - y)
            return grad

        X = array([[0, 1, 2], [-1, 5, 3], [2, 0, 1]])
        initial_theta = array([[0.3], [2.7], [1.6]])
        y = array([[0.3], [1.2], [0.5]])
        num_iters = 4
        alpha = 0.001

        assert_allclose(array([[0.31748], [2.58283], [1.51720]]),
                        BGD(X, y, grad, initial_theta,
                            alpha, num_iters),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_BGD3(self, err):
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
                        BGD(X, y, grad, initial_theta,
                            alpha, num_iters, lubba=lubba,
                            schleem=schleem, wubba=wubba,
                            plumbus=plumbus),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_SGD1(self, err):
        def grad(X, y, theta):
            m = len(y)
            grad = (1 / m) * (X.T).dot(X.dot(theta) - y)
            return grad

        X = array([[0, 1, 2], [-1, 5, 3], [2, 0, 1]])
        initial_theta = array([[0], [0], [0]])
        y = array([[0.3], [1.2], [0.5]])
        num_iters = 1
        alpha = 1

        assert_allclose(array([[6.1000], [-10.2000], [-3.7000]]),
                        SGD(X, y, grad, initial_theta,
                            alpha, num_iters),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_SGD2(self, err):
        def grad(X, y, theta):
            m = len(y)
            grad = (1 / m) * (X.T).dot(X.dot(theta) - y)
            return grad

        X = array([[0, 1, 2], [-1, 5, 3], [2, 0, 1]])
        initial_theta = array([[0], [0], [0]])
        y = array([[0.3], [1.2], [0.5]])
        num_iters = 10
        alpha = 0.01

        assert_allclose(array([[0.042237], [0.162748], [0.133705]]),
                        SGD(X, y, grad, initial_theta,
                            alpha, num_iters),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_SGD3(self, err):
        def grad(X, y, theta):
            m = len(y)
            grad = (1 / m) * (X.T).dot(X.dot(theta) - y)
            return grad

        X = array([[0, 1, 2], [-1, 5, 3], [2, 0, 1]])
        initial_theta = array([[0.3], [2.7], [1.6]])
        y = array([[0.3], [1.2], [0.5]])
        num_iters = 5
        alpha = 0.001

        assert_allclose(array([[0.36143], [2.28673], [1.30772]]),
                        SGD(X, y, grad, initial_theta,
                            alpha, num_iters),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_SGD4(self, err):
        def grad(X, y, theta, schleem, plumbus, wubba, lubba):
            m = len(y)
            grad = (schleem / (m * wubba))
            grad = grad * (X.T).dot(X.dot(theta) - y)
            grad = grad + plumbus / (2 * lubba)
            return grad

        X = array([[0, 1, 2], [-1, 5, 3], [2, 0, 1]])
        initial_theta = array([[0.3], [2.7], [1.6]])
        y = array([[0.3], [1.2], [0.5]])
        num_iters = 8
        plumbus = 1.2
        schleem = 0.9
        wubba = 2.4
        lubba = 3
        alpha = 0.005

        assert_allclose(array([[0.42789], [1.63920], [0.84140]]),
                        SGD(X, y, grad, initial_theta,
                            alpha, num_iters, lubba=lubba,
                            schleem=schleem, wubba=wubba,
                            plumbus=plumbus),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_MBGD1(self, err):
        def grad(X, y, theta):
            m = len(y)
            grad = (1 / m) * (X.T).dot(X.dot(theta) - y)
            return grad

        X = array([[0, 1, 2], [-1, 5, 3.6], [2, 0.5, 1],
                   [2, -1, 4], [3, 6, -3], [0, 7.8, 3.5],
                   [2.5, 3, 4.3], [3.2, 5.7, -3], [0, 7.8, 3.5], [9, 8, 7]])
        initial_theta = array([[0], [0], [0]])
        y = array([[0.3], [1.2], [0.5], [0.8], [1.5],
                   [-0.75], [0.43], [0.62], [0.85], [-0.3]])
        num_iters = 1
        alpha = 0.1
        b = 3

        assert_allclose(array([[-18.314], [-15.212], [-14.151]]),
                        MBGD(X, y, grad, initial_theta,
                             alpha, num_iters, b),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_MBGD2(self, err):
        def grad(X, y, theta):
            m = len(y)
            grad = (1 / m) * (X.T).dot(X.dot(theta) - y)
            return grad

        X = array([[0, 1, 2], [-1, 5, 3.6], [2, 0.5, 1],
                   [2, -1, 4], [3, 6, -3], [0, 7.8, 3.5],
                   [2.5, 3, 4.3], [3.2, 5.7, -3], [0, 7.8, 3.5], [9, 8, 7]])
        initial_theta = array([[3], [-2], [0.7]])
        y = array([[0.3], [1.2], [0.5], [0.8], [1.5],
                   [-0.75], [0.43], [0.62], [0.85], [-0.3]])
        num_iters = 1
        alpha = 0.05
        b = 3

        assert_allclose(array([[-2.9970], [-5.3434], [-3.5491]]),
                        MBGD(X, y, grad, initial_theta,
                             alpha, num_iters, b),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_MBGD3(self, err):
        def grad(X, y, theta):
            m = len(y)
            grad = (1 / m) * (X.T).dot(X.dot(theta) - y)
            return grad

        X = array([[0, 1, 2], [-1, 5, 3.6], [2, 0.5, 1],
                   [2, -1, 4], [3, 6, -3], [0, 7.8, 3.5],
                   [2.5, 3, 4.3], [3.2, 5.7, -3], [0, 7.8, 3.5], [9, 8, 7]])
        initial_theta = array([[3], [-2], [0.7]])
        y = array([[0.3], [1.2], [0.5], [0.8], [1.5],
                   [-0.75], [0.43], [0.62], [0.85], [-0.3]])
        num_iters = 3
        alpha = 0.05
        b = 4

        assert_allclose(array([[-4.6898], [-4.1763], [-6.0834]]),
                        MBGD(X, y, grad, initial_theta,
                             alpha, num_iters, b),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_MBGD4(self, err):
        def grad(X, y, theta):
            m = len(y)
            grad = (1 / m) * (X.T).dot(X.dot(theta) - y)
            return grad

        X = array([[0, 1, 2], [-1, 5, 3.6], [2, 0.5, 1],
                   [2, -1, 4], [3, 6, -3], [0, 7.8, 3.5],
                   [2.5, 3, 4.3], [3.2, 5.7, -3], [0, 7.8, 3.5], [9, 8, 7]])
        initial_theta = array([[3], [-2], [0.7]])
        y = array([[0.3], [1.2], [0.5], [0.8], [1.5],
                   [-0.75], [0.43], [0.62], [0.85], [-0.3]])
        num_iters = 1
        alpha = 0.05
        b = 5

        assert_allclose(array([[0.67576], [-2.01056], [-0.80705]]),
                        MBGD(X, y, grad, initial_theta,
                             alpha, num_iters, b),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_MBGD5(self, err):
        def grad(X, y, theta, schleem, plumbus, wubba, lubba):
            m = len(y)
            grad = (schleem / (m * wubba))
            grad = grad * (X.T).dot(X.dot(theta) - y)
            grad = grad + plumbus / (2 * lubba)
            return grad

        X = array([[0, 1, 2], [-1, 5, 3.6], [2, 0.5, 1],
                   [2, -1, 4], [3, 6, -3], [0, 7.8, 3.5],
                   [2.5, 3, 4.3], [3.2, 5.7, -3], [0, 7.8, 3.5], [9, 8, 7]])
        initial_theta = array([[3], [-2], [0.7]])
        y = array([[0.3], [1.2], [0.5], [0.8], [1.5],
                   [-0.75], [0.43], [0.62], [0.85], [-0.3]])
        num_iters = 1
        plumbus = 1.2
        schleem = 0.9
        wubba = 2.4
        lubba = 3
        alpha = 0.05
        b = 5

        assert_allclose(array([[2.27894], [-1.54560], [0.30656]]),
                        MBGD(X, y, grad, initial_theta,
                             alpha, num_iters, b, lubba=lubba,
                             schleem=schleem, wubba=wubba,
                             plumbus=plumbus),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_MBGD6(self, err):
        def grad(X, y, theta, schleem, plumbus, wubba, lubba):
            m = len(y)
            grad = (schleem / (m * wubba))
            grad = grad * (X.T).dot(X.dot(theta) - y)
            grad = grad + plumbus / (2 * lubba)
            return grad

        X = array([[0, 1, 2], [-1, 5, 3.6], [2, 0.5, 1],
                   [2, -1, 4], [3, 6, -3], [0, 7.8, 3.5],
                   [2.5, 3, 4.3], [3.2, 5.7, -3], [0, 7.8, 3.5], [9, 8, 7]])
        initial_theta = array([[3], [-2], [0.7]])
        y = array([[0.3], [1.2], [0.5], [0.8], [1.5],
                   [-0.75], [0.43], [0.62], [0.85], [-0.3]])
        num_iters = 5
        plumbus = 1.2
        schleem = 0.9
        wubba = 2.4
        lubba = 3
        alpha = 0.1
        b = 5

        assert_allclose(array([[-0.0062510], [-0.3414393], [-0.3127686]]),
                        MBGD(X, y, grad, initial_theta,
                             alpha, num_iters, b, lubba=lubba,
                             schleem=schleem, wubba=wubba,
                             plumbus=plumbus),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_mean_normalization(self):
        Y = array([[5, 4, 0, 0],
                   [3, 0, 0, 0],
                   [4, 0, 0, 0],
                   [3, 0, 0, 0],
                   [3, 0, 0, 0]])
        R = array([[1, 1, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 0]])

        Y_norm, Y_mean = mean_normlztn(Y, R)

        assert_allclose(Y_norm,
                        array([[0.5, -0.5, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(Y_mean,
                        array([[4.5], [3], [4], [3], [3]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_feature_normalize(self, err):
        X = array([[1.5, 8., 7.5], [1., 6., 9.]])
        X_norm, mu, sigma = feature_normalize(X)

        assert_allclose(array([1.25, 7, 8.25]),
                        mu,
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(array([0.35355339, 1.41421356, 1.06066017]),
                        sigma,
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(array([[0.707, 0.707, -0.707],
                               [-0.707, -0.707, 0.707]]),
                        X_norm,
                        rtol=0, atol=0.001, equal_nan=False)
