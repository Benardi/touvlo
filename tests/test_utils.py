from math import radians

import pytest
from numpy import array, cos, sin, exp
from numpy.testing import assert_allclose

from ml_algorithms.utils import numerical_grad, g_grad


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
