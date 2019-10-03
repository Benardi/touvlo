import os

import pytest
from numpy import ones, zeros, float64, array, append, isclose, genfromtxt
from numpy.testing import assert_allclose

from touvlo.supv.lgx_rg import (cost_func, grad, predict_prob, predict, h,
                                reg_cost_func, reg_grad)
from touvlo.utils import numerical_grad

TESTDATA3 = os.path.join(os.path.dirname(__file__), 'data3.csv')
TESTDATA4 = os.path.join(os.path.dirname(__file__), 'data4.csv')


class TestLogisticRegression:

    @pytest.fixture(scope="module")
    def data3(self):
        return genfromtxt(TESTDATA3, delimiter=',')

    @pytest.fixture(scope="module")
    def data4(self):
        return genfromtxt(TESTDATA4, delimiter=',')

    @pytest.fixture(scope="module")
    def err(self):
        return 1e-4


# COST FUNCTION

    def test_cost_func_data3_1(self, data3):

        y = data3[:, -1:]
        X = data3[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = zeros((n + 1, 1), dtype=float64)

        assert isclose(0.693, cost_func(X, y, theta),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data3_2(self, data3):
        y = data3[:, -1:]
        X = data3[:, :-1]
        m, _ = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-0.1], [0.4], [-0.4]])

        assert isclose(4.227, cost_func(X, y, theta),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data3_3(self, data3):
        y = data3[:, -1:]
        X = data3[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-24], [0.2], [0.2]])

        assert isclose(0.218, cost_func(X, y, theta),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data4_1(self, data4):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = zeros((n + 1, 1), dtype=float64)

        assert isclose(0.693, cost_func(X, y, theta),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data4_2(self, data4):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-0.1], [0.4], [-0.4], [-0.4], [0.4],
                       [-0.4], [0.4], [-0.4], [-0.4]])

        assert_allclose(40.216,
                        cost_func(X, y, theta),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data4_3(self, data4):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -0.1 * ones((n + 1, 1), dtype=float64)

        assert_allclose(14.419,
                        cost_func(X, y, theta),
                        rtol=0, atol=0.001, equal_nan=False)

# REGULARIZED COST FUNCTION

    def test_reg_cost_func_data4_1(self, data4):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -0.1 * ones((n + 1, 1), dtype=float64)
        _lambda = 10

        assert_allclose(14.419,
                        reg_cost_func(X, y, theta, _lambda),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_reg_cost_func_data4_2(self, data4):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -0.1 * ones((n + 1, 1), dtype=float64)
        _lambda = 100

        assert_allclose(14.424,
                        reg_cost_func(X, y, theta, _lambda),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_reg_cost_func_data4_3(self, data4):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -0.1 * ones((n + 1, 1), dtype=float64)
        _lambda = 1000000

        assert_allclose(66.502,
                        reg_cost_func(X, y, theta, _lambda),
                        rtol=0, atol=0.001, equal_nan=False)

# GRADIENT

    def test_grad_data3_1(self, data3):
        y = data3[:, -1:]
        X = data3[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = zeros((n + 1, 1), dtype=float64)

        assert_allclose([[-0.1000], [-12.0092], [-11.2628]],
                        grad(X, y, theta), rtol=0, atol=5e-05,
                        equal_nan=False)

    def test_grad_data3_2(self, data3):
        y = data3[:, -1:]
        X = data3[:, :-1]
        m, _ = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-0.1], [0.4], [-0.4]])

        assert_allclose([[-0.134], [-8.275], [-18.564]],
                        grad(X, y, theta), rtol=0, atol=0.001,
                        equal_nan=False)

    def test_grad_data3_3(self, data3):
        y = data3[:, -1:]
        X = data3[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-24], [0.2], [0.2]])

        assert_allclose([[0.043], [2.566], [2.647]],
                        grad(X, y, theta), rtol=0, atol=0.001,
                        equal_nan=False)

    def test_grad_data3_4(self, data3, err):
        y = data3[:, -1:]
        X = data3[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -3 * ones((n + 1, 1), dtype=float64)

        def J(theta):
            return cost_func(X, y, theta)

        assert_allclose(grad(X, y, theta),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_grad_data3_5(self, data3, err):
        y = data3[:, -1:]
        X = data3[:, :-1]
        m, _ = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-0.87], [0.32], [-0.54]])

        def J(theta):
            return cost_func(X, y, theta)

        assert_allclose(grad(X, y, theta),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_grad_data3_6(self, data3, err):
        y = data3[:, -1:]
        X = data3[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-20], [0.164], [-0.23]])

        def J(theta):
            return cost_func(X, y, theta)

        assert_allclose(grad(X, y, theta),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_grad_data4_1(self, data4):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = zeros((n + 1, 1), dtype=float64)

        assert_allclose(array([[0.151], [0.225], [11.154], [9.838], [2.534],
                               [4.887], [3.733], [0.044], [3.686]]),
                        grad(X, y, theta),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_grad_data4_2(self, data4):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, _ = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-0.1], [0.4], [-0.4], [-0.4], [0.4],
                       [-0.4], [0.4], [-0.4], [-0.4]])

        assert_allclose(array([[-0.349], [-1.698], [-49.293],
                               [-24.715], [-7.734], [-35.013],
                               [-12.263], [-0.192], [-12.935]]),
                        grad(X, y, theta),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_grad_data4_3(self, data4):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -0.1 * ones((n + 1, 1), dtype=float64)

        assert_allclose(array([[-0.349], [-1.698], [-49.293],
                               [-24.715], [-7.734], [-35.013],
                               [-12.263], [-0.192], [-12.935]]),
                        grad(X, y, theta),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_grad_data4_4(self, data4, err):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = zeros((n + 1, 1), dtype=float64)

        def J(theta):
            return cost_func(X, y, theta)

        assert_allclose(grad(X, y, theta),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_grad_data4_5(self, data4, err):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, _ = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-0.198], [0.25], [-0.234], [-0.793], [0.123],
                       [-0.378], [0.423], [-0.678], [-0.3]])

        def J(theta):
            return cost_func(X, y, theta)

        assert_allclose(grad(X, y, theta),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_grad_data4_6(self, data4, err):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -0.145 * ones((n + 1, 1), dtype=float64)

        def J(theta):
            return cost_func(X, y, theta)

        assert_allclose(grad(X, y, theta),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001, equal_nan=False)

# REGULARIZED GRADIENT

    def test_reg_grad_data4_1(self, data4):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -0.1 * ones((n + 1, 1), dtype=float64)
        _lambda = 10

        assert_allclose(array([[-0.349], [-1.699], [-49.294],
                               [-24.716], [-7.736], [-35.014],
                               [-12.265], [-0.193], [-12.936]]),
                        reg_grad(X, y, theta, _lambda),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_reg_grad_data4_2(self, data4):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -0.1 * ones((n + 1, 1), dtype=float64)
        _lambda = 100

        assert_allclose(array([[-0.349], [-1.710], [-49.305],
                               [-24.727], [-7.747], [-35.026],
                               [-12.276], [-0.205], [-12.948]]),
                        reg_grad(X, y, theta, _lambda),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_reg_grad_data4_3(self, data4):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -0.1 * ones((n + 1, 1), dtype=float64)
        _lambda = 1000000

        assert_allclose(array([[-0.348958], [-131.906250],
                               [-179.501291], [-154.923177],
                               [-137.942708], [-165.221354],
                               [-142.471614], [-130.400435],
                               [-143.143226]]),
                        reg_grad(X, y, theta, _lambda),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_reg_grad_data4_4(self, data4, err):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -0.567 * ones((n + 1, 1), dtype=float64)
        _lambda = 17.89

        def J(theta):
            return reg_cost_func(X, y, theta, _lambda)

        assert_allclose(reg_grad(X, y, theta, _lambda),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_reg_grad_data4_5(self, data4, err):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -0.176 * ones((n + 1, 1), dtype=float64)
        _lambda = 78.56

        def J(theta):
            return reg_cost_func(X, y, theta, _lambda)

        assert_allclose(reg_grad(X, y, theta, _lambda),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_reg_grad_data4_6(self, data4, err):
        y = data4[:, -1:]
        X = data4[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -0.238 * ones((n + 1, 1), dtype=float64)
        _lambda = 975032

        def J(theta):
            return reg_cost_func(X, y, theta, _lambda)

        assert_allclose(reg_grad(X, y, theta, _lambda),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001, equal_nan=False)

# HYPOTHESIS

    def test_h_prob1(self):
        X = array([[1, 45, 85]])
        theta = array([[-25.161], [0.206], [0.201]])
        assert isclose(0.767,
                       h(X, theta),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_h_prob2(self):
        X = array([[1, 20, 93]])
        theta = array([[0], [0], [0]])

        assert isclose(0.5,
                       h(X, theta),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_h_prob3(self):
        X = array([[1, 10, -50]])
        theta = array([[5.161], [0.206], [0.201]])

        assert isclose(0.0558,
                       h(X, theta),
                       rtol=0, atol=0.001, equal_nan=False)

# PREDICT PROB

    def test_predict_prob1(self):
        X = array([[1, 45, 85]])
        theta = array([[-25.161], [0.206], [0.201]])

        assert isclose(0.767,
                       predict_prob(X, theta),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_predict_prob2(self):
        X = array([[1, 20, 93]])
        theta = array([[0], [0], [0]])

        assert isclose(0.5,
                       predict_prob(X, theta),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_predict_prob3(self):
        X = array([[1, 10, -50]])
        theta = array([[5.161], [0.206], [0.201]])

        assert isclose(0.0558,
                       predict_prob(X, theta),
                       rtol=0, atol=0.001, equal_nan=False)

# PREDICT

    def test_predict_1(self):
        X = array([[1, 45, 85]])
        theta = array([[-25.161], [0.206], [0.201]])

        assert predict(X, theta)

    def test_predict_2(self):
        X = array([[1, 20, 93]])
        theta = array([[0], [0], [0]])

        assert predict(X, theta)

    def test_predict3(self):
        X = array([[1, 10, -50]])
        theta = array([[5.161], [0.206], [0.201]])

        assert not predict(X, theta)
