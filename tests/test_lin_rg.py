import os

import pytest
from numpy.testing import assert_allclose
from numpy import ones, zeros, float64, array, append, genfromtxt

from ml_algorithms.lin_rg import (normal_eqn, cost_func,
                                  reg_cost_func, grad,
                                  reg_grad, predict, h)
from ml_algorithms.utils import numerical_grad

TESTDATA1 = os.path.join(os.path.dirname(__file__), 'data1.csv')
TESTDATA2 = os.path.join(os.path.dirname(__file__), 'data2.csv')


class TestLinearRegression:

    @pytest.fixture(scope="module")
    def data1(self):
        return genfromtxt(TESTDATA1, delimiter=',')

    @pytest.fixture(scope="module")
    def data2(self):
        return genfromtxt(TESTDATA2, delimiter=',')

    @pytest.fixture(scope="module")
    def err(self):
        return 1e-4


# NORMAL EQUATION

    def test_normal_eqn_data1(self, data1):

        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=int)
        X = append(intercept, X, axis=1)

        assert_allclose([[-3.896], [1.193]],
                        normal_eqn(X, y),
                        rtol=0, atol=0.001)

    def test_normal_eqn_data2(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=int)
        X = append(intercept, X, axis=1)

        assert_allclose([[89597.909], [139.210], [-8738.019]],
                        normal_eqn(X, y),
                        rtol=0, atol=0.001)

# COST FUNCTION

    def test_cost_func_data1_1(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = zeros((n + 1, 1), dtype=float64)

        assert_allclose([[32.073]],
                        cost_func(X, y, theta),
                        rtol=0, atol=0.001)

    def test_cost_func_data1_2(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = ones((n + 1, 1), dtype=float64)

        assert_allclose([[10.266]],
                        cost_func(X, y, theta),
                        rtol=0, atol=0.001)

    def test_cost_func_data1_3(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-1], [2]])

        assert_allclose([[54.242]],
                        cost_func(X, y, theta),
                        rtol=0, atol=0.001)

    def test_cost_func_data2_1(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = zeros((n + 1, 1), dtype=float64)

        assert_allclose([[65591548106.457]],
                        cost_func(X, y, theta),
                        rtol=0, atol=0.001)

    def test_cost_func_data2_2(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = ones((n + 1, 1), dtype=float64)

        assert_allclose([[64828197300.798]],
                        cost_func(X, y, theta),
                        rtol=0, atol=0.001)

    def test_cost_func_data2_3(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-25.3], [32], [7.8]])

        assert_allclose([[43502644952.311]],
                        cost_func(X, y, theta),
                        rtol=0, atol=0.001)

# REGULARIZED COST FUNCTION

    def test_reg_cost_func_data1_1(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = ones((n + 1, 1), dtype=float64)
        _lambda = 0

        assert_allclose([[10.266]],
                        reg_cost_func(X, y, theta, _lambda),
                        rtol=0, atol=0.001)

    def test_reg_cost_func_data1_2(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = ones((n + 1, 1), dtype=float64)
        _lambda = 100

        assert_allclose([[10.781984]],
                        reg_cost_func(X, y, theta, _lambda),
                        rtol=0, atol=0.001)

    def test_reg_cost_func_data1_3(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-1], [2]])
        _lambda = 750

        assert_allclose([[69.706373]],
                        reg_cost_func(X, y, theta, _lambda),
                        rtol=0, atol=0.001)

    def test_reg_cost_func_data2_1(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = ones((n + 1, 1), dtype=float64)
        _lambda = 0

        assert_allclose([[64828197300.798]],
                        reg_cost_func(X, y, theta, _lambda),
                        rtol=0, atol=0.001)

    def test_reg_cost_func_data2_2(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = ones((n + 1, 1), dtype=float64)
        _lambda = 1000000

        assert_allclose([[64828218577.393623]],
                        reg_cost_func(X, y, theta, _lambda),
                        rtol=0, atol=0.001)

    def test_reg_cost_func_data2_3(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-25.3], [32], [7.8]])
        _lambda = 1000000

        assert_allclose([[43514185803.375198]],
                        reg_cost_func(X, y, theta, _lambda),
                        rtol=0, atol=0.001)

# GRADIENT

    def test_grad_data1_1(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = zeros((n + 1, 1), dtype=float64)

        assert_allclose([[-5.839], [-65.329]],
                        grad(X, y, theta),
                        rtol=0, atol=0.001)

    def test_grad_data1_2(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = ones((n + 1, 1), dtype=float64)

        assert_allclose([[3.321], [24.235]],
                        grad(X, y, theta),
                        rtol=0, atol=0.001)

    def test_grad_data1_3(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-1], [2]])

        assert_allclose([[9.480], [89.319]],
                        grad(X, y, theta),
                        rtol=0, atol=0.001)

    def test_grad_data1_4(self, data1, err):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = (1 / 3) * ones((n + 1, 1), dtype=float64)

        def J(theta):
            return cost_func(X, y, theta)

        assert_allclose(grad(X, y, theta),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001)

    def test_grad_data1_5(self, data1, err):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = - 7.43 * ones((n + 1, 1), dtype=float64)

        def J(theta):
            return cost_func(X, y, theta)

        assert_allclose(grad(X, y, theta),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001)

    def test_grad_data1_6(self, data1, err):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[3.46], [-2.76]])

        def J(theta):
            return cost_func(X, y, theta)

        assert_allclose(grad(X, y, theta),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001)

    def test_grad_data2_1(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = zeros((n + 1, 1), dtype=float64)

        assert_allclose([[-340412.659], [-764209128.191], [-1120367.702]],
                        grad(X, y, theta),
                        rtol=0, atol=0.001)

    def test_grad_data2_2(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = ones((n + 1, 1), dtype=float64)

        assert_allclose([[-338407.808], [-759579615.064], [-1113679.894]],
                        grad(X, y, theta),
                        rtol=0, atol=0.001)

    def test_grad_data2_3(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-25.3], [32], [7.8]])

        assert_allclose([[-276391.445], [-616340858.434], [-906796.414]],
                        grad(X, y, theta),
                        rtol=0, atol=0.001)

# REGULARIZED GRADIENT

    def test_reg_grad_data1_1(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = ones((n + 1, 1), dtype=float64)
        _lambda = 0

        assert_allclose([[3.321], [24.235]],
                        reg_grad(X, y, theta, _lambda),
                        rtol=0, atol=0.001)

    def test_reg_grad_data1_2(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = ones((n + 1, 1), dtype=float64)
        _lambda = 100

        assert_allclose([[3.320665], [25.265821]],
                        reg_grad(X, y, theta, _lambda),
                        rtol=0, atol=0.001)

    def test_reg_grad_data1_3(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-1], [2]])
        _lambda = 750

        assert_allclose([[9.480465], [104.783153]],
                        reg_grad(X, y, theta, _lambda),
                        rtol=0, atol=0.001)

    def test_reg_grad_data1_4(self, data1, err):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -8.4 * ones((n + 1, 1), dtype=float64)
        _lambda = 0.762

        def J(theta):
            return reg_cost_func(X, y, theta, _lambda)

        assert_allclose(reg_grad(X, y, theta, _lambda),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001)

    def test_reg_grad_data1_5(self, data1, err):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = 3.2 * ones((n + 1, 1), dtype=float64)
        _lambda = 154

        def J(theta):
            return reg_cost_func(X, y, theta, _lambda)

        assert_allclose(reg_grad(X, y, theta, _lambda),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001)

    def test_reg_grad_data1_6(self, data1, err):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-12.4], [23.56]])
        _lambda = 943

        def J(theta):
            return reg_cost_func(X, y, theta, _lambda)

        assert_allclose(reg_grad(X, y, theta, _lambda),
                        numerical_grad(J, theta, err),
                        rtol=0, atol=0.001)

    def test_reg_grad_data2_1(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = ones((n + 1, 1), dtype=float64)
        _lambda = 0

        assert_allclose([[-338407.808], [-759579615.064], [-1113679.894]],
                        reg_grad(X, y, theta, _lambda),
                        rtol=0, atol=0.001)

    def test_reg_grad_data2_2(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = ones((n + 1, 1), dtype=float64)
        _lambda = 1000000

        assert_allclose([[-338407.808], [-759558338.468], [-1092403.298]],
                        reg_grad(X, y, theta, _lambda),
                        rtol=0, atol=0.001)

    def test_reg_grad_data2_3(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-25.3], [32], [7.8]])
        _lambda = 1000000

        assert_allclose([[-276391.444681],
                         [-615660007.370213],
                         [-740838.968085]],
                        reg_grad(X, y, theta, _lambda),
                        rtol=0, atol=0.001)

# PREDICT

    def test_predict_1(self):
        X = array([[3.5]])
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-3.6303], [1.1664]])

        assert_allclose([[0.4521]],
                        predict(X, theta),
                        rtol=0, atol=0.001)

    def test_predict_2(self):
        X = array([[3.5]])
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = zeros((n + 1, 1), dtype=float64)

        assert_allclose([[0]],
                        predict(X, theta),
                        rtol=0, atol=0.001)

    def test_predict_3(self):
        X = array([[-3.5, 2.7]])
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = ones((n + 1, 1), dtype=float64)

        assert_allclose([[0.2]],
                        predict(X, theta),
                        rtol=0, atol=0.001)

    def test_predict_4(self):
        X = array([[-3.5, 2.7]])
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -1 * ones((n + 1, 1), dtype=float64)

        assert_allclose([[-0.2]],
                        predict(X, theta),
                        rtol=0, atol=0.001)

# HYPOTHESYS

    def test_h_1(self):
        X = array([[3.5]])
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = array([[-3.6303], [1.1664]])

        assert_allclose([[0.4521]],
                        h(X, theta),
                        rtol=0, atol=0.001)

    def test_h_2(self):
        X = array([[3.5]])
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = zeros((n + 1, 1), dtype=float64)

        assert_allclose([[0]],
                        h(X, theta),
                        rtol=0, atol=0.001)

    def test_h_3(self):
        X = array([[-3.5, 2.7]])
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = ones((n + 1, 1), dtype=float64)

        assert_allclose([[0.2]],
                        h(X, theta),
                        rtol=0, atol=0.001)

    def test_h_4(self):
        X = array([[-3.5, 2.7]])
        m, n = X.shape
        intercept = ones((m, 1), dtype=float64)
        X = append(intercept, X, axis=1)
        theta = -1 * ones((n + 1, 1), dtype=float64)

        assert_allclose([[-0.2]],
                        h(X, theta),
                        rtol=0, atol=0.001)
