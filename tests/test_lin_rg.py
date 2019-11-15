import os

import pytest
from numpy.testing import assert_allclose
from numpy import ones, zeros, float64, array, append, genfromtxt
from numpy.linalg import LinAlgError

from touvlo.lin_rg import (normal_eqn, cost_func, reg_cost_func, grad,
                           reg_grad, predict, h, LinearRegression,
                           RidgeLinearRegression, reg_normal_eqn)
from touvlo.utils import numerical_grad

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
        m, _ = X.shape
        intercept = ones((m, 1), dtype=int)
        X = append(intercept, X, axis=1)

        assert_allclose([[-3.896], [1.193]],
                        normal_eqn(X, y),
                        rtol=0, atol=0.001)

    def test_normal_eqn_data2(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, _ = X.shape
        intercept = ones((m, 1), dtype=int)
        X = append(intercept, X, axis=1)

        assert_allclose([[89597.909], [139.210], [-8738.019]],
                        normal_eqn(X, y),
                        rtol=0, atol=0.001)

    def test_reg_normal_eqn_data1_1(self, data1):

        y = data1[:, -1:]
        X = data1[:, :-1]
        m, _ = X.shape
        intercept = ones((m, 1), dtype=int)
        X = append(intercept, X, axis=1)
        _lambda = 0

        assert_allclose([[-3.896], [1.193]],
                        reg_normal_eqn(X, y, _lambda),
                        rtol=0, atol=0.001)

    def test_reg_normal_eqn_data1_2(self, data1):

        y = data1[:, -1:]
        X = data1[:, :-1]
        m, _ = X.shape
        intercept = ones((m, 1), dtype=int)
        X = append(intercept, X, axis=1)
        _lambda = 1

        assert_allclose([[-3.889], [1.192]],
                        reg_normal_eqn(X, y, _lambda),
                        rtol=0, atol=0.001)

    def test_reg_normal_eqn_data2(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m, _ = X.shape
        intercept = ones((m, 1), dtype=int)
        X = append(intercept, X, axis=1)
        _lambda = 100

        assert_allclose([[74104.492], [135.249], [-1350.731]],
                        reg_normal_eqn(X, y, _lambda),
                        rtol=0, atol=0.001)

    def test_normal_eqn_singular(self, data2):
        y = array([[0], [0], [0]])
        X = array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        m, _ = X.shape
        intercept = ones((m, 1), dtype=int)
        X = append(intercept, X, axis=1)

        with pytest.raises(LinAlgError) as excinfo:
            normal_eqn(X, y)
        msg = excinfo.value.args[0]
        assert msg == ("Singular matrix")

    def test_reg_normal_eqn_singular1(self, data2):
        y = array([[0], [0], [0]])
        X = array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        m, _ = X.shape
        intercept = ones((m, 1), dtype=int)
        X = append(intercept, X, axis=1)
        _lambda = 0

        with pytest.raises(LinAlgError) as excinfo:
            reg_normal_eqn(X, y, _lambda),
        msg = excinfo.value.args[0]
        assert msg == ("Singular matrix")

    def test_reg_normal_eqn_singular2(self, data2):
        y = array([[0], [0], [0]])
        X = array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        m, _ = X.shape
        intercept = ones((m, 1), dtype=int)
        X = append(intercept, X, axis=1)
        _lambda = 0.1

        assert_allclose([[0], [0], [0], [0]],
                        reg_normal_eqn(X, y, _lambda),
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
        m, _ = X.shape
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

# LINEAR REGRESSION CLASS

    def test_LinearRegression_constructor1(self, data1):
        theta = array([[1.], [0.6], [1.]])
        lr = LinearRegression(theta)

        assert_allclose(array([[1.], [0.6], [1.]]),
                        lr.theta,
                        rtol=0, atol=0.001)

    def test_LinearRegression_constructor2(self):
        lr = LinearRegression()

        assert lr.theta is None

    def test_LinearRegression_cost_data1_1(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        _, n = X.shape
        theta = ones((n + 1, 1), dtype=float64)
        lr = LinearRegression(theta)

        assert_allclose([[10.266]],
                        lr.cost(X, y),
                        rtol=0, atol=0.001)

    def test_LinearRegression_normal_fit(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        lr = LinearRegression()
        lr.fit(X, y, strategy="normal_equation")

        assert_allclose([[-3.896], [1.193]],
                        lr.theta,
                        rtol=0, atol=0.001)

    def test_LinearRegression_fit_BGD(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        lr = LinearRegression()
        lr.fit(X, y, strategy="BGD", alpha=1, num_iters=1)

        assert_allclose([[340412.659], [764209128.191], [1120367.702]],
                        lr.theta,
                        rtol=0, atol=0.001)

    def test_LinearRegression_fit_SGD(self, err):
        X = array([[0, 1, 2], [-1, 5, 3], [2, 0, 1]])
        y = array([[0.3], [1.2], [0.5]])
        lr = LinearRegression()
        lr.fit(X, y, strategy="SGD", alpha=1, num_iters=1)

        assert_allclose(array([[2.3], [11.2], [-11.7], [-2.2]]),
                        lr.theta,
                        rtol=0, atol=0.001, equal_nan=False)

    def test_LinearRegression_fit_MBGD1(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m = len(X)
        lr = LinearRegression()
        lr.fit(X, y, strategy="MBGD", alpha=1, num_iters=1, b=m)

        assert_allclose([[340412.659], [764209128.191], [1120367.702]],
                        lr.theta,
                        rtol=0, atol=0.001)

    def test_LinearRegression_fit_MBGD2(self, err):
        X = array([[0, 1, 2], [-1, 5, 3], [2, 0, 1]])
        y = array([[0.3], [1.2], [0.5]])
        lr = LinearRegression()
        lr.fit(X, y, strategy="MBGD", alpha=1, num_iters=1, b=1)

        assert_allclose(array([[2.3], [11.2], [-11.7], [-2.2]]),
                        lr.theta,
                        rtol=0, atol=0.001, equal_nan=False)

    def test_LinearRegression_predict(self):
        X = array([[3.5]])
        theta = array([[-3.6303], [1.1664]])
        lr = LinearRegression(theta)

        assert_allclose([[0.4521]],
                        lr.predict(X),
                        rtol=0, atol=0.001)

# RIDGE LINEAR REGRESSION CLASS

    def test_RidgeLinearRegression_constructor1(self, data1):
        theta = ones((3, 1), dtype=float64)
        lr = RidgeLinearRegression(theta, _lambda=13.50)

        assert lr._lambda == 13.50
        assert_allclose(array([[1.], [1.], [1.]]),
                        lr.theta,
                        rtol=0, atol=0.001)

    def test_RidgeLinearRegression_constructor2(self):
        lr = RidgeLinearRegression()

        assert lr.theta is None
        assert lr._lambda == 0

    def test_RidgeLinearRegression_cost_data1_1(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        _, n = X.shape
        theta = ones((n + 1, 1), dtype=float64)
        lr = RidgeLinearRegression(theta, _lambda=0)

        assert_allclose([[10.266]],
                        lr.cost(X, y),
                        rtol=0, atol=0.001)

    def test_RidgeLinearRegression_cost_data1_2(self, data1):

        y = data1[:, -1:]
        X = data1[:, :-1]
        _, n = X.shape
        theta = ones((n + 1, 1), dtype=float64)
        lr = RidgeLinearRegression(theta, _lambda=100)

        assert_allclose([[10.781984]],
                        lr.cost(X, y),
                        rtol=0, atol=0.001)

    def test_RidgeLinearRegression_cost_data2_1(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        _, n = X.shape
        theta = ones((n + 1, 1), dtype=float64)
        lr = RidgeLinearRegression(theta, _lambda=0)

        assert_allclose([[64828197300.798]],
                        lr.cost(X, y),
                        rtol=0, atol=0.001)

    def test_RidgeLinearRegression_cost_data2_2(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        _, n = X.shape
        theta = ones((n + 1, 1), dtype=float64)
        lr = RidgeLinearRegression(theta, _lambda=1000000)

        assert_allclose([[64828218577.393623]],
                        lr.cost(X, y),
                        rtol=0, atol=0.001)

    def test_RidgeLinearRegression_normal_fit_data1_1(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        lr = RidgeLinearRegression(_lambda=0)
        lr.fit(X, y, strategy="normal_equation")

        assert_allclose([[-3.896], [1.193]],
                        lr.theta,
                        rtol=0, atol=0.001)

    def test_RidgeLinearRegression_normal_fit_data1_2(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        lr = RidgeLinearRegression(_lambda=1)
        lr.fit(X, y, strategy="normal_equation")

        assert_allclose([[-3.889], [1.192]],
                        lr.theta,
                        rtol=0, atol=0.001)

    def test_RidgeLinearRegression_fit_BGD1(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        lr = RidgeLinearRegression(_lambda=0)
        lr.fit(X, y, strategy="BGD", alpha=1, num_iters=1)

        assert_allclose([[340412.659], [764209128.191], [1120367.702]],
                        lr.theta,
                        rtol=0, atol=0.001)

    def test_RidgeLinearRegression_fit_BGD2(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        _, n = X.shape
        theta = ones((n + 1, 1), dtype=float64)
        lr = RidgeLinearRegression(theta, _lambda=100)

        lr.fit(X, y, strategy="BGD", alpha=1, num_iters=1)

        assert_allclose([[-2.321], [-24.266]],
                        lr.theta,
                        rtol=0, atol=0.001)

    def test_RidgeLinearRegression_fit_SGD1(self, err):
        X = array([[0, 1, 2], [-1, 5, 3], [2, 0, 1]])
        y = array([[0.3], [1.2], [0.5]])
        lr = RidgeLinearRegression(_lambda=0)
        lr.fit(X, y, strategy="SGD", alpha=1, num_iters=1)

        assert_allclose(array([[2.3], [11.2], [-11.7], [-2.2]]),
                        lr.theta,
                        rtol=0, atol=0.001, equal_nan=False)

    def test_RidgeLinearRegression_fit_SGD2(self, err):
        X = array([[0, 1, 2], [-1, 5, 3], [2, 0, 1]])
        y = array([[0.3], [1.2], [0.5]])
        lr = RidgeLinearRegression(_lambda=10)
        lr.fit(X, y, strategy="SGD", alpha=1, num_iters=1)

        assert_allclose(array([[8.3], [-0.8], [132.3], [123.8]]),
                        lr.theta,
                        rtol=0, atol=0.001, equal_nan=False)

    def test_RidgeLinearRegression_fit_MBGD1(self, data2):
        y = data2[:, -1:]
        X = data2[:, :-1]
        m = len(X)
        lr = RidgeLinearRegression(_lambda=0)
        lr.fit(X, y, strategy="MBGD", alpha=1, num_iters=1, b=m)

        assert_allclose([[340412.659], [764209128.191], [1120367.702]],
                        lr.theta,
                        rtol=0, atol=0.001)

    def test_RidgeLinearRegression_fit_MBGD2(self, err):
        X = array([[0, 1, 2], [-1, 5, 3], [2, 0, 1]])
        y = array([[0.3], [1.2], [0.5]])
        lr = RidgeLinearRegression(_lambda=0)
        lr.fit(X, y, strategy="MBGD", alpha=1, num_iters=1, b=1)

        assert_allclose(array([[2.3], [11.2], [-11.7], [-2.2]]),
                        lr.theta,
                        rtol=0, atol=0.001, equal_nan=False)

    def test_RidgeLinearRegression_fit_MBGD3(self, data1):
        y = data1[:, -1:]
        X = data1[:, :-1]
        m, n = X.shape
        theta = ones((n + 1, 1), dtype=float64)
        lr = RidgeLinearRegression(theta, _lambda=100)
        lr.fit(X, y, strategy="MBGD", alpha=1, num_iters=1, b=m)

        assert_allclose([[-2.321], [-24.266]],
                        lr.theta,
                        rtol=0, atol=0.001)

    def test_RidgeLinearRegression_fit_MBGD4(self, data1):
        X = array([[0, 1, 2], [-1, 5, 3], [2, 0, 1]])
        y = array([[0.3], [1.2], [0.5]])
        lr = RidgeLinearRegression(_lambda=10)
        lr.fit(X, y, strategy="MBGD", alpha=1, num_iters=1, b=1)

        assert_allclose(array([[8.3], [-0.8], [132.3], [123.8]]),
                        lr.theta,
                        rtol=0, atol=0.001)

    def test_RidgeLinearRegression_fit_unknown(self, data1):
        X = array([[0, 1, 2], [-1, 5, 3], [2, 0, 1]])
        y = array([[0.3], [1.2], [0.5]])
        lr = RidgeLinearRegression(_lambda=10)

        with pytest.raises(ValueError) as excinfo:
            lr.fit(X, y, strategy="oompa_loompa")

        msg = excinfo.value.args[0]
        assert msg == ("'oompa_loompa' (type '<class 'str'>') was passed. ",
                       'The strategy parameter for the fit function should ',
                       "be 'BGD' or 'SGD' or 'MBGD' or 'normal_equation'.")

    def test_RidgeLinearRegression_predict1(self):
        X = array([[3.5]])
        theta = array([[-3.6303], [1.1664]])
        lr = RidgeLinearRegression(theta)

        assert_allclose([[0.4521]],
                        lr.predict(X),
                        rtol=0, atol=0.001)

    def test_RidgeLinearRegression_predict2(self):
        X = array([[3.5]])
        theta = array([[-3.6303], [1.1664]])
        lr = RidgeLinearRegression(theta, _lambda=10)

        assert_allclose([[0.4521]],
                        lr.predict(X),
                        rtol=0, atol=0.001)
