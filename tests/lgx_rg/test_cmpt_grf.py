import os

import pytest
from numpy import ones, zeros, float64, array, isclose, genfromtxt
from numpy.testing import assert_allclose

from touvlo.lgx_rg.cmpt_grf import cost_func, grad, predict, h

TESTDATA3 = os.path.join(os.path.dirname(__file__), '../data3.csv')
TESTDATA4 = os.path.join(os.path.dirname(__file__), '../data4.csv')


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
        Y = data3[:, -1:].T
        X = data3[:, :-1].T
        n, m = X.shape

        w = zeros((n, 1), dtype=float64)
        b = 0

        assert isclose(0.693,
                       cost_func(X, Y, w=w, b=b),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data3_2(self, data3):
        Y = data3[:, -1:].T
        X = data3[:, :-1].T
        w = array([[0.4], [-0.4]])
        b = -0.1

        assert isclose(4.227, cost_func(X, Y, w=w, b=b),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data3_3(self, data3):
        Y = data3[:, -1:].T
        X = data3[:, :-1].T
        w = array([[0.2], [0.2]])
        b = -24

        assert isclose(0.218, cost_func(X, Y, w=w, b=b),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data4_1(self, data4):
        Y = data4[:, -1:].T
        X = data4[:, :-1].T
        n, m = X.shape
        w = zeros((n, 1), dtype=float64)
        b = 0

        assert isclose(0.693, cost_func(X, Y, w=w, b=b),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data4_2(self, data4):
        Y = data4[:, -1:].T
        X = data4[:, :-1].T
        w = array([[0.4], [-0.4], [-0.4], [0.4],
                   [-0.4], [0.4], [-0.4], [-0.4]])
        b = -0.1

        assert_allclose(40.216,
                        cost_func(X, Y, w=w, b=b),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data4_3(self, data4):
        Y = data4[:, -1:].T
        X = data4[:, :-1].T
        n, m = X.shape
        w = -0.1 * ones((n, 1), dtype=float64)
        b = -0.1

        assert_allclose(14.419,
                        cost_func(X, Y, w=w, b=b),
                        rtol=0, atol=0.001, equal_nan=False)

# GRADIENT

    def test_grad_data3_1(self, data3):
        Y = data3[:, -1:].T
        X = data3[:, :-1].T
        m, n = X.shape
        w = zeros((m, 1), dtype=float64)
        b = 0

        dw, db = grad(X, Y, w, b)

        assert_allclose([[-12.0092], [-11.2628]],
                        dw,
                        rtol=0, atol=5e-05, equal_nan=False)

        assert isclose(-0.1000,
                       db,
                       rtol=0, atol=5e-05, equal_nan=False)

    def test_grad_data3_2(self, data3):
        Y = data3[:, -1:].T
        X = data3[:, :-1].T
        w = array([[0.4], [-0.4]])
        b = -0.1

        dw, db = grad(X, Y, w, b)

        assert_allclose([[-8.275], [-18.564]],
                        dw,
                        rtol=0, atol=0.001, equal_nan=False)

        assert isclose(-0.134,
                       db,
                       rtol=0, atol=0.001, equal_nan=False)

    def test_grad_data3_3(self, data3):
        y = data3[:, -1:].T
        X = data3[:, :-1].T
        w = array([[0.2], [0.2]])
        b = -24

        dw, db = grad(X, y, w, b)
        assert_allclose([[2.566], [2.647]],
                        dw,
                        rtol=0, atol=0.001,
                        equal_nan=False)

        assert isclose(0.043,
                       db,
                       rtol=0, atol=0.001,
                       equal_nan=False)

    def test_grad_data4_1(self, data4):
        y = data4[:, -1:].T
        X = data4[:, :-1].T
        m, n = X.shape
        w = zeros((m, 1), dtype=float64)
        b = 0

        dw, db = grad(X, y, w, b)

        assert_allclose(array([[0.225], [11.154], [9.838], [2.534],
                               [4.887], [3.733], [0.044], [3.686]]),
                        dw,
                        rtol=0, atol=0.001, equal_nan=False)

        assert isclose(0.151,
                       db,
                       rtol=0, atol=0.001, equal_nan=False)

    def test_grad_data4_2(self, data4):
        y = data4[:, -1:].T
        X = data4[:, :-1].T
        w = array([[0.4], [-0.4], [-0.4], [0.4],
                   [-0.4], [0.4], [-0.4], [-0.4]])
        b = -0.1

        dw, db = grad(X, y, w, b)

        assert_allclose(array([[-1.698], [-49.293], [-24.715], [-7.734],
                               [-35.013], [-12.263], [-0.192], [-12.935]]),
                        dw,
                        rtol=0, atol=0.001, equal_nan=False)

        assert isclose(-0.349,
                       db,
                       rtol=0, atol=0.001, equal_nan=False)

    def test_grad_data4_3(self, data4):
        y = data4[:, -1:].T
        X = data4[:, :-1].T
        m, n = X.shape
        w = -0.1 * ones((m, 1), dtype=float64)
        b = -0.1

        dw, db = grad(X, y, w, b)

        assert_allclose(array([[-1.698], [-49.293], [-24.715], [-7.734],
                               [-35.013], [-12.263], [-0.192], [-12.935]]),
                        dw,
                        rtol=0, atol=0.001, equal_nan=False)

        assert isclose(-0.349,
                       db,
                       rtol=0, atol=0.001, equal_nan=False)

# HYPOTHESIS

    def test_h_prob1(self):
        X = array([[45], [85]])
        w = array([[0.206], [0.201]])
        b = -25.161

        assert isclose(0.767,
                       h(X, w, b),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_h_prob2(self):
        X = array([[20], [93]])
        w = array([[0], [0]])
        b = 0

        assert isclose(0.5,
                       h(X, w, b),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_h_prob3(self):
        X = array([[10], [-50]])
        w = array([[0.206], [0.201]])
        b = 5.161

        assert isclose(0.0558,
                       h(X, w, b),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_h_prob4(self):
        X = array([[45], [85]])
        w = array([[0.206], [0.201]])
        b = -25.161

        assert isclose(0.767,
                       h(X, w, b),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_h_prob5(self):
        X = array([[20], [93]])
        w = array([[0], [0]])
        b = 0

        assert isclose(0.5,
                       h(X, w, b),
                       rtol=0, atol=0.001, equal_nan=False)

    def test_h_prob6(self):
        X = array([[10], [-50]])
        w = array([[0.206], [0.201]])
        b = 5.161
        assert isclose(0.0558,
                       h(X, w, b),
                       rtol=0, atol=0.001, equal_nan=False)

# PREDICT

    def test_predict_1(self):
        X = array([[45], [85]])
        w = array([[0.206], [0.201]])
        b = -25.161

        assert predict(X, w, b)

    def test_predict_2(self):
        X = array([[20], [93]])
        w = array([[0], [0]])
        b = 0

        assert predict(X, w, b)

    def test_predict3(self):
        X = array([[10], [-50]])
        w = array([[0.206], [0.201]])
        b = 5.161
        assert not predict(X, w, b)
