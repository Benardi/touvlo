import unittest
import os

import numpy as np
from numpy.testing import assert_allclose

from ml_algorithms.linear_regression import (normal_eqn, cost_function,
                                             grad, predict, h)

TESTDATA1 = os.path.join(os.path.dirname(__file__), 'data1.csv')
TESTDATA2 = os.path.join(os.path.dirname(__file__), 'data2.csv')


class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        self.data1 = np.genfromtxt(TESTDATA1, delimiter=',')
        self.data2 = np.genfromtxt(TESTDATA2, delimiter=',')

# NORMAL EQUATION

    def test_normal_eqn_data1(self):
        y = self.data1[:, -1:]
        X = self.data1[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int)
        X = np.append(intercept, X, axis=1)

        assert_allclose([[-3.896], [1.193]],
                        normal_eqn(X, y),
                        rtol=0, atol=0.001)

    def test_normal_eqn_data2(self):
        y = self.data2[:, -1:]
        X = self.data2[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int)
        X = np.append(intercept, X, axis=1)

        assert_allclose([[89597.909], [139.210], [-8738.019]],
                        normal_eqn(X, y),
                        rtol=0, atol=0.001)

# COST FUNCTION

    def test_cost_func_data1_1(self):
        y = self.data1[:, -1:]
        X = self.data1[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.zeros((n + 1, 1), dtype=np.int64)

        assert_allclose([[32.073]],
                        cost_function(X, y, theta),
                        rtol=0, atol=0.001)

    def test_cost_func_data1_2(self):
        y = self.data1[:, -1:]
        X = self.data1[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.ones((n + 1, 1), dtype=np.int64)

        assert_allclose([[10.266]],
                        cost_function(X, y, theta),
                        rtol=0, atol=0.001)

    def test_cost_func_data1_3(self):
        y = self.data1[:, -1:]
        X = self.data1[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.array([[-1], [2]])

        assert_allclose([[54.242]],
                        cost_function(X, y, theta),
                        rtol=0, atol=0.001)

    def test_cost_func_data2_1(self):
        y = self.data2[:, -1:]
        X = self.data2[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.zeros((n + 1, 1), dtype=np.int64)

        assert_allclose([[65591548106.457]],
                        cost_function(X, y, theta),
                        rtol=0, atol=0.001)

    def test_cost_func_data2_2(self):
        y = self.data2[:, -1:]
        X = self.data2[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.ones((n + 1, 1), dtype=np.int64)

        assert_allclose([[64828197300.798]],
                        cost_function(X, y, theta),
                        rtol=0, atol=0.001)

    def test_cost_func_data2_3(self):
        y = self.data2[:, -1:]
        X = self.data2[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.array([[-25.3], [32], [7.8]])

        assert_allclose([[43502644952.311]],
                        cost_function(X, y, theta),
                        rtol=0, atol=0.001)

# GRADIENT

    def test_grad_data1_1(self):
        y = self.data1[:, -1:]
        X = self.data1[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.zeros((n + 1, 1), dtype=np.int64)

        assert_allclose([[-5.839], [-65.329]],
                        grad(X, y, theta),
                        rtol=0, atol=0.001)

    def test_grad_data1_2(self):
        y = self.data1[:, -1:]
        X = self.data1[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.ones((n + 1, 1), dtype=np.int64)

        assert_allclose([[3.321], [24.235]],
                        grad(X, y, theta),
                        rtol=0, atol=0.001)

    def test_grad_data1_3(self):
        y = self.data1[:, -1:]
        X = self.data1[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.array([[-1], [2]])

        assert_allclose([[9.480], [89.319]],
                        grad(X, y, theta),
                        rtol=0, atol=0.001)

    def test_grad_data2_1(self):
        y = self.data2[:, -1:]
        X = self.data2[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.zeros((n + 1, 1), dtype=np.int64)

        assert_allclose([[-340412.659], [-764209128.191], [-1120367.702]],
                        grad(X, y, theta),
                        rtol=0, atol=0.001)

    def test_grad_data2_2(self):
        y = self.data2[:, -1:]
        X = self.data2[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.ones((n + 1, 1), dtype=np.int64)

        assert_allclose([[-338407.808], [-759579615.064], [-1113679.894]],
                        grad(X, y, theta),
                        rtol=0, atol=0.001)

    def test_grad_data2_3(self):
        y = self.data2[:, -1:]
        X = self.data2[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.array([[-25.3], [32], [7.8]])

        assert_allclose([[-276391.445], [-616340858.434], [-906796.414]],
                        grad(X, y, theta),
                        rtol=0, atol=0.001)

# PREDICT

    def test_predict_1(self):
        X = np.array([[3.5]])
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.array([[-3.6303], [1.1664]])

        assert_allclose([[0.4521]],
                        predict(X, theta),
                        rtol=0, atol=0.001)

    def test_predict_2(self):
        X = np.array([[3.5]])
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.zeros((n + 1, 1), dtype=np.int64)

        assert_allclose([[0]],
                        predict(X, theta),
                        rtol=0, atol=0.001)

    def test_predict_3(self):
        X = np.array([[-3.5, 2.7]])
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.ones((n + 1, 1), dtype=np.int64)

        assert_allclose([[0.2]],
                        predict(X, theta),
                        rtol=0, atol=0.001)

    def test_predict_4(self):
        X = np.array([[-3.5, 2.7]])
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = -1 * np.ones((n + 1, 1), dtype=np.int64)

        assert_allclose([[-0.2]],
                        predict(X, theta),
                        rtol=0, atol=0.001)

# HYPOTHESYS

    def test_h_1(self):
        X = np.array([[3.5]])
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.array([[-3.6303], [1.1664]])

        assert_allclose([[0.4521]],
                        h(X, theta),
                        rtol=0, atol=0.001)

    def test_h_2(self):
        X = np.array([[3.5]])
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.zeros((n + 1, 1), dtype=np.int64)

        assert_allclose([[0]],
                        h(X, theta),
                        rtol=0, atol=0.001)

    def test_h_3(self):
        X = np.array([[-3.5, 2.7]])
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.ones((n + 1, 1), dtype=np.int64)

        assert_allclose([[0.2]],
                        h(X, theta),
                        rtol=0, atol=0.001)

    def test_h_4(self):
        X = np.array([[-3.5, 2.7]])
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = -1 * np.ones((n + 1, 1), dtype=np.int64)

        assert_allclose([[-0.2]],
                        h(X, theta),
                        rtol=0, atol=0.001)


if __name__ == '__main__':
    unittest.main()
