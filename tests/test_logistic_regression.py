import unittest
import os

import numpy as np
from numpy.testing import assert_allclose

from ml_algorithms.logistic_regression import (cost_function, grad,
                                               predict_prob, predict, h)

TESTDATA3 = os.path.join(os.path.dirname(__file__), 'data3.csv')
TESTDATA4 = os.path.join(os.path.dirname(__file__), 'data4.csv')


class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        self.data3 = np.genfromtxt(TESTDATA3, delimiter=',')
        self.data4 = np.genfromtxt(TESTDATA4, delimiter=',')


# COST FUNCTION

    def test_cost_func_data3_1(self):

        y = self.data3[:, -1:]
        X = self.data3[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.zeros((n + 1, 1), dtype=np.int64)

        assert np.isclose(0.693, cost_function(X, y, theta),
                          rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data3_2(self):
        y = self.data3[:, -1:]
        X = self.data3[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.array([[-0.1], [0.4], [-0.4]])

        assert np.isclose(4.227, cost_function(X, y, theta),
                          rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data3_3(self):
        y = self.data3[:, -1:]
        X = self.data3[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.array([[-24], [0.2], [0.2]])

        assert np.isclose(0.218, cost_function(X, y, theta),
                          rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data4_1(self):
        y = self.data4[:, -1:]
        X = self.data4[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.zeros((n + 1, 1), dtype=np.int64)

        assert np.isclose(0.693, cost_function(X, y, theta),
                          rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data4_2(self):
        y = self.data4[:, -1:]
        X = self.data4[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.array([[-0.1], [0.4], [-0.4], [-0.4], [0.4],
                          [-0.4], [0.4], [-0.4], [-0.4]])

        assert_allclose(40.216,
                        cost_function(X, y, theta),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_cost_func_data4_3(self):
        y = self.data4[:, -1:]
        X = self.data4[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = -0.1 * np.ones((n + 1, 1), dtype=np.int64)

        assert_allclose(14.419,
                        cost_function(X, y, theta),
                        rtol=0, atol=0.001, equal_nan=False)

# GRADIENT

    def test_grad_data3_1(self):
        y = self.data3[:, -1:]
        X = self.data3[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.zeros((n + 1, 1), dtype=np.int64)

        assert_allclose([[-0.1000], [-12.0092], [-11.2628]],
                        grad(X, y, theta), rtol=0, atol=5e-05,
                        equal_nan=False)

    def test_grad_data3_2(self):
        y = self.data3[:, -1:]
        X = self.data3[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.array([[-0.1], [0.4], [-0.4]])

        assert_allclose([[-0.134], [-8.275], [-18.564]],
                        grad(X, y, theta), rtol=0, atol=0.001,
                        equal_nan=False)

    def test_grad_data3_3(self):
        y = self.data3[:, -1:]
        X = self.data3[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.array([[-24], [0.2], [0.2]])

        assert_allclose([[0.043], [2.566], [2.647]],
                        grad(X, y, theta), rtol=0, atol=0.001,
                        equal_nan=False)

    def test_grad_data4_1(self):
        y = self.data4[:, -1:]
        X = self.data4[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.zeros((n + 1, 1), dtype=np.int64)

        assert_allclose(np.array([[0.151], [0.225], [11.154], [9.838], [2.534],
                                  [4.887], [3.733], [0.044], [3.686]]),
                        grad(X, y, theta),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_grad_data4_2(self):
        y = self.data4[:, -1:]
        X = self.data4[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.array([[-0.1], [0.4], [-0.4], [-0.4], [0.4],
                          [-0.4], [0.4], [-0.4], [-0.4]])

        assert_allclose(np.array([[-0.349], [-1.698], [-49.293],
                                  [-24.715], [-7.734], [-35.013],
                                  [-12.263], [-0.192], [-12.935]]),
                        grad(X, y, theta),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_grad_data4_3(self):
        y = self.data4[:, -1:]
        X = self.data4[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = -0.1 * np.ones((n + 1, 1), dtype=np.int64)

        assert_allclose(np.array([[-0.349], [-1.698], [-49.293],
                                  [-24.715], [-7.734], [-35.013],
                                  [-12.263], [-0.192], [-12.935]]),
                        grad(X, y, theta),
                        rtol=0, atol=0.001, equal_nan=False)

# HYPOTHESIS

    def test_h_prob1(self):
        X = np.array([[1, 45, 85]])
        theta = np.array([[-25.161], [0.206], [0.201]])
        assert np.isclose(0.767,
                          h(X, theta),
                          rtol=0, atol=0.001, equal_nan=False)

    def test_h_prob2(self):
        X = np.array([[1, 20, 93]])
        theta = np.array([[0], [0], [0]])

        assert np.isclose(0.5,
                          h(X, theta),
                          rtol=0, atol=0.001, equal_nan=False)

    def test_h_prob3(self):
        X = np.array([[1, 10, -50]])
        theta = np.array([[5.161], [0.206], [0.201]])

        assert np.isclose(0.0558,
                          h(X, theta),
                          rtol=0, atol=0.001, equal_nan=False)

# PREDICT PROB

    def test_predict_prob1(self):
        X = np.array([[1, 45, 85]])
        theta = np.array([[-25.161], [0.206], [0.201]])

        assert np.isclose(0.767,
                          predict_prob(X, theta),
                          rtol=0, atol=0.001, equal_nan=False)

    def test_predict_prob2(self):
        X = np.array([[1, 20, 93]])
        theta = np.array([[0], [0], [0]])

        assert np.isclose(0.5,
                          predict_prob(X, theta),
                          rtol=0, atol=0.001, equal_nan=False)

    def test_predict_prob3(self):
        X = np.array([[1, 10, -50]])
        theta = np.array([[5.161], [0.206], [0.201]])

        assert np.isclose(0.0558,
                          predict_prob(X, theta),
                          rtol=0, atol=0.001, equal_nan=False)

# PREDICT

    def test_predict_1(self):
        X = np.array([[1, 45, 85]])
        theta = np.array([[-25.161], [0.206], [0.201]])

        assert predict(X, theta)

    def test_predict_2(self):
        X = np.array([[1, 20, 93]])
        theta = np.array([[0], [0], [0]])

        assert predict(X, theta)

    def test_predict3(self):
        X = np.array([[1, 10, -50]])
        theta = np.array([[5.161], [0.206], [0.201]])

        assert not predict(X, theta)


if __name__ == '__main__':
    unittest.main()
