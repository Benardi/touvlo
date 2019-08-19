import unittest
import os

import numpy as np
from numpy.testing import assert_allclose

from ml_algorithms.logistic_regression import (cost_function, grad,
                                               predict_prob, predict)

TESTDATA1 = os.path.join(os.path.dirname(__file__), 'ex2data1.csv')


class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        self.data1 = np.genfromtxt(TESTDATA1, delimiter=',')

    def test_cost_function1(self):

        y = self.data1[:, -1:]
        X = self.data1[:, :-1]
        m, n = X.shape

        intercept = np.ones((m, 1), dtype=np.int64)

        X = np.append(intercept, X, axis=1)
        theta = np.zeros((n + 1, 1), dtype=np.int64)

        assert np.isclose(0.693, cost_function(theta, X, y),
                          rtol=0, atol=5e-04, equal_nan=False)

    def test_grad1(self):

        y = self.data1[:, -1:]
        X = self.data1[:, :-1]
        m, n = X.shape

        intercept = np.ones((m, 1), dtype=np.int64)

        X = np.append(intercept, X, axis=1)
        theta = np.zeros((n + 1, 1), dtype=np.int64)

        assert_allclose([[-0.1000], [-12.0092], [-11.2628]],
                        grad(theta, X, y, m), rtol=0, atol=5e-05,
                        equal_nan=False)

    def test_cost_function2(self):

        y = self.data1[:, -1:]
        X = self.data1[:, :-1]
        m, n = X.shape

        intercept = np.ones((m, 1), dtype=np.int64)

        X = np.append(intercept, X, axis=1)
        theta = np.array([[-24], [0.2], [0.2]])

        assert np.isclose(0.218, cost_function(theta, X, y),
                          rtol=0, atol=5e-04, equal_nan=False)

    def test_grad2(self):

        y = self.data1[:, -1:]
        X = self.data1[:, :-1]
        m, n = X.shape

        intercept = np.ones((m, 1), dtype=np.int64)

        X = np.append(intercept, X, axis=1)
        theta = np.array([[-24], [0.2], [0.2]])

        assert_allclose([[0.043], [2.566], [2.647]],
                        grad(theta, X, y, m), rtol=0, atol=0.001,
                        equal_nan=False)

    def test_predict_prob(self):
        X = np.array([[1, 45, 85]])
        theta = np.array([[-25.161], [0.206], [0.201]])
        assert np.isclose(0.775, predict_prob(theta, X), rtol=0.01,
                          atol=0, equal_nan=False)

    def test_predict(self):
        X = np.array([[1, 45, 85]])
        theta = np.array([[-25.161], [0.206], [0.201]])
        assert predict(theta, X) == 1


if __name__ == '__main__':
    unittest.main()
