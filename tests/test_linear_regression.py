import unittest
import os

import numpy as np
from numpy.testing import assert_allclose

from ml_algorithms.linear_regression import normal_eqn, cost_function

TESTDATA1 = os.path.join(os.path.dirname(__file__), 'data1.csv')
TESTDATA2 = os.path.join(os.path.dirname(__file__), 'data2.csv')
TESTDATA3 = os.path.join(os.path.dirname(__file__), 'data3.csv')


class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        self.data1 = np.genfromtxt(TESTDATA1, delimiter=',')
        self.data2 = np.genfromtxt(TESTDATA2, delimiter=',')
        self.data3 = np.genfromtxt(TESTDATA3, delimiter=',')

    def test_normal_eqn(self):
        y = self.data2[:, -1:]
        X = self.data2[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int)
        X = np.append(intercept, X, axis=1)

        assert_allclose([[8.9598e+04], [1.3921e+02], [-8.7380e+03]],
                        normal_eqn(X, y), rtol=1e-03)

    def test_cost_function1(self):
        y = self.data1[:, -1:]
        X = self.data1[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.zeros((n + 1, 1), dtype=np.int64)

        assert_allclose([[32.073]], cost_function(X, y, theta), rtol=1e-3)

    def test_cost_function2(self):
        y = self.data1[:, -1:]
        X = self.data1[:, :-1]
        m, n = X.shape
        intercept = np.ones((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)
        theta = np.array([[-1], [2]])

        assert_allclose([[54.24]], cost_function(X, y, theta), rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
