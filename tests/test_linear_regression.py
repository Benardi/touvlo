import unittest
import os

import numpy as np

from ml_algorithms.linear_regression import compute_cost


TESTDATA1 = os.path.join(os.path.dirname(__file__), 'data1.csv')
TESTDATA2 = os.path.join(os.path.dirname(__file__), 'data2.csv')


class TestEquationSolution(unittest.TestCase):

    def setUp(self):
        self.data1 = np.genfromtxt(TESTDATA1, delimiter=',')
        self.data2 = np.genfromtxt(TESTDATA2, delimiter=',')

    def test_compute_cost(self):
        y = self.data1[:, -1:]
        m = len(y)

        X = self.data1[:, :-1]
        intercept = np.zeros((m, 1), dtype=np.int64)
        X = np.append(intercept, X, axis=1)

        theta = np.zeros((X.shape[1], 1), dtype=np.int64)

        assert np.isclose(32.07, compute_cost(X, y, theta),
                          rtol=1e-04, atol=1e-08, equal_nan=False)


if __name__ == '__main__':
    unittest.main()
