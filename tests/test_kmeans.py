import unittest

from numpy import array
from numpy.testing import assert_allclose, assert_almost_equal


from ml_algorithms.kmeans import (find_closest_centroids, euclidean_dist,
                                  compute_centroids)


class Kmeans(unittest.TestCase):

    def test_euclidean_dist1(self):
        p = array([1, 7, 4, 2, -1])
        q = array([8, 6, 3, 4, 10])
        dist = euclidean_dist(p, q)
        assert_almost_equal(dist, 13.2665, decimal=4)

    def test_euclidean_dist2(self):
        p = array([1, 7, 4, 2, -1])
        q = array([8, 6, 3, 4, 10])

        self.assertEqual(euclidean_dist(p, q),
                         euclidean_dist(q, p))

    def test_euclidean_dist3(self):
        p = array([1, 7, 4, 2, -1])

        self.assertEqual(euclidean_dist(p, p), 0)

    def test_find_closest_centroids1(self):
        X = array([[1.8421, 4.6076], [5.6586, 4.8000], [6.3526, 3.2909],
                   [2.9040, 4.6122], [3.2320, 4.9399], [1.2479, 4.9327],
                   [1.9762, 4.4349], [2.2345, 5.0555], [2.9834, 4.8405],
                   [2.9797, 4.8067]])
        initial_centroids = array(
            [[2.9040, 4.6122], [1.2479, 4.9327], [2.9797, 4.8067]])

        assert_allclose(array([[1], [2], [0], [0], [2], [1],
                               [1], [2], [2], [2]]),
                        find_closest_centroids(X, initial_centroids),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_find_closest_centroids2(self):
        X = array([[5.89562, 2.89844], [5.61754, 2.59751], [5.63176, 3.04759],
                   [5.50259, 3.11869], [6.48213, 2.55085], [7.30279, 3.38016],
                   [6.99198, 2.98707], [4.82553, 2.77962], [6.11768, 2.85476],
                   [0.94049, 5.71557]])
        initial_centroids = array([[5.63176, 3.04759], [6.48213, 2.55085],
                                   [6.99198, 2.98707]])

        assert_allclose(array([[0], [0], [0], [0], [1],
                               [2], [2], [0], [1], [0]]),
                        find_closest_centroids(X, initial_centroids),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_compute_centroids1(self):
        X = array([[1.8421, 4.6076], [5.6586, 4.8000], [6.3526, 3.2909],
                   [2.9040, 4.6122], [3.2320, 4.9399], [1.2479, 4.9327],
                   [1.9762, 4.4349], [2.2345, 5.0555], [2.9834, 4.8405],
                   [2.9797, 4.8067]])
        idx = array([[1], [2], [0], [1], [2], [0], [1], [2], [0], [2]])
        K = 3

        assert_allclose(array([[3.5280, 4.3547],
                               [2.2408, 4.5516],
                               [3.5262, 4.9005]]),
                        compute_centroids(X, idx, K),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_compute_centroids2(self):
        X = array([[5.89562, 2.89844], [5.61754, 2.59751], [5.63176, 3.04759],
                   [5.50259, 3.11869], [6.48213, 2.55085], [7.30279, 3.38016],
                   [6.99198, 2.98707], [4.82553, 2.77962], [6.11768, 2.85476],
                   [0.94049, 5.71557]])
        idx = array([[3], [1], [2], [0], [1], [3], [1], [0], [1], [2]])
        K = 4

        assert_allclose(array([[5.1641, 2.9492], [6.3023, 2.7475],
                               [3.2861, 4.3816], [6.5992, 3.1393]]),
                        compute_centroids(X, idx, K),
                        rtol=0, atol=0.001, equal_nan=False)
