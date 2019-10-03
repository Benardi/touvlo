from math import inf, sqrt

from numpy import zeros, int64, power, sum, mean
from numpy.random import permutation


def euclidean_dist(p, q):
    """Calculates Euclidean distance between 2 n-dimensional points.

    :param p: First n-dimensional point.
    :type p: numpy.array

    :param q: Second n-dimensional point.
    :type q: numpy.array

    :returns: Distance between 2 points.
    :rtype: float
    """
    dist = p - q
    dist = power(dist, 2)
    dist = sum(dist)
    dist = sqrt(dist)

    return dist


def find_closest_centroids(X, initial_centroids):
    """Assigns to each example the indice of the closest centroid.

    :param X: Features' dataset
    :type X: numpy.array

    :param initial_centroids: List of initialized centroids.
    :type initial_centroids: list(numpy.array)

    :returns: Column vector of assigned centroids' indices.
    :rtype: numpy.array
    """
    m = len(X)
    K = len(initial_centroids)
    idx = zeros((m, 1), dtype=int64)

    for i in range(m):
        best_c = -1
        min_dist = inf
        for k in range(K):
            dist = euclidean_dist(X[i, :], initial_centroids[k])
            if dist < min_dist:
                best_c = k
                min_dist = dist

        idx[i, :] = best_c

    return idx


def compute_centroids(X, idx, K):
    """Computes centroids from the mean of its cluster's members.

    :param X: Features' dataset
    :type X: numpy.array

    :param idx: Column vector of assigned centroids' indices.
    :type idx: numpy.array

    :param K: Number of centroids.
    :type K: int

    :returns: Column vector of newly computed centroids
    :rtype: numpy.array
    """
    m, n = X.shape
    centroids = zeros((K, n))
    for k in range(K):
        centroids[k] = mean(X[(idx == k).flatten()], axis=0)

    return centroids


def init_centroids(X, K):
    """Computes centroids from the mean of its cluster's members.

    :param X: Features' dataset
    :type X: numpy.array

    :param idx: Column vector of assigned centroids' indices.
    :type idx: numpy.array

    :param K: Number of centroids.
    :type K: int

    :returns: Column vector of centroids randomly picked from dataset
    :rtype: numpy.array
    """
    centroids = permutation(X)
    centroids = centroids[0:K, :]
    return centroids


def run_kmeans(X, K, max_iters):
    """Applies kmeans using a single random initialization.

    :param X: Features' dataset
    :type X: numpy.array

    :param K: Number of centroids.
    :type K: int

    :param max_iters: Number of times the algorithm will be fitted.
    :type max_iters: int

    :returns:
        - centroids -Column vector of centroids
        - idx -Column vector of assigned centroids' indices.

    :rtype:
        - centroids (:py:class: numpy.array)
        - idx (:py:class: numpy.array)
    """
    centroids = init_centroids(X, K)

    for _ in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)

    return centroids, idx


def cost_function(X, idx, centroids):
    """Calculates the cost function for K means.

    :param X: Features' dataset
    :type X: numpy.array

    :param idx: Column vector of assigned centroids' indices.
    :type idx: numpy.array

    :returns: Column vector of centroids
    :rtype: numpy.array

    :returns: Computed cost
    :rtype: float
    """
    cost = 0
    m = len(X)
    for i in range(m):
        centroid = centroids[idx[i][0]]
        cost += power(euclidean_dist(X[i], centroid), 2)

    cost = cost / m
    return cost


def run_intensive_kmeans(X, K, max_iters, n_inits):
    """Applies kmeans using multiple random initializations.

    :param X: Features' dataset
    :type X: numpy.array

    :param K: Number of centroids.
    :type K: int

    :param max_iters: Number of times the algorithm will be fitted.
    :type max_iters: int

    :param n_inits: Number of random initialization.
    :type n_inits: int

    :returns:
        - centroids -Column vector of centroids
        - idx -Column vector of assigned centroids' indices.

    :rtype:
        - centroids (:py:class: numpy.array)
        - idx (:py:class: numpy.array)
    """
    min_cost = inf
    best_idx = None
    best_centroids = None
    for _ in range(n_inits):
        centroids, idx = run_kmeans(X, K, max_iters)
        cost = cost_function(X, idx, centroids)
        if cost < min_cost:
            best_idx = idx
            best_centroids = centroids

    return best_centroids, best_idx


def elbow_method(X, K_values, max_iters, n_inits):
    """Calculates the cost for each given K.

    :param X: Features' dataset
    :type X: numpy.array

    :param K_values: List of possible number of centroids.
    :type K: list(int)

    :param max_iters: Number of times the algorithm will be fitted.
    :type max_iters: int

    :param n_inits: Number of random initialization.
    :type n_inits: int

    :returns:
        - K_values - List of possible numbers of centroids.
        - cost_values -Computed cost for each K.

    :rtype:
        - K_values (:py:class: list(int))
        - cost_values (:py:class: list(float))
    """
    cost_values = []
    for K in K_values:
        centroids, idx = run_intensive_kmeans(X, K, max_iters, n_inits)
        cost = cost_function(X, idx, centroids)
        cost_values.append(cost)

    return K_values, cost_values
