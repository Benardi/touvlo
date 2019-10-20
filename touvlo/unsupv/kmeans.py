"""
.. module:: kmeans
    :synopsis: Provides routines to apply K-means clustering.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from math import inf, sqrt

from numpy import zeros, int64, power, sum, mean
from numpy.random import permutation


def euclidean_dist(p, q):
    """Calculates Euclidean distance between 2 n-dimensional points.

    Args:
        p (numpy.array): First n-dimensional point.
        q (numpy.array): Second n-dimensional point.

    Returns:
        float: Distance between 2 points.
    """
    dist = p - q
    dist = power(dist, 2)
    dist = sum(dist)
    dist = sqrt(dist)

    return dist


def find_closest_centroids(X, initial_centroids):
    """Assigns to each example the indice of the closest centroid.

    Args:
        X (numpy.array): Features' dataset
        initial_centroids (numpy.array): List of initialized centroids.

    Returns:
        numpy.array: Column vector of assigned centroids' indices.
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

    Args:
        X (numpy.array): Features' dataset
        idx (numpy.array): Column vector of assigned centroids' indices.
        K (int): Number of centroids.

    Returns:
        numpy.array: Column vector of newly computed centroids
    """
    m, n = X.shape
    centroids = zeros((K, n))
    for k in range(K):
        centroids[k] = mean(X[(idx == k).flatten()], axis=0)

    return centroids


def init_centroids(X, K):
    """Computes centroids from the mean of its cluster's members.

    Args:
        X (numpy.array): Features' dataset
        idx (numpy.array): Column vector of assigned centroids' indices.
        K (int): Number of centroids.

    Returns:
        numpy.array: Column vector of centroids randomly picked from dataset
    """
    centroids = permutation(X)
    centroids = centroids[0:K, :]
    return centroids


def run_kmeans(X, K, max_iters):
    """Applies kmeans using a single random initialization.

    Args:
        X (numpy.array): Features' dataset
        K (int): Number of centroids.
        max_iters (int): Number of times the algorithm will be fitted.

    Returns:
        (numpy.array, numpy.array): A 2-tuple of centroids, a column vector of
            centroids, and idx, a column vector of assigned centroids' indices.
    """
    centroids = init_centroids(X, K)

    for _ in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)

    return centroids, idx


def cost_function(X, idx, centroids):
    """Calculates the cost function for K means.

    Args:
        X (numpy.array): Features' dataset
        idx (numpy.array): Column vector of assigned centroids' indices.

    Returns:
        float: Computed cost
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

    Args:
        X (numpy.array): Features' dataset
        K (int): Number of centroids.
        max_iters (int): Number of times the algorithm will be fitted.
        n_inits (int): Number of random initialization.

    Returns:
        (numpy.array, numpy.array): A 2-tuple of centroids, a column vector of
            centroids, and idx, a column vector of assigned centroids' indices.
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

    Args:
        X (numpy.array): Features' dataset
        K_values (list(int)): List of possible number of centroids.
        max_iters (int): Number of times the algorithm will be fitted.
        n_inits (int): Number of random initialization.

    Returns:
        (list(int), list(float)): A 2-tuple of K_values, a list of possible
            numbers of centroids, and cost_values, a computed cost for each K.
    """
    cost_values = []
    for K in K_values:
        centroids, idx = run_intensive_kmeans(X, K, max_iters, n_inits)
        cost = cost_function(X, idx, centroids)
        cost_values.append(cost)

    return K_values, cost_values
