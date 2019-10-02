from math import inf, sqrt

from numpy import zeros, int64, power, sum, mean
from numpy.random import permutation


def euclidean_dist(p, q):
    dist = p - q
    dist = power(dist, 2)
    dist = sum(dist)
    dist = sqrt(dist)

    return dist


def find_closest_centroids(X, initial_centroids):

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
    m, n = X.shape
    centroids = zeros((K, n))
    for k in range(K):
        centroids[k] = mean(X[(idx == k).flatten()], axis=0)

    return centroids


def init_centroids(X, K):
    centroids = permutation(X)
    centroids = centroids[0:K, :]
    return centroids


def run_kmeans(X, K, max_iters):
    centroids = init_centroids(X, K)

    for _ in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)

    return centroids, idx


def cost_function(X, idx, centroids):
    cost = 0
    m = len(X)
    for i in range(m):
        centroid = centroids[idx[i][0]]
        cost += power(euclidean_dist(X[i], centroid), 2)

    cost = cost / m
    return cost


def run_intensive_kmeans(X, K, max_iters, n_inits):
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
    cost_values = []
    for K in K_values:
        centroids, idx = run_intensive_kmeans(X, K, max_iters, n_inits)
        cost = cost_function(X, idx, centroids)
        cost_values.append(cost)

    return zip(K_values, cost_values)
