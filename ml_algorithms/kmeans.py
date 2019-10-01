from math import inf, sqrt

from numpy import zeros, int64, power, sum, mean


def euclidean_dist(p, q):
    dist = p - q
    dist = power(dist, 2)
    dist = sum(dist)
    dist = sqrt(dist)

    return dist


def find_closest_centroids(X, initial_centroids):

    m = len(X)
    idx = zeros((m, 1), dtype=int64)

    K = len(initial_centroids)
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
