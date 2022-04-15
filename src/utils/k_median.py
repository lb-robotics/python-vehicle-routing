import numpy as np
from scipy.spatial.distance import euclidean, cdist
from numpy.linalg import norm


class KMedian:

  def __init__(self, points: np.ndarray, num_clusters: int) -> None:
    self.num_clusters = num_clusters
    self.points = points

  def kmeans_iter(self, centroids: np.ndarray) -> tuple:
    """
    Performs one iteration of the k-means clustering algorithm.
    
    INPUT:
      centroids - (num_clusters,2) a list of the centroid values
    
    OUTPUT:
      assignments - an updated list of cluster assignments 
      updated_centroids - an updated list of the new centroid values
    """
    X = self.points
    k = self.num_clusters

    assignments = np.zeros(len(X))
    updated_centroids = np.zeros((k, 2))

    for i in range(len(X)):
      assignments[i] = 0
      min_dist = euclidean(X[i], centroids[0])
      for j in range(len(centroids)):
        if euclidean(X[i], centroids[j]) < min_dist:
          assignments[i] = j
          min_dist = euclidean(X[i], centroids[j])

    updated_centroids = []
    for i in range(self.num_clusters):
      updated_centroids.append(
          KMedian.geometric_median(self.points[assignments == i]))

    return assignments, np.stack(updated_centroids)

  def run(self) -> tuple:
    """
    Performs k-median clustering by calling kmeans_iter until no centroid value changes.
    
    INPUT:

    OUTPUT:
      assignments - (N,) of indices of cluster assignments in X
      centroids - (k,2) centroid values for each cluster
      iters - the number of iterations it took for k-means to converge
    """
    centroids = self.kmeanspp_init()
    prev_centroids = None
    iters = 0

    while prev_centroids is None or np.any(centroids != prev_centroids):
      prev_centroids = centroids
      (assignments, centroids) = self.kmeans_iter(prev_centroids)
      iters += 1

    return assignments, centroids, iters

  def kmeanspp_init(self) -> np.ndarray:
    """Runs K-means++ initialization method over centroids.

    Returns:
        np.ndarray: (num_clusters,2) centroids initialized for each cluster
    """
    k = self.num_clusters

    # 1. Randomly choose one data point as the first centroid
    list_centroids = [
        self.points[np.random.choice(np.arange(len(self.points)))]
    ]
    num_chosen_centroids = 1

    # 2. Loop until all k centroids have been chosen
    for _ in range(1, k):
      closest_dist_to_centroids = []
      # 2a. Loop over all data points & find its closest distance to centroids
      for point in self.points:
        min_dist = np.inf
        for i in range(num_chosen_centroids):
          centroid = list_centroids[i]
          if norm(point - centroid) < min_dist:
            min_dist = norm(point - centroid)
        closest_dist_to_centroids.append(min_dist)

      # 2b. Choose a new data point as a new centroid, according to the
      # distribution where farthest data point has the largest probability to
      # be chosen
      closest_dist_to_centroids = np.array(closest_dist_to_centroids) / np.sum(
          closest_dist_to_centroids)
      list_centroids.append(self.points[np.random.choice(
          len(self.points), p=closest_dist_to_centroids)])

    return np.stack(list_centroids)

  @staticmethod
  def geometric_median(X: np.ndarray, eps=1e-5) -> np.ndarray:
    """Computes geometric median given a set of points in 2D.

    Reference: https://stackoverflow.com/a/30305181

    Args:
        X (np.ndarray): (N,2) the set of points in 2D.
        eps (float, optional): threshold for finding the median. Defaults to 1e-5.

    Returns:
        np.ndarray: (2,) 2D coordinate of the geometric median.
    """
    y = np.mean(X, 0)

    while True:
      D = cdist(X, [y])
      nonzeros = (D != 0)[:, 0]

      Dinv = 1 / D[nonzeros]
      Dinvs = np.sum(Dinv)
      W = Dinv / Dinvs
      T = np.sum(W * X[nonzeros], 0)

      num_zeros = len(X) - np.sum(nonzeros)
      if num_zeros == 0:
        y1 = T
      elif num_zeros == len(X):
        return y
      else:
        R = (T - y) * Dinvs
        r = norm(R)
        rinv = 0 if r == 0 else num_zeros / r
        y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

      if euclidean(y, y1) < eps:
        return y1

      y = y1
