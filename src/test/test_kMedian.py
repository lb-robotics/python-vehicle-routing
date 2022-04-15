import sys
import os

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# unit test
if __name__ == '__main__':
  from utils.k_median import KMedian
  from Sampler import PointSampler
  from matplotlib import pyplot as plt

  sampler = PointSampler([-1, -1], [1, 1], 'uniform')
  points = sampler.sample(1000)
  clusterer = KMedian(points, 4)
  (clusters, centroids, iters) = clusterer.run()

  plt.figure()
  plt.scatter(points[:, 0], points[:, 1])
  plt.scatter(centroids[:, 0], centroids[:, 1])
  plt.show()
