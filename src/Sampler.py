import numpy as np
from numpy.random import uniform, normal


class PointSampler:
  """ Point sampler within an area """

  def __init__(self,
               xy_min: list,
               xy_max: list,
               distribution_type='uniform') -> None:
    self.type_uniform = 'uniform'
    self.type_normal = 'normal'
    self.available_types = [self.type_uniform, self.type_normal]

    self.distribution_type = distribution_type
    self.xy_min = xy_min
    self.xy_max = xy_max

  def sample(self, num_points: int) -> np.ndarray:
    """ Samples num_points according to distribution type """
    if self.distribution_type == self.type_uniform:
      return self.sample_uniform(num_points)
    elif self.distribution_type == self.type_normal:
      return self.sample_normal(num_points)
    else:
      raise NotImplementedError('Current type is not supported')

  def sample_uniform(self, num_points: int) -> np.ndarray:
    """ Samples points from uniform distribution """
    return uniform(self.xy_min, self.xy_max, (num_points, 2))

  def sample_normal(self, num_points: int) -> np.ndarray:
    """ Samples points from normal distribution """
    # ensure at least 99% points are within region and centered at region center
    center = 0.5 * (np.array(self.xy_min) + np.array(self.xy_max))
    stddev = (1 / 3) * 0.5 * (np.array(self.xy_max) - np.array(self.xy_min))
    return np.clip(normal(center, stddev, size=(num_points, 2)),
                   np.array(self.xy_min), np.array(self.xy_max))
