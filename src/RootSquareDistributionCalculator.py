import numpy as np
import math


class RootSquareDistributionCalculator:

  def __init__(self,
               depot: np.ndarray,
               xy_min: list,
               xy_max: list,
               distribution='uniform') -> None:
    self.depot = depot

    # area of interest (rectangle)
    self.xy_min = xy_min
    self.xy_max = xy_max

    # distribution type
    self.dist_uniform = 'uniform'
    self.dist_normal = 'normal'
    self.list_supported_dist = [self.dist_uniform, self.dist_normal]
    if distribution not in self.list_supported_dist:
      raise NotImplementedError("Current distribution is not supported: %s" %
                                distribution)
    self.distribution_type = distribution

  def calculate(self) -> float:
    """Calculate Root Square Integral of the distribution

    Args:

    Raises:
    NotImplementedError

    Returns:
    float: Root Square Integral of the distribution
    """
    if self.distribution_type == 'uniform':
      return self.xy_max[0] - self.xy_min[0]
    elif self.distribution_type == 'normal':
      a = self.xy_max[0] - self.xy_min[0]
      return 2 * math.sqrt(2 * np.pi) / math.erf(a / 2 / math.sqrt(2)) * (
          (math.erf(a / 4))**2)
