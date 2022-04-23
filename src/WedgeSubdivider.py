import numpy as np
from numpy.linalg import norm
from Sampler import PointSampler


def cart2pol(xy: np.ndarray):
  assert (xy.shape[0] == 2)
  rho = norm(xy, axis=0)
  phi = np.arctan2(xy[1], xy[0])
  return np.stack([rho, phi])


class WedgeSubdivider:

  def __init__(self,
               depot: np.ndarray,
               num_wedges: int,
               xy_min: list,
               xy_max: list,
               distribution='uniform',
               mode='monte-carlo') -> None:
    self.depot = depot
    self.num_wedges = num_wedges
    self.dTheta = 0.005  # sweep step, in rad

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

    # subdiving mode
    self.mode_montecarlo = 'monte-carlo'
    self.list_supported_mode = [self.mode_montecarlo]
    if mode not in self.list_supported_mode:
      raise NotImplementedError("Current mode is not supported: %s" % mode)
    self.current_mode = mode

  def subdivide(self) -> np.ndarray:
    """Run subdivision algorithm (angle sweep in polar coordinates) 

    Args:

    Raises:
        NotImplementedError

    Returns:
        np.ndarray: (num_wedges,) starting angles of each subdivision 
    """

    if self.current_mode == self.mode_montecarlo:
      sampler = PointSampler(self.xy_min, self.xy_max, self.distribution_type)
      points = sampler.sample(100000)
      points_rebased = points.T - self.depot.reshape(2, 1)
      return self.subdivide_montecarlo(points_rebased)
    else:
      raise NotImplementedError("Current mode: %s is not supported" %
                                self.current_mode)

  def subdivide_montecarlo(self, points: np.ndarray) -> np.ndarray:
    points_polar = cart2pol(points)
    num_points = points.shape[1]

    list_subpoints = []

    start_theta = -np.pi
    list_angles = [start_theta]
    while start_theta < np.pi:
      end_theta = start_theta
      while end_theta < np.pi and np.sum(
          np.logical_and(points_polar[1] >= start_theta, points_polar[1] <=
                         end_theta)) < (num_points / self.num_wedges):
        end_theta += self.dTheta
      if end_theta < np.pi:
        list_angles.append(end_theta)
      print("New subdivision: %f to %f" % (start_theta, end_theta))
      list_subpoints.append(
          np.sum(
              np.logical_and(points_polar[1] >= start_theta,
                             points_polar[1] <= end_theta)))
      start_theta = end_theta

    ret = np.stack(list_angles)
    print("All subdivision angles:", ret)
    print("Num points in each region:", np.stack(list_subpoints))
    return ret
