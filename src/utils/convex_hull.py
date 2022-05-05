from scipy.spatial import Delaunay
import numpy as np


def in_hull(p: np.ndarray, hull: np.ndarray):
  """
  Test if points in `p` are in `hull`

  `p` should be a `NxK` coordinates of `N` points in `K` dimensions
  `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
  coordinates of `M` points in `K`dimensions for which Delaunay triangulation
  will be computed

  Referenced from: https://stackoverflow.com/a/16898636
  """
  if hull is None:
    return False

  if not isinstance(hull, Delaunay):
    hull = Delaunay(hull)

  return hull.find_simplex(p) >= 0
