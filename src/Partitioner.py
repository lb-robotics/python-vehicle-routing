import numpy as np
import pyvoro
from numpy.linalg import norm


class EquitablePartitioner:
  """ 
  Implements a distributed equitable partitioner based on power diagrams.

  See paper: 
  M. Pavone, A. Arsie, E. Frazzoli and F. Bullo, "Distributed Algorithms for 
  Environment Partitioning in Mobile Robotic Networks," in IEEE Transactions 
  on Automatic Control, vol. 56, no. 8, pp. 1834-1848, Aug. 2011, doi: 
  10.1109/TAC.2011.2112410.
  """

  def __init__(self,
               xy_min: list,
               xy_max: list,
               distribution_type='uniform') -> None:
    # distribution type
    self.dist_type_uniform = 'uniform'
    self.list_available_dist_type = [self.dist_type_uniform]
    if distribution_type not in self.list_available_dist_type:
      raise NotImplementedError(
          "Current distribution type is not supported: %s" % distribution_type)
    self.dist_type = distribution_type

    # Power diagram parameters
    self.generators = None  # should be np.array of (N,2)
    self.radii = None  # should be np.array of (N)
    self.N = 0  # number of generators

    # area to partition
    self.xy_minmax = np.stack([np.array(xy_min), np.array(xy_max)])
    self.cells = None

  def getLineCollections(self) -> list:
    """ Get current Voronoi cells for animation """
    line_list = []
    if self.cells is not None:
      for cell in self.cells:
        vertices = cell['vertices']
        for face in cell['faces']:
          line_list.append(
              (vertices[face['vertices'][0]], vertices[face['vertices'][1]]))
    return line_list

  def getGenerators(self) -> np.ndarray:
    return self.generators

  def getVertices(self, gid: int) -> np.ndarray:
    """ Returns the vertices of the partition corresponding to some cell id """
    gid_cell = self.cells[gid]
    return np.stack(gid_cell["vertices"])

  def getAllAreas(self) -> np.ndarray:
    """ Returns volumes of all cells """
    if self.cells is None:
      return None

    areas = []
    for cell in self.cells:
      areas.append(cell["volume"])
    return np.array(areas)

  def setGenerators(self, generators: np.ndarray):
    """ Sets the locations of generators. The locations of generators are fixed. """
    assert (generators.shape[1] == 2)
    self.N = generators.shape[0]
    self.generators = generators

  def updateRadii(self, radii: np.ndarray):
    """ Sets the radii of generators. """
    assert (len(radii) == self.N)
    self.radii = radii

  def computeGrad(self, gid: int) -> float:
    """ Computes the gradient of energy function w.r.t. radius of generator gid. """
    if self.generators is None:
      raise ValueError("self.generators is not set")
    if self.radii is None:
      raise ValueError("self.radii is not set")

    if self.dist_type == self.dist_type_uniform:
      return self.computeGrad_uniform(gid)
    else:
      raise NotImplementedError(
          "Current distribution type is not supported: %s" % self.dist_type)

  def computeGrad_uniform(self, gid: int) -> float:
    """
    Assuming uniform measure, computes equation (Example 3.6):

    \partial H/\partial w_i = |Q|/(2*lambda_Q) * sum_{j \in N_i}(delta_ij/gamma_ij * (1/|V_j|**2 - 1/|V_i|**2))
    """
    # 1. generate power diagram based on current generators and values
    cells = pyvoro.compute_2d_voronoi(
        self.generators,
        [self.xy_minmax[:, 0].tolist(), self.xy_minmax[:, 1].tolist()],
        0.05,
        radii=np.sqrt(self.radii - np.min(self.radii)))
    gid_cell = cells[gid]
    assert np.all(gid_cell["original"] == self.generators[gid])
    self.cells = cells

    # 2. computes parameters
    lambda_Q = 1  # integral of prob. distribution over entire region is always 1

    dxy = self.xy_minmax[1] - self.xy_minmax[0]
    area_Q = dxy[0] * dxy[1]

    # 3. Sum over all adjacent cells
    nbr_sum = 0
    for face in gid_cell["faces"]:
      if face["adjacent_cell"] < 0:
        continue
      delta_ij = norm(gid_cell["vertices"][face["vertices"][0]] -
                      gid_cell["vertices"][face["vertices"][1]])
      gamma_ij = norm(gid_cell["original"] -
                      cells[face["adjacent_cell"]]["original"])
      area_Vj = cells[face["adjacent_cell"]]["volume"]
      area_Vi = gid_cell["volume"]
      nbr_sum += (delta_ij / gamma_ij) * (1 / (area_Vj**2) - 1 / (area_Vi**2))

    # 4. compute gradient
    grad_gid = area_Q / (2 * lambda_Q) * nbr_sum

    return grad_gid
