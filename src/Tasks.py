from threading import Thread
from queue import Empty
import numpy as np
import time
from Graph import Graph


class Tasks(Thread):

  def __init__(self, G: Graph, lambda_p: float, mode='random'):
    """ Constructor """
    Thread.__init__(self)

    self.G = G
    self.lambda_p = lambda_p  # Poisson process rate parameter

    self.done = False

    self.mode_random = 'random'  # randoms
    self.mode_fcfs = 'fcfs'  # first-come-first-serve
    self.mode_dc = 'dc'  # divide-and-conquer
    self.mode_m_sqm = 'm_sqm'  # m-SQM

    self.available_modes = [
        self.mode_random, self.mode_fcfs, self.mode_dc, self.mode_m_sqm
    ]
    self.current_mode = mode

    # sanity check
    mode_recognized = False
    for available_mode in self.available_modes:
      if self.current_mode == available_mode:
        mode_recognized = True
        break
    if not mode_recognized:
      raise NotImplementedError(
          'Current assignment mode is not supported: %s' % self.current_mode)

  #################################################################
  # YOUR WORK HERE: generate and assign tasks
  # This example code just assigns tasks in order
  #################################################################
  def run(self):
    while (not self.done):
      # time until next task
      T = -np.log(np.random.rand(1)) / self.lambda_p
      time.sleep(T[0])

      # location of next task
      t = np.multiply(np.random.rand(2), np.array([2, 2])) - np.array([1, 1])

      if self.current_mode == self.mode_random:
        self.assignRandom(t)
      elif self.current_mode == self.mode_fcfs:
        self.assignFCFS(t)
      elif self.current_mode == self.mode_dc:
        self.assignDC(t)
      elif self.current_mode == self.mode_m_sqm:
        self.assignMSQM(t)
      else:
        raise NotImplementedError(
            'Current task assignment mode is not supported')

  def assignRandom(self, t: np.ndarray):
    # assign to a random vehicle
    inode = np.random.randint(self.G.Nv)
    self.G.V[inode].assignTask(t)

  def assignFCFS(self, t: np.ndarray):
    # assign to the nearest vehicle, FCFS
    nearest_inode = -1
    nearest_dist = np.inf
    for inode in range(self.G.Nv):
      pos = self.G.V[inode].getState()
      dist = np.linalg.norm(t - pos)
      if dist < nearest_dist:
        nearest_dist = dist
        nearest_inode = inode

    self.G.V[nearest_inode].assignTask(t)

  def assignMSQM(self, t: np.ndarray):
    # assign to the nearest m median, FCFS
    distances = np.linalg.norm((t.reshape((1, 2)) - self.G.centroids), axis=1)
    node_index = np.argmin(distances)

    self.G.V[node_index].assignTask(t)

  def assignDC(self, t: np.ndarray):
    # assign task to corresponding partition based on its coordinates
    #
    # Current designed partition:
    #   Divide the entire space [-1,1]x[-1,1] into self.G.Nv wedges, centering at (0,0)
    theta = np.arctan2(t[1], t[0])
    dTheta = 2 * np.pi / self.G.Nv
    target_inode = int(np.floor((theta - (-np.pi)) / dTheta))
    self.G.V[target_inode].assignTask(t)

  def terminate(self):
    self.done = True
