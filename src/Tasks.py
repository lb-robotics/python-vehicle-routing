from threading import Thread
from queue import Empty
import numpy as np
import time
from Graph import Graph
import math
from RootSquareDistributionCalculator import RootSquareDistributionCalculator

BETA_TSP_2 = 0.7120


class Tasks(Thread):

  def __init__(self,
               G: Graph,
               lambda_p: float,
               mean_s: float = 1.0,
               mode='random'):
    """ Constructor """
    Thread.__init__(self)

    self.G = G
    self.lambda_p = lambda_p  # Poisson process rate parameter
    self.mean_s = mean_s  # mean of service time, \bar{s}

    self.done = False

    self.mode_random = 'random'  # randoms
    self.mode_fcfs = 'fcfs'  # first-come-first-serve
    self.mode_dc = 'dc'  # divide-and-conquer
    self.mode_m_sqm = 'm_sqm'  # m-SQM
    self.mode_utsp = 'utsp'  # UTSP

    self.available_modes = [
        self.mode_random, self.mode_fcfs, self.mode_dc, self.mode_m_sqm,
        self.mode_utsp
    ]
    self.current_mode = mode

    # sanity check
    if mode not in self.available_modes:
      raise NotImplementedError(
          'Current assignment mode is not supported: %s' % self.current_mode)

    # ########### UTSP Policy ###########
    # define parameters
    self.utsp_r = 10  # r, number of wedges
    self.tasksetqueue = []  # list of list (task set), each set is of size n/r
    self.tasks_remained = None  # list of remained tasks per wedge, excluding those in taskqueue

    distributionCalculator = RootSquareDistributionCalculator(
        np.zeros(2), [-1, -1], [1, 1], 'uniform')
    rootSquareDistribution = distributionCalculator.calculate()
    rho = self.lambda_p * self.mean_s / G.Nv
    # n, hyperparameter, set to the minimum number that ensures stability.
    self.utsp_n = math.ceil(
        ((self.lambda_p * BETA_TSP_2 * rootSquareDistribution)**2) /
        ((G.Nv * G.V[0].speed * (1 - rho))**2))
    self.taskset_size = self.utsp_n // self.utsp_r

    # initialize wedge divider
    self.utsp_wedge_angles = None
    if self.current_mode == self.mode_utsp:
      print('========== UTSP policy precomputation ==========')
      from WedgeSubdivider import WedgeSubdivider
      subdivider = WedgeSubdivider(np.zeros(2), self.utsp_r, [-1, -1], [1, 1],
                                   'uniform')
      self.utsp_wedge_angles = subdivider.subdivide()
      while len(self.utsp_wedge_angles) > self.utsp_r:
        self.utsp_wedge_angles = np.delete(self.utsp_wedge_angles, -1)
      self.tasks_remained = [[] for _ in range(self.utsp_r)]
      print('========== UTSP precomputation finished ==========')
      self.G.draw_wedges_utsp(np.zeros(2), self.utsp_wedge_angles)
    # ###################################

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

      # service time of next task
      #   assume Gaussian distribution with self.mean_s and variance 1.0
      s = np.random.normal(loc=self.mean_s, scale=self.mean_s / 3)
      s = np.clip(s, 0.0, None)

      if self.current_mode == self.mode_random:
        self.assignRandom((t, s))
      elif self.current_mode == self.mode_fcfs:
        self.assignFCFS((t, s))
      elif self.current_mode == self.mode_dc:
        self.assignDC((t, s))
      elif self.current_mode == self.mode_m_sqm:
        self.assignMSQM((t, s))
      elif self.current_mode == self.mode_utsp:
        self.assignUTSP((t, s))
      else:
        raise NotImplementedError(
            'Current task assignment mode is not supported')

  def assignRandom(self, t: tuple):
    # assign to a random vehicle
    inode = np.random.randint(self.G.Nv)
    self.G.V[inode].assignTask(t)

  def assignFCFS(self, t: tuple):
    # assign to the nearest vehicle, FCFS
    t_loc, t_s = t
    nearest_inode = -1
    nearest_dist = np.inf
    for inode in range(self.G.Nv):
      pos = self.G.V[inode].getState()
      dist = np.linalg.norm(t_loc - pos)
      if dist < nearest_dist:
        nearest_dist = dist
        nearest_inode = inode

    self.G.V[nearest_inode].assignTask(t)

  def assignMSQM(self, t: tuple):
    # assign to the nearest m median, FCFS
    t_loc, t_s = t
    distances = np.linalg.norm((t_loc.reshape((1, 2)) - self.G.centroids),
                               axis=1)
    node_index = np.argmin(distances)

    self.G.V[node_index].assignTask(t)

  def assignDC(self, t: tuple):
    # assign task to corresponding partition based on its coordinates
    #
    # Current designed partition:
    #   Divide the entire space [-1,1]x[-1,1] into self.G.Nv wedges, centering at (0,0)
    t_loc, t_s = t
    theta = np.arctan2(t_loc[1], t_loc[0])
    dTheta = 2 * np.pi / self.G.Nv
    target_inode = int(np.floor((theta - (-np.pi)) / dTheta))
    self.G.V[target_inode].assignTask(t)

  def assignUTSP(self, t: tuple):
    # 1. queue them up in each subdivided wedge based on location
    t_loc, t_s = t
    theta = np.arctan2(t_loc[1], t_loc[0])
    wedge_idx = np.searchsorted(self.utsp_wedge_angles, theta,
                                side='right') - 1
    self.tasks_remained[wedge_idx].append(t)

    # 2. if a queue is full, collect it and get ready to assign to vehicle
    if len(self.tasks_remained[wedge_idx]) >= self.taskset_size:
      tasks = [
          self.tasks_remained[wedge_idx].pop(0)
          for _ in range(self.taskset_size)
      ]
      self.tasksetqueue.append(tasks)

    # 3. assign tasks to first-available vehicle
    if len(self.tasksetqueue) > 0:
      for inode in range(self.G.Nv):
        if len(self.G.V[inode].taskqueue) == 0:
          tasks = self.tasksetqueue.pop(0)
          self.G.V[inode].assignTasks(tasks)
        if len(self.tasksetqueue) == 0:
          break

    # 4. Calculate current remaining tasks for plotting
    num_remaining_tasks = 0
    for taskset in self.tasksetqueue:
      num_remaining_tasks += len(taskset)
    for tasks_remained in self.tasks_remained:
      num_remaining_tasks += len(tasks_remained)
    self.G.utsp_num_remaining_tasks = num_remaining_tasks

  def terminate(self):
    self.done = True
