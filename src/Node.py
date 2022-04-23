from threading import Thread
from queue import Empty
from xml.sax.handler import DTDHandler
import numpy as np
import time
from tsp_solver.greedy_numpy import solve_tsp

from Edge import Edge


class Node(Thread):

  def __init__(self, uid, mode='random'):
    """ Constructor """
    Thread.__init__(self)

    # basic information about network connectivity
    self.uid = uid  # node UID (an integer)
    self.out_nbr = []  # list of outgoing edges (see Edge class)
    self.in_nbr = []  # list of incoming edges (see Edge class)

    self.state = np.array([0, 0])  # state vars of interest ([x, y])
    self.done = False  # termination flag

    self.nominaldt = 0.05  # desired time step
    self.dt = 0  # time step
    self.speed = 0.5
    self.reach_goal_eps = 0.05  # threshold to decide whether goal is reached

    self.taskqueue = []  # list of tasks to perform in order
    self.taskqueue_serviceTime = []
    self.hub = np.zeros(2)

    ############### mode ###############
    self.mode_random = 'random'  # randoms
    self.mode_fcfs = 'fcfs'  # first-come-first-serve
    self.mode_dc = 'dc'  # divide-and-conquer
    self.mode_m_sqm = 'm_sqm'  # m-SQM

    self.available_modes = [
        self.mode_random, self.mode_fcfs, self.mode_dc, self.mode_m_sqm
    ]
    self.current_mode = mode

    # sanity check
    if mode not in self.available_modes:
      raise NotImplementedError(
          'Current assignment mode is not supported: %s' % self.current_mode)
    ####################################

    # Divide and Conquer
    self.tsp_path = []
    self.past_tasks = []

  def __str__(self):
    """ Printing """
    return "Node %d has %d in_nbr and %d out_nbr" % (self.uid, len(
        self.in_nbr), len(self.out_nbr))

  ################################################
  #
  # Modify the graph
  #
  ################################################

  def addOutgoing(self, e: Edge):
    """ Add an edge for outgoing messages """
    self.out_nbr.append(e)

  def addIncoming(self, e: Edge):
    """ Add an edge for incoming messages """
    self.in_nbr.append(e)

  ################################################
  #
  # Set states externally
  #
  ################################################

  def setState(self, s: np.ndarray):
    """ update the state of the node """
    self.state = s
    self.hub = s

  def getState(self) -> np.ndarray:
    """ return the state of the node """
    return self.state

  def terminate(self):
    """ stop sim """
    self.done = True

  ################################################
  #
  # Task assignment
  #
  ################################################

  def assignTask(self, t: tuple):
    """ Add a task to the queue """
    t_loc, t_s = t
    self.taskqueue.append(t_loc)
    self.taskqueue_serviceTime.append(t_s)

  ################################################
  #
  # Run the vehicle
  #
  ################################################

  def run(self):
    """ Send messages, Retrieve message, Transition """
    while (not self.done):
      start = time.time()

      self.systemdynamics()
      self.updategoal()

      end = time.time()
      time.sleep(max(self.nominaldt - (end - start), 0))

  def systemdynamics(self):
    """ Move the vehicle towards the goal """
    if self.current_mode == self.mode_dc:
      self.systemdynamics_dc()
    elif self.current_mode == self.mode_m_sqm:
      self.systemdynamics_m_sqm()
    elif self.current_mode == self.mode_random:
      self.systemdynamics_fcfs()
    elif self.current_mode == self.mode_fcfs:
      self.systemdynamics_fcfs()
    else:
      raise NotImplementedError("Current mode is not supported")

  def updategoal(self):
    """ Updates goal to the next goal if this one has been reached """
    if self.current_mode == self.mode_dc:
      self.updategoal_dc()
    else:
      self.updategoal_fcfs()

  def compute_tsp(self):
    """ Computes a TSP path over all tasks in the taskqueue """
    # construct distance matrix
    N = len(self.taskqueue)

    if N > 1:
      dist_matrix = np.zeros((N, N))
      for i in range(N - 1):
        for j in range(i, N):
          dist_matrix[j, i] = np.linalg.norm(self.taskqueue[i] -
                                             self.taskqueue[j])
      self.tsp_path = solve_tsp(dist_matrix)
    else:
      self.tsp_path.append(0)

  def systemdynamics_dc(self):
    # 1. If taskqueue is NOT empty, visit next TSP waypoint;
    # 2. If taskqueue is empty, return back to centroid of all past tasks
    if len(self.taskqueue) > 0 and len(self.tsp_path) > 0:
      this_goal = self.taskqueue[self.tsp_path[0]]
      velocity = this_goal - self.state
      velocity = velocity * (self.speed / np.linalg.norm(velocity))
      if np.linalg.norm(this_goal - self.state) > self.reach_goal_eps:
        self.state = self.state + self.nominaldt * velocity
    elif len(self.taskqueue) == 0 and len(self.past_tasks) > 0:
      this_goal = np.mean(np.stack(self.past_tasks), axis=0)
      velocity = this_goal - self.state
      velocity = velocity * (self.speed / np.linalg.norm(velocity))
      if np.linalg.norm(this_goal - self.state) > self.reach_goal_eps:
        self.state = self.state + self.nominaldt * velocity

  def systemdynamics_m_sqm(self):
    # 1. If taskqueue is NOT empty, visit next task;
    # 2. If taskqueue is empty, return back to hub
    if len(self.taskqueue) > 0:
      this_goal = self.taskqueue[0]
    else:
      this_goal = self.hub
    velocity = this_goal - self.state
    if np.linalg.norm(velocity) > 0:
      velocity = velocity * (self.speed / np.linalg.norm(velocity))

    if np.linalg.norm(this_goal - self.state) > self.reach_goal_eps:
      self.state = self.state + self.nominaldt * velocity

  def systemdynamics_fcfs(self):
    if (len(self.taskqueue) > 0):
      this_goal = self.taskqueue[0]
      velocity = this_goal - self.state
      velocity = velocity * (self.speed / np.linalg.norm(velocity))

      self.state = self.state + self.nominaldt * velocity

  def updategoal_dc(self):
    if len(self.taskqueue) > 0:
      # 1. If TSP path is computed, visit next TSP waypoint;
      # 2. If TSP path is empty (finished/not computed), compute new TSP path
      if len(self.tsp_path) > 0:
        this_goal = self.taskqueue[self.tsp_path[0]]
        if (np.linalg.norm(this_goal - self.state) < self.reach_goal_eps):
          # service the task
          t_s = self.taskqueue_serviceTime.pop(self.tsp_path[0])
          time.sleep(t_s)

          self.past_tasks.append(self.taskqueue[self.tsp_path[0]])
          self.taskqueue.pop(self.tsp_path[0])
          task_id = self.tsp_path.pop(0)

          path = np.array(self.tsp_path)
          path[path > task_id] = path[path > task_id] - 1
          self.tsp_path = path.tolist()

          print("%d: Task done! Starting new task" % self.uid)
      else:
        self.compute_tsp()

  def updategoal_fcfs(self):
    if len(self.taskqueue) > 0:
      # FCFS, no TSP
      this_goal = self.taskqueue[0]
      if (np.linalg.norm(this_goal - self.state) < self.reach_goal_eps):
        t_s = self.taskqueue_serviceTime.pop(0)
        time.sleep(t_s)
        self.taskqueue.pop(0)
        print("%d: Task done! Starting new task" % self.uid)
