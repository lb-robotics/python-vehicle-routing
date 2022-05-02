from threading import Thread
from queue import Empty
import numpy as np
from numpy.linalg import norm
import time
from tsp_solver.greedy_numpy import solve_tsp

from utils.k_median import KMedian
from Partitioner import EquitablePartitioner

from Edge import Edge

from GlobalTaskQueue import global_taskqueue


class Node(Thread):

  def __init__(self,
               uid,
               xy_min: list,
               xy_max: list,
               mode='random',
               dist_type='uniform'):
    """ Constructor """
    Thread.__init__(self)

    # basic information about network connectivity
    self.uid = uid  # node UID (an integer)
    self.out_nbr = []  # list of outgoing edges (see Edge class)
    self.in_nbr = []  # list of incoming edges (see Edge class)

    # boundary of the world
    self.xy_min = xy_min
    self.xy_max = xy_max

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
    self.mode_utsp = 'utsp'  # UTSP
    self.mode_nc = 'nc'  # No-Communication

    self.available_modes = [
        self.mode_random, self.mode_fcfs, self.mode_dc, self.mode_m_sqm,
        self.mode_utsp, self.mode_nc
    ]
    self.current_mode = mode

    # sanity check
    if mode not in self.available_modes:
      raise NotImplementedError(
          'Current assignment mode is not supported: %s' % self.current_mode)
    ####################################

    ################ Divide and Conquer ###############
    self.tsp_path = []
    self.past_tasks = []  # will be re-used in No-Communication policy
    self.done_partition = False  # self partitioning termination flag
    self.all_done_partition = False  # all nodes partitioning termination flag
    self.partition_eps = 4e-6  # threshold to decide whether weight has reached critical point
    self.partition_weight = 0.009  # for power diagram partitioning
    self.partition_stepsize = 0.01  # for weight computation gradient descent
    self.partition_vertices = None  # vertex of the designated partition, should be Mx2
    self.partitioner = None

    if self.current_mode == self.mode_dc:
      self.partitioner = EquitablePartitioner(self.xy_min, self.xy_max,
                                              dist_type)
    ###################################################

    # UTSP
    self.taskset_size = -1

    ################ No-Communication ################
    self.next_task = None  # a tuple of (t_loc, t_s)
    self.next_task_tid = -1
    ##################################################

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

  def assignTasks(self, ts: list):
    """ Add a list of tasks to the queue (specifically for UTSP) """
    if self.current_mode != self.mode_utsp:
      raise RuntimeError(
          "Node in mode [%s] is not supposed to use this function" %
          self.current_mode)
    if self.taskset_size < 0:
      self.taskset_size = len(ts)
    for t_loc, t_s in ts:
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

      if self.current_mode == self.mode_dc:
        self.run_dc()
      else:
        self.systemdynamics()
        self.updategoal()

      end = time.time()
      time.sleep(max(self.nominaldt - (end - start), 0))

  def send(self):
    """ Send messages """
    if self.current_mode == self.mode_dc:
      self.send_dc()

  def transition(self):
    """ Update the states based on comms """
    if self.current_mode == self.mode_dc:
      self.transition_dc()

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
    elif self.current_mode == self.mode_utsp:
      self.systemdynamics_utsp()
    elif self.current_mode == self.mode_nc:
      self.systemdynamics_nc()
    else:
      raise NotImplementedError("Current mode is not supported")

  def updategoal(self):
    """ Updates goal to the next goal if this one has been reached """
    if self.current_mode == self.mode_dc:
      self.updategoal_dc()
    elif self.current_mode == self.mode_utsp:
      self.updategoal_utsp()
    elif self.current_mode == self.mode_nc:
      self.updategoal_nc()
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
      this_goal = KMedian.geometric_median(np.stack(self.past_tasks))
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

  def systemdynamics_utsp(self):
    # 1. If taskqueue is NOT empty, visit next TSP waypoint;
    # 2. If taskqueue is empty, return back to centroid of all past tasks
    if len(self.taskqueue) > 0 and len(self.tsp_path) > 0:
      this_goal = self.taskqueue[self.tsp_path[0]]
      velocity = this_goal - self.state
      velocity = velocity * (self.speed / np.linalg.norm(velocity))
      if np.linalg.norm(this_goal - self.state) > self.reach_goal_eps:
        self.state = self.state + self.nominaldt * velocity

  def systemdynamics_nc(self):
    if self.next_task is None:
      if not self.past_tasks:
        return
      this_goal = KMedian.geometric_median(np.stack(self.past_tasks))
    else:
      this_goal = self.next_task[0]

    # forward system dynamics
    velocity = this_goal - self.state
    velocity = velocity * (self.speed / np.linalg.norm(velocity))
    if np.linalg.norm(this_goal - self.state) > self.reach_goal_eps:
      self.state = self.state + self.nominaldt * velocity

  def updategoal_dc(self):
    if self.done_partition and len(self.taskqueue) > 0:
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

  def updategoal_utsp(self):
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

          # print("%d: Task done! Starting new task" % self.uid)
      else:
        if len(self.taskqueue) == self.taskset_size:
          self.compute_tsp()

  def updategoal_nc(self):
    # 1. Check if the goal task still exists
    if self.next_task_tid >= 0 and not global_taskqueue.hasTask(
        self.next_task_tid):
      self.next_task_tid = -1
      self.next_task = None

    # 2. Check if the goal task has been reached
    if self.next_task_tid >= 0:
      this_goal, t_s = self.next_task
      if (np.linalg.norm(this_goal - self.state) < self.reach_goal_eps):
        t_loc, _ = global_taskqueue.finishTask(self.next_task_tid)
        time.sleep(t_s)
        print("%d: Task done! Starting new task" % self.uid)
        self.past_tasks.append(t_loc)
        self.next_task_tid = -1
        self.next_task = None

    # 3. Assign new task to vehicle
    self.next_task_tid = global_taskqueue.getClosestTaskIdx(self.state)
    if self.next_task_tid >= 0:
      self.next_task = global_taskqueue.getTask(self.next_task_tid)
      if self.next_task is None:
        self.next_task_tid = -1
    else:
      self.next_task = None

  def run_dc(self):
    """ Main loop for m-DC policy """
    if self.all_done_partition:
      self.systemdynamics()
      self.updategoal()
    else:
      self.send()
      self.transition()

  def send_dc(self):
    """ m-DC policies sends uid, state, partitioning weight, done to all neighbors """
    for onbr in self.out_nbr:
      onbr.put(
          (self.uid, self.state, self.partition_weight, self.done_partition))

  def transition_dc(self):
    generators = [None for _ in range(len(self.in_nbr) + 1)]
    generators[self.uid] = self.state

    radii = [None for _ in range(len(self.in_nbr) + 1)]
    radii[self.uid] = self.partition_weight

    partition_finish_nodes = [self.done_partition]

    for inbr in self.in_nbr:
      uid, generator, radius, done = inbr.get()
      generators[uid] = generator
      radii[uid] = radius
      partition_finish_nodes.append(done)

    generators = np.stack(generators)
    radii = np.array(radii)
    partition_finish_nodes = np.array(partition_finish_nodes)

    if np.all(partition_finish_nodes):
      self.all_done_partition = True

    if not self.done_partition:
      # continue to update partition
      if self.partitioner.generators is None:
        self.partitioner.setGenerators(generators)
      if len(radii) > 0:
        self.partitioner.updateRadii(radii)
      grad_w = self.partitioner.computeGrad(self.uid)

      new_partition_weight = self.partition_weight - grad_w * self.partition_stepsize
      if norm(self.partition_weight -
              new_partition_weight) < self.partition_eps:
        print("Node %d done partitioning!" % self.uid)
        self.done_partition = True
        self.partition_vertices = self.partitioner.getVertices(self.uid)
      else:
        self.partition_weight = new_partition_weight
