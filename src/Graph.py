from Node import *
from Edge import *

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection

from GlobalTaskQueue import global_taskqueue


class Graph:

  def __init__(self, mode='random', filename=None):
    """ Constructor """
    self.Nv = 0
    self.V = []
    self.E = []
    self.root = None
    self.num_active_tasks = []
    self.centroids = None

    # for plotting
    self.animatedt = 100  # milliseconds
    self.fig = plt.figure()
    self.ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    self.ax.set_aspect('equal', 'box')
    self.pts, = self.ax.plot([], [], 'bo')
    self.pts_t, = self.ax.plot([], [], 'r*')
    self.anim = None
    # for partitioning
    self.lines = LineCollection([], linestyles="-.", color='green')
    self.ax.add_collection(self.lines)

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

    # for UTSP num active tasks calculation
    self.utsp_num_remaining_tasks = -1

    # for reading in graphs if they come from a file
    if not (filename is None):
      # read the graph from a file
      with open(filename) as f:
        # nodes
        line = f.readline()
        self.Nv = int(line)
        for inode in range(self.Nv):
          self.addNode(Node(inode))

        # edges
        line = f.readline()
        while line:
          data = line.split()

          in_nbr = int(data[0])
          out_nbr = int(data[1])
          cost = float(data[2])

          self.addEdge(in_nbr, out_nbr, cost)

          line = f.readline()

      f.close()

  def __str__(self):
    """ Printing """
    return "Graph: %d nodes, %d edges" % (self.Nv, len(self.E))

  ################################################
  #
  # Modify the graph
  #
  ################################################

  def addNode(self, n):
    """ Add a node to the graph """
    self.V.append(n)
    self.Nv += 1

  def addEdge(self, i, o, c):
    """ Add an edge between two nodes """
    e = Edge(i, o, c)
    self.V[i].addOutgoing(e)
    self.V[o].addIncoming(e)
    self.E.append(e)

  ################################################
  #
  # Start and Stop computations
  #
  ################################################

  def run(self):
    """ Run the alg on all of the nodes """
    # Start running the threads
    for i in range(self.Nv):
      self.V[i].start()

  def stop(self):
    """ Send a stop signal """
    # Send a stop signal
    for i in range(self.Nv):
      self.V[i].terminate()
    # Wait until all the nodes are done
    for i in range(self.Nv):
      self.V[i].join()

  ################################################
  #
  # Animation helpers
  #
  ################################################

  def gatherNodeLocations(self):
    """ Collect state information from all the nodes """
    x = []
    y = []
    for i in range(self.Nv):
      x.append(self.V[i].state[0])
      y.append(self.V[i].state[1])
    return x, y

  def gatherTaskLocations(self):
    """ Collect state information from all the nodes """
    x = []
    y = []

    if self.current_mode == self.mode_nc:
      for task_loc, _ in global_taskqueue.taskqueue.values():
        x.append(task_loc[0])
        y.append(task_loc[1])
    else:
      for i in range(self.Nv):
        if (self.V[i].current_mode == self.mode_dc
            or self.current_mode == self.mode_utsp) and len(
                self.V[i].tsp_path) > 0 and len(self.V[i].taskqueue) > 0:
          x.append(self.V[i].taskqueue[self.V[i].tsp_path[0]][0])
          y.append(self.V[i].taskqueue[self.V[i].tsp_path[0]][1])
        elif (len(self.V[i].taskqueue) > 0):
          x.append(self.V[i].taskqueue[0][0])
          y.append(self.V[i].taskqueue[0][1])
    return x, y

  def gatherNumActiveTasks(self):
    """ Collect number of active tasks over time """
    num_active_tasks = 0
    for i in range(self.Nv):
      num_active_tasks += len(self.V[i].taskqueue)

    if self.current_mode == self.mode_utsp:
      num_active_tasks += self.utsp_num_remaining_tasks
    if self.current_mode == self.mode_nc:
      num_active_tasks = len(global_taskqueue.taskqueue)

    return num_active_tasks

  def setupAnimation(self):
    """ Initialize the animation """
    self.anim = animation.FuncAnimation(self.fig,
                                        self.animate,
                                        interval=self.animatedt,
                                        blit=False)

    plt.show()

  def animate(self, i: int):
    """ Animation helper function """
    if self.current_mode == self.mode_dc:
      self.lines.set_segments(self.V[0].partitioner.getLineCollections())

    x, y = self.gatherNodeLocations()
    self.pts.set_data(x, y)

    x, y = self.gatherTaskLocations()
    self.pts_t.set_data(x, y)

    self.num_active_tasks.append(self.gatherNumActiveTasks())
    return self.pts, self.lines

  def draw_wedges_utsp(self, depot: np.ndarray, angles: np.ndarray):
    ymin, ymax = self.ax.get_ylim()
    x0, y0 = depot
    for angle in angles:
      m = np.tan(angle)
      if angle >= 0:
        xmax = x0 + (ymax - y0) / m
        self.ax.plot([xmax, x0], [ymax, y0], '--')
      else:
        xmin = x0 + (ymin - y0) / m
        self.ax.plot([xmin, x0], [ymin, y0], '--')
