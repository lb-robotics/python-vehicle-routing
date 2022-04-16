from Node import *
from Edge import *

from matplotlib import pyplot as plt
from matplotlib import animation


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
    mode_recognized = False
    for available_mode in self.available_modes:
      if self.current_mode == available_mode:
        mode_recognized = True
        break
    if not mode_recognized:
      raise NotImplementedError(
          'Current assignment mode is not supported: %s' % self.current_mode)
    ####################################

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
    for i in range(self.Nv):
      if self.V[i].current_mode == self.mode_dc and len(
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
    x, y = self.gatherNodeLocations()
    self.pts.set_data(x, y)

    x, y = self.gatherTaskLocations()
    self.pts_t.set_data(x, y)

    self.num_active_tasks.append(self.gatherNumActiveTasks())
    return self.pts,
