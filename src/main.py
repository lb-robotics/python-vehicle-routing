from Node import *
from Edge import *
from Graph import *
from Tasks import *

import numpy as np
import time


def generateRandomGraph(N: int, use_tsp=False) -> Graph:
  G = Graph()

  for inode in range(N):
    # randomly generate node states
    n = Node(inode, use_tsp=use_tsp)

    if use_tsp:
      # divide and conquer --- initialize each vehicle inside the designed partition
      #
      # Current designed partition:
      #   Divide the entire space [-1,1]x[-1,1] into N wedges, centering at (0,0)
      dTheta = 2 * np.pi / N
      pol = np.array([1, -np.pi + inode * dTheta + dTheta / 2])  # rho, phi
      xy = np.array([pol[0] * np.cos(pol[1]), pol[0] * np.sin(pol[1])])
      n.setState(xy)
    else:
      n.setState(
          np.multiply(np.random.rand(2), np.array([2, 2])) - np.array([1, 1]))

    G.addNode(n)

  return G


### MAIN
if __name__ == '__main__':
  list_available_modes = ['random', 'fcfs', 'dc']
  current_mode = 'random'

  # generate a random graph with 10 nodes
  use_tsp = (current_mode == 'dc')
  G = generateRandomGraph(10, use_tsp)

  # set up thread for generating and assigning tasks
  T = Tasks(G, 5, mode=current_mode)

  print("========== Starting now ==========")
  print("Close the figure to stop the simulation")
  G.run()  # start threads in nodes
  T.start()  # start generating tasks
  G.setupAnimation()  # set up plotting
  print("Sending stop signal.....")
  G.stop()  # send stop signal
  T.terminate()
  T.join()
  print("========== Terminated ==========")

  # gather number of active tasks over time....
  num_active_tasks = np.array(G.num_active_tasks)
  timesteps = (G.animatedt / 1000) * np.arange(len(num_active_tasks))

  plt.figure()
  plt.plot(timesteps, num_active_tasks)
  plt.xlabel('Timestamp [s]')
  plt.ylabel('No. active tasks')
  plt.title('Number of Active Tasks Over Time')
  plt.show()
