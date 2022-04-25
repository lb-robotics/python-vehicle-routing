from Node import *
from Edge import *
from Graph import *
from Tasks import *

import numpy as np
import time

from Sampler import PointSampler
from utils.k_median import KMedian


def generateRandomGraph(N: int, mode='random') -> Graph:
  G = Graph(mode=mode)

  if mode == 'm_sqm':
    # m-SQM --- precompute m-median points
    sampler = PointSampler([-1, -1], [1, 1], distribution_type='uniform')
    num_samples = 10000
    points_sampled = sampler.sample(num_samples)
    k_median = KMedian(points_sampled, N)
    print('========== m-SQM policy precomputation ==========')
    _, centroids, iters = k_median.run()
    print('k-median clustering: %d points, %d iters' % (num_samples, iters))
    print('========== m-SQM precomputation finished ==========')
    G.centroids = centroids

  for inode in range(N):
    n = Node(inode, [-1.5, -1.5], [1.5, 1.5], mode, dist_type='uniform')

    # generate node states
    if mode == 'm_sqm':
      # m-SQM --- initialize each vehicle to be the m-median locations within the area
      n.setState(centroids[inode])
    else:
      n.setState(
          np.multiply(np.random.rand(2), np.array([2, 2])) - np.array([1, 1]))

    G.addNode(n)

    # add all-to-all edges
    for iedge in range(inode):
      G.addEdge(iedge, inode, 0)
      G.addEdge(inode, iedge, 0)

  return G


### MAIN
if __name__ == '__main__':
  list_available_modes = ['random', 'fcfs', 'dc', 'm_sqm', 'utsp']
  current_mode = 'utsp'

  # generate a random graph with 10 nodes, when in dc 5 is maximum
  G = generateRandomGraph(10, mode=current_mode)

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
