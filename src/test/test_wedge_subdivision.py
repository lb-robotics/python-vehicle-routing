import sys
import os

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# unit test
if __name__ == '__main__':
  from WedgeSubdivider import WedgeSubdivider
  import numpy as np
  from matplotlib import pyplot as plt

  depot = np.zeros(2)
  subdivider = WedgeSubdivider(depot, 10, [-1, -1], [1, 1], 'uniform')
  angles = subdivider.subdivide()

  plt.figure()
  ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1))
  ax.set_aspect('equal', 'box')

  ymin, ymax = plt.gca().get_ylim()
  x0, y0 = depot
  for angle in angles:
    m = np.tan(angle)
    if angle >= 0:
      xmax = x0 + (ymax - y0) / m
      ax.plot([xmax, x0], [ymax, y0])
    else:
      xmin = x0 + (ymin - y0) / m
      ax.plot([xmin, x0], [ymin, y0])

  plt.show()
