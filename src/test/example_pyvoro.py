import sys
import os

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# unit test
if __name__ == '__main__':
  import pyvoro
  import numpy as np

  x_limits = [-1.5, 1.5]
  y_limits = [-1.5, 1.5]
  num_samples = 10
  generators = np.random.uniform(low=-1.3, high=1.3, size=(num_samples, 2))
  radius = np.random.rand(num_samples)

  cells = pyvoro.compute_2d_voronoi(
      generators.tolist(),  # point positions, 2D vectors this time.
      [x_limits, y_limits],  # box size, again only 2D this time.
      0.1,  # block size; same as before.
      radii=np.sqrt(radius).tolist(
      )  # particle radii -- optional and keyword-compatible.
  )

  import json
  os.makedirs("tmp", exist_ok=True)
  with open("tmp/cells.json", 'w') as outfile:
    json.dump(cells, outfile)
    print("cells have been dumped to tmp/cells.json")
  print(generators)

  from matplotlib import pyplot as plt
  from matplotlib.collections import LineCollection

  plt.figure()
  ax = plt.axes(xlim=tuple(x_limits), ylim=tuple(y_limits))
  ax.set_aspect('equal', 'box')
  ax.scatter(generators[:, 0], generators[:, 1])
  line_list = []
  for cell in cells:
    vertices = cell['vertices']
    for face in cell['faces']:
      line_list.append(
          (vertices[face['vertices'][0]], vertices[face['vertices'][1]]))
  line_collection = LineCollection(line_list, lw=1., colors='k')
  ax.add_collection(line_collection)
  plt.show()
