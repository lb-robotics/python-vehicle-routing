import sys
import os

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# unit test
if __name__ == '__main__':
  import pyvoro
  import numpy as np

  num_samples = 10
  generators = np.random.uniform(low=2., high=8., size=(num_samples, 2))
  radius = np.array([0, 0, 0, 0, 0, 0])

  cells = pyvoro.compute_2d_voronoi(
      generators,  # point positions, 2D vectors this time.
      [[0.0, 10.0], [0.0, 10.0]],  # box size, again only 2D this time.
      0.1,  # block size; same as before.
      radii=radius  # particle radii -- optional and keyword-compatible.
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
  ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
  ax.axis('equal')
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
