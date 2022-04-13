from queue import Queue


class Edge(Queue):

  def __init__(self, in_nbr: int, out_nbr: int, cost: float):
    """ Constructor """
    Queue.__init__(self)
    self.in_nbr = in_nbr
    self.out_nbr = out_nbr
    self.cost = cost

  def __str__(self):
    """ Printing """
    return "Edge (%d, %d), cost = %f" % (self.in_nbr, self.out_nbr, self.cost)
