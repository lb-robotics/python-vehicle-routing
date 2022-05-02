from readerwriterlock import rwlock
import numpy as np


class GlobalTaskQueue:
  """ 
  A global task queue that allows only one instance to exist in the whole
  process. Implements the idea of Singleton and accessed by all robots & task
  generator. 

  This should only be used in No-Communication (NC) policy.
  """

  def __init__(self) -> None:
    self.lock = rwlock.RWLockFair()  # no threads should be starved
    self.taskqueue = {}
    self.next_tid = 0

  def addTask(self, new_t: tuple):
    """ Adds a new task to the taskqueue. Invokes writer lock. """
    with self.lock.gen_wlock():
      # self.taskqueue.update({self.next_tid, new_t})
      self.taskqueue[self.next_tid] = new_t
      self.next_tid += 1

  def finishTask(self, tid: int) -> tuple:
    """
    Removes and returns the task with index [tid] from taskqueue. Invokes
    writer lock.
    """
    with self.lock.gen_wlock():
      t = self.taskqueue.pop(tid)
      return t

  def getTask(self, tid: int) -> tuple:
    """ Reads the task with index [tid] from taskqueue. Invokes reader lock. """
    t = None
    with self.lock.gen_rlock():
      t = self.taskqueue[tid]
    return t

  def getClosestTaskIdx(self, pos: np.ndarray) -> int:
    """
    Computes and returns the closest task index (in Euclidean sense) given a 2D
    position. Invokes reader lock. Returns -1 if taskqueue is empty.
    """
    with self.lock.gen_rlock():
      closest_tid = -1
      closest_distance = np.inf

      if not self.taskqueue:
        return closest_tid

      for tid, t in self.taskqueue.items():
        t_loc, t_s = t
        distance = np.linalg.norm(t_loc - pos)
        if distance < closest_distance:
          closest_distance = distance
          closest_tid = tid

      return closest_tid

  def hasTask(self, tid: int) -> bool:
    """
    Checks if a task still exists (unserved) in the taskqueue. Invokes
    reader lock.
    """
    with self.lock.gen_rlock():
      is_exist = (tid in self.taskqueue)
      return is_exist

  def isEmpty(self) -> bool:
    """ Checks if the taskqueue is empty. Invokes reader lock. """
    with self.lock.gen_rlock():
      is_empty = not bool(self.taskqueue)
      return is_empty


global_taskqueue = GlobalTaskQueue()
