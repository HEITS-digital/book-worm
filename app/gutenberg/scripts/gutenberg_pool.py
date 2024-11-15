import queue
from .gutenberg import Gutenberg


class GutenbergPool:
    def __init__(self, max_size=5):
        """
        Initialize the object pool.
        :param create_func: a function to create a new object for the pool
        :param max_size: maximum number of objects in the pool
        """
        self.pool = queue.Queue(maxsize=max_size)

        # Pre-fill the pool with objects
        for _ in range(max_size):
            self.pool.put(self.create())

    def acquire(self):
        """
        Acquire an object from the pool.
        Waits if no object is available.
        """
        return self.pool.get(block=True)  # blocks until an item is available

    def release(self, obj):
        """
        Return an object to the pool.
        :param obj: the object to return to the pool
        """
        self.pool.put(obj)

    def create(self):
        return Gutenberg()
