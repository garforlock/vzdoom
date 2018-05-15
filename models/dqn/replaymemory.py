import numpy as np
from collections import deque


class ReplayMemory:

    def __init__(self, n_steps, capacity=10000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()

    def sample_batch(self, batch_size):  # creates an iterator that returns random batches
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs + 1) * batch_size <= len(self.buffer):
            yield vals[ofs * batch_size:(ofs + 1) * batch_size]
            ofs += 1

    def run_steps(self, samples):
        while samples > 0:
            entry = next(self.n_steps_iter)  # 10 consecutive steps
            self.buffer.append(entry)  # we put 200 for the current episode
            samples -= 1
        while len(self.buffer) > self.capacity:  # we accumulate no more than the capacity (10000)
            self.buffer.popleft()
