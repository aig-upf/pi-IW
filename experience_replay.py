from utils import random_index
from collections import deque


class ExperienceReplay:
    def __init__(self, capacity=None):
        self.capacity = capacity

    def __len__(self):
        try:
            return len(next(iter(self._data.values())))  # length of any value of the dict (i.e. any of the lists)
        except AttributeError:
            return 0

    def items(self):
        try:
            return self._data.items()
        except AttributeError:
            raise ValueError("Trying to iterate over an empty dataset")

    def append(self, example):
        try:
            for k, v in example.items():
                self._data[k].append(v)
        except AttributeError:
            self._data = {k: deque([v], maxlen=self.capacity) for k,v in example.items()}

    def extend(self, examples):
        try:
            for k, v in examples.items():
                self._data[k].extend(v)
        except AttributeError:
            self._data = {k: deque(v, maxlen=self.capacity) for k,v in examples.items()}

    def sample(self, size):
        idx = random_index(len(self), size, replace=False)
        return {k: [col[i] for i in idx] for k,col in self._data.items()}
