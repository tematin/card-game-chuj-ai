import numpy as np
from abc import abstractmethod, ABC
from encoders import concatenate_embeddings


class RegularFitter:
    def get_data(self, X, y):
        return X, y


class GroupedFitter:
    def __init__(self, episodes_in_batch=1):
        self.episodes_in_batch = episodes_in_batch
        self.X = []
        self.y = []

    def get_data(self, X, y):
        self.X.append(X)
        self.y.append(y)
        if len(self.X) >= self.episodes_in_batch:
            X = concatenate_embeddings(self.X)
            y = np.concatenate(self.y).reshape(-1, 1)
            self.X = []
            self.y = []
            return X, y
        else:
            return None


class ReplayFitter:
    def __init__(self, replay_memory, fitter, replay_size):
        self.replay_memory = replay_memory
        self.fitter = fitter
        self.replay_size = replay_size

    def get_data(self, X ,y):
        data = self.fitter.get_data(X, y)
        if data is None:
            return None
        X, y = data
        replay_X, replay_y = self.replay_memory.add_and_extract(X, y, self.replay_size)
        return concatenate_embeddings([X, replay_X]), np.vstack([y, replay_y])


class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.replay_memory_X = None
        self.replay_memory_y = None

    def add_to_replay(self, X, y):
        if self.replay_memory_X is None:
            self.replay_memory_X = X
            self.replay_memory_y = y
        else:
            self.replay_memory_X.append(X)
            self.replay_memory_y = np.vstack([self.replay_memory_y, y])
        self._truncate_memory()

    def _truncate_memory(self):
        self.replay_memory_X = self.replay_memory_X[-self.size:]
        self.replay_memory_y = self.replay_memory_y[-self.size:]

    def get_from_replay(self, size):
        idx = np.random.choice(np.arange(self.replay_memory_X.data_count),
                               size=size)
        return self.replay_memory_X[idx], self.replay_memory_y[idx]

    def add_and_extract(self, X, y, size):
        if self.replay_memory_X is None:
            self.add_to_replay(X, y)
            return X, y
        else:
            replay_X, replay_y = self.get_from_replay(size)
            self.add_to_replay(X, y)
            return concatenate_embeddings([X, replay_X]), np.vstack([y, replay_y])

