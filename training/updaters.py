import numpy as np
from game.constants import CARDS_PER_PLAYER


class MonteCarlo:
    def get_update_data(self, episode):
        X = episode.embeddings
        y = np.cumsum(episode.rewards).reshape(-1, 1)
        return X, y


class Sarsa:
    def __init__(self, lmbda, expected=True):
        self.lmbda = lmbda
        self.expected = expected
        i = np.arange(CARDS_PER_PLAYER)
        self.value_discounts = (1 - lmbda) * (lmbda ** i)
        self.reward_discounts = 1 - np.cumsum(self.value_discounts)
        self.reward_discounts = np.insert(self.reward_discounts, 0, 1)

    def get_update_data(self, episode):
        y = []
        length = episode.length
        if self.expected:
            vals = episode.expected_values
        else:
            vals = episode.played_values
        for i in range(length):
            reward = np.sum(episode.rewards[i:] * self.reward_discounts[:(length - i)])
            value = np.sum(vals[(i + 1):] * self.value_discounts[:(length - i - 1)])
            y.append(reward + value)
        return episode.embeddings, np.array(y).reshape(-1, 1)


class Q:
    def get_update_data(self, episode):
        y = episode.rewards + np.append(episode.greedy_values[1:], 0)
        return episode.embeddings, y.reshape(-1, 1)
