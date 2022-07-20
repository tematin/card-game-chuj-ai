import numpy as np


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = float(epsilon)

    def get_probabilities(self, values):
        p = np.full_like(values, self.epsilon / len(values))
        p[np.argmax(values)] += 1 - self.epsilon
        return p


class Softmax:
    def __init__(self, temperature):
        self.t = temperature

    def get_probabilities(self, values):
        p = np.exp((values - values.max()) / self.t)
        p /= p.sum()
        return p


class ExplorationCombiner:
    def __init__(self, explorations, probabilities):
        self.explorations = explorations
        self.probabilities = np.array(probabilities).reshape(-1, 1)

    def get_probabilities(self, values):
        probs = []
        for exp in self.explorations:
            probs.append(exp.get_probabilities(values))
        p = (self.probabilities * np.stack(probs)).sum(0).flatten()
        return p / p.sum()
