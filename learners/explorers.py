from typing import List

import numpy as np
from abc import abstractmethod


class Explorer:

    @abstractmethod
    def get_probabilities(self, values: np.ndarray) -> np.ndarray:
        pass

    def decay(self) -> None:
        pass


class Greedy(Explorer):

    def get_probabilities(self, values: np.ndarray) -> np.ndarray:
        p = np.zeros_like(values, dtype=float)
        p[np.argmax(values)] = 1
        return p


class Random(Explorer):

    def get_probabilities(self, values: np.ndarray) -> np.ndarray:
        return np.ones_like(values, dtype=float) / len(values)


class EpsilonGreedy(Explorer):

    def __init__(self, epsilon: float, decay: float = 1, undecayable: float = 0):
        self._original_epsilon = float(epsilon)
        self._epsilon = float(epsilon) - undecayable
        self._undecayable = undecayable
        self._decay = decay

    def get_probabilities(self, values: np.ndarray) -> np.ndarray:
        epsilon = self._epsilon + self._undecayable

        p = np.full_like(values, epsilon / len(values), dtype=float)
        p[np.argmax(values)] += 1 - epsilon

        return p

    def decay(self) -> None:
        self._epsilon *= self._decay

    @property
    def params(self) -> dict:
        return {
            'epsilon': self._original_epsilon,
            'decay': self._decay,
            'undecayable': self._undecayable
        }


class Softmax(Explorer):

    def __init__(self, temperature: float, decay: float = 1,
                 undecayable: float = 0) -> None:
        self._original_t = temperature
        self._t = temperature - undecayable
        self._decay = decay
        self._undecayable = undecayable

    def get_probabilities(self, values: np.ndarray) -> np.ndarray:
        p = np.exp((values - values.max()) / (self._t + self._undecayable))
        p /= p.sum()

        return p

    def decay(self) -> None:
        self._t *= self._decay

    @property
    def params(self) -> dict:
        return {
            'temperature': self._original_t,
            'undecayable': self._undecayable,
            'decay': self._decay
        }


class ExplorationCombiner(Explorer):

    def __init__(self, explorations: List[Explorer], probabilities: List[float]) -> None:
        self._explorations = explorations
        self._probabilities = np.array(probabilities).reshape(-1, 1)

    def get_probabilities(self, values: np.ndarray) -> np.ndarray:
        probs = []
        for exp in self._explorations:
            probs.append(exp.get_probabilities(values))
        p = (self._probabilities * np.stack(probs)).sum(0).flatten()
        return p / p.sum()

    def decay(self) -> None:
        for explorer in self._explorations:
            explorer.decay()

    @property
    def params(self) -> dict:
        return {
            'explorations': self._explorations,
            'probabilities': self._probabilities
        }


class ExplorationSwitcher(Explorer):

    def __init__(self, explorations: List[Explorer], probabilities: List[float]) -> None:
        self._explorations = explorations
        self._probabilities = np.array(probabilities)
        self._change_explorer()

    def get_probabilities(self, values: np.ndarray) -> np.ndarray:
        return self._explorations[self._idx].get_probabilities(values)

    def decay(self) -> None:
        self._change_explorer()
        for explorer in self._explorations:
            explorer.decay()

    def _change_explorer(self) -> None:
        self._idx = np.random.choice(np.arange(len(self._probabilities)),
                                     p=self._probabilities)

    @property
    def params(self) -> dict:
        return {
            'explorations': self._explorations,
            'probabilities': self._probabilities
        }
