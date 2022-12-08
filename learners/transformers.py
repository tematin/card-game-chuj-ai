from abc import ABC, abstractmethod
from typing import Tuple, List, Union
import numpy as np

from learners.feature_generators import FeatureGenerator
from learners.representation import concatenate_feature_list, index_array_list
from baselines.baselines import Agent
from game.environment import Environment


class Transformer(ABC):
    @abstractmethod
    def transform(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        pass

    @abstractmethod
    def inverse_transform(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        pass


class MultiDimensionalScaler(Transformer):
    def __init__(self, collapse_axis):
        self.collapse_axis = collapse_axis

    def fit(self, x: np.ndarray) -> None:
        self._mean = x.mean(self.collapse_axis, keepdims=True)
        self._std = x.std(self.collapse_axis, keepdims=True)

    def transform(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return (x - self._mean) / self._std

    def inverse_transform(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return (x * self._std) + self._mean

    @property
    def params(self) -> dict:
        return {
            'mean': self._mean,
            'std': self._std
        }


class SimpleScaler(Transformer):
    def fit(self, x: np.ndarray) -> None:
        self._mean = x.mean()
        self._std = x.std()

    def transform(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return (x - self._mean) / self._std

    def inverse_transform(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return (x * self._std) + self._mean

    @property
    def params(self) -> dict:
        return {
            'mean': self._mean,
            'std': self._std
        }


def generate_dataset(
        env: Environment,
        agent: Agent,
        episodes: int,
        feature_generator: FeatureGenerator,
        exclude_actions: bool = False
) -> Tuple[List[np.ndarray], np.ndarray]:
    features = []
    final_rewards = []

    for _ in range(episodes):
        observation = env.reset()

        rewards = []

        done = False
        while not done:
            card = agent.play(observation)

            if exclude_actions:
                feature = feature_generator.state(observation)
            else:
                action_feature, cards = feature_generator.state_action(observation)
                idx = cards.index(card)
                feature = index_array_list(action_feature, idx)

            observation, reward, done = env.step(card)

            features.append(feature)
            rewards.append(reward)

        rewards = np.cumsum(np.array(rewards)[::-1])[::-1]
        final_rewards.append(rewards.reshape(-1, 1))

    return concatenate_feature_list(features), np.concatenate(final_rewards, axis=0)
