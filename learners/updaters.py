import numpy as np
from typing import Tuple, Optional, List
from abc import abstractmethod, ABC
from dataclasses import dataclass, field

from .representation import TrainTuple


@dataclass
class UpdateStep:
    feature: Optional[List[np.ndarray]]
    value: float
    reward: float


class StepUpdater(ABC):
    @abstractmethod
    def get_updates(self, steps: List[UpdateStep]) -> TrainTuple:
        pass

    @property
    @abstractmethod
    def length(self) -> int:
        pass


class Step(StepUpdater):

    def __init__(self, discount: float):
        self._discount = discount

    def get_updates(self, steps: List[UpdateStep]) -> TrainTuple:
        return TrainTuple(
            steps[0].feature,
            steps[1].reward + self._discount * steps[1].value
        )

    @property
    def length(self) -> int:
        return 2

    @property
    def params(self):
        return {'discount': self._discount}


class NStep(StepUpdater):

    def __init__(self, discount: float, steps: int) -> None:
        self._discount = discount
        self._steps = steps
        self._discounted_array = _discount_array(steps, discount)

    def get_updates(self, steps: List[UpdateStep]) -> TrainTuple:
        return TrainTuple(
            steps[0].feature,
            (rewards(steps)[1:self._steps + 1] * self._discounted_array).sum()
            + self._discounted_array[-1]
            * steps[self._steps].value
        )

    @property
    def length(self) -> int:
        return self._steps + 1

    @property
    def params(self):
        return {
            'discount': self._discount,
            'steps': self._steps
        }


class TDLambda(StepUpdater):

    def __init__(self, discount: float, cutoff_steps: int,
                 lmbda: float) -> None:
        self._discount = discount
        self._cutoff_steps = cutoff_steps
        self._discounted_array = _discount_array(cutoff_steps + 1, discount)
        self._lambda_weights = _discount_array(cutoff_steps, lmbda) * (1 - lmbda)
        self._lambda_weights[-1] += 1 - self._lambda_weights.sum()

    def get_updates(self, steps: List[UpdateStep]) -> TrainTuple:
        discounted_rewards = (rewards(steps[1:self._cutoff_steps + 1])
                              * self._discounted_array[:-1])
        discounted_values = (values(steps[1:self._cutoff_steps + 1])
                             * self._discounted_array[1:])

        calculated_rewards = []
        for i in range(self._cutoff_steps):
            calculated_rewards.append(
                discounted_rewards[:(i + 1)].sum()
                + discounted_values[i]
            )
        calculated_rewards = np.array(calculated_rewards)
        reward = (calculated_rewards * self._lambda_weights).sum()

        return TrainTuple(steps[0].feature, reward)

    @property
    def length(self) -> int:
        return self._cutoff_steps + 1

    @property
    def params(self) -> dict:
        return {
            'discount': self._discount,
            'cutoff_steps': self._cutoff_steps
        }


def rewards(x: List[UpdateStep]) -> np.ndarray:
    return np.array([i.reward for i in x])


def values(x: List[UpdateStep]) -> np.ndarray:
    return np.array([i.value for i in x])


def _discount_array(length: int, discount: float):
    return discount ** np.arange(length)


class ValueCalculator:
    @abstractmethod
    def value(self, action_values: np.ndarray,
              action_took: int,
              action_probs: np.ndarray) -> float:
        pass

    @abstractmethod
    def double(self, action_values: np.ndarray,
               action_took: int,
               action_probs: np.ndarray,
               reference_values: np.ndarray) -> float:
        pass


class MaximumValue(ValueCalculator):

    def value(self, action_values: np.ndarray,
              action_took: int,
              action_probs: np.ndarray) -> float:
        return max(action_values)

    def double(self, action_values: np.ndarray,
               action_took: int,
               action_probs: np.ndarray,
               reference_values: np.ndarray) -> float:
        idx = np.argmax(action_values)
        return reference_values[idx]

    @property
    def params(self) -> dict:
        return {'type': 'maximum_value'}


class ActionTookValue(ValueCalculator):
    def value(self, action_values: np.ndarray,
              action_took: int,
              action_probs: np.ndarray) -> float:
        return action_values[action_took]

    def double(self, action_values: np.ndarray,
               action_took: int,
               action_probs: np.ndarray,
               reference_values: np.ndarray) -> float:
        return reference_values[action_took]

    @property
    def params(self) -> dict:
        return {'type': 'action_took_value'}


class ExpectedValue(ValueCalculator):
    def value(self, action_values: np.ndarray,
              action_took: int,
              action_probs: np.ndarray) -> float:
        return (action_values * action_probs).sum()

    def double(self, action_values: np.ndarray,
               action_took: int,
               action_probs: np.ndarray,
               reference_values: np.ndarray) -> float:
        return (reference_values * action_probs).sum()

    @property
    def params(self) -> dict:
        return {'type': 'expected_value'}
