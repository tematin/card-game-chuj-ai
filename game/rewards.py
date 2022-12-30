from abc import ABC, abstractmethod
from typing import List

from game.utils import GamePhase



class Reward(ABC):
    def reset(self, observation):
        pass

    @abstractmethod
    def step(self, observation):
        pass


class OrdinaryReward(Reward):
    def __init__(self, alpha):
        self.alpha = alpha

    def reset(self, observation):
        self._total_took = 0
        self._total_given = 0

    def step(self, observation):
        if observation['phase'] != GamePhase.PLAY:
            return 0

        score = observation['score']

        took = score[0]
        given = sum(score) - took

        delta_took = took - self._total_took
        delta_given = given - self._total_given

        self._total_took = took
        self._total_given = given

        return self.alpha * delta_given - (1 - self.alpha) * delta_took


class EndReward(Reward):
    def __init__(self, alpha):
        self.alpha = alpha

    def step(self, observation):
        if observation['phase'] != GamePhase.PLAY:
            return 0

        if len(observation['hand']) > 0:
            return 0

        score = observation['score']

        took = score[0]
        given = sum(score) - took

        return self.alpha * given - (1 - self.alpha) * took


class DurchDeclarationPenalty(Reward):
    def __init__(self, penalty: float) -> None:
        self._penalty = penalty
        self._could_declare = False

    def reset(self, observation):
        self._could_declare = False

    def step(self, observation):
        if observation['phase'] == GamePhase.DURCH:
            self._could_declare = True
            return 0

        if self._could_declare:
            self._could_declare = False
            if observation['declared_durch'][0]:
                return self._penalty
            else:
                return 0
        else:
            return 0


class RewardsCombiner(Reward):
    def __init__(self, rewards: List[Reward]) -> None:
        self._rewards = rewards

    def reset(self, observation):
        for reward in self._rewards:
            reward.reset(observation)

    def step(self, observation):
        total_reward = 0
        for reward in self._rewards:
            total_reward += reward.step(observation)
        return total_reward
