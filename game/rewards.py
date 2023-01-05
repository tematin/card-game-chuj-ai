from abc import ABC, abstractmethod
from typing import List

import numpy as np

from game.constants import DURCH_SCORE, CARDS_PER_PLAYER
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

        if sum(observation['declared_durch']) != 0:
            return 0

        took, given = self._resolve_score(observation)

        delta_took = took - self._total_took
        delta_given = given - self._total_given

        self._total_took = took
        self._total_given = given

        return self.alpha * delta_given - (1 - self.alpha) * delta_took

    def _resolve_score(self, observation):
        score = observation['score']
        took, gave = score[0], sum(score) - score[0]

        if len(observation['hand']) > 0:
            return took, gave
        elif observation['eligible_durch'][0]:
            return DURCH_SCORE, 0
        elif sum(observation['eligible_durch']) > 0:
            return 0, DURCH_SCORE
        else:
            return took, gave


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


class DeclaredDurchRewards(Reward):
    def __init__(self, success_reward: float, failure_reward: float,
                 rival_success_reward: float, rival_failure_reward: float) -> None:
        self._success_reward = success_reward / CARDS_PER_PLAYER
        self._failure_reward = failure_reward
        self._rival_success_reward = rival_success_reward / CARDS_PER_PLAYER
        self._rival_failure_reward = rival_failure_reward

    def reset(self, observation):
        self._declared = False
        self._durch_failed = False
        self._player_declared = None
        self._per_step_reward = 0

    def step(self, observation):
        if self._durch_failed:
            return 0

        if (not self._declared
                and observation['phase'] == GamePhase.PLAY
                and sum(observation['declared_durch']) > 0):
            self._declared = True
            self._player_declared = np.argmax(observation['declared_durch'])

        if not self._declared:
            return 0

        if not observation['eligible_durch'][self._player_declared]:
            self._durch_failed = True
            ret = -self._per_step_reward

            if self._player_declared == 0:
                ret += self._failure_reward
            else:
                ret += self._rival_failure_reward
            return ret

        if self._player_declared == 0:
            self._per_step_reward += self._success_reward
            return self._success_reward
        else:
            self._per_step_reward += self._rival_success_reward
            return self._rival_success_reward


class DurchReward(Reward):
    def __init__(self, reward: float, rival_reward: float) -> None:
        self._reward = reward
        self._rival_reward = rival_reward

    def step(self, observation):
        if len(observation['hand']) > 0:
            return 0
        elif observation['eligible_durch'][0]:
            return self._reward
        elif sum(observation['eligible_durch']) > 0:
            return self._rival_reward
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
