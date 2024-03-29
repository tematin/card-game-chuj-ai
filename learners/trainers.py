from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from copy import deepcopy
from typing import List, Optional, Union, Any, Tuple
from abc import abstractmethod, ABC

from learners.approximators import Approximator
from learners.transformers import Transformer
from learners.representation import index_observation
from learners.updaters import StepUpdater, UpdateStep, ValueCalculator
from learners.feature_generators import FeatureGenerator
from learners.explorers import Explorer
from learners.memory import Memory, MemoryStep, EmptyMemory, PassthroughMemory
from baselines.baselines import Agent
from game.utils import Card, GamePhase
from debug.timer import timer


class Trainer(ABC):
    @abstractmethod
    def reset(self, reward: float, run_id: int) -> None:
        pass

    @abstractmethod
    def step(self, observations: List[dict], action_list: List[List[Any]],
             reward: List[Optional[float]], idx: List[int]) -> List[Any]:
        pass

    def save(self, directory: Path) -> None:
        pass

    def load(self, directory: Path) -> None:
        pass

    @abstractmethod
    def agent(self) -> Agent:
        pass


class ValueAgent(Agent, ABC):
    @abstractmethod
    def values(self, observation: dict, actions: List[Any]) -> np.ndarray:
        pass

    def parallel_values(self, observations: List[dict],
                        action_list: List[List[Any]]) -> List[np.ndarray]:
        return [self.values(o, a) for o, a in zip(observations, action_list)]

    def play(self, observation: dict, actions: List[Any]) -> Any:
        vals = self.values(observation, actions)

        idx = np.argmax(vals)

        return actions[idx]


def fill_memory(memory: Memory, reward: float):
    for i in range(memory.yield_length - 1):
        memory.set(
            MemoryStep(
                features=None,
                action_took=None,
                reward=reward if i == 0 else 0
            ),
            skip=True
        )


def fill_update_steps(update_steps: Memory, reward: float):
    for i in range(update_steps.yield_length - 1):
        update_steps.set(
            UpdateStep(
                features=None,
                value=0,
                reward=reward if i == 0 else 0
            ),
            skip=True
        )


class TrainedDoubleQ(ValueAgent):
    def __init__(self, approximators: Tuple[Approximator, Approximator],
                 feature_generator: FeatureGenerator) -> None:
        self._q = approximators
        self._feature_generator = feature_generator

    def load(self, directory: Path) -> None:
        self._q[0].load(directory / 'q1')
        self._q[1].load(directory / 'q2')
        self._feature_generator.load(directory / 'feature_generator')

    def debug(self, observation: dict, actions: List[Any]) -> dict:
        features = self._feature_generator.state_action(observation, actions)
        q1_vals = self._q[0].get(features)
        q2_vals = self._q[1].get(features)
        q_vals = (q1_vals + q2_vals) / 2

        return {
            'actions': actions,
            'q1_vals': q1_vals,
            'q2_vals': q2_vals,
            'q_avg': q_vals,
        }

    def parallel_play(self, observations: List[dict],
                      action_list: List[List[Any]]) -> List[Any]:
        values = self.parallel_values(observations, action_list)
        actions_to_play = []
        for run_id in range(len(observations)):
            idx = np.argmax(values[run_id])

            actions_to_play.append(action_list[run_id][idx])

        return actions_to_play

    def values(self, observation: dict, actions: List[Any]) -> np.ndarray:
        features = self._feature_generator.state_action(observation, actions)

        q1_vals = self._q[0].get(features)
        q2_vals = self._q[1].get(features)
        return (q1_vals + q2_vals) / 2

    def parallel_values(self, observations: List[dict],
                        action_list: List[List[Any]]) -> List[np.ndarray]:
        assert len(observations) == len(action_list)
        features = [self._feature_generator.state_action(observation, actions)
                    for observation, actions in zip(observations, action_list)]

        q1_vals = self._q[0].get_from_list(features)
        q2_vals = self._q[1].get_from_list(features)

        q_val_list = []
        for run_id in range(len(observations)):
            q_vals = (q1_vals[run_id] + q2_vals[run_id]) / 2
            q_val_list.append(q_vals)

        return q_val_list


class DoubleTrainer(Trainer, TrainedDoubleQ):

    def __init__(self, q: Approximator,
                 updater: StepUpdater,
                 value_calculator: ValueCalculator,
                 feature_generator: FeatureGenerator,
                 explorer: Explorer,
                 memory: Memory,
                 assign_rule: str,
                 run_count: int) -> None:

        super().__init__(approximators=(q, deepcopy(q)),
                         feature_generator=feature_generator)

        self._updater = updater
        self._memories = [[deepcopy(memory) for _ in range(2)]
                          for _ in range(run_count)]
        self._run_count = run_count
        self._explorer = explorer
        self._value_calculator = value_calculator
        self._assign_rule = assign_rule
        self._episodes = 0

    def reset(self, reward: float, run_id: int) -> None:
        for i in range(2):
            fill_memory(self._memories[run_id][i], reward)
            self._memories[run_id][i].reset_episode()
            self._q[i].decay()
            if run_id == 0:
                self._train_from_memory(i)

        self._explorer.decay()
        self._episodes += 1

    @timer.trace("Trainer Step")
    def step(self, observations: List[dict], action_list: List[List[Any]],
             reward: List[Optional[float]], run_ids: List[int]) -> List[Any]:
        features = [self._feature_generator.state_action(observation, actions)
                    for observation, actions in zip(observations, action_list)]

        q1_vals = self._q[0].get_from_list(features)
        q2_vals = self._q[1].get_from_list(features)

        actions_to_play = []
        for i, run_id in enumerate(run_ids):
            q_vals = (q1_vals[i] + q2_vals[i]) / 2

            probs = self._explorer.get_probabilities(q_vals)

            idx = np.random.choice(np.arange(len(probs)), p=probs)

            memory_step = MemoryStep(
                features=features[i],
                action_took=idx,
                reward=reward[i]
            )

            self._store_in_memory(memory_step, run_id)
            actions_to_play.append(action_list[i][idx])

        return actions_to_play

    def _store_in_memory(self, memory_step: MemoryStep, run_id: int) -> None:
        if self._assign_rule == 'both':
            skip = [False, False]
        elif self._assign_rule == 'random':
            rand_bool = bool(np.random.randint(2))
            skip = [rand_bool, not rand_bool]
        elif self._assign_rule == 'episode_random':
            rand_bool = bool(self._episodes % 2)
            skip = [rand_bool, not rand_bool]
        else:
            raise ValueError("Wrong assign rule")

        self._memories[run_id][0].set(memory_step, skip=skip[0])
        self._memories[run_id][1].set(memory_step, skip=skip[1])

    @timer.trace("Memory Train")
    def _train_from_memory(self, segment: int) -> None:
        memory_steps_list = []
        for run_memories in self._memories:
            memory_steps_list.extend(run_memories[segment].get())

        features = []
        for memory_list in memory_steps_list:
            for memory in memory_list:
                if memory.features:
                    features.append(memory.features)

        if not features:
            return

        main_q_vals = self._q[segment].get_from_list(features, update_mode=True)
        other_q_vals = self._q[1 - segment].get_from_list(features, update_mode=True)

        for memory_list in memory_steps_list:
            update_steps = []

            for memory in memory_list:

                if memory.features:
                    main_q_val = main_q_vals.pop(0)
                    other_q_val = other_q_vals.pop(0)

                    step = UpdateStep(
                        features=index_observation(memory.features,
                                                   memory.action_took),
                        value=self._value_calculator.double(
                            action_values=main_q_val,
                            action_took=memory.action_took,
                            action_probs=self._explorer.get_probabilities(main_q_val),
                            reference_values=other_q_val
                        ),
                        reward=memory.reward
                    )

                else:
                    step = UpdateStep(
                        features=None,
                        value=0,
                        reward=memory.reward
                    )
                update_steps.append(step)

            data = self._updater.get_updates(update_steps)

            self._q[segment].update(*data)

    @property
    def params(self) -> dict:
        return {
            'q1': self._q[0],
            'updater': self._updater,
            'memories': self._memories[0],
            'feature_generator': self._feature_generator,
            'explorer': self._explorer,
            'value_calculator': self._value_calculator,
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(exist_ok=True)
        self._q[0].save(directory / 'q1')
        self._q[1].save(directory / 'q2')
        self._feature_generator.save(directory / 'feature_generator')

    def agent(self) -> Agent:
        return TrainedDoubleQ(
            approximators=deepcopy(self._q),
            feature_generator=deepcopy(self._feature_generator)
        )
