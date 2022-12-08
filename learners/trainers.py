from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from copy import deepcopy
from typing import List, Optional, Union
from abc import abstractmethod, ABC

from learners.transformers import Transformer
from learners.representation import index_array_list
from learners.updaters import StepUpdater, UpdateStep, ValueCalculator
from learners.feature_generators import FeatureGenerator
from learners.explorers import Explorer
from learners.approximators import Approximator
from learners.memory import Memory, MemoryStep, EmptyMemory, PassthroughMemory
from baselines.baselines import Agent
from game.utils import Card


class Trainer(ABC):
    @abstractmethod
    def reset(self, reward: float) -> None:
        pass

    @abstractmethod
    def step(self, observation: dict, reward=None) -> Card:
        pass

    def save(self, directory: Path) -> None:
        pass

    def debug(self, observation: dict) -> dict:
        return {}


def fill_memory(memory: Memory, reward: float):
    for i in range(memory.yield_length - 1):
        memory.set(
            MemoryStep(
                features=[],
                action_took=None,
                reward=reward if i == 0 else 0
            ),
            skip=True
        )


def fill_update_steps(update_steps: Memory, reward: float):
    for i in range(update_steps.yield_length - 1):
        update_steps.set(
            UpdateStep(
                feature=None,
                value=0,
                reward=reward if i == 0 else 0
            ),
            skip=True
        )


class SimpleTrainer(Trainer, Agent):

    def __init__(self, q: Union[Approximator, List[Approximator]],
                 updater: StepUpdater,
                 value_calculator: ValueCalculator,
                 feature_generator: FeatureGenerator,
                 explorer: Explorer,
                 feature_transformers: List[Transformer],
                 memory: Optional[Memory],
                 ) -> None:

        self.train = True
        self._q = q
        self._updater = updater
        self._memory = memory

        self._feature_generator = feature_generator
        self._explorer = explorer
        self._value_calculator = value_calculator

        self._feature_transformers = feature_transformers

    def reset(self, reward: float) -> None:
        fill_memory(self._memory, reward)
        self._memory.reset_episode()

        self._explorer.decay()
        self._q.decay()

    def step(self, observation, reward=None) -> Card:
        features, actions = self._feature_generator.state_action(observation)
        features = [t.transform(x) for t, x in zip(self._feature_transformers, features)]

        q_vals = self._q.get(features)

        if not self.train:
            idx = np.argmax(q_vals)

        else:
            probs = self._explorer.get_probabilities(q_vals)
            idx = np.random.choice(np.arange(len(probs)), p=probs)

            self._memory.set(
                MemoryStep(
                    features=features,
                    action_took=idx,
                    reward=reward
                ),
            )

        if self.train:
            self._train_from_memory()

        return actions[idx]

    def _train_from_memory(self) -> None:
        memory_steps_list = self._memory.get()

        for memory_list in memory_steps_list:
            update_steps = []

            for memory in memory_list:
                if memory.features:
                    q_vals = self._q.get(memory.features)

                    step = UpdateStep(
                        feature=index_array_list(memory.features, memory.action_took),
                        value=self._value_calculator.value(
                            action_values=q_vals,
                            action_took=memory.action_took,
                            action_probs=self._explorer.get_probabilities(q_vals),
                        ),
                        reward=memory.reward,
                    )
                else:
                    step = UpdateStep(
                        feature=None,
                        value=0,
                        reward=memory.reward
                    )

                update_steps.append(step)

            data = self._updater.get_updates(update_steps)

            self._q.update(data.features, np.array(data.target).reshape(1, 1))

    @property
    def params(self) -> dict:
        return {
            'q': self._q,
            'updater': self._updater,
            'value_calculator': self._value_calculator,
            'memories': self._memory,
            'feature_generator': self._feature_generator,
            'explorer': self._explorer,
            'feature_transformers': self._feature_transformers,
        }

    def save(self, directory: Path) -> None:
        self._q.save(directory / 'q')

    def load(self, directory: Path) -> None:
        self._q.load(directory / 'q')

    def play(self, observation: dict) -> Card:
        self.train = False
        card = self.step(observation)
        self.train = True
        return card


class DoubleTrainer(Trainer, Agent):

    def __init__(self, q: Union[Approximator, List[Approximator]],
                 updater: StepUpdater,
                 value_calculator: ValueCalculator,
                 feature_generator: FeatureGenerator,
                 explorer: Explorer,
                 feature_transformers: List[Transformer],
                 memory: Memory,
                 ) -> None:

        self.train = True
        self._q = [q, deepcopy(q)]
        self._updater = updater
        self._memories = [memory, deepcopy(memory)]

        self._feature_generator = feature_generator
        self._explorer = explorer
        self._value_calculator = value_calculator

        self._feature_transformers = feature_transformers

    def reset(self, reward: float) -> None:
        for i in range(2):
            fill_memory(self._memories[i], reward)
            self._memories[i].reset_episode()
            self._q[i].decay()
        self._explorer.decay()

    def step(self, observation: dict, reward=None) -> Card:
        features, actions = self._feature_generator.state_action(observation)
        features = [t.transform(x) for t, x in zip(self._feature_transformers, features)]

        q1_vals = self._q[0].get(features)
        q2_vals = self._q[1].get(features)
        q_vals = (q1_vals + q2_vals) / 2

        if not self.train:
            idx = np.argmax(q_vals)

        else:
            probs = self._explorer.get_probabilities(q_vals)

            idx = np.random.choice(np.arange(len(probs)), p=probs)

            rand_bool = bool(np.random.randint(2))
            memory_step = MemoryStep(
                features=features,
                action_took=idx,
                reward=reward
            )

            self._memories[0].set(memory_step, skip=rand_bool)
            self._memories[1].set(memory_step, skip=not rand_bool)

        if self.train:
            for i in range(2):
                self._train_from_memory(i)

        return actions[idx]

    def debug(self, observation: dict) -> dict:
        features, actions = self._feature_generator.state_action(observation)
        q1_vals = self._q[0].get(features)
        q2_vals = self._q[1].get(features)
        q_vals = (q1_vals + q2_vals) / 2

        return {
            'actions': actions,
            'q1_vals': q1_vals,
            'q2_vals': q2_vals,
            'q_avg': q_vals,
        }

    def _train_from_memory(self, segment: int) -> None:
        memory_steps_list = self._memories[segment].get()

        for memory_list in memory_steps_list:
            update_steps = []

            for memory in memory_list:
                if memory.features:
                    q_vals = self._q[segment].get(memory.features)
                    other_q_vals = self._q[1 - segment].get(memory.features)

                    step = UpdateStep(
                        feature=index_array_list(memory.features, memory.action_took),
                        value=self._value_calculator.double(
                            action_values=q_vals,
                            action_took=memory.action_took,
                            action_probs=self._explorer.get_probabilities(q_vals),
                            reference_values=other_q_vals
                        ),
                        reward=memory.reward,
                    )
                else:
                    step = UpdateStep(
                        feature=None,
                        value=0,
                        reward=memory.reward
                    )

                update_steps.append(step)

            data = self._updater.get_updates(update_steps)

            self._q[segment].update(data.features, np.array(data.target).reshape(1, 1))

    @property
    def params(self) -> dict:
        return {
            'q': self._q[0],
            'updater': self._updater,
            'memories': self._memories[0],
            'feature_generator': self._feature_generator,
            'explorer': self._explorer,
            'value_calculator': self._value_calculator,
            'feature_transformers': self._feature_transformers,
        }

    def save(self, directory: Path) -> None:
        self._q[0].save(directory / 'q1')
        self._q[1].save(directory / 'q2')

    def load(self, directory: Path) -> None:
        self._q[0].load(directory / 'q1')
        self._q[1].load(directory / 'q2')

    def play(self, observation: dict) -> Card:
        self.train = False
        card = self.step(observation)
        self.train = True
        return card
