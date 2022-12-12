from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from copy import deepcopy
from typing import List, Optional, Union
from abc import abstractmethod, ABC

from learners.transformers import Transformer
from learners.representation import index_observation
from learners.updaters import StepUpdater, UpdateStep, ValueCalculator
from learners.feature_generators import FeatureGenerator
from learners.explorers import Explorer
from learners.approximators import Approximator, ApproximatorSplitter
from learners.memory import Memory, MemoryStep, EmptyMemory, PassthroughMemory
from baselines.baselines import Agent
from game.utils import Card, Observation


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
                observation=None,
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
                reward=reward if i == 0 else 0,
                phase=None
            ),
            skip=True
        )


class DoubleTrainer(Trainer, Agent):

    def __init__(self, q: ApproximatorSplitter,
                 updater: StepUpdater,
                 value_calculator: ValueCalculator,
                 feature_generator: FeatureGenerator,
                 explorer: Explorer,
                 memory: Memory,
                 ) -> None:

        self.train = True
        self._q = [q, deepcopy(q)]
        self._updater = updater
        self._memories = [memory, deepcopy(memory)]

        self._feature_generator = feature_generator
        self._explorer = explorer
        self._value_calculator = value_calculator

    def reset(self, reward: float) -> None:
        for i in range(2):
            fill_memory(self._memories[i], reward)
            self._memories[i].reset_episode()
            self._q[i].decay()
        self._explorer.decay()

    def step(self, observation: Observation, reward=None) -> Card:
        observation.features = self._feature_generator.state_action(observation)

        q1_vals = self._q[0].get(observation)
        q2_vals = self._q[1].get(observation)
        q_vals = (q1_vals + q2_vals) / 2

        if not self.train:
            idx = np.argmax(q_vals)

        else:
            probs = self._explorer.get_probabilities(q_vals)

            idx = np.random.choice(np.arange(len(probs)), p=probs)

            rand_bool = bool(np.random.randint(2))

            memory_step = MemoryStep(
                observation=observation,
                action_took=idx,
                reward=reward
            )

            self._memories[0].set(memory_step, skip=rand_bool)
            self._memories[1].set(memory_step, skip=not rand_bool)

        if self.train:
            for i in range(2):
                self._train_from_memory(i)

        return observation.actions[idx]

    def debug(self, observation: Observation) -> dict:
        observation.features = self._feature_generator.state_action(observation)
        q1_vals = self._q[0].get(observation)
        q2_vals = self._q[1].get(observation)
        q_vals = (q1_vals + q2_vals) / 2

        return {
            'actions': observation.actions,
            'q1_vals': q1_vals,
            'q2_vals': q2_vals,
            'q_avg': q_vals,
        }

    def _train_from_memory(self, segment: int) -> None:
        memory_steps_list = self._memories[segment].get()

        for memory_list in memory_steps_list:
            update_steps = []

            for memory in memory_list:

                if memory.observation:
                    q_vals = self._q[segment].get(memory.observation, update_mode=True)
                    other_q_vals = self._q[1 - segment].get(memory.observation, update_mode=True)

                    step = UpdateStep(
                        features=index_observation(memory.observation.features,
                                                   memory.action_took),
                        value=self._value_calculator.double(
                            action_values=q_vals,
                            action_took=memory.action_took,
                            action_probs=self._explorer.get_probabilities(q_vals),
                            reference_values=other_q_vals
                        ),
                        reward=memory.reward,
                        phase=memory.observation.phase
                    )

                else:
                    step = UpdateStep(
                        features=None,
                        value=0,
                        reward=memory.reward,
                        phase=None
                    )

                update_steps.append(step)

            data = self._updater.get_updates(update_steps)

            self._q[segment].update(data)

    @property
    def params(self) -> dict:
        return {
            'q': self._q[0],
            'updater': self._updater,
            'memories': self._memories[0],
            'feature_generator': self._feature_generator,
            'explorer': self._explorer,
            'value_calculator': self._value_calculator,
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


def validate_memory(memory):
    for slot in memory._steps._items:
        for step in slot:
            if step.features is None:
                pass
            else:
                assert step.features.features[0].shape[0] > 0
