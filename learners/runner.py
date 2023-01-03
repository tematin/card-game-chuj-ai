import json
from copy import deepcopy
from typing import Optional, Any, List, Dict
from pathlib import Path
import shutil

import numpy as np

from baselines.baselines import Agent
from game.environment import Tester, Environment
from learners.trainers import Trainer
from debug.timer import timer


class TrainRun:
    _average_length = 100

    def __init__(self, agent: Trainer,
                 testers: Dict[str, Tester],
                 environment: Environment,
                 eval_freq: int = 1000,
                 run_count: int = 1,
                 checkpoint_dir: Optional[Path] = None,
                 tracker: Any = None) -> None:
        self._agent = agent
        self._testers = testers
        self._eval_freq = eval_freq
        self._tracker = tracker
        self._run_count = run_count

        self._env = environment
        self._env.reset()

        self._checkpoint_dir = checkpoint_dir
        if self._checkpoint_dir is not None:
            self._checkpoint_dir.mkdir(exist_ok=True)
            shutil.copy(Path('C:/Python/Repos/card-game-chuj-ai/main.py'), self._checkpoint_dir)

        self._score = []
        self._score_idx = []
        self._trained_episodes = 0

    def _train_episode(self) -> None:
        observations, actions, rewards, done, idx = self._env.reset()
        last_rewards = [0] * len(rewards)

        while not all(done):
            actions_to_play = self._agent.step(observations, actions, rewards, idx)
            observations, actions, rewards, done, idx = self._env.step(actions_to_play)

            for reward, d, i in zip(rewards, done, idx):
                if d:
                    self._agent.reset(reward, i)

            observations = self._remove_done(observations, done)
            actions = self._remove_done(actions, done)
            rewards = self._remove_done(rewards, done)
            idx = self._remove_done(idx, done)

        self._trained_episodes += len(last_rewards)

    def _remove_done(self, x: List, mask: List[bool]) -> List:
        return [item for item, m in zip(x, mask) if not m]

    @property
    def trained_episodes(self) -> int:
        return self._trained_episodes

    @timer.trace("Total")
    def train(self, episode_count: int) -> None:
        for i in range(episode_count):
            self._train_episode()

            if self._tracker is not None:
                self._tracker.register()

            print(f"\r{str(i + 1).rjust(len(str(episode_count)))} / {episode_count}. ",
                  end="")

            if self._trained_episodes % self._eval_freq == 0:
                print(f"\nAfter episodes: {self._trained_episodes}")
                score = self._evaluate()
                self._score_idx.append(self._trained_episodes)
                self._score.append(score)

                if self._checkpoint_dir is not None:
                    self._save_model()
                    self._save_agent_json()

    def _evaluate(self) -> Dict[str, float]:
        ret = {}
        for key, tester in self._testers.items():
            print('')
            print(key)
            ret[key] = tester.evaluate(self._agent, verbose=1)
        return ret

    def _save_agent_json(self) -> None:
        agent_params = extract_params(self._agent)

        with open(self._checkpoint_dir / 'agent.json', 'w') as f:
            json.dump(agent_params, f)

    def _save_model(self) -> None:
        checkpoint_path = (self._checkpoint_dir /
                           f'episode_{self.trained_episodes}')
        checkpoint_path.mkdir(exist_ok=True)

        self._agent.save(checkpoint_path)

        performance = {
            'score': self._score,
            'score_idx': self._score_idx,
        }

        with open(checkpoint_path / 'performance.json', 'w') as f:
            json.dump(performance, f)


def extract_params(obj):
    if hasattr(obj, 'params'):
        extracted_vals = {
            k: extract_params(v)
            for k, v in obj.params.items()
        }
        extracted_vals['name'] = type(obj).__name__
        return extracted_vals
    elif isinstance(obj, dict):
        return {k: extract_params(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [extract_params(x) for x in obj]
    else:
        return str(obj)
