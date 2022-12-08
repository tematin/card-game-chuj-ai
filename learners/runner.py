import json
from typing import Optional, Any
from pathlib import Path
import shutil

from baselines.baselines import Agent
from game.environment import Tester, Environment
from learners.trainers import Trainer


class TrainRun:
    _average_length = 100

    def __init__(self, agent: Trainer,
                 tester: Tester,
                 environment: Environment,
                 eval_freq: int = 1000,
                 checkpoint_dir: Optional[Path] = None,
                 benchmark: Optional[Agent] = None,
                 tracker: Any = None) -> None:
        self._env = environment
        self._agent = agent
        self._tester = tester
        self._eval_freq = eval_freq
        self._benchmark = benchmark
        self._tracker = tracker

        self._checkpoint_dir = checkpoint_dir
        if self._checkpoint_dir is not None:
            self._checkpoint_dir.mkdir(exist_ok=True)
            shutil.copy(Path('C:/Python/Repos/chuj/main.py'), self._checkpoint_dir)

        self._rewards = []
        self._score = []
        self._score_idx = []
        self._trained_episodes = 0
        self._recent_average_reward = 0

    def _train_episode(self) -> float:
        total_rewards = 0

        self._agent.train = True
        observation = self._env.reset()

        done = False
        reward = 0

        while not done:
            card = self._agent.step(observation, reward=reward)
            observation, reward, done = self._env.step(card)
            total_rewards += reward

        self._agent.reset(reward)

        self._trained_episodes += 1

        return total_rewards

    @property
    def trained_episodes(self) -> int:
        return self._trained_episodes

    def train(self, episode_count: int) -> None:
        for i in range(episode_count):
            reward = self._train_episode()

            if self._tracker is not None:
                self._tracker.register()

            self._recent_average_reward += reward
            self._rewards.append(reward)
            if len(self._rewards) > self._average_length:
                self._recent_average_reward -= self._rewards[-self._average_length]

            if i % 10 == 0:
                print(f"\r{str(i).rjust(len(str(episode_count)))} / {episode_count}. "
                      f"Reward: {self._recent_average_reward / self._average_length}",
                      end="")

            if (self._trained_episodes % self._eval_freq == 0
                    and self._benchmark is not None):
                print(f"\nAfter episodes: {self._trained_episodes}")
                self._agent.train = False

                score = self._tester.evaluate(self._agent,
                                              adversary=self._benchmark,
                                              verbose=1)
                self._score_idx.append(self._trained_episodes)
                self._score.append(score)

                if self._checkpoint_dir is not None:
                    self._save_model()
                    self._save_agent_json()

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
            'rewards': self._rewards
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
