import itertools
import json
from copy import deepcopy
from typing import Optional, Any, List, Dict
from pathlib import Path
import shutil

import numpy as np

from baselines.baselines import Agent
from game.constants import PLAYERS
from game.environment import Tester, Environment, ThreeWayTester
from game.rewards import Reward
from learners.trainers import Trainer
from debug.timer import timer


class AgentLeague:
    def __init__(self, seed_agents: List[Agent], max_agents: int,
                 game_count: int, agent_names: Optional[List[str]] = None) -> None:
        self._agents = seed_agents
        self._names = agent_names or [f'agent_{i}' for i in range(self.agent_count)]
        self._naming_idx = self.agent_count

        self._tester = ThreeWayTester(game_count=game_count, run_count=20)

        self._max_agents = max_agents

        self._performances = {}
        self._fill_performances()

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    @property
    def names(self) -> List[str]:
        return self._names

    def _fill_performances(self) -> None:
        for idx in itertools.combinations(range(self.agent_count), PLAYERS):
            if idx in self._performances:
                continue

            scores = self._tester.evaluate([self._agents[i] for i in idx])
            self._performances[idx] = scores

    def add_agent(self, agent: Agent, name: Optional[str] = None) -> None:
        self._agents.append(agent)
        self._names.append(name or f'naming_{self._naming_idx}')
        self._naming_idx += 1
        self._fill_performances()

    def _remove_agent(self, agent_idx: int) -> None:
        ret = {}

        for idx, scores in self._performances.items():
            if agent_idx in idx:
                continue

            idx = tuple([x - 1 if x > agent_idx else x for x in idx])
            ret[idx] = scores

        self._performances = ret
        self._agents.pop(agent_idx)
        self._names.pop(agent_idx)

    def prune(self) -> None:
        while self.agent_count > self._max_agents:
            points = self._points()
            idx = np.argmin(points)
            self._remove_agent(idx)

    def _avg_score(self) -> np.ndarray:
        total_scores = np.zeros(self.agent_count)

        for idxs, scores in self._performances.items():
            for agent_idx, score in zip(idxs, scores):
                total_scores[agent_idx] += score

        total_scores /= ((3 / self.agent_count) * len(self._performances))

        return total_scores

    def _points(self) -> np.ndarray:
        total_worst = np.zeros(self.agent_count)
        total_best = np.zeros(self.agent_count)

        for idxs, scores in self._performances.items():
            max_idx = np.argmax(scores)
            total_worst[idxs[max_idx]] += 1

            min_idx = np.argmin(scores)
            total_best[idxs[min_idx]] += 1

        return total_best - 2 * total_worst

    def sample(self):
        idx = np.random.randint(self.agent_count)
        return self._agents[idx]


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


class LeagueTrainRun:
    _average_length = 100

    def __init__(self, agent: Trainer,
                 testers: Dict[str, Tester],
                 league: AgentLeague,
                 reward: Reward,
                 eval_freq: int = 1000,
                 run_count: int = 1,
                 checkpoint_dir: Optional[Path] = None,
                 tracker: Any = None) -> None:
        self._agent = agent
        self._testers = testers
        self._eval_freq = eval_freq
        self._tracker = tracker
        self._run_count = run_count

        self._league = league
        self._reward = reward

        self._checkpoint_dir = checkpoint_dir
        if self._checkpoint_dir is not None:
            self._checkpoint_dir.mkdir(exist_ok=True)
            shutil.copy(Path('C:/Python/Repos/card-game-chuj-ai/main.py'), self._checkpoint_dir)

        self._score = []
        self._score_idx = []
        self._trained_episodes = 0

    def _train_episode(self) -> None:
        env = self._create_environment()

        observations, actions, rewards, done, idx = env.reset()
        last_rewards = [0] * len(rewards)

        while not all(done):
            actions_to_play = self._agent.step(observations, actions, rewards, idx)
            observations, actions, rewards, done, idx = env.step(actions_to_play)

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

    def _create_environment(self):
        rival = self._league.sample()
        return Environment(
            reward=self._reward,
            rival=rival,
            run_count=self._run_count
        )

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

                    self._league.prune()
                    self._league.add_agent(self._agent.agent(), f'ep_{self._trained_episodes}')
                    print(self._league.names)

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

