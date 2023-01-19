import itertools
from typing import Any, Optional, List, Tuple

import numpy as np
from copy import deepcopy

from tqdm import tqdm

from baselines.baselines import Agent
from game.constants import PLAYERS
from game.game import PlayPhase, TrackedGameRound, Hand
from game.rewards import Reward
from game.utils import generate_hands, GamePhase, Card


class Environment:
    def __init__(self, reward: Reward, rival: Agent, run_count: int) -> None:
        self._rewards = [deepcopy(reward) for _ in range(run_count)]
        self._rival = rival
        self._run_count = run_count

    def _init_new_games(self, hands: List[List[Hand]]) -> None:
        self._games = []
        for run_id in range(self._run_count):
            round_hands = hands[run_id]

            self._games.append(TrackedGameRound(
                starting_player=np.random.choice(np.arange(PLAYERS)),
                hands=round_hands,
            ))

        self._finished = np.full(self._run_count, False)

    def reset(self, hands: Optional[List[List[Hand]]] = None):
        if hands is None:
            hands = [generate_hands() for _ in range(self._run_count)]

        self._init_new_games(hands)
        self._play_for_rivals()

        observations, action_list = self._observe(self._games, player=0)
        for reward, observation in zip(self._rewards, observations):
            reward.reset(observation)

        rewards = [0] * self._run_count
        done = [False] * self._run_count
        idx = np.arange(self._run_count)

        return observations, action_list, rewards, done, idx

    def step(self, actions_to_play: List[Optional[Any]]):
        active_idx = np.where(~self._finished)[0]
        active_games = [self._games[x] for x in active_idx]
        assert len(active_games) == len(actions_to_play)

        for game, action in zip(active_games, actions_to_play):
            if action is None:
                assert game.end
                continue
            assert game.phasing_player == 0
            game.play(action)

        self._play_for_rivals()

        observations, actions_list = self._observe(active_games, player=0)
        rewards = self._process_reward(observations, active_idx)

        ends = [g.end for g in active_games]

        self._finished[active_idx[ends]] = True

        return observations, actions_list, rewards, ends, active_idx

    def _play_for_rivals(self):
        while (mask := np.array([g.phasing_player != 0 and not g.end
                                 for g in self._games])).sum():
            mask_idx = np.where(mask)[0]
            active_games = [self._games[x] for x in mask_idx]

            observations, action_list = self._observe(active_games)

            actions_to_play = self._rival.parallel_play(observations, action_list)

            for game, action in zip(active_games, actions_to_play):
                game.play(action)

    def _observe(self, games: List[TrackedGameRound],
                 player: Optional[int] = None) -> Tuple[List, List]:
        observations = []
        action_list = []

        for game in games:
            observation, actions = game.observe(player=player)

            observations.append(observation)
            action_list.append(actions)

        return observations, action_list

    def _process_reward(self, observations: List[dict], idx: List[int]):
        rewards = []
        for observation, run_id in zip(observations, idx):
            rewards.append(self._rewards[run_id].step(observation))
        return rewards


class OneThreadEnvironment:
    def __init__(self, reward, rival):
        self._reward = reward
        self._rival = rival

        self._game = None

    def reset(self):
        hands = generate_hands()
        self._game = TrackedGameRound(
            starting_player=np.random.choice(np.arange(PLAYERS)),
            hands=hands,
        )

        self._play_for_rivals()

        observation, actions = self._game.observe(player=0)
        self._reward.reset(observation)

        return observation, actions, 0, False

    def step(self, action):
        self._game.play(action)

        self._play_for_rivals()

        observation, actions = self._game.observe(player=0)

        reward = self._reward.step(observation)

        return observation, actions, reward, self._game.end

    def _play_for_rivals(self):
        while self._game.phasing_player != 0 and not self._game.end:
            observation, actions = self._game.observe()

            action = self._rival.play(observation, actions)
            self._game.play(action)


def finish_game(tracked_game, players):
    while not tracked_game.end:
        observation, actions = tracked_game.observe()
        action = players[tracked_game.phasing_player].play(observation, actions)
        tracked_game.play(action)

    return tracked_game


class RewardTester:
    def __init__(self, reward: Reward, adversary: Agent, run_count: int, episodes: int) -> None:
        self._reward = reward
        self._run_count = run_count
        self._iters = int(episodes / run_count)
        self._episodes = self._iters * run_count
        self._adversary = adversary
        self._hands_list = [generate_hands() for _ in range(episodes)]

    def _remove_done(self, x: List, mask: List[bool]) -> List:
        return [item for item, m in zip(x, mask) if not m]

    def _run(self, env: Environment, agent: Agent, hands: List[List[Hand]]) -> np.ndarray:
        total_rewards = np.zeros(self._run_count)
        observations, actions, _, done, idx = env.reset(hands)

        while not all(done):
            actions_to_play = agent.parallel_play(observations, actions)
            observations, actions, rewards, done, idx = env.step(actions_to_play)
            total_rewards[idx] += rewards

            observations = self._remove_done(observations, done)
            actions = self._remove_done(actions, done)

        return total_rewards

    def evaluate(self, agent: Agent, verbose: int = 0) -> float:
        env = Environment(
            reward=self._reward,
            rival=self._adversary,
            run_count=self._run_count
        )

        total_rewards = []
        iter_item = range(self._iters)
        if verbose > 1:
            iter_item = tqdm(iter_item)

        hands_idx = 0

        for _ in iter_item:
            hands = self._hands_list[hands_idx:(hands_idx + self._run_count)]
            hands_idx += self._run_count
            rewards = self._run(env, agent, deepcopy(hands))
            total_rewards.append(rewards)

        total_rewards = np.concatenate(total_rewards)

        reward_per_episode = np.mean(total_rewards)
        exp_scale = np.std(total_rewards) / np.sqrt(len(total_rewards))

        if verbose:
            print(f"Reward earned: {reward_per_episode:.4f} +- {exp_scale:.4f}")

        return reward_per_episode


class Tester:
    def __init__(self, game_count, adversary, return_ratio=False, seed=123456):
        if seed is not None:
            np.random.seed(seed)
        self._hands_list = [generate_hands() for _ in range(game_count)]
        self._std_scale = 1 / np.sqrt(game_count)
        self._return_ratio = return_ratio
        self._adversary = adversary

    def _simulate(self, player: Agent, verbose: int):
        points = []
        game_value = []
        player_orders = [(player, self._adversary, self._adversary),
                         (self._adversary, player, self._adversary),
                         (self._adversary, self._adversary, player)]

        iter = tqdm(self._hands_list) if verbose > 1 else self._hands_list

        for hand in iter:
            for i in range(3):
                hand_copy = deepcopy(hand)

                game = TrackedGameRound(
                    starting_player=0,
                    hands=hand_copy,
                )
                game = finish_game(game, player_orders[i])
                points_gained = game.points

                points.append(points_gained[i])
                game_value.append(np.sum(points_gained))

        return np.array(points).reshape(-1, 3), np.array(game_value).reshape(-1, 3)

    def evaluate(self, player, verbose=0):
        points, game_value = self._simulate(player, verbose)

        avg_points = points.mean()
        std_points = points.mean(1).std() * self._std_scale

        ratio = sum(points) / sum(game_value)
        avg_ratio = ratio.mean()

        ratios = []
        for _ in range(1000):
            idx = np.random.randint(len(points), size=len(points))
            ratio = points[idx].sum() / game_value[idx].sum()
            ratios.append(ratio)

        std_ratio = np.std(ratios)

        if verbose:
            print(f"Score Achieved: {avg_points:.2f} +- {std_points:.2f}")
            print(f"Ratio Achieved: {avg_ratio:.1%} +- {std_ratio:.1%}")
            print(f"Durch Made: {(points < 0).sum()}")
            print(f"Total Durch Made: {(game_value < 0).sum()}")
            print(f"Average Total Score: {game_value.mean()}")

        if self._return_ratio:
            return avg_ratio
        else:
            return avg_points


class ThreeWayTester:
    def __init__(self, game_count: int, run_count: int):
        iter_count = game_count // run_count
        self._hands_list = [[generate_hands() for _ in range(run_count)]
                            for _ in range(iter_count)]
        self._run_count = run_count

    def _play_paralel_games(self, players: List[Agent], hands: List[List[Hand]]):
        games = np.array([TrackedGameRound(starting_player=0, hands=deepcopy(hand))
                          for hand in hands])

        while not all([g.end for g in games]):
            for player in range(PLAYERS):
                mask = [g.phasing_player == player and not g.end for g in games]
                if sum(mask) == 0:
                    continue

                obs = [g.observe() for g in games[mask]]
                observations, action_list = list(zip(*obs))

                actions_to_play = players[player].parallel_play(observations, action_list)

                for game, action in zip(games[mask], actions_to_play):
                    game.play(action)

        return np.array([g.points for g in games])

    def evaluate(self, players: List[Agent]):
        score_list = []

        for hands in tqdm(self._hands_list):
            for idx in itertools.permutations(np.arange(PLAYERS)):
                reordered_players = [players[i] for i in idx]

                scores = self._play_paralel_games(reordered_players, hands)

                re_idx = np.argsort(idx)
                score_list.append(scores[:, re_idx])

        scores = np.concatenate(score_list, axis=0).sum(0)
        return scores / scores.sum()


def analyze_game_round(agent, hands=None):
    if hands is None:
        hands = generate_hands()
    else:
        hands = deepcopy(hands)

    game = TrackedGameRound(
        starting_player=np.random.choice(np.arange(PLAYERS)),
        hands=hands
    )

    while True:
        observation, actions = game.observe()
        action = agent.play(observation, actions)
        print('')
        print("Choice Made", action)
        if game.phasing_player == 0:
            print('')
            for k, v in observation.items():
                print(k)
                print(v)

            print('------------------')

            debug_samples = agent.debug(observation, actions)
            for k, v in debug_samples.items():
                print(k)
                print(v)

            print('------------------')

        game.play(action)

        if game.end:
            print(game.points)
            break