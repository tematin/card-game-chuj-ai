import numpy as np
from copy import deepcopy

from .constants import PLAYERS
from .game import PlayPhase, TrackedGameRound
from .utils import generate_hands, GamePhase


class Environment:
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

        return observation, actions

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


class Tester:
    def __init__(self, game_count, seed=123456):
        if seed is not None:
            np.random.seed(seed)
        self.hands_list = [generate_hands() for _ in range(game_count)]
        self.std_scale = 1 / np.sqrt(game_count)

    def _simulate(self, player, adversary):
        points = []
        game_value = []
        player_orders = [(player, adversary, adversary),
                         (adversary, player, adversary),
                         (adversary, adversary, player)]

        for hand in self.hands_list:
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

    def evaluate(self, player, adversary, verbose=0, return_ratio=False):
        points, game_value = self._simulate(player, adversary)

        avg_points = points.mean()
        std_points = points.mean(1).std()

        ratio = points / game_value
        avg_ratio = ratio.mean()
        std_ratio = ratio.mean(1).std()

        if verbose:
            print(f"Score Achieved: {avg_points:.2f} +- {self.std_scale * 2 * std_points:.2f}")
            print(f"Ratio Achieved: {avg_ratio:.1%} +- {self.std_scale * 2 * std_ratio:.1%}")
            print(f"Durch Made: {(points < 0).sum()}")
            print(f"Total Durch Made: {(game_value < 0).sum()}")
            print(f"Average Total Score: {game_value.mean()}")
            print('')

        if return_ratio:
            return avg_ratio
        else:
            return avg_points


def analyze_game_round(agent):
    hands = generate_hands()
    game = TrackedGameRound(
        starting_player=np.random.choice(np.arange(PLAYERS)),
        hands=hands
    )

    while True:
        observation = game.observe()
        obs_dict = observation.features
        action = agent.step(observation)
        print('')
        print("Choice Made", action)
        if game.phasing_player == 0:
            print('')
            for k, v in obs_dict.items():
                print(k)
                print(v)

            print('------------------')

            debug_samples = {}#agent.debug(observation)
            for k, v in debug_samples.items():
                print(k)
                print(v)

            print('------------------')

        game.play(action)

        if game.end:
            print(game.points)
            break
