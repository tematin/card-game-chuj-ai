import numpy as np
from copy import deepcopy

from .constants import PLAYERS
from .game import MainPhase, TrackedGameRound
from .utils import generate_hands


class Environment:
    def __init__(self, trackers, reward, rival):
        self._trackers = trackers
        self._reward = reward
        self._rival = rival

        self._game = None

    def reset(self):
        hands = generate_hands()
        self._game = TrackedGameRound(
            starting_player=np.random.choice(np.arange(PLAYERS)),
            hands=hands,
            trackers=self._trackers
        )

        self._play_for_rivals()

        observation = self._game.observe(player=0)
        self._reward.reset(observation)

        return observation

    def step(self, card):
        self._game.play(card)

        self._play_for_rivals()

        observation = self._game.observe(player=0)
        reward = self._reward.step(observation)

        return observation, reward, self._game.end

    def _play_for_rivals(self):
        while self._game.phasing_player != 0 and not self._game.end:
            observation = self._game.observe()

            card = self._rival.play(observation)
            self._game.play(card)


def finish_game(tracked_game, players):
    while not tracked_game.end:
        observation = tracked_game.observe()
        card = players[tracked_game.phasing_player].play(observation)
        tracked_game.play(card)

    return tracked_game


class Tester:
    def __init__(self, game_count, trackers, seed=123456):
        if seed is not None:
            np.random.seed(seed)
        self.hands_list = [generate_hands() for _ in range(game_count)]
        self.std_scale = 1 / np.sqrt(game_count)
        self._trackers = trackers

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
                    trackers=self._trackers
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


def analyze_game_round(players, initial_player=0):
    game = MainPhase(initial_player)
    while True:
        observation = game.observe()
        card = players[game.phasing_player].play(observation)
        if game.phasing_player == 0:
            try:
                print(list(game.tracker.history.history[-1]))
            except Exception as e:
                pass
            print('')
            print('')
            print("Hand", np.sort(list(game.hands[0])))
            print("Pot", list(game.pot))
            print("Choice Made", card)
            print("Adversary Hands")
            print(np.sort(list(game.hands[1])))
            print(np.sort(list(game.hands[2])))
            print('---')
            print(game.get_points())
            idx = np.argsort(observation.eligible_choices)
            print(players[game.phasing_player].get_embedding_value_pair(observation)[1][idx])

        game.play(card)

        if game.end:
            print(game.get_points())
            break
