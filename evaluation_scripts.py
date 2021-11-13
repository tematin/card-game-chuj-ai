from game.game import GameRound
import numpy as np
from tqdm import tqdm
from copy import deepcopy


def finish_game(game, players):
    while not game.end:
        observation = game.observe()
        card = players[game.phasing_player].play(observation)
        game.play(card)
    return game


def simulate_game(starting_player, players):
    game = GameRound(starting_player)
    game = finish_game(game, players)
    return game


def get_cached_games(count):
    cached_games = []
    for _ in range(count):
        cached_games.append(GameRound(0))
    return cached_games


def analyze_game_round(players, initial_player=0):
    game = GameRound(initial_player)
    while True:
        observation = game.observe()
        card = players[game.phasing_player].play(observation)
        if game.phasing_player == 0:
            try:
                print(list(game.record.history[-1]))
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
            print(list(game.record.history[-1]))
            print(game.get_points())
            break


class Tester:
    def __init__(self, game_count):
        self.game_list = get_cached_games(game_count)
        self.std_scale = 1 / np.sqrt(game_count)

    def _basic_evaluate(self, player, adversary, verbose=0):
        points = []
        game_value = []
        player_orders = [(player, adversary, adversary),
                         (adversary, player, adversary),
                         (adversary, adversary, player)]

        game_list = tqdm(self.game_list) if verbose else self.game_list

        for game in game_list:
            for i in range(3):
                game_copy = deepcopy(game)
                game_copy = finish_game(game_copy, player_orders[i])
                points_gained = game_copy.get_points()

                points.append(points_gained[i])
                game_value.append(np.sum(points_gained))

        return np.array(points).reshape(-1, 3), np.array(game_value).reshape(-1, 3)

    def evaluate(self, player, adversary, verbose=0, return_ratio=False):
        points, game_value = self._basic_evaluate(player, adversary, verbose=verbose)

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
