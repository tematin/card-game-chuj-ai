from game import GameRound, PLAYERS
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from itertools import permutations


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
    j = 0
    for _ in range(count):
        j = (j + 1) % PLAYERS
        cached_games.append(GameRound(j))
    return cached_games


def evaluate_on_cached_games(cached_games, players):
    totals = []
    count = 0
    for game in tqdm(cached_games):
        for re_idx, player_order in zip(permutations(range(len(players))),
                                        permutations(players)):
            game_copy = deepcopy(game)
            game_copy = finish_game(game_copy, player_order)
            points_gained = np.array(game_copy.get_points())
            totals.append(points_gained[np.argsort(re_idx)])
            count += 1
    totals = np.array(totals)
    return totals.mean(0), totals.std(0, ddof=1) / np.sqrt(count)


def evaluate_on_cached_games_against(cached_games, player, adversary, return_score=False):
    took = []
    gave = []
    player_orders = [(player, adversary, adversary),
                     (adversary, player, adversary),
                     (adversary, adversary, player)]
    for game in tqdm(cached_games):
        for i in range(3):
            game_copy = deepcopy(game)
            game_copy = finish_game(game_copy, player_orders[i])
            points_gained = np.array(game_copy.get_points())
            took.append(points_gained[i])
            gave.append(points_gained.sum() - points_gained[i])
    took = np.array(took).reshape(-1, 3)
    gave = np.array(gave).reshape(-1, 3)
    if return_score:
        return gave.mean() - took.mean(), (gave.mean(1).std() - took.mean(1).std()) / np.sqrt(len(cached_games))
    else:
        return took.mean(), took.mean(1).std() / np.sqrt(len(cached_games))


def monte_carlo_evaluation(players, iterations):
    j = 0
    total = []
    for _ in tqdm(range(iterations)):
        j = (j + 1) % PLAYERS
        game = simulate_game(j, players)
        total.append(game.get_points())
    total = np.array(total)
    return total.mean(0), total.std(0, ddof=1) / np.sqrt(iterations)


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
