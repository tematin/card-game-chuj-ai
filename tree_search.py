from collections import defaultdict
from baselines import LowPlayer, RandomPlayer
from encoders import get_possible_cards
from game import GameRound
import numpy as np


game = GameRound(0)
player = LowPlayer()

for i in range(5):
    game.play(player.play(game.observe()))
    game.play(player.play(game.observe()))
    game.play(player.play(game.observe()))

observation = game.observe()


class SimulatedGame(GameRound):
    def __init__(self, observation, right_hand, left_hand):
        self.phasing_player = observation.phasing_player
        self.pot = observation.pot
        self.record = observation.record
        self.end = False

        self.hands = [None, None, None]
        self.hands[self.phasing_player] = observation.hand
        self.hands[(self.phasing_player + 1) % 3] = right_hand
        self.hands[(self.phasing_player + 2) % 3] = left_hand


def get_simulated_game(observation, iter):
    right_player_possible_cards = set(get_possible_cards(0)(observation))
    left_player_possible_cards = set(get_possible_cards(1)(observation))

    unknown_cards = right_player_possible_cards.intersection(left_player_possible_cards)
    right_cards = right_player_possible_cards.difference(unknown_cards)
    left_cards = left_player_possible_cards.difference(unknown_cards)

    total_length = len(unknown_cards) + len(right_cards) + len(left_cards)
    to_assign_to_left = int(np.floor(total_length / 2)) - len(left_cards)

    unknown_cards = list(unknown_cards)
    for _ in range(iter):
        np.random.shuffle(unknown_cards)
        yield SimulatedGame(observation,
                            right_cards.union(unknown_cards[to_assign_to_left:]),
                            left_cards.union(unknown_cards[:to_assign_to_left]))


while not simulated_game.end:
    simulated_game.play(player.play(simulated_game.observe()))

