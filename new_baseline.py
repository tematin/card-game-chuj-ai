from baselines import LowPlayer, RandomPlayer
from game.game import GameRound
from evaluation_scripts import Tester
from encoders import get_possible_cards, encode_card_2d
import numpy as np


class Baseline:
    def __init__(self, decision_weights):
        self.decision_weights = decision_weights

    def play(self, observation):
        score = []
        for card in observation.eligible_choices:
            score.append(self.calculate_score(observation, card))
        print(score)
        return observation.eligible_choices[np.argmax(score)]

    def calculate_score(self, observation, card):
        return self.decision_weights * self.hand_leftover_features(observation, card)

    def hand_leftover_features(self, observation, card):
        return_features = []

        not_played = encode_card_2d(get_possible_cards(0)(observation))
        hand = encode_card_2d(observation.hand)
        hand[card.colour, card.value] = 0

        return_features.append((np.cumsum(hand, axis=1) - np.cumsum(not_played, axis=1)).max(1))
        return_features.append((np.cumsum(hand, axis=1) - 0.5 * np.cumsum(not_played, axis=1)).max(1))
        return_features.append(hand.sum(1))

        return np.concatenate(return_features)


tester = Tester(1000)

random = RandomPlayer()
low = LowPlayer()

baseline = Baseline(np.random.rand(12))

tester.evaluate(low, random, verbose=1)


game = GameRound(0)
observation = game.observe()
card = baseline.play(observation)
game.play(card)
