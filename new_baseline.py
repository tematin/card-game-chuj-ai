from baselines import LowPlayer, RandomPlayer
from game.game import GameRound
from evaluation_scripts import Tester
from encoders import get_possible_cards, encode_card_2d


tester = Tester(1000)

random = RandomPlayer()
baseline = LowPlayer()


tester.evaluate(baseline, random, verbose=1)


class Baseline:
    def __init__(self, colour_):
        self.color

    def play(self, observation):
        if observation.pot.is_empty():
            self._first_card()


    def dange

game = GameRound(0)
obs = game.observe()

obs.hand

free_cards = encode_card_2d(get_possible_cards(0)(obs))
cards = encode_card_2d(obs.hand)

free = free_cards[0]
card = cards[0]



