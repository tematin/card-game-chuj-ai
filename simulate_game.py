from game.game import GameRound
from baselines import LowPlayer
import numpy as np

adversary = LowPlayer()

game = GameRound(0)
observation = game.observe()


print('┌─────────┐')
print('│ 7       │')
print('│         │')
print('│         │')
print('│         │')
print('│       7 │')
print('└─────────┘')


class MessagePrint:
    ROWS = 30
    COLUMNS = 132

    def __init__(self):
        self.board = np.full((self.ROWS, self.COLUMNS), ' ')

    def add_card(self, card, row, column):



    def flush(self):
        for i in range(self.ROWS):
            s = ''
            for j in range(self.COLUMNS):
                s += self.board[i, j]
            print(s)

    def card_to_string_value(self, card):
        return {0: ' VI ',
                1: 'VII ',
                2: 'VIII',
                3: ' IX ',
                4: ' X  ',
                5: ' D  ',
                6: ' H  ',
                7: ' K  ',
                8: ' A  '}[card.value]

    def card_to_string_colour(self, card):
        return {0: '♥',
                1: '☘',
                2: '⛊',
                3: '⚈'}[card.colour]

