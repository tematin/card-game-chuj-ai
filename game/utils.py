from dataclasses import dataclass
from enum import Enum
from typing import Union, List

import numpy as np

from game.constants import VALUES, COLOURS, CARDS_PER_PLAYER, PLAYERS


def advance_player(player, moves=1):
    return (player + moves) % PLAYERS


def is_eligible_choice(pot, hand, card):
    if pot.is_empty():
        return True
    elif card.colour == pot.get_pot_colour():
        return True
    else:
        if not hand.has_colour(pot.get_pot_colour()):
            return True
        else:
            return False


class Card:
    colour_to_str = {0: '♥',
                     1: '☘',
                     2: '♠',
                     3: '⚈'}
    value_to_str = {5: 'D',
                    6: 'H',
                    7: 'K',
                    8: 'A'}

    str_to_colour = {v: k for k, v in colour_to_str.items()}
    str_to_value = {v: k for k, v in value_to_str.items()}

    def __init__(self, *args):
        if len(args) == 2:
            self._colour_value_init(*args)
        elif len(args) == 1:
            self._string_init(*args)

    def _colour_value_init(self, colour, value):
        if not 0 <= value < VALUES:
            raise ValueError("Wrong value specified")
        if not 0 <= colour < COLOURS:
            raise ValueError("Wrong colour specified")
        self._colour = colour
        self._value = value
        self._doubled = False

    def _string_init(self, s):
        colour_code, value_code = s
        self._colour_value_init(int(colour_code), int(value_code))

    def double(self):
        self._doubled = True

    def is_higher_value(self, other):
        if other.colour == self._colour:
            return other.value >= self._value
        return False

    def __int__(self):
        return self._colour * VALUES + self._value

    def __eq__(self, other):
        return (self._colour == other.colour) & (self._value == other.value)

    def __lt__(self, other):
        return int(self) < int(other)

    def __hash__(self):
        return int(self)

    def get_point_value(self):
        if self._colour == 0:
            val = 1
        elif self._colour == 1 and self._value == 6:
            val = 4
        elif self._colour == 2 and self._value == 6:
            val = 8
        else:
            val = 0

        if self._doubled:
            val *= 2

        return val

    def __str__(self):
        return str(self._colour) + str(self._value)

    def __repr__(self):
        colour_code = self.colour_to_str[self._colour]
        value_code = self.value_to_str.get(self._value, str(self._value + 6))

        point_value = self.get_point_value()
        if point_value > 2:
            point_code = '#'
        else:
            point_code = ''

        return point_code + colour_code + value_code

    @property
    def colour(self):
        return self._colour

    @property
    def value(self):
        return self._value


def get_deck():
    ret = []
    for value in range(VALUES):
        for colour in range(COLOURS):
            ret.append(Card(colour, value))
    return ret


def generate_hands():
    deck = get_deck()
    np.random.shuffle(deck)
    return [list(np.sort(deck[i:(i + CARDS_PER_PLAYER)]))
            for i in range(0, CARDS_PER_PLAYER * PLAYERS, CARDS_PER_PLAYER)]


def get_eligible_choices(pot, hand):
    if pot.is_empty():
        return list(hand)
    else:
        pot_colour = pot.get_pot_colour()
        if hand.has_colour(pot_colour):
            return [c for c in hand if c.colour == pot_colour]
        else:
            return list(hand)


class GamePhase(Enum):
    MOVING = 0
    DURCH = 1
    DECLARATION = 2
    PLAY = 3
