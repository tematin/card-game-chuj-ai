from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from collections import Counter

from game.utils import Card, GamePhase, Observation


class Agent(ABC):
    @abstractmethod
    def play(self, observation: Observation) -> Any:
        pass


class RandomPlayer(Agent):
    def play(self, observation: Observation) -> Any:
        if observation.phase == GamePhase.DURCH:
            if np.random.rand() < 0.985:
                return False
            else:
                return True

        if observation.phase == GamePhase.DECLARATION:
            if np.random.rand() < 0.5:
                return ()

        idx = np.random.randint(len(observation.actions))
        return observation.actions[idx]


class PartialRandomPlayer(Agent):
    def play(self, observation: Observation) -> Any:
        phase = observation.phase
        if phase == GamePhase.PLAY:
            idx = np.random.randint(len(observation.actions))
            return observation.actions[idx]
        elif phase == GamePhase.MOVING:
            return self._move_cards(observation.features['hand'])
        elif phase == GamePhase.DURCH:
            return False
        elif phase == GamePhase.DECLARATION:
            return ()

    def _move_cards(self, hand: List[Card]) -> List[Card]:
        values = [x.value for x in hand]
        idx = np.argsort(values)[-2:]
        return [hand[idx[0]], hand[idx[1]]]


class LowPlayer(Agent):
    def play(self, observation: Observation) -> Any:
        phase = observation.phase
        if phase == GamePhase.PLAY:
            return self._play_phase(observation.features)
        elif phase == GamePhase.MOVING:
            return self._move_cards(observation.features['hand'])
        elif phase == GamePhase.DURCH:
            return False
        elif phase == GamePhase.DECLARATION:
            return ()

    def _move_cards(self, hand: List[Card]) -> List[Card]:
        values = [x.value for x in hand]
        idx = np.argsort(values)[-2:]
        return [hand[idx[0]], hand[idx[1]]]

    def _play_phase(self, features: dict) -> Card:
        pot = features['pot']
        hand = features['hand']

        if pot.is_empty():
            smallest_colour = get_smallest_colour(hand)
            return get_smallest_card(hand, smallest_colour)
        else:
            choices = get_sorted_cards_of_colour(hand, pot.get_pot_colour())
            if len(choices) == 0:
                card, point = get_highest_points_card(hand, return_value=True)
                if point >= 2:
                    return card
                return get_largest_card(hand)
            else:
                if choices[0].is_higher_value(pot.get_highest_card()):
                    vals = [c.value for c in choices]
                    pos = np.searchsorted(vals, pot.get_highest_card().value)
                    under = choices[:pos]
                    points = [c.get_point_value() for c in under]
                    idx = np.argmax(points)
                    if points[idx] >= 2:
                        return under[idx]
                    else:
                        return under[-1]
                else:
                    return choices[-1]


def get_smallest_card(hand, colour=None):
    if colour is not None:
        return get_sorted_cards_of_colour(hand, colour)[0]
    else:
        idx = np.argmin([c.value for c in hand])
        return hand[idx]


def get_largest_card(hand, colour=None):
    if colour is not None:
        return get_sorted_cards_of_colour(hand, colour)[-1]
    else:
        idx = np.argmax([c.value for c in hand])
        return hand[idx]


def get_highest_points_card(hand, return_value=False):
    vals = [c.get_point_value() for c in hand]
    idx = np.argmax(vals)
    if return_value:
        return hand[idx], vals[idx]
    else:
        return hand[idx]


def get_smallest_colour(hand):
    c = get_colour_counts(hand)
    return c.most_common()[-1][0]


def get_colour_counts(hand):
    c = Counter()
    for card in hand:
        c[card.colour] += 1
    return c


def get_sorted_cards_of_colour(hand, colour):
    return np.sort([c for c in hand if c.colour == colour])
