from __future__ import annotations

import numpy as np
from typing import List, Tuple, Callable
from abc import abstractmethod


from game.utils import Card, get_deck
from game.constants import VALUES, COLOURS, CARD_COUNT


class FeatureGenerator:
    @abstractmethod
    def state_action(self, observation: dict
                     ) -> Tuple[List[np.ndarray], List[Card]]:
        pass

    @abstractmethod
    def state(self, observation: dict) -> List[np.ndarray]:
        pass


class Lambda2DEmbedder(FeatureGenerator):
    def __init__(self, card_extractors: List[Callable[[dict], List[Card]]],
                 other_extractors: List[Callable[[dict], np.ndarray]],
                 include_values: bool = False) -> None:
        self._card_extractors = card_extractors
        self._other_extractors = other_extractors
        self._include_values = include_values

        if include_values:
            self._values = np.zeros((COLOURS, VALUES))
            for card in get_deck():
                self._values[card.colour, card.value] += card.get_point_value()

    def state_action(self, observation: dict
                     ) -> Tuple[List[np.ndarray], List[Card]]:
        options_embedded = np.stack([encode_card_2d(card)
                                     for card in observation['eligible_choices']])
        options_embedded = np.expand_dims(options_embedded, axis=1)

        other_embeddings = self._get_other_embedding(observation)
        other_embeddings = np.tile(other_embeddings,
                                   reps=(options_embedded.shape[0], 1))

        card_embedding = self._get_2d_embedding(observation)
        card_embedding = np.tile(card_embedding,
                                 reps=(options_embedded.shape[0], 1, 1, 1))
        card_embedding = np.concatenate([card_embedding, options_embedded], axis=1)

        return [card_embedding, other_embeddings], observation['eligible_choices']

    def state(self, observation: dict) -> List[np.ndarray]:
        card_embedding = self._get_2d_embedding(observation)

        other_embeddings = self._get_other_embedding(observation)

        return [card_embedding, other_embeddings]

    def _get_2d_embedding(self, observation: dict) -> np.ndarray:
        card_embedding = [encode_card_2d(f(observation)) for f in self._card_extractors]
        if self._include_values:
            card_embedding.append(self._values)

        card_embedding = np.stack(card_embedding)
        card_embedding = np.expand_dims(card_embedding, axis=0)

        return card_embedding

    def _get_other_embedding(self, observation: dict) -> np.ndarray:
        return np.concatenate([np.array(f(observation), dtype=float).flatten()
                               for f in self._other_extractors]).reshape(1, -1)

    @property
    def params(self) -> dict:
        return {
            'encoders': self._card_extractors,
            'other_encoders': self._other_extractors,
            'include_values': self._include_values
        }


def encode_card(cards):
    if isinstance(cards, Card):
        cards = [cards]
    ret = np.zeros(CARD_COUNT, dtype=int)
    for card in cards:
        if card is None:
            continue
        ret[int(card)] += 1
    return ret


def encode_card_2d(cards):
    if isinstance(cards, Card):
        cards = [cards]
    ret = np.zeros((COLOURS, VALUES), dtype=int)
    for card in cards:
        if card is None:
            continue
        ret[card.colour, card.value] += 1
    return ret


def get_hand(observation):
    return observation['hand']


def get_highest_pot_card(observation):
    return [observation['pot'].get_highest_card()]


def get_first_pot_card(observation):
    return [observation['pot'].get_first_card()]


def get_pot_cards(observation):
    return observation['pot']


def get_pot_value(observation):
    return [observation['pot'].get_point_value()]


def get_pot_size_indicators(observation):
    return [len(observation['pot']) == 1,
            len(observation['pot']) == 2]


def get_split_pot_value(observation):
    ret = np.zeros(4)
    red_idx = 0
    for card in observation['pot']:
        if card.get_point_value() == 1:
            ret[red_idx] += 1
            red_idx += 1
        elif card.get_point_value() == 4:
            ret[-2] += 1
        elif card.get_point_value() == 8:
            ret[-1] += 1
    return ret


def get_current_score(observation):
    return observation['score']


def get_eligible_durch(observation):
    return observation['eligible_durch']


def get_possible_cards(player):
    def inner(observation):
        return observation['possible_cards'][player]

    return inner

