from __future__ import annotations

import numpy as np
from typing import List, Tuple, Callable, Union, Optional
from abc import abstractmethod


from game.utils import Card, get_deck, Observation, GamePhase
from game.constants import VALUES, COLOURS, CARD_COUNT
from learners.transformers import Transformer


class FeatureGenerator:
    @abstractmethod
    def state_action(self, observation: Observation
                     ) -> List[np.ndarray]:
        pass

    @abstractmethod
    def state(self, observation: Observation) -> List[np.ndarray]:
        pass


class FeatureGeneratorSplitter(FeatureGenerator):
    def __init__(
            self,
            feature_generators: dict[GamePhase, FeatureGenerator],
            feature_transformers: dict[GamePhase, List[Transformer]]
    ) -> None:
        self._feature_generators = feature_generators
        self._feature_transformers = feature_transformers

    def state_action(self, observation: Observation
                     ) -> List[np.ndarray]:
        out = self._feature_generators[observation.phase].state_action(observation)
        transformers = self._feature_transformers[observation.phase]

        return [t.transform(x) for x, t in zip(out, transformers)]

    def state(self, observation: Observation) -> List[np.ndarray]:
        out = self._feature_generators[observation.phase].state(observation)
        transformers = self._feature_transformers[observation.phase]

        return [t.transform(x) for x, t in zip(out, transformers)]


class Lambda2DEmbedder(FeatureGenerator):
    def __init__(self, card_extractors: List[Union[Callable[[dict], List[Card]], str]],
                 other_extractors: List[Union[Callable[[dict], np.ndarray], str]],
                 include_values: bool = False,
                 action_list: Optional[list[str]] = None) -> None:
        self._card_extractors = self._replace_str_with_callables(card_extractors)
        self._other_extractors = self._replace_str_with_callables(other_extractors)
        self._include_values = include_values
        self._action_list = action_list

        if include_values:
            self._values = np.zeros((COLOURS, VALUES))
            for card in get_deck():
                self._values[card.colour, card.value] += card.get_point_value()

    def _replace_str_with_callables(
            self, func_list: List[Union[Callable[[dict], List[Card]], str]]
    ) -> List[Callable[[dict], List[Card]]]:
        return [get_key(x) if isinstance(x, str) else x for x in func_list]

    def state_action(self, observation: Observation
                     ) -> List[np.ndarray]:
        action_count = len(observation.actions)

        other_embeddings = self._get_other_embedding(observation.features)
        other_embeddings = np.tile(other_embeddings, reps=(action_count, 1))

        card_embedding = self._get_2d_embedding(observation.features)
        card_embedding = np.tile(card_embedding, reps=(action_count, 1, 1, 1))

        if self._action_list is None:
            options_embedded = np.stack([encode_card_2d(card)
                                         for card in observation.actions])
            options_embedded = np.expand_dims(options_embedded, axis=1)
            card_embedding = np.concatenate([card_embedding, options_embedded], axis=1)
        else:
            option

        return [card_embedding, other_embeddings]

    def _append_action(self, ):

    def state(self, observation: Observation) -> List[np.ndarray]:
        card_embedding = self._get_2d_embedding(observation.features)
        other_embeddings = self._get_other_embedding(observation.features)

        return [card_embedding, other_embeddings]

    def _get_2d_embedding(self, observation: dict) -> np.ndarray:
        card_embedding = []
        for extractor in self._card_extractors:
            cards = extractor(observation)
            if len(cards) == 0:
                card_embedding.append(encode_card_2d(cards))
            if isinstance(cards[0], list):
                card_embedding.extend([encode_card_2d(x) for x in cards])
            else:
                card_embedding.append(encode_card_2d(cards))

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


def get_key(key):
    def retrieval_function(observation):
        return observation[key]
    return retrieval_function


def get_highest_pot_card(observation):
    return [observation['pot'].get_highest_card()]


def get_first_pot_card(observation):
    return [observation['pot'].get_first_card()]


def get_pot_value(observation):
    return [observation['pot'].get_point_value()]


def get_pot_size_indicators(observation):
    return [len(observation['pot']) == 1,
            len(observation['pot']) == 2]
