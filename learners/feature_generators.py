from __future__ import annotations

from pathlib import Path

import numpy as np
from typing import List, Tuple, Callable, Union, Optional, Any, Dict
from abc import abstractmethod

from baselines.baselines import Agent
from game.environment import Environment, OneThreadEnvironment

from game.utils import Card, get_deck, GamePhase
from game.constants import VALUES, COLOURS, CARD_COUNT, CARDS_PER_PLAYER
from learners.representation import index_features, concatenate_feature_list
from learners.transformers import Transformer, ListTransformer, DummyListTransformer
from debug.timer import timer


class FeatureGenerator:
    @abstractmethod
    def state_action(self, observations: Dict, actions: List[Any]) -> List[np.ndarray]:
        pass

    def save(self, directory: Path) -> None:
        pass

    def load(self, directory: Path) -> None:
        pass


class TransformedFeatureGenerator(FeatureGenerator):

    def __init__(self, feature_generator: FeatureGenerator,
                 feature_transformer: ListTransformer) -> None:
        self._feature_generator = feature_generator
        self._feature_transformer = feature_transformer

    def state_action(self, features: Dict, actions: List[Any]) -> List[np.ndarray]:
        features = self._feature_generator.state_action(features, actions)
        return self._feature_transformer.transform(features)

    def save(self, directory: Path) -> None:
        directory.mkdir(exist_ok=True)
        self._feature_generator.save(directory / 'feature_generator')
        self._feature_transformer.save(directory / 'feature_transformer')

    def load(self, directory: Path) -> None:
        self._feature_generator.load(directory / 'feature_generator')
        self._feature_transformer.load(directory / 'feature_transformer')


class Lambda2DEmbedder(FeatureGenerator):

    def __init__(
            self,
            card_extractors: List[Callable[[Dict, List[Any]], np.ndarray]],
            flat_extractors: List[Callable[[Dict, List[Any]], np.ndarray]]
    ) -> None:
        self._card_extractors = card_extractors
        self._flat_extractors = flat_extractors

        self._card_feature_shape = (len(self._card_extractors), 4, 9)

    def state_action(self, features: Dict, actions: List[Any]) -> List[np.ndarray]:
        action_count = len(actions)

        card_embedding = np.empty((action_count, *self._card_feature_shape))
        for i, card_extractor in enumerate(self._card_extractors):
            card_embedding[:, i] = card_extractor(features, actions)

        other_embeddings = []
        for f in self._flat_extractors:
            ret = np.array(f(features, actions))
            if ret.ndim == 1:
                ret = np.tile(ret, (action_count, 1))
            other_embeddings.append(ret)
        other_embeddings = np.concatenate(other_embeddings, axis=1)

        return [card_embedding, other_embeddings]

    @property
    def params(self) -> dict:
        return {
            'card_encoders': self._card_extractors,
            'flat_encoders': self._flat_extractors
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


def get_cards_remaining(observation, actions):
    ret = np.zeros(CARDS_PER_PLAYER - 1)
    ret[:(len(observation['hand']) - 1)] = 1
    return ret


def get_card_by_key(key):
    def retrieval_function(observation, actions):
        return encode_card_2d(observation[key])
    return retrieval_function


def get_flat_by_key(key):
    def retrieval_function(observation, actions):
        return observation[key]
    return retrieval_function


def get_possible_cards(idx):
    def possible_card(observation, action):
        return encode_card_2d(observation['possible_cards'][idx])
    return possible_card


def get_highest_pot_card(observation, actions):
    return encode_card_2d([observation['pot'].get_highest_card()])


def get_pot_card(order):
    def get_first_pot_card(observation, actions):
        try:
            cards = [observation['pot'][order]]
        except IndexError:
            cards = []
        return encode_card_2d(cards)
    return get_first_pot_card


def get_pot_value(observation, actions):
    return [observation['pot'].get_point_value()]


def get_pot_size_indicators(observation, actions):
    if observation['phase'] != GamePhase.PLAY:
        return [0, 0, 0]
    else:
        return [len(observation['pot']) == 0,
                len(observation['pot']) == 1,
                len(observation['pot']) == 2]


def get_moved_cards(observation, actions):
    if observation['phase'] == GamePhase.MOVING:
        return np.stack([encode_card_2d(x) for x in actions])
    else:
        return encode_card_2d(observation['moved_cards'])


def get_durch_phase_action(observation, actions):
    if observation['phase'] == GamePhase.DURCH:
        if actions[0]:
            return np.array([[1, 0], [0, 1]])
        else:
            return np.array([[0, 1], [1, 0]])
    else:
        return [0, 0]


def get_play_phase_action(observation, actions):
    if observation['phase'] == GamePhase.PLAY:
        return np.stack([encode_card_2d([x]) for x in actions])
    else:
        return encode_card_2d([])


def get_declaration_phase_action(observation, actions):
    if observation['phase'] == GamePhase.DECLARATION:
        ret = []
        for action in actions:
            idx = 0
            for card in action:
                idx += card.colour
            feats = [0, 0, 0, 0]
            feats[idx] += 1
            ret.append(feats)
        return ret
    else:
        return [0, 0, 0, 0]


def generate_dataset(
        env: OneThreadEnvironment,
        agent: Agent,
        episodes: int,
        feature_generator: FeatureGenerator
) -> Tuple[List[np.ndarray], np.ndarray]:
    feature_dataset = []
    reward_dataset = []

    for _ in range(episodes):
        episode_rewards = []
        observation, actions, _, done = env.reset()

        while not done:
            action = agent.play(observation, actions)
            idx = actions.index(action)

            action_feature = feature_generator.state_action(observation, actions)
            feature = index_features(action_feature, idx)

            observation, actions, reward, done = env.step(action)

            feature_dataset.append(feature)
            episode_rewards.append(reward)

        episode_rewards = np.cumsum(np.array(episode_rewards)[::-1])[::-1]
        reward_dataset.append(episode_rewards)

    feature_dataset = concatenate_feature_list(feature_dataset)
    reward_dataset = np.concatenate(reward_dataset)

    return feature_dataset, reward_dataset
