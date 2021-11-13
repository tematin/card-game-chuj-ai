import numpy as np
from game.constants import COLOURS, VALUES, PLAYERS
from game.game import Card, get_deck

CARD_COUNT = COLOURS * VALUES


class Embeddings:
    def __init__(self, X):
        self.X = X
        if isinstance(X, list):
            self.length = len(X)
        else:
            self.length = 0

    def append(self, other):
        if self.length != other.length:
            raise ValueError("Incompatible lengths")
        if self.length == 0:
            self.X = np.concatenate([self.X, other.X], 0)
        else:
            self.X = [np.concatenate([x, y], 0) for x, y in zip(self.X, other.X)]

    def __getitem__(self, item):
        if self.length == 0:
            return Embeddings(self.X[item])
        else:
            return Embeddings([x[item] for x in self.X])

    @property
    def data_count(self):
        if self.length == 0:
            return self.X.shape[0]
        else:
            return self.X[0].shape[0]


def concatenate_embeddings(embeddings):
    length = embeddings[0].length
    if length == 0:
        return Embeddings(np.concatenate([x.X for x in embeddings], 0))
    else:
        return Embeddings([np.concatenate([x.X[i] for x in embeddings], 0)
                           for i in range(length)])


class LambdaEmbedder:
    def __init__(self, card_extractors, other_extractors, is_2d=False):
        self.card_extractors = card_extractors
        self.other_extractors = other_extractors
        self.is_2d = is_2d

    def get_state_embedding(self, observation):
        return Embeddings(self._get_raw_state_embedding(observation))

    def _get_raw_state_embedding(self, observation):
        card_embedding = [encode_card(f(observation)) for f in self.card_extractors]
        other_embedding = [f(observation) for f in self.other_extractors]
        return np.concatenate(card_embedding + other_embedding).reshape(1, -1)

    def get_action_embedding(self, observation):
        return Embeddings(self._get_raw_action_embedding(observation))

    def _get_raw_action_embedding(self, card):
        return encode_card(card).reshape(1, -1)

    def get_state_action_embedding(self, observation, card):
        return Embeddings(np.hstack([self._get_raw_state_embedding(observation),
                                     self._get_raw_action_embedding(card)]))

    def get_state_actions_embedding(self, observation):
        state_embedding = self._get_raw_state_embedding(observation)
        action_embeddings = np.vstack([self._get_raw_action_embedding(card)
                                       for card in observation.eligible_choices])
        return Embeddings(np.hstack([np.tile(state_embedding,
                                             (action_embeddings.shape[0], 1)),
                                     action_embeddings]))


class Lambda2DEmbedder:
    def __init__(self, card_extractors, other_extractors):
        self.card_extractors = card_extractors
        self.other_extractors = other_extractors

    def get_state_embedding(self, observation):
        return Embeddings(list(self._get_raw_state_embedding(observation)))

    def _get_raw_state_embedding(self, observation):
        card_embedding = np.stack([encode_card_2d(f(observation))
                                   for f in self.card_extractors])
        other_embedding = np.concatenate([np.array(f(observation))
                                          for f in self.other_extractors])
        return np.expand_dims(card_embedding, 0), other_embedding.reshape(1, -1)

    def get_action_embedding(self, card):
        return Embeddings(self._get_raw_action_embedding(card))

    def _get_raw_action_embedding(self, card):
        return np.expand_dims(encode_card_2d(card), [0, 1])

    def get_state_action_embedding(self, observation, card):
        card_embeddings, rest = self._get_raw_state_embedding(observation)
        action_embeddings = self._get_raw_action_embedding(card)
        return Embeddings([np.concatenate([card_embeddings, action_embeddings], 1),
                           rest])

    def get_state_actions_embedding(self, observation):
        card_embedding, rest = self._get_raw_state_embedding(observation)
        action_embeddings = np.concatenate([self._get_raw_action_embedding(card)
                                            for card in observation.eligible_choices],
                                           axis=0)
        card_embedding = np.tile(card_embedding, (action_embeddings.shape[0], 1, 1, 1))
        return Embeddings([np.concatenate([card_embedding, action_embeddings], axis=1),
                           np.tile(rest, (action_embeddings.shape[0], 1))])


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
    return observation.hand


def get_highest_pot_card(observation):
    return [observation.pot.get_highest_card()]


def get_historically_played_cards(observation):
    return observation.tracker.played_cards.played_cards


def get_pot_cards(observation):
    return list(observation.pot)


def get_pot_value(observation):
    return [observation.pot.get_point_value()]


def get_card_took_flag(observation):
    return observation.tracker.durch.took_card


def get_possible_cards(player):
    def inner_get_possible_cards(observation):
        return [card for card in get_deck()
                if card not in observation.hand
                and card not in observation.tracker.played_cards.played_cards
                and not observation.tracker.missed_colours.missed_colours[(observation.phasing_player + player) % PLAYERS, card.colour]]
    return inner_get_possible_cards

