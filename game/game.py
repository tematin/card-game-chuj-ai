from enum import Enum
from itertools import permutations, combinations
from typing import List

import numpy as np
from .constants import COLOURS, VALUES, PLAYERS, CARDS_PER_PLAYER, DURCH_SCORE
from .stat_trackers import Tracker
from .utils import advance_player, is_eligible_choice, get_eligible_choices, Card


class Hand:
    def __init__(self, cards):
        if not isinstance(cards, list):
            raise TypeError("Wrong type instantiation")
        self._cards = cards

    def remove_card(self, card):
        self._cards.remove(card)

    def add_card(self, card):
        self._cards.append(card)

    def has_colour(self, colour):
        for card in self._cards:
            if card.colour == colour:
                return True
        return False

    def is_empty(self):
        return len(self._cards) == 0

    def __getitem__(self, item):
        return self._cards[item]

    def __iter__(self):
        for card in self._cards:
            yield card

    def __len__(self):
        return len(self._cards)

    def __repr__(self):
        return '{' + ', '.join([c.__repr__() for c in self._cards]) + '}'


class Pot:
    def __init__(self, initial_player):
        self._initial_player = initial_player
        self._cards = []
        self._highest_card = None

    def add_card(self, card):
        if len(self._cards) == 0:
            self._highest_card = 0
        else:
            if self._cards[self._highest_card].is_higher_value(card):
                self._highest_card = len(self._cards)

        self._cards.append(card)
        if len(self._cards) > PLAYERS:
            raise ValueError("Too many cards in Pot")

    def get_pot_owner(self):
        return advance_player(self._initial_player, self._highest_card)

    def get_point_value(self):
        return sum([card.get_point_value() for card in self._cards])

    def is_empty(self):
        return len(self._cards) == 0

    def is_full(self):
        return len(self._cards) == PLAYERS

    def get_pot_colour(self):
        return self._cards[0].colour

    def get_highest_card(self):
        if self.is_empty():
            return None
        return self._cards[self._highest_card]

    def get_first_card(self):
        if self.is_empty():
            return None
        return self._cards[0]

    def __repr__(self):
        return self._cards.__repr__()

    def __iter__(self):
        for card in self._cards:
            yield card

    def __getitem__(self, item):
        return self._cards[item]

    def __len__(self):
        return len(self._cards)

    @property
    def initial_player(self):
        return self._initial_player


class MainPhase:
    def __init__(self, starting_player, hands):
        self._phasing_player = starting_player
        self._hands = [Hand(i) for i in hands]
        self._pot = Pot(starting_player)
        self._pot_history = []
        self._end = False

    def play(self, card):
        if self._end:
            raise RuntimeError("Game already ended")

        if not is_eligible_choice(self._pot, self._hands[self.phasing_player], card):
            raise RuntimeError("Foul play")
        self._play(card)

    def _play(self, card):
        self._hands[self._phasing_player].remove_card(card)
        self._pot.add_card(card)

        if self._pot.is_full():
            self._clear()
        else:
            self._phasing_player = advance_player(self._phasing_player)

    def _clear(self):
        self._phasing_player = self._pot.get_pot_owner()
        self._pot_history.append(self._pot)
        self._pot = Pot(self._phasing_player)

        self._trick_end = False

        if self._hands[self.phasing_player].is_empty():
            self._end = True
            self._points = self._get_points()

    def _get_points(self):
        scores = np.zeros(PLAYERS)
        pots_took = np.zeros(PLAYERS)

        for pot in self._pot_history:
            owner = pot.get_pot_owner()
            pots_took[owner] += 1
            scores[owner] += pot.get_point_value()

        idx = np.argmax(pots_took)
        if pots_took[idx] == CARDS_PER_PLAYER:
            scores = np.zeros(PLAYERS)
            scores[idx] += DURCH_SCORE
            return scores
        else:
            return scores

    @property
    def points(self):
        return self._points

    @property
    def end(self):
        return self._end

    @property
    def phasing_player(self):
        return self._phasing_player

    @property
    def current_player_hand(self):
        return self._hands[self._phasing_player]

    @property
    def hands(self):
        return self._hands

    @property
    def pot(self):
        return self._pot

    @property
    def pot_history(self):
        return self._pot_history

    def eligible_choices(self):
        return get_eligible_choices(
            self.pot, self.current_player_hand
        )


class CardMovingPhase:
    def __init__(self, starting_player, hands):
        self._starting_player = starting_player
        self._hands = hands
        self._moved_cards = [list() for _ in range(PLAYERS)]
        self._phasing_player = 0
        self._end = False

    def play(self, cards):
        if self._end:
            raise RuntimeError("Game ready for next phase")

        if len(cards) != 2:
            raise ValueError("Wrong number of cards")

        for card in cards:
            if card not in self._hands:
                raise ValueError("Invalid move")
            self._moved_cards[self._phasing_player].append(card)
            self._hands[self._phasing_player].remove_card(card)

        self._phasing_player += 1
        if self._phasing_player == PLAYERS:
            self._end = True

            for i in range(PLAYERS):
                for card in self._moved_cards[i]:
                    self._hands[(i + 1) % PLAYERS].add_card(card)

    def next_stage(self):
        return GameTypeDeclarationPhase(hands=self._hands,
                                        starting_player=self._starting_player)

    @property
    def end(self):
        return self._end

    @property
    def phasing_player(self):
        return self._phasing_player

    @property
    def current_player_hand(self):
        return self._hands[self._phasing_player]

    def eligible_choices(self):
        return combinations(self.current_player_hand, 2)


class GameTypeDeclarationPhase:
    _allowed_doubling_cards = [Card(colour=1, value=6),
                               Card(colour=2, value=6)]

    def __init__(self, hands, starting_player) -> None:
        self._hands = hands
        self._starting_player = starting_player

        self._end = False
        self._phasing_player = 0
        self._round = 0

        self._doubled_cards = []

    def play(self, cards):
        if self._end:
            raise RuntimeError("Game ready for next phase")

        for card in cards:
            if card not in self._hands:
                raise ValueError("Invalid move")

            if card not in self._allowed_doubling_cards:
                raise ValueError("Invalid move")

            self._doubled_cards.append(card)

        self._phasing_player += 1
        if self._phasing_player == PLAYERS:
            if self._round == 0 and len(self._doubled_cards) > 0:
                self._round = 1
                self._phasing_player = 0
            else:
                self._end = True

    @property
    def end(self):
        return self._end

    @property
    def phasing_player(self):
        return self._phasing_player

    def next_stage(self):
        return MainPhase(hands=self._hands,
                         starting_player=self._starting_player)

    @property
    def current_player_hand(self):
        return self._hands[self._phasing_player]

    def eligible_choices(self):
        ret = []
        allowed_cards = [x for x in self._hands[self._phasing_player]
                        if x in self._allowed_doubling_cards]
        for i in range(len(allowed_cards)):
            ret.extend(list(combinations(allowed_cards, i)))

        return ret


class TrackedGameRound:
    def __init__(self, starting_player: int, hands: List[Hand],
                 trackers: List[Tracker]) -> None:
        self._game = CardMovingPhase(
            starting_player=starting_player,
            hands=hands
        )

        self._trackers = trackers
        for tracker in trackers:
            tracker.reset(hands)

    def observe(self, player=None):
        if player is None:
            player = self._game.phasing_player

        observation = {
            'hand': self._game.current_player_hand,
            'eligible_choices': self._game.eligible_choices()
        }

        if isinstance(self._game, MainPhase):
            observation['pot'] = self._game.pot

        for tracker in self._trackers:
            observation.update(tracker.get_observations(player))

        return observation

    def play(self, action):
        for tracker in self._trackers:
            tracker.pre_play_update(self._game, action)

        self._game.play(action)

        if isinstance(self._game, MainPhase):
            for tracker in self._trackers:
                tracker.post_play_update(self._game)

    @property
    def phasing_player(self):
        return self._game.phasing_player

    @property
    def end(self):
        return self._game.end

    @property
    def points(self):
        if isinstance(self._game, MainPhase):
            return self._game.points
        else:
            return np.zeros(PLAYERS)
