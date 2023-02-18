from itertools import combinations
from typing import List, Optional, Any, Tuple

import numpy as np

from game.constants import PLAYERS, CARDS_PER_PLAYER, DURCH_SCORE
from game.stat_trackers import GameTrackerManager, get_default_tracker
from game.utils import advance_player, is_eligible_choice, get_eligible_choices, Card, \
    get_deck, GamePhase


class Hand:
    def __init__(self, cards):
        if not isinstance(cards, list):
            raise TypeError("Wrong type instantiation")
        self._cards = cards

    def remove_card(self, card):
        i = self._cards.index(card)
        card = self._cards[i]
        self._cards.remove(card)
        return card

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


class PlayPhase:
    phase = GamePhase.PLAY

    def __init__(self, starting_player, hands, declared_durch=None):
        self._phasing_player = starting_player
        self._declared_durch = declared_durch
        if declared_durch is not None:
            self._phasing_player = declared_durch

        self._hands = hands
        self._pot = Pot(starting_player)
        self._pot_history = []
        self._end = False

    def play(self, card):
        if self._end:
            raise RuntimeError("Game already ended")

        current_hand = self._hands[self._phasing_player]
        if not is_eligible_choice(self._pot, current_hand, card):
            raise RuntimeError("Foul play")

        card = self._hands[self._phasing_player].remove_card(card)
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

        if self._declared_durch is None:
            idx = np.argmax(pots_took)
            if pots_took[idx] == CARDS_PER_PLAYER:
                scores = np.zeros(PLAYERS)
                scores[idx] += DURCH_SCORE
        else:
            sum_scores = scores.sum()
            scores = np.zeros(PLAYERS)
            if pots_took[self._declared_durch] == CARDS_PER_PLAYER:
                scores[self._declared_durch] = DURCH_SCORE * 2
            else:
                scores[self._declared_durch] = sum_scores
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


class MovingPhase:
    phase = GamePhase.MOVING

    def __init__(self, starting_player, hands):
        self._starting_player = starting_player
        self._hands = [Hand(x) for x in hands]
        self._moved_cards = [list() for _ in range(PLAYERS)]
        self._phasing_player = starting_player
        self._end = False

    def play(self, cards):
        if self._end:
            raise RuntimeError("Game ready for next phase")

        if len(cards) != 2:
            raise ValueError("Wrong number of cards")

        for card in cards:
            if card not in self.current_player_hand:
                raise ValueError("Invalid move")
            self._moved_cards[self._phasing_player].append(card)
            self._hands[self._phasing_player].remove_card(card)

        self._phasing_player = (self._phasing_player + 1) % PLAYERS
        if self._phasing_player == self._starting_player:
            self._end = True

            for i in range(PLAYERS):
                for card in self._moved_cards[i]:
                    self._hands[(i + 1) % PLAYERS].add_card(card)

    def next_stage(self):
        return DurchPhase(hands=self._hands,
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

    @property
    def hands(self):
        return self._hands

    def eligible_choices(self):
        return list(combinations(self.current_player_hand, 2))


class DurchPhase:
    phase = GamePhase.DURCH

    def __init__(self, hands, starting_player) -> None:
        self._hands = hands
        self._starting_player = starting_player

        self._end = False
        self._phasing_player = starting_player

        self._declared_durch = None

    def play(self, declare):
        if self._end:
            raise RuntimeError("Game ready for next phase")

        if declare:
            self._end = True
            self._declared_durch = self._phasing_player
            return

        self._phasing_player = (self._phasing_player + 1) % PLAYERS

        if self._phasing_player == self._starting_player:
            self._end = True

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

    def next_stage(self):
        if self._declared_durch is None:
            return DeclarationPhase(
                hands=self._hands,
                starting_player=self._starting_player
            )
        else:
            return PlayPhase(
                hands=self._hands,
                starting_player=self._declared_durch,
                declared_durch=self._declared_durch
            )

    def eligible_choices(self):
        return [1, 0]


class DeclarationPhase:
    phase = GamePhase.DECLARATION
    _allowed_doubling_cards = [Card(1, 6),
                               Card(2, 6)]

    def __init__(self, hands, starting_player) -> None:
        self._hands = hands
        self._starting_player = starting_player

        self._end = False
        self._phasing_player = starting_player
        self._round = 0

        self._doubled_cards = []

    def play(self, cards):
        if self._end:
            raise RuntimeError("Game ready for next phase")

        for card in cards:
            if card not in self.current_player_hand:
                raise ValueError("Invalid move")

            if card not in self._allowed_doubling_cards:
                raise ValueError("Invalid move")

            self._doubled_cards.append(card)

        self._phasing_player = (self._phasing_player + 1) % PLAYERS
        if self._phasing_player == self._starting_player:
            if self._round == 0 and len(self._doubled_cards) == 1:
                self._round = 1

            else:
                self._double_values()
                self._end = True

    def _double_values(self):
        if len(self._doubled_cards) == 2:
            self._doubled_cards.extend([x for x in get_deck() if x.colour == 0])

        for hand in self._hands:
            for card in hand:
                if card in self._doubled_cards:
                    card.double()

    @property
    def end(self):
        return self._end

    @property
    def phasing_player(self):
        return self._phasing_player

    def next_stage(self):
        return PlayPhase(hands=self._hands,
                         starting_player=self._starting_player)

    @property
    def current_player_hand(self):
        return self._hands[self._phasing_player]

    @property
    def hands(self):
        return self._hands

    def eligible_choices(self):
        ret = []
        allowed_cards = [x for x in self.current_player_hand
                         if x in self._allowed_doubling_cards
                         and x not in self._doubled_cards]
        for i in range(len(allowed_cards) + 1):
            ret.extend(sorted(list(combinations(allowed_cards, i))))

        return ret


class TrackedGameRound:
    def __init__(self, starting_player: int, hands: List[Hand],
                 tracker: Optional[GameTrackerManager] = None) -> None:
        self._game = MovingPhase(
            starting_player=starting_player,
            hands=hands
        )

        if tracker is None:
            tracker = get_default_tracker()
        self._tracker = tracker

        self._tracker.reset(hands, starting_player)

        self._end = False

    def observe(self, player=None) -> Tuple[dict, List[Any]]:
        if player is None:
            player = self._game.phasing_player

        observation = {
            'hand': self._game.hands[player],
            'phase': self._game.phase,
        }

        if self._game.phase == GamePhase.PLAY:
            observation['pot'] = self._game.pot
        else:
            observation['pot'] = Pot(0)

        tracker_observation = self._tracker.get_observations(player)
        observation.update(tracker_observation)

        return observation, self._game.eligible_choices()

    def play(self, action):
        self._tracker.pre_play_update(
            phase=self.phase,
            player=self.phasing_player,
            action=action,
            pot=self._game.pot if self._game.phase == GamePhase.PLAY else None
        )

        self._game.play(action)

        if self._game.phase == GamePhase.PLAY and self._game.pot.is_empty():
            self._tracker.post_play_update(self._game.pot_history[-1])

        if self._game.end:
            if self._game.phase == GamePhase.PLAY:
                self._end = True
            else:
                self._game = self._game.next_stage()

    @property
    def phase(self):
        return self._game.phase

    @property
    def phasing_player(self):
        return self._game.phasing_player

    @property
    def end(self):
        return self._end

    @property
    def points(self):
        if self._game.phase == GamePhase.PLAY:
            return self._game.points
        else:
            return np.zeros(PLAYERS)
