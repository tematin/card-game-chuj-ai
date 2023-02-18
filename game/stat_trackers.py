from abc import ABC, abstractmethod

import numpy as np
from .constants import PLAYERS
from .utils import get_deck, Card, GamePhase


def get_all_trackers():
    return [
        DurchDeclarationTracker(),
        ScoreTracker(),
        MovedCardsTracker(),
        ReceivedCarsTracker(),
        DoubledCardsTracker(),
        RemainingPossibleCardsTracker(),
        DurchEligibilityTracker(),
        HistoryTracker(),
        PlayedCardsTracker(),
        StartingPlayerTracker()
    ]


class Tracker(ABC):
    def reset(self, hands, starting_player):
        pass

    def pre_play_update(self, game, card):
        if game.phase == GamePhase.PLAY:
            self._play_phase_pre_play_update(game, card)
        elif game.phase == GamePhase.DECLARATION:
            self._declaration_pre_play_update(game, card)
        elif game.phase == GamePhase.MOVING:
            self._moving_pre_play_update(game, card)
        elif game.phase == GamePhase.DURCH:
            self._durch_pre_play_update(game, card)

    def _play_phase_pre_play_update(self, game, card):
        pass

    def _declaration_pre_play_update(self, game, cards):
        pass

    def _durch_pre_play_update(self, game, declared):
        pass

    def _moving_pre_play_update(self, game, cards):
        pass

    def post_play_update(self, game):
        pass

    @abstractmethod
    def get_observations(self, player):
        pass


class HistoryTracker(Tracker):
    def reset(self, hands, starting_player):
        self._durch_phase = []
        self._declaration_phase = []
        self._play_phase = []

    def _play_phase_pre_play_update(self, game, card):
        self._play_phase.append((game.phasing_player, card))

    def _declaration_pre_play_update(self, game, cards):
        self._declaration_phase.append((game.phasing_player, cards))

    def _durch_pre_play_update(self, game, declared):
        self._durch_phase.append((game.phasing_player, declared))

    def get_observations(self, player):
        return {
            'declaration_history': self._declaration_phase,
            'durch_history': self._durch_phase,
            'play_history': self._play_phase
        }

    def _rotate_player(self, player, items):
        return [(_offset_player(x, player), y) for x, y in items]


class PlayedCardsTracker(Tracker):
    def reset(self, hands, starting_player):
        self._played_cards = [[] for _ in range(PLAYERS)]

    def _play_phase_pre_play_update(self, game, card):
        self._played_cards[game.phasing_player].append(card)

    def get_observations(self, player):
        return {
            'played_cards': _offset_array(self._played_cards, player)
        }


class StartingPlayerTracker(Tracker):
    def reset(self, hands, starting_player):
        self._starting_player = starting_player

    def get_observations(self, player):
        return {
            'starting_player': _offset_player(self._starting_player, player)
        }


class RemainingPossibleCardsTracker(Tracker):
    def reset(self, hands, starting_player):
        self._possible_cards = [[get_deck() for _ in range(PLAYERS - 1)]
                                for _ in range(PLAYERS)]
        for i in range(PLAYERS):
            for card in hands[i]:
                for j in range(PLAYERS - 1):
                    self._possible_cards[i][j].remove(card)

    def _play_phase_pre_play_update(self, game, card):
        player = game.phasing_player

        for i in range(PLAYERS):
            for j in range(PLAYERS - 1):
                try:
                    self._possible_cards[i][j].remove(card)
                except ValueError:
                    pass

        if game.pot.is_empty():
            return

        pot_colour = game.pot.get_pot_colour()
        if pot_colour == card.colour:
            return

        for i in range(PLAYERS):
            if i == player:
                continue
            j = (player - i) % PLAYERS - 1
            self._possible_cards[i][j] = [c for c in self._possible_cards[i][j]
                                          if c.colour != pot_colour]

    def _moving_pre_play_update(self, game, cards):
        player = game.phasing_player

        for card in cards:
            for i in range(0, PLAYERS - 1):
                self._possible_cards[(player + 1) % PLAYERS][i].remove(card)

            self._possible_cards[player][0].append(card)

    def _declaration_pre_play_update(self, game, cards):
        player = game.phasing_player

        for i in range(PLAYERS):
            if i == player:
                continue
            for j in range(PLAYERS - 1):
                if (i + j + 1) % PLAYERS == player:
                    continue
                for card in cards:
                    if card in self._possible_cards[i][j]:
                        self._possible_cards[i][j].remove(card)

    def get_observations(self, player):
        return {'possible_cards': self._possible_cards[player]}


class ScoreTracker(Tracker):
    def reset(self, hands, starting_player):
        self._score = [0 for _ in range(PLAYERS)]

    def post_play_update(self, game):
        if not game.pot.is_empty() or len(game.pot_history) == 0:
            return

        last_pot = game.pot_history[-1]
        self._score[last_pot.get_pot_owner()] += last_pot.get_point_value()

    def get_observations(self, player):
        return {
            'score': _offset_array(self._score, player)
        }


class DurchEligibilityTracker(Tracker):
    def reset(self, hands, starting_player):
        self._took_card = np.full(PLAYERS, False)

    def post_play_update(self, game):
        if not game.pot.is_empty() or len(game.pot_history) == 0:
            return

        last_pot = game.pot_history[-1]
        pot_owner = last_pot.get_pot_owner()
        self._took_card[pot_owner] = True

    def get_observations(self, player):
        if sum(self._took_card) == 0:
            ret = [True for _ in range(PLAYERS)]
        elif sum(self._took_card) == 1:
            ret = [False for _ in range(PLAYERS)]
            ret[np.argmax(self._took_card)] = True
            ret = _offset_array(ret, player)
        else:
            ret = [False for _ in range(PLAYERS)]

        return {'eligible_durch': ret}


class MovedCardsTracker(Tracker):
    def reset(self, hands, starting_player):
        self._moved_cards = [[]] * PLAYERS

    def _moving_pre_play_update(self, game, cards):
        player = game.phasing_player
        self._moved_cards[player] = cards

    def get_observations(self, player):
        return {'moved_cards': self._moved_cards[player]}


class ReceivedCarsTracker(Tracker):
    def reset(self, hands, starting_player):
        self._moved_cards = [[]] * PLAYERS

    def _moving_pre_play_update(self, game, cards):
        player = game.phasing_player
        self._moved_cards[(player + 1) % PLAYERS] = cards

    def get_observations(self, player):
        return {'received_cards': self._moved_cards[player]}


class DoubledCardsTracker(Tracker):
    def reset(self, hands, starting_player):
        self._doubled = [False, False]
        self._player = [None, None]

    def _declaration_pre_play_update(self, game, cards):
        if Card(1, 6) in cards:
            self._doubled[0] = True
            self._player[0] = game.phasing_player

        if Card(2, 6) in cards:
            self._doubled[1] = True
            self._player[1] = game.phasing_player

    def get_observations(self, player):
        return {
            'doubled': self._doubled,
            'player_doubled': self._player
        }


class DurchDeclarationTracker(Tracker):
    def reset(self, hands, starting_player):
        self._declared = np.full(PLAYERS, 0.0)

    def _durch_pre_play_update(self, game, declared):
        if declared:
            self._declared[game.phasing_player] = 1

    def get_observations(self, player):
        return {'declared_durch': _offset_array(self._declared, player)}


def _offset_array(x, player):
    return [x[(player + j) % PLAYERS] for j in range(PLAYERS)]


def _offset_player(absolute_player: int, relative_for: int) -> int:
    return (absolute_player - relative_for) % PLAYERS
