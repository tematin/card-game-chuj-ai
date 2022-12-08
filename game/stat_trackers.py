from abc import ABC

import numpy as np
from .constants import PLAYERS
from .utils import get_deck
from .game import MainPhase, CardMovingPhase, GameTypeDeclarationPhase

class Tracker(ABC):
    def reset(self, hands):
        pass

    def pre_play_update(self, game, card):
        if isinstance(game, CardMovingPhase):
            self._main_phase_pre_play_update(game, card)
        elif isinstance(game, GameTypeDeclarationPhase):
            self._card_moving_pre_play_update(game, card)
        elif isinstance(game, MainPhase):
            self._main_phase_pre_play_update(game, card)

    def _main_phase_pre_play_update(self, game, card):
        pass

    def _card_moving_pre_play_update(self, game, card):
        pass

    def _type_declaration_pre_play_update(self, game, card):
        pass

    def post_play_update(self, game):
        pass

    def get_observations(self, player):
        pass


class RemainingPossibleCards(Tracker):
    def __init__(self):
        self._possible_cards = None

    def reset(self, hands):
        self._possible_cards = [[get_deck() for _ in range(PLAYERS - 1)]
                                for _ in range(PLAYERS)]
        for i in range(PLAYERS):
            for card in hands[i]:
                for j in range(PLAYERS - 1):
                    self._possible_cards[i][j].remove(card)

    def _main_phase_pre_play_update(self, game, card):
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
            j = (i - player) % PLAYERS - 1
            self._possible_cards[i][j] = [c for c in self._possible_cards[i][j]
                                          if c.colour != pot_colour]

    def get_observations(self, player):
        return {'possible_cards': self._possible_cards[player]}


class ScoreTracker(Tracker):
    def reset(self, hands):
        self._score = [0 for _ in range(PLAYERS)]

    def post_play_update(self, game):
        if not game.pot.is_empty() or len(game.pot_history) == 0:
            return

        last_pot = game.pot_history[-1]
        self._score[last_pot.get_pot_owner()] += last_pot.get_point_value()

    def get_observations(self, player):
        return {
            'score': [self._score[(player + j) % PLAYERS] for j in range(PLAYERS)]
        }


class DurchEligibilityTracker(Tracker):
    def reset(self, hands):
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
            ret[(player + np.argmax(self._took_card)) % PLAYERS] = True
        else:
            ret = [False for _ in range(PLAYERS)]

        return {'eligible_durch': ret}
