import time
from copy import deepcopy
from itertools import combinations
from typing import Any, Dict, Optional, List, Tuple

from baselines.baselines import LowPlayer
from game.constants import PLAYERS, CARDS_PER_PLAYER
from game.game import Hand, Pot, TrackedGameRound
from game.stat_trackers import PartialGameTrackerManager, get_partial_default_tracker
from game.utils import GamePhase, generate_hands, Card


class PartialGameRound:
    _declarable_cards = [Card(1, 6), Card(2, 6)]

    def __init__(self, hand: List[Card], starting_player: int,
                 trackers: Optional[PartialGameTrackerManager] = None) -> None:
        self._end = False
        self._played_cards = 0
        self._hand = Hand(hand)
        self._starting_player = starting_player
        self._phasing_player = 0
        self._phase = GamePhase.MOVING
        self._cards_declared = []
        self._declaration_round = 0
        self._pot = Pot(self._starting_player)

        if trackers is None:
            trackers = get_partial_default_tracker()
        self._trackers = trackers
        self._trackers.reset(hand=hand, starting_player=starting_player)

    def play(self, action: Any) -> None:
        if self._phase == GamePhase.MOVING:
            if self._phasing_player == 0:
                self._move_card(action)
            else:
                self._receive_card(action)

        elif self._phase == GamePhase.DURCH:
            self._declare_durch(action)

        elif self._phase == GamePhase.DECLARATION:
            self._card_declaration(action)

        elif self._phase == GamePhase.PLAY:
            self._play_card(action)
            self._update_end()

        else:
            raise RuntimeError()

    def _update_end(self):
        self._played_cards += 1
        if self._played_cards == PLAYERS * CARDS_PER_PLAYER:
            self._end = True

    def _move_card(self, action) -> None:
        for card in action:
            self._hand.remove_card(card)
        self._phasing_player = PLAYERS - 1
        self._trackers.moved_cards(action)

    def _receive_card(self, action) -> None:
        for card in action:
            if card in self._hand:
                raise RuntimeError("Invalid Move")
            self._hand.add_card(card)
        self._phasing_player = self._starting_player
        self._phase = GamePhase.DURCH
        self._trackers.received_cards(action)

    def _declare_durch(self, action) -> None:
        self._trackers.pre_play_update(GamePhase.DURCH, self._phasing_player, action)
        if action:
            self._phase = GamePhase.PLAY
            self._pot = Pot(self._phasing_player)
        else:
            self._advance_player()
            if self._phasing_player == self._starting_player:
                self._phase = GamePhase.DECLARATION

    def _card_declaration(self, action) -> None:
        self._trackers.pre_play_update(
            GamePhase.DECLARATION, self._phasing_player, action)

        self._cards_declared.extend(action)

        if len(set(self._cards_declared)) != len(self._cards_declared):
            raise RuntimeError("Invalid Move")

        self._advance_player()

        if self._phasing_player == self._starting_player:
            if self._declaration_round == 1 or not self._cards_declared:
                self._phase = GamePhase.PLAY
            self._declaration_round += 1

    def _play_card(self, action) -> None:
        self._trackers.pre_play_update(
            GamePhase.PLAY, self._phasing_player, action, self._pot)

        if self._is_doubled(action):
            action.double()

        if self._phasing_player == 0:
            self._hand.remove_card(action)

        self._pot.add_card(action)

        if self._pot.is_full():
            self._phasing_player = self._pot.get_pot_owner()
            self._trackers.post_play_update(self._pot)
            self._pot = Pot(self._phasing_player)
        else:
            self._advance_player()

    def _is_doubled(self, card):
        return (
            card in self._cards_declared
            or card.colour == 0 and len(self._cards_declared) == 2
        )

    def _advance_player(self) -> None:
        self._phasing_player = (self._phasing_player + 1) % PLAYERS

    def observe(self) -> Tuple[Dict, List[Any]]:
        observation = {
            'hand': self._hand,
            'phase': self._phase,
            'pot': self._pot
        }

        observation.update(self._trackers.get_observations())
        return observation, self._eligible_actions(observation['possible_cards'])

    def _eligible_actions(self, possible_cards) -> List[Any]:
        if self._phase == GamePhase.MOVING:
            return list(combinations(self._hand, 2))
        elif self._phase == GamePhase.DURCH:
            return [1, 0]
        elif self._phase == GamePhase.DECLARATION:
            card_list = {
                0: self._hand,
                1: possible_cards[0],
                2: possible_cards[1]
            }[self._phasing_player]

            ret = []
            allowed_cards = [x for x in card_list
                             if x in self._declarable_cards
                             and x not in self._cards_declared]
            for i in range(len(allowed_cards) + 1):
                ret.extend(sorted(list(combinations(allowed_cards, i))))

            return ret
        else:
            if self._phasing_player == 0:
                if self._pot.is_empty():
                    return list(self._hand)

                pot_colour = self._pot.get_pot_colour()
                eligible_cards = [x for x in self._hand if x.colour == pot_colour]

                if eligible_cards:
                    return eligible_cards
                else:
                    return list(self._hand)

            else:
                return possible_cards[self._phasing_player - 1]

    @property
    def phasing_player(self) -> int:
        return self._phasing_player

    @property
    def phase(self) -> GamePhase:
        return self._phase

    @property
    def end(self) -> bool:
        return self._end
