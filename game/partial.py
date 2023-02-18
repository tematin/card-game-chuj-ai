from typing import Any, Dict

from game.constants import PLAYERS
from game.game import Hand, Pot
from game.utils import GamePhase


class PartialGame:
    def __init__(self, hand: Hand, starting_player: int) -> None:
        self._hand = hand
        self._starting_player = starting_player
        self._phasing_player = 0
        self._phase = GamePhase.MOVING
        self._cards_declared = []
        self._declaration_round = 0
        self._pot = Pot(self._starting_player)

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

        else:
            raise RuntimeError()

    def _move_card(self, action) -> None:
        for card in action:
            self._hand.remove_card(card)
        self._phasing_player = PLAYERS - 1

    def _receive_card(self, action) -> None:
        for card in action:
            if card in self._hand:
                raise RuntimeError("Invalid Move")
            self._hand.add_card(card)
        self._phasing_player = self._starting_player
        self._phase = GamePhase.DURCH

    def _declare_durch(self, action) -> None:
        if action:
            self._phase = GamePhase.PLAY
            self._pot = Pot(self._phasing_player)
        else:
            self._advance_player()
            if self._phasing_player == self._starting_player:
                self._phase = GamePhase.DECLARATION

    def _card_declaration(self, action) -> None:
        self._cards_declared.extend(action)

        if len(set(self._cards_declared)) != len(self._cards_declared):
            raise RuntimeError("Invalid Move")

        self._advance_player()

        if self._phasing_player == self._starting_player:
            if self._declaration_round == 1 or not self._cards_declared:
                self._phase = GamePhase.PLAY
            self._declaration_round += 1

    def _play_card(self, action) -> None:
        self._pot.add_card(action)
        if self._phasing_player == 0:
            self._hand.remove_card(action)

        if self._pot.is_full():
            self._phasing_player = self._pot.get_pot_owner()
        else:
            self._advance_player()

    def _advance_player(self) -> None:
        self._phasing_player = (self._phasing_player + 1) % PLAYERS

    def observe(self) -> Dict:
        return {
            'hand': self._hand,
            'phase': self._phase
        }

