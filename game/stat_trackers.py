from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from .constants import PLAYERS
from .utils import get_deck, Card, GamePhase


def get_default_tracker():
    return GameTrackerManager([
        DurchDeclarationTracker(),
        ScoreTracker(),
        DoubledCardsTracker(),
        DurchEligibilityTracker(),
        HistoryTracker(),
        PlayedCardsTracker(),
        StartingPlayerTracker()
    ], [
        MovedCardsTracker(),
        ReceivedCarsTracker(),
        RemainingPossibleCardsTracker(),
    ])


class GameTrackerManager:
    def __init__(self, public_trackers, private_trackers):
        self._public_trackers = public_trackers
        self._private_trackers = []
        for i in range(PLAYERS):
            trackers = [RelativePrivateTracker(deepcopy(tracker), i)
                        for tracker in private_trackers]
            self._private_trackers.append(trackers)

    def reset(self, hands, starting_player):
        for tracker in self._public_trackers:
            tracker.reset(starting_player)

        for player, tracker_list in enumerate(self._private_trackers):
            for tracker in tracker_list:
                tracker.reset(hands[player], starting_player)

    def pre_play_update(self, phase, player, action, pot):
        if phase == GamePhase.DURCH:
            for tracker in self._yield_all_trackers():
                tracker.durch_pre_play_update(player, action)
        elif phase == GamePhase.DECLARATION:
            for tracker in self._yield_all_trackers():
                tracker.declaration_pre_play_update(player, action)
        elif phase == GamePhase.PLAY:
            for tracker in self._yield_all_trackers():
                tracker.play_phase_pre_play_update(player, action, pot)
        elif phase == GamePhase.MOVING:
            for tracker in self._private_trackers[player]:
                tracker.moved_cards(action)
            for tracker in self._private_trackers[(player + 1) % PLAYERS]:
                tracker.received_cards(action)
        else:
            raise RuntimeError()

    def post_play_update(self, pot):
        for tracker in self._yield_all_trackers():
            tracker.post_play_update(pot)

    def _yield_all_trackers(self):
        for tracker in self._public_trackers:
            yield tracker

        for tracker_list in self._private_trackers:
            for tracker in tracker_list:
                yield tracker

    def get_observations(self, player):
        ret = {}
        for tracker in self._public_trackers:
            ret.update(tracker.get_observations(player))

        for tracker in self._private_trackers[player]:
            ret.update(tracker.get_observations())

        return ret


class PublicTracker(ABC):
    def reset(self, starting_player):
        pass

    def play_phase_pre_play_update(self, player, card, pot):
        pass

    def declaration_pre_play_update(self, player, declared_cards):
        pass

    def durch_pre_play_update(self, player, declared):
        pass

    def post_play_update(self, pot):
        pass

    @abstractmethod
    def get_observations(self, player):
        pass


class PrivateTracker(ABC):
    @abstractmethod
    def reset(self, hand, starting_player):
        pass

    def moved_cards(self, cards):
        pass

    def received_cards(self, cards):
        pass

    def play_phase_pre_play_update(self, player, card, pot):
        pass

    def declaration_pre_play_update(self, player, declared_cards):
        pass

    def durch_pre_play_update(self, player, declared):
        pass

    def post_play_update(self, pot):
        pass

    @abstractmethod
    def get_observations(self):
        pass


class RelativePrivateTracker:
    def __init__(self, tracker: PrivateTracker, player: int) -> None:
        self._tracker = tracker
        self._player = player

    @abstractmethod
    def reset(self, hand, starting_player):
        starting_player = _offset_player(starting_player, self._player)
        self._tracker.reset(hand, starting_player)

    def moved_cards(self, cards):
        self._tracker.moved_cards(cards)

    def received_cards(self, cards):
        self._tracker.received_cards(cards)

    def play_phase_pre_play_update(self, player, card, pot):
        player = _offset_player(player, self._player)
        self._tracker.play_phase_pre_play_update(player, card, pot)

    def declaration_pre_play_update(self, player, declared_cards):
        player = _offset_player(player, self._player)
        self._tracker.declaration_pre_play_update(player, declared_cards)

    def durch_pre_play_update(self, player, declared):
        player = _offset_player(player, self._player)
        self._tracker.durch_pre_play_update(player, declared)

    def post_play_update(self, pot):
        pass

    def get_observations(self):
        return self._tracker.get_observations()


class HistoryTracker(PublicTracker):
    def reset(self, starting_player):
        self._durch_phase = []
        self._declaration_phase = []
        self._play_phase = []

    def play_phase_pre_play_update(self, player, card, pot):
        self._play_phase.append((player, card))

    def declaration_pre_play_update(self, player, declared_cards):
        self._declaration_phase.append((player, declared_cards))

    def durch_pre_play_update(self, player, card):
        self._durch_phase.append((player, card))

    def get_observations(self, player):
        return {
            'declaration_history': self._declaration_phase,
            'durch_history': self._durch_phase,
            'play_history': self._play_phase
        }

    def _rotate_player(self, player, items):
        return [(_offset_player(x, player), y) for x, y in items]


class PlayedCardsTracker(PublicTracker):
    def reset(self, starting_player):
        self._played_cards = [[] for _ in range(PLAYERS)]

    def play_phase_pre_play_update(self, player, card, pot):
        self._played_cards[player].append(card)

    def get_observations(self, player):
        return {
            'played_cards': _offset_array(self._played_cards, player)
        }


class StartingPlayerTracker(PublicTracker):
    def reset(self, starting_player):
        self._starting_player = starting_player

    def get_observations(self, player):
        return {
            'starting_player': _offset_player(self._starting_player, player)
        }


class RemainingPossibleCardsTracker(PrivateTracker):
    def reset(self, hand, starting_player):
        self._possible_cards = [get_deck() for _ in range(PLAYERS - 1)]
        for card in hand:
            for i in range(PLAYERS - 1):
                self._possible_cards[i].remove(card)

    def moved_cards(self, cards):
        self._possible_cards[0].extend(cards)

    def received_cards(self, cards):
        for card in cards:
            for i in range(0, PLAYERS - 1):
                self._possible_cards[i].remove(card)

    def declaration_pre_play_update(self, player, declared_cards):
        if player == 0:
            return

        for i in range(PLAYERS - 1):
            if player == i + 1:
                continue
            for card in declared_cards:
                if card in self._possible_cards[i]:
                    self._possible_cards[i].remove(card)

    def get_observations(self):
        return {'possible_cards': self._possible_cards}


class ScoreTracker(PublicTracker):
    def reset(self, starting_player):
        self._score = [0 for _ in range(PLAYERS)]

    def post_play_update(self, pot):
        self._score[pot.get_pot_owner()] += pot.get_point_value()

    def get_observations(self, player):
        return {
            'score': _offset_array(self._score, player)
        }


class DurchEligibilityTracker(PublicTracker):
    def reset(self, starting_player):
        self._took_card = np.full(PLAYERS, False)

    def post_play_update(self, pot):
        pot_owner = pot.get_pot_owner()
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


class MovedCardsTracker(PrivateTracker):
    def reset(self, hand, starting_player):
        self._moved_cards = []

    def moved_cards(self, cards):
        self._moved_cards = cards

    def get_observations(self):
        return {'moved_cards': self._moved_cards}


class ReceivedCarsTracker(PrivateTracker):
    def reset(self, hand, starting_player):
        self._moved_cards = []

    def received_cards(self, cards):
        self._moved_cards = cards

    def get_observations(self):
        return {'received_cards': self._moved_cards}


class DoubledCardsTracker(PublicTracker):
    def reset(self, starting_player):
        self._doubled = [False, False]
        self._player = [None, None]

    def declaration_pre_play_update(self, player, declared_cards):
        if Card(1, 6) in declared_cards:
            self._doubled[0] = True
            self._player[0] = player

        if Card(2, 6) in declared_cards:
            self._doubled[1] = True
            self._player[1] = player

    def get_observations(self, player):
        return {
            'doubled': self._doubled,
            'player_doubled': self._player
        }


class DurchDeclarationTracker(PublicTracker):
    def reset(self, starting_player):
        self._declared = np.full(PLAYERS, 0.0)

    def durch_pre_play_update(self, player, declared):
        if declared:
            self._declared[player] = 1

    def get_observations(self, player):
        return {'declared_durch': _offset_array(self._declared, player)}


def _offset_array(x, player):
    return [x[(player + j) % PLAYERS] for j in range(PLAYERS)]


def _offset_player(absolute_player: int, relative_for: int) -> int:
    return (absolute_player - relative_for) % PLAYERS
