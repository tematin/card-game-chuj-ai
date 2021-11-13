import numpy as np
from .constants import COLOURS, PLAYERS


class SuperTracker:
    def callback(self, pot, card, player):
        pass


class HistoryTracker(SuperTracker):
    def __init__(self):
        self.history = []

    def callback(self,  pot, card, player):
        if pot.is_full():
            self.history.append(pot)


class ScoreTracker(SuperTracker):
    def __init__(self):
        self.score = np.full(PLAYERS, 0)

    def callback(self, pot, card, player):
        if pot.is_full():
            pot_owner = pot.get_pot_owner()
            self.score[pot_owner] += pot.get_point_value()


class PlayedCardsTracker(SuperTracker):
    def __init__(self):
        self.played_cards = []

    def callback(self,  pot, card, player):
        self.played_cards.append(card)


class MissedColoursTracker(SuperTracker):
    def __init__(self):
        self.missed_colours = np.full((PLAYERS, COLOURS), False)

    def callback(self, pot, card, player):
        if len(pot) > 1:
            if card.colour != pot.get_pot_colour():
                self.missed_colours[player,
                                    pot.get_pot_colour()] = True


class DurchEligibilityTracker:
    def __init__(self):
        self.took_card = np.full(PLAYERS, False)

    def callback(self, pot, card, player):
        if pot.is_full():
            pot_owner = pot.get_pot_owner()
            self.took_card[pot_owner] = True

    def can_play_durch(self, player):
        ret = True
        for i in range(PLAYERS):
            if i == player:
                continue
            ret = ret & (not self.took_card[i])
        return ret


class MultiTracker(SuperTracker):
    def __init__(self):
        self.score = ScoreTracker()
        self.durch = DurchEligibilityTracker()
        self.missed_colours = MissedColoursTracker()
        self.played_cards = PlayedCardsTracker()
        self.history = HistoryTracker()

        self.trackers = [self.score,
                         self.durch,
                         self.missed_colours,
                         self.played_cards,
                         self.history]

    def callback(self, pot, card, player):
        for tracker in self.trackers:
            tracker.callback(pot, card, player)
