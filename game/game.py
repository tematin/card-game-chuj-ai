import numpy as np
from .constants import COLOURS, VALUES, PLAYERS, CARDS_PER_PLAYER, DURCH_SCORE
from .stat_trackers import ScoreTracker, DurchEligibilityTracker, MultiTracker


def get_deck():
    ret = []
    for value in range(VALUES):
        for colour in range(COLOURS):
            ret.append(Card(colour, value))
    return ret


def generate_hands():
    deck = get_deck()
    np.random.shuffle(deck)
    return [Hand(list(np.sort(deck[i:(i + CARDS_PER_PLAYER)])))
            for i in range(0, CARDS_PER_PLAYER * PLAYERS, CARDS_PER_PLAYER)]


def advance_player(player, moves=1):
    return (player + moves) % PLAYERS


def is_eligible_choice(pot, hand, card):
    if pot.is_empty():
        return True
    elif card.colour == pot.get_pot_colour():
        return True
    else:
        if not hand.has_colour(pot.get_pot_colour()):
            return True
        else:
            return False


def get_eligible_choices(pot, hand):
    if pot.is_empty():
        return hand.hand
    else:
        pot_colour = pot.get_pot_colour()
        if hand.has_colour(pot_colour):
            return [c for c in hand if c.colour == pot_colour]
        else:
            return hand.hand


class Card:
    colour_to_str = {0: '♥',
                     1: '☘',
                     2: '⛊',
                     3: '⚈'}
    value_to_str = {5: 'D',
                    6: 'H',
                    7: 'K',
                    8: 'A'}

    def __init__(self, colour, value):
        if not 0 <= value < VALUES:
            raise ValueError("Wrong value specified")
        if not 0 <= colour < COLOURS:
            raise ValueError("Wrong colour specified")
        self.colour = colour
        self.value = value
        self.multiplier = 1

    def is_higher_value(self, other):
        if other.colour == self.colour:
            return other.value >= self.value
        return False

    def __int__(self):
        return self.colour * 9 + self.value

    def __eq__(self, other):
        return (self.colour == other.colour) & (self.value == other.value)

    def __lt__(self, other):
        return int(self) < int(other)

    def __hash__(self):
        return int(self)

    def get_point_value(self):
        if self.colour == 0:
            val = 1
        elif self.colour == 1 and self.value == 6:
            val = 4
        elif self.colour == 2 and self.value == 6:
            val = 8
        else:
            val = 0
        return val * self.multiplier

    def __repr__(self):
        colour_code = self.colour_to_str[self.colour]
        value_code = self.value_to_str.get(self.colour, str(self.value + 6))

        point_value = self.get_point_value()
        if point_value > 2:
            point_code = '#'
        else:
            point_code = ''

        return point_code + colour_code + value_code


class Hand:
    def __init__(self, hand):
        if not isinstance(hand, list):
            raise TypeError("Wrong type instantiation")
        self.hand = hand

    def remove_card(self, card):
        self.hand.remove(card)

    def has_colour(self, colour):
        for card in self.hand:
            if card.colour == colour:
                return True
        return False

    def is_empty(self):
        return len(self.hand) == 0

    def __getitem__(self, item):
        return self.hand[item]

    def __iter__(self):
        for card in self.hand:
            yield card

    def __repr__(self):
        return '{' + ', '.join([c.__repr__() for c in self.hand]) + '}'


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

    def __iter__(self):
        for card in self._cards:
            yield card

    def __len__(self):
        return len(self._cards)


class GameRound:
    def __init__(self, starting_player):
        self.phasing_player = starting_player
        self.hands = generate_hands()
        self.pot = Pot(starting_player)
        self.phase = ""
        self.end = False
        self.trick_end = False

        self.tracker = MultiTracker()

    def play(self, card=None):
        if self.end:
            raise RuntimeError("Game already ended")

        if self.trick_end:
            if card is not None:
                raise RuntimeError("Trick ended no cards allowed")
            self._clear()
        else:
            if card is None:
                raise RuntimeError("Card required")
            if not is_eligible_choice(self.pot, self.hands[self.phasing_player], card):
                raise RuntimeError("Foul play")
            self._play(card)

    def _play(self, card):
        self.hands[self.phasing_player].remove_card(card)
        self.pot.add_card(card)

        self.tracker.callback(self.pot, card, self.phasing_player)

        if self.pot.is_full():
            self.trick_end = True
            self.phasing_player = 0
        else:
            self.phasing_player = advance_player(self.phasing_player)

    def _clear(self):
        if self.phasing_player == 2:
            self.phasing_player = self.pot.get_pot_owner()
            self.pot = Pot(self.phasing_player)

            self.trick_end = False

            if self.hands[self.phasing_player].is_empty():
                self.end = True
        else:
            self.phasing_player = advance_player(self.phasing_player)

    def observe(self):
        return Observation(tracker=self.tracker,
                           pot=self.pot,
                           hand=self.hands[self.phasing_player],
                           phasing_player=self.phasing_player,
                           right_hand=self.hands[advance_player(self.phasing_player)],
                           left_hand=self.hands[advance_player(self.phasing_player, 2)])

    def requires_action(self):
        return not self.trick_end

    def get_points(self):
        if self.end and sum(self.tracker.durch.took_card) == 1:
            score = np.full(PLAYERS, 0)
            score[np.argmax(self.tracker.durch.took_card)] = DURCH_SCORE
            return score
        else:
            return self.tracker.score.score


class Observation:
    def __init__(self, tracker, pot, hand, phasing_player, right_hand, left_hand):
        self.tracker = tracker
        self.pot = pot
        self.hand = hand
        self.phasing_player = phasing_player
        if pot.is_full():
            self.eligible_choices = [None]
        else:
            self.eligible_choices = get_eligible_choices(pot, hand)

        self.right_hand = right_hand
        self.left_hand = left_hand


