import numpy as np

COLOURS = 4
VALUES = 9
PLAYERS = 3
CARDS_PER_PLAYER = 12


class Card:
    colour_to_str = {0: 'R',
                     1: 'Y',
                     2: 'G',
                     3: 'B'}
    value_to_str = {5: 'D',
                    6: 'H'}

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
        colour_code = self.colour_to_letter[self.colour]

        if self.value == 8:
            value_code = 'A'
        elif self.value == 7:
            value_code = 'K'
        elif self.value == 6:
            value_code = 'H'
        elif self.value == 5:
            value_code = 'D'
        else:
            value_code = str(self.value + 6)

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


def get_deck():
    ret = []
    for value in range(VALUES):
        for colour in range(COLOURS):
            ret.append(Card(colour, value))
    return ret


def generate_hands():
    deck = get_deck()
    np.random.shuffle(deck)
    return [Hand(deck[i:(i + CARDS_PER_PLAYER)])
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


class GameRecord:
    DURCH_SCORE = -10

    def __init__(self):
        self.history = []
        self.score = [0, 0, 0]
        self.took_card = [False, False, False]
        self.played_cards = []
        self.missed_colours = np.full((PLAYERS, COLOURS), False)

    def add_record(self, pot, card, player):
        if len(pot) > 1:
            if card.colour != pot.get_pot_colour():
                self.missed_colours[player, pot.get_pot_colour()] = True

        if pot.is_full():
            self.history.append(pot)
            self.played_cards.extend(pot)
            pot_owner = pot.get_pot_owner()
            self.score[pot_owner] += pot.get_point_value()
            self.took_card[pot_owner] = True
            if len(self.history) == CARDS_PER_PLAYER:
                self._finalize_score()

    def _finalize_score(self):
        if sum(self.took_card) == 1:
            self.score = [0, 0, 0]
            self.score[np.where(self.took_card)[0][0]] = self.DURCH_SCORE

    def can_play_durch(self, player):
        ret = True
        for i in range(PLAYERS):
            if i == player:
                continue
            ret = ret & (not self.took_card[i])


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

    def __iter__(self):
        for card in self._cards:
            yield card

    def __len__(self):
        return len(self._cards)


class GameRound:
    def __init__(self, starting_player):
        self.phasing_player = starting_player
        self.record = GameRecord()
        self.hands = generate_hands()
        self.pot = Pot(starting_player)
        self.phase = ""
        self.end = False

    def play(self, card):
        if self.end:
            raise RuntimeError("Game already ended")
        if not is_eligible_choice(self.pot, self.hands[self.phasing_player], card):
            raise RuntimeError("Foul play")
        self.hands[self.phasing_player].remove_card(card)
        self.pot.add_card(card)
        self.record.add_record(self.pot, card, self.phasing_player)
        if self.pot.is_full():
            self.phasing_player = self.pot.get_pot_owner()
            self.pot = Pot(self.phasing_player)
        else:
            self.phasing_player = advance_player(self.phasing_player)

        if self.hands[self.phasing_player].is_empty():
            self.end = True

    def observe(self):
        return Observation(self.record, self.pot, self.hands[self.phasing_player], self.phasing_player)

    def get_points(self):
        return self.record.score


class Observation:
    def __init__(self, record, pot, hand, phasing_player):
        self.record = record
        self.pot = pot
        self.hand = hand
        self.phasing_player = phasing_player
        self.eligible_choices = get_eligible_choices(pot, hand)
