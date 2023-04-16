import time
from copy import deepcopy
from typing import Any, Optional

import eel
import jinja2
import numpy as np

from baselines.agents import phase_one
from baselines.baselines import Agent
from game.constants import PLAYERS
from game.game import TrackedGameRound
from game.partial import PartialGameRound
from game.utils import generate_hands, Card, GamePhase, get_deck
from pprint import pprint
from .receive import get_frontend_action


def _render_template(name, **kwargs):
    template_loader = jinja2.FileSystemLoader(searchpath="./portal/templates")
    template_env = jinja2.Environment(loader=template_loader)

    template = template_env.get_template(f"{name}.html")
    return template.render(**kwargs)


def _infer_card_count(observation):
    cards = len(observation['hand'])
    card_counts = [cards] * PLAYERS

    pot_starter = observation['pot'].initial_player
    for i in range(len(observation['pot'])):
        card_counts[pot_starter] -= 1
        pot_starter = (pot_starter + 1) % PLAYERS

    return card_counts


def _fill_card_covers(cards, total_length):
    fill_length = total_length - len(cards)
    return cards + ['cover'] * fill_length


def _observation_html(observation):
    hand = observation['hand']
    possible_cards = observation['possible_cards']
    card_count = _infer_card_count(observation)

    known_cards = [
        sorted(set(possible_cards[0]) - set(possible_cards[1])),
        sorted(set(possible_cards[1]) - set(possible_cards[0]))
    ]

    all_cards = set(possible_cards[0]).union(possible_cards[1]).union(hand)
    declared = [[], [], []]
    for i in range(2):
        card = Card(i + 1, 6)
        if observation['doubled'][i] and card in all_cards:
            player = observation['player_doubled'][i]
            declared[player].append(card)
            card_count[player] -= 1
            if player != 0:
                known_cards[player - 1].remove(card)

    for i in range(2):
        known_cards[i] = _fill_card_covers(known_cards[i], card_count[i + 1])

    eligible_durch = observation['eligible_durch']
    if sum(eligible_durch) > 1:
        eligible_durch = [False, False, False]

    pot_cards = [None, None, None]
    p = observation['pot'].initial_player
    for card in observation['pot']:
        pot_cards[p] = card
        p = (p + 1) % PLAYERS

    html = _render_template(
        'observation',
        pot=pot_cards,
        first_player_cards=known_cards[0],
        second_player_cards=known_cards[1],
        first_player_board=declared[1],
        second_player_board=declared[2],
        remaining_cards=sorted(set(possible_cards[0]).union(set(possible_cards[1]))),
        declared_durch=observation['declared_durch'],
        durch_possible=eligible_durch,
        score=observation['score']
    )

    eel.update_html("observation", html)


def _controlling_observation_html(observation, phasing_player):
    hand = observation['hand']
    possible_cards = deepcopy(observation['possible_cards'])

    all_cards = set(possible_cards[0]).union(possible_cards[1]).union(hand)

    declared = [[], [], []]
    for i in range(2):
        card = Card(i + 1, 6)
        if observation['doubled'][i] and card in all_cards:
            player = observation['player_doubled'][i]
            declared[player].append(card)
            if player != 0:
                possible_cards[player - 1].remove(card)

    eligible_durch = observation['eligible_durch']
    if sum(eligible_durch) > 1:
        eligible_durch = [False, False, False]

    pot_cards = [None, None, None]
    p = observation['pot'].initial_player
    for card in observation['pot']:
        pot_cards[p] = card
        p = (p + 1) % PLAYERS

    html = _render_template(
        'a',
        pot=pot_cards,
        first_player_cards=sorted(possible_cards[0]),
        second_player_cards=sorted(possible_cards[1]),
        first_player_board=declared[1],
        second_player_board=declared[2],
        declared_durch=observation['declared_durch'],
        durch_possible=eligible_durch,
        score=observation['score'],
        first_player_selectable=(phasing_player == 1),
        second_player_selectable=(phasing_player == 2),
    )
    eel.update_html("observation", html)


def _hand_html(observation, actions=None, values=None, selectable=True):
    if values is None:
        val_dict = {}
    else:
        val_dict = {str(a): str(np.round(v, 2)) for a, v in zip(actions, values)}

    hand = observation['hand']
    hand = sorted([str(c) for c in hand])

    html = _render_template('cards', cards=hand, values=val_dict, selectable=selectable)
    eel.update_html("hand", html)


def _score_html(score):
    html = _render_template('score', score=score)
    eel.update_html("score", html)


def _moving_dictionary_values(actions, values):
    ret = []
    for action, value in zip(actions, values):
        item = [[str(x) for x in action], float(value)]
        ret.append(item)
    return ret


def _init_move(phase, actions, values):
    if phase == GamePhase.MOVING:
        _init_moving_phase_turn(actions, values)
    elif phase == GamePhase.DURCH:
        _init_durch_phase_turn(values)
    elif phase == GamePhase.DECLARATION:
        _init_declaration_phase_turn(actions, values)
    elif phase == GamePhase.PLAY:
        _init_play_phase_turn(actions)


def _init_moving_phase_turn(actions, values):
    html = _render_template(
        'buttons',
        ids=['next'],
        texts=['Potvrdit'],
        types=['primary'],
        values=['']
    )
    eel.update_html("action", html)

    value_list = _moving_dictionary_values(actions, values)
    eel.init_cards_moving_choice(value_list)


def _init_durch_phase_turn(values):
    html = _render_template(
        'buttons',
        ids=['Declare', 'Pass'],
        texts=['Vyhlasit Durcha', 'Pass'],
        types=['danger', 'primary'],
        values=[np.round(x, 2) for x in values]
    )
    eel.update_html("action", html)
    eel.init_declaration()


def _init_declaration_phase_turn(actions, values):
    html = _render_template(
        'buttons',
        ids=['next'],
        texts=['Potvrdit'],
        types=['primary'],
        values=['']
    )

    eel.update_html("action", html)

    value_list = _moving_dictionary_values(actions, values)
    eel.init_card_declaration(value_list)


def _init_play_phase_turn(actions):
    eel.update_html("action", "")
    eel.init_regular_play([str(x) for x in actions])


def _frontend_action_to_action(phase, action):
    if phase == GamePhase.MOVING:
        return [Card(x) for x in action]
    elif phase == GamePhase.DURCH:
        return action == 'true'
    elif phase == GamePhase.DECLARATION:
        return tuple([Card(x) for x in action])
    elif phase == GamePhase.PLAY:
        return Card(action[0])


def _display_finished_pot_observation(old_observation, new_observation, action, return_obs=False):
    finished_pot = deepcopy(old_observation['pot'])
    finished_pot.add_card(action)

    finished_pot_observation = deepcopy(new_observation)
    finished_pot_observation['pot'] = finished_pot

    if return_obs:
        return finished_pot_observation

    _observation_html(finished_pot_observation)
    _hand_html(finished_pot_observation)


class PlayAgainstGameManager:
    def __init__(self, starting_player: int, agent: Agent, turn_wait: float) -> None:
        self._starting_player = starting_player
        self._agent = agent
        self._turn_wait = turn_wait
        self._score = [0, 0, 0]

    def run(self) -> None:
        eel.init('portal')
        eel.start('main.html', block=False)

        self._game = TrackedGameRound(self._starting_player, generate_hands())

        while True:
            while not self._game.end:
                observation, actions = self._game.observe(player=0)
                if self._players_turn():
                    values = self._agent.values(observation, actions)

                    _observation_html(observation)
                    _hand_html(observation, actions, values)

                    _init_move(self._game.phase, actions, values)

                    action = get_frontend_action()
                    action = _frontend_action_to_action(self._game.phase, action)
                else:
                    _observation_html(observation)
                    _hand_html(observation)
                    if self._game.phase == GamePhase.PLAY:
                        eel.sleep(self._turn_wait)
                    agent_observation, agent_actions = self._game.observe()
                    action = self._agent.play(agent_observation, agent_actions)

                self._play_action(action)

            self._score = [int(x + y) for x, y in zip(self._game.points, self._score)]
            self._starting_player = (self._starting_player + 1) % PLAYERS
            self._game = TrackedGameRound(self._starting_player, generate_hands())

            _score_html(self._score)

    def _play_action(self, action: Any) -> None:
        old_observation, _ = self._game.observe(player=0)
        old_observation = deepcopy(old_observation)
        self._game.play(action)
        observation, _ = self._game.observe(player=0)

        if len(old_observation['pot']) == PLAYERS - 1:
            _display_finished_pot_observation(old_observation, observation, action)
            eel.sleep(self._turn_wait)

    def _players_turn(self) -> bool:
        return self._game.phasing_player == 0


class SimulatePlayGameManager:
    def __init__(self, starting_player: int, agent: Agent) -> None:
        self._starting_player = starting_player
        self._agent = agent

    def run(self) -> None:
        eel.init('portal')
        eel.start('main.html', block=False)

        while True:
            self._run_game_round()
            self._starting_player = (self._starting_player + 1) % PLAYERS

    def _run_game_round(self):
        self._init_game()

        while not self._game.end:
            observation, actions = self._game.observe()
            observation = deepcopy(observation)

            if self._game.phase == GamePhase.DECLARATION and len(actions) == 1:
                action = actions[0]
                self._game.play(action)
                continue

            if self._game.phasing_player == 0:
                values = self._agent.values(observation, actions)
                selectable = True
            else:
                values = []
                selectable = False
            _hand_html(observation, actions, values, selectable=selectable)
            _controlling_observation_html(observation, self._game.phasing_player)
            _init_move(self._game.phase, actions, values)

            action = get_frontend_action()
            action = _frontend_action_to_action(self._game.phase, action)

            self._game.play(action)

            if len(observation['pot']) == PLAYERS - 1:
                new_observation, _ = self._game.observe()
                mod_obs = _display_finished_pot_observation(observation, new_observation, action, return_obs=True)
                _hand_html(
                    mod_obs, actions, values,
                    selectable=self._game.phasing_player == 0
                )
                _controlling_observation_html(mod_obs, self._game.phasing_player)
                eel.sleep(1)

    def _init_game(self):
        all_cards = [[], [], [], []]
        for card in get_deck():
            all_cards[card.colour].append(card)

        html = _render_template('cards_by_row', cards_list=all_cards)
        eel.update_html("observation", html)

        html = _render_template(
            'buttons',
            ids=['next'],
            texts=['Potvrdit'],
            types=['primary'],
            values=['']
        )
        eel.update_html("action", html)

        eel.init_starting_hand()

        action = get_frontend_action()
        hand = [Card(x) for x in action]
        self._game = PartialGameRound(
            hand=hand,
            starting_player=self._starting_player
        )
