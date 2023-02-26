from copy import deepcopy

import eel
import jinja2
import numpy as np

from baselines.agents import phase_one
from game.constants import PLAYERS
from game.game import TrackedGameRound
from game.utils import generate_hands, Card, GamePhase


@eel.expose
def log(x):
    print(x)


action = None


@eel.expose
def register_action(x):
    global action
    action = x


def get_frontend_action():
    global action

    action = None
    while action is None:
        eel.sleep(0.1)

    return action


def render_template(name, **kwargs):
    template_loader = jinja2.FileSystemLoader(searchpath="./portal/templates")
    template_env = jinja2.Environment(loader=template_loader)

    template = template_env.get_template(f"{name}.html")
    return template.render(**kwargs)


def infer_card_count(observation):
    cards = len(observation['hand'])
    card_counts = [cards] * PLAYERS

    pot_starter = observation['pot'].initial_player
    for i in range(len(observation['pot'])):
        card_counts[pot_starter] -= 1
        pot_starter = (pot_starter + 1) % PLAYERS

    return card_counts


def fill_card_covers(cards, total_length):
    fill_length = total_length - len(cards)
    return cards + ['cover'] * fill_length


def display_observations(observation):
    hand = observation['hand']
    possible_cards = observation['possible_cards']
    card_count = infer_card_count(observation)

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
        known_cards[i] = fill_card_covers(known_cards[i], card_count[i + 1])

    eligible_durch = observation['eligible_durch']
    if sum(eligible_durch) > 1:
        eligible_durch = [False, False, False]

    pot_cards = [None, None, None]
    p = observation['pot'].initial_player
    for card in observation['pot']:
        pot_cards[p] = card
        p = (p + 1) % PLAYERS

    observation_html = render_template(
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

    eel.update_html("observation", observation_html)


def display_hand(observation, actions, values=[]):
    val_dict = {str(a): str(np.round(v, 2)) for a, v in zip(actions, values)}

    hand = observation['hand']
    hand = sorted([str(c) for c in hand])

    output = render_template('cards', cards=hand, values=val_dict)
    eel.update_html("hand", output)


def add_action_to_pot(observation, action):
    observation['pot'] = deepcopy(observation['pot'])
    observation['pot'].add_card(action)
    return observation


def remove_action_from_hand(observation, action):
    observation['hand'] = deepcopy(observation['hand'])
    observation['hand'].remove_card(action)
    return observation


def play_others(game, agent):
    while game.phasing_player != 0:
        action = agent.play(*game.observe())
        game.play(action)


def moving_dictionary_values(actions, values):
    ret = []
    for action, value in zip(actions, values):
        item = [[str(x) for x in action], float(value)]
        ret.append(item)
    print(ret)
    return ret


agent = phase_one('10_190')

#agent = LowPlayer()

eel.init('portal')
eel.start('main.html', block=False)


score = [0, 0, 0]
starting = -1

while True:
    out = render_template('score', score=score)
    eel.update_html("score", out)

    starting = (starting + 1) % PLAYERS
    game = TrackedGameRound(starting, generate_hands())
    play_others(game, agent)

    while not game.end:
        observation, actions = game.observe()
        values = agent.debug(observation, actions)['q_avg']
        display_observations(observation)
        display_hand(observation, actions, values)

        if game.phase == GamePhase.MOVING:
            out = render_template(
                'buttons',
                ids=['next'],
                texts=['Potvrdit'],
                types=['primary'],
                values=['']
            )
            val_list = moving_dictionary_values(actions, values)

            eel.update_html("action", out)
            eel.init_cards_moving_choice(val_list)

            actions = get_frontend_action()
            game.play([Card(x) for x in actions])

        elif game.phase == GamePhase.DURCH:
            out = render_template(
                'buttons',
                ids=['Declare', 'Pass'],
                texts=['Vyhlasit Durcha', 'Pass'],
                types=['danger', 'primary'],
                values=[np.round(x, 2) for x in values]
            )
            eel.update_html("action", out)
            eel.init_declaration()

            action = get_frontend_action()
            game.play(action == 'true')

        elif game.phase == GamePhase.DECLARATION:
            if len(actions) == 1:
                action = actions[0]
            else:
                out = render_template(
                    'buttons',
                    ids=['next'],
                    texts=['Potvrdit'],
                    types=['primary'],
                    values=['']
                )

                eel.update_html("action", out)

                val_list = moving_dictionary_values(actions, values)
                eel.init_card_declaration(val_list)

                action = get_frontend_action()
                action = tuple([Card(x) for x in action])
            game.play(action)

        elif game.phase == GamePhase.PLAY:
            eel.update_html("action", "")
            eel.init_regular_play([str(x) for x in actions])

            action = get_frontend_action()
            action = Card(action[0])
            if len(observation['pot']) == PLAYERS - 1:
                observation = add_action_to_pot(observation, action)
                observation = remove_action_from_hand(observation, action)
                display_observations(observation)
                display_hand(observation, actions)
                eel.sleep(1)
            game.play(action)

        while game.phasing_player != 0 and not game.end:
            if game.phase == GamePhase.PLAY:
                zero_obs, actions = game.observe(player=0)
                display_observations(zero_obs)
                display_hand(zero_obs, actions)
                eel.sleep(1)
            observation, actions = game.observe()
            action = agent.play(observation, actions)

            if len(observation['pot']) == PLAYERS - 1:
                zero_obs, _ = game.observe(player=0)
                zero_obs = add_action_to_pot(zero_obs, action)
                display_observations(zero_obs)
                eel.sleep(1)

            game.play(action)

    score = [int(x + y) for x, y in zip(game.points, score)]
