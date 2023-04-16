import eel

from game.partial import PartialGameRound
from game.utils import Card
from pprint import pprint


agent = None


def set_agent(a):
    global agent
    agent = a


@eel.expose
def evaluate_hand(x):
    hand = [Card(xx) for xx in x]
    game = PartialGameRound(hand=hand, starting_player=0)
    observation, actions = game.observe()

    value = float(max(agent.values(observation, actions)))
    print(value)

    return value
