from copy import deepcopy
from pprint import pprint
from typing import List, Any

import numpy as np
from tqdm import tqdm

from baselines.agents import phase_one
from baselines.baselines import LowPlayer, Agent
from game.constants import CARDS_PER_PLAYER, PLAYERS
from game.environment import Tester
from game.game import TrackedGameRound, Hand
from game.rewards import OrdinaryReward, DeclaredDurchRewards, RewardsCombiner, Reward
from game.utils import generate_hands, GamePhase

from game.old_game import TrackedGameRound as OldTrackedGameRound

from learners.explorers import Explorer, Softmax, ExplorationCombiner, Random
from learners.trainers import ValueAgent

agent = phase_one('10_190')

hands = generate_hands()

game = TrackedGameRound(
    hands=deepcopy(hands),
    starting_player=1
)

old_game = OldTrackedGameRound(
    hands=deepcopy(hands),
    starting_player=1
)

while not game.end:
    observation, actions = game.observe()
    old_observation, old_actions = old_game.observe()

    for key in observation:
        if key == 'pot':
            equal = observation[key]._cards == old_observation[key]._cards
        elif key == 'hand':
            equal = set(observation[key]) == set(old_observation[key])
        else:
            equal = observation[key] == old_observation[key]
        if not equal:
            print(key)
            print(observation[key])
            print(old_observation[key])
            assert False

    action = agent.play(observation, actions)
    game.play(action)
    old_game.play(action)


pprint(observation, width=140)
pprint(old_observation, width=140)
