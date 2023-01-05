from pprint import pprint

import numpy as np

from baselines.baselines import LowPlayer
from game.constants import PLAYERS
from game.game import TrackedGameRound, Hand
from game.rewards import RewardsCombiner, OrdinaryReward, DurchDeclarationPenalty, \
    DeclaredDurchRewards, DurchReward
from game.utils import generate_hands, GamePhase, Card

reward = RewardsCombiner([
    OrdinaryReward(0.3),
    DurchDeclarationPenalty(-3),
    DeclaredDurchRewards(
            success_reward=5,
            failure_reward=-12,
            rival_failure_reward=6.3,
            rival_success_reward=-14.7
        )
])


hands = [
    [Card(0, x) for x in range(9)] + [Card(3, 3), Card(3, 7), Card(3, 8)],
    [Card(1, x) for x in range(9)] + [Card(3, 4), Card(3, 5), Card(3, 6)],
    [Card(2, x) for x in range(9)] + [Card(3, 2), Card(3, 1), Card(3, 0)],
]

hands = [
    [Card(x, 6) for x in range(4)] + [Card(x, 8) for x in range(4)] + [Card(x, 7) for x in range(4)],
    [Card(x, 1) for x in range(4)] + [Card(x, 2) for x in range(4)] + [Card(x, 0) for x in range(4)],
    [Card(x, 3) for x in range(4)] + [Card(x, 5) for x in range(4)] + [Card(x, 4) for x in range(4)],
]


#hands = generate_hands()

game = TrackedGameRound(
    starting_player=0,
    hands=hands
)
tot = 0
pp = 2

observation, actions = game.observe()
reward.reset(observation)

game.play((Card(3, 7), Card(3, 8)))
game.play((Card(3, 5), Card(3, 4)))
game.play((Card(3, 2), Card(3, 1)))


observation, actions = game.observe()

while not game.end:
    if game.phasing_player != 1 or observation['phase'] != GamePhase.PLAY:
        action = LowPlayer().play(observation, actions)
    else:
        action = actions[np.random.randint(len(actions))]

    game.play(action)
    observation, actions = game.observe()

    if game.phasing_player == pp:
        print(observation['hand'])
        r = reward.step(observation)
        print(r)
        tot += r

r = reward.step(observation)

print(game.points)
print(tot)
