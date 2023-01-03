import numpy as np

from baselines.baselines import RandomPlayer
from game.environment import Environment
from game.rewards import OrdinaryReward, RewardsCombiner, DurchDeclarationPenalty, \
    DeclaredDurchRewards

reward = RewardsCombiner([
    OrdinaryReward(0.15),
    DurchDeclarationPenalty(-3),
    DeclaredDurchRewards(
        success_reward=7,
        failure_reward=-14.8,
        rival_failure_reward=3.15,
        rival_success_reward=-14
    )
])


env = Environment(
    reward=reward,
    rival=RandomPlayer(),
    run_count=1
)

o, a, r, d, i = env.reset()

while sum(d) < len(d):
    actions = [x[np.random.randint(len(x))] for x, m in zip(a, d) if not m]
    o, a, r, d, i = env.step(actions)
    print(o[0]['declared_durch'])
    print(o[0]['eligible_durch'])
    print(r)
