import numpy as np

from baselines.baselines import RandomPlayer, Agent
from game.environment import Environment
from game.rewards import OrdinaryReward, RewardsCombiner, DurchDeclarationPenalty, \
    DeclaredDurchRewards
from game.utils import GamePhase, Card

reward_obj = RewardsCombiner([
    OrdinaryReward(0.3),
    DeclaredDurchRewards(
        success_reward=7,
        failure_reward=-14.8,
        rival_failure_reward=3.15,
        rival_success_reward=-14
    )
])

o, a, r, d, i = env.reset()

while sum(d) < len(d):
    actions = [x[np.random.randint(len(x))] for x, m in zip(a, d) if not m]
    o, a, r, d, i = env.step(actions)
    print(o[0]['declared_durch'])
    print(o[0]['eligible_durch'])
    print(r)



env = Environment(
    reward=reward_obj,
    rival=R(),
    run_count=5
)


class R(Agent):
    def play(self, observation: dict, actions):
        if observation['phase'] == GamePhase.DURCH:
            if np.random.rand() < 0.5:
                return False
            else:
                return True

        if observation['phase'] == GamePhase.DECLARATION:
            if np.random.rand() < 0.5:
                return ()

        idx = np.random.randint(len(actions))
        return actions[idx]


class A:
    def step(self, observations, actions, rewards, idx):
        print('step')
        print(rewards)
        print(idx)
        return [x[np.random.randint(len(x))] for x in actions]

    def reset(self, r, i):
        print('reset')
        print(r)
        print(i)

def _remove_done(x, mask):
    return [item for item, m in zip(x, mask) if not m]

agent = A()

observations, actions, rewards, done, idx = env.reset()
last_rewards = [0] * len(rewards)

while not all(done):
    actions_to_play = agent.step(observations, actions, rewards, idx)
    print(actions_to_play)
    observations, actions, rewards, done, idx = env.step(actions_to_play)

    for reward, d, i in zip(rewards, done, idx):
        if d:
            agent.reset(reward, i)

    observations = _remove_done(observations, done)
    actions = _remove_done(actions, done)
    rewards = _remove_done(rewards, done)
    idx = _remove_done(idx, done)

print([sum(env._games[x].points) for x in range(1)])
print([env._games[x].points for x in range(1)])

hands = [
    [Card(x, 3) for x in range(4)] + [Card(x, 5) for x in range(4)] + [
        Card(x, 4) for x in range(4)],
    [Card(x, 1) for x in range(4)] + [Card(x, 2) for x in range(4)] + [
        Card(x, 0) for x in range(4)],
    [Card(x, 6) for x in range(4)] + [Card(x, 8) for x in range(4)] + [
        Card(x, 7) for x in range(4)],
]

hands = [
    [Card(x, 6) for x in range(4)] + [Card(x, 8) for x in range(4)] + [
        Card(x, 7) for x in range(4)],
    [Card(x, 1) for x in range(4)] + [Card(x, 2) for x in range(4)] + [
        Card(x, 0) for x in range(4)],
    [Card(x, 3) for x in range(4)] + [Card(x, 5) for x in range(4)] + [
        Card(x, 4) for x in range(4)],
]
