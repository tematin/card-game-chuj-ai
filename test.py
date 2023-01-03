from copy import deepcopy
from pathlib import Path
from typing import List, Any

import torch
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from tqdm import tqdm

from baselines.baselines import RandomPlayer, LowPlayer, Agent
from game.environment import Tester, Environment, OneThreadEnvironment, \
    analyze_game_round
from game.rewards import OrdinaryReward, RewardsCombiner, DurchDeclarationPenalty, \
    DeclaredDurchRewards, DurchReward
from game.utils import GamePhase
from learners.explorers import EpsilonGreedy, ExplorationCombiner, Random, Softmax, ExplorationSwitcher
from learners.feature_generators import Lambda2DEmbedder, get_highest_pot_card, \
    get_pot_value, get_pot_size_indicators, generate_dataset, get_pot_card, \
    get_card_by_key, get_possible_cards, get_moved_cards, get_play_phase_action, \
    get_flat_by_key, get_declaration_phase_action, get_durch_phase_action, \
    TransformedFeatureGenerator, FeatureGenerator
from learners.memory import ReplayMemory
from learners.runner import TrainRun
from learners.updaters import Step, MaximumValue
from model.model import MainNetwork, SimpleNetwork
from learners.trainers import DoubleTrainer, TrainedDoubleQ
from learners.approximators import Torch, Buffer, SoftUpdateTorch, TargetTransformer, \
    Approximator, TransformedApproximator
from learners.transformers import MultiDimensionalScaler, SimpleScaler, ListTransformer
from debug.timer import timer


target_transformer = SimpleScaler()

#model = MainNetwork(channels=45, dense_sizes=[[256, 256], [64, 32]], depth=1).to("cuda")
model = MainNetwork(channels=60, dense_sizes=[[256, 256], [128, 128], [64, 32]], depth=2).to("cuda")
X = [np.random.rand(16, 8, 4, 9), np.random.rand(16, 21)]
model(*[torch.tensor(x).float().to("cuda") for x in X]).mean()


approximator = TransformedApproximator(
    approximator=SoftUpdateTorch(
        tau=1e-3,
        torch_model=Torch(
            model=model,
            loss=nn.HuberLoss(delta=1.5),
            optimizer=torch.optim.Adam,
            optimizer_kwargs={'lr': 6e-4},
            scheduler=StepLR,
            scheduler_kwargs={'step_size': 100, 'gamma': 0.995}
        ),
    ),
    transformers=[
        Buffer(128),
        TargetTransformer(target_transformer)
    ]
)

feature_transformer = ListTransformer([
    MultiDimensionalScaler((0, 2, 3)),
    MultiDimensionalScaler((0,))]
)

base_embedder = Lambda2DEmbedder(
    [get_pot_card(0),
     get_pot_card(1),
     get_card_by_key('hand'),
     get_possible_cards(0),
     get_possible_cards(1),
     get_card_by_key('received_cards'),
     get_moved_cards,
     get_play_phase_action
     ],
    [get_pot_value,
     get_pot_size_indicators,
     get_flat_by_key('score'),
     get_flat_by_key('doubled'),
     get_flat_by_key('eligible_durch'),
     get_flat_by_key('declared_durch'),
     get_durch_phase_action,
     get_declaration_phase_action
     ],
)

feature_generator = TransformedFeatureGenerator(
    feature_transformer=feature_transformer,
    feature_generator=base_embedder
)

agent = TrainedDoubleQ(approximator, feature_generator)
agent.load(Path('baselines/first_baseline'))

agent2 = TrainedDoubleQ(approximator, feature_generator)
agent2.load(Path('runs/baseline_run_2/episode_75000'))


class RewardTester:
    def __init__(self, env: Environment, run_count: int):
        self._env = env
        self._run_count = run_count

    def _remove_done(self, x: List, mask: List[bool]) -> List:
        return [item for item, m in zip(x, mask) if not m]

    def _run(self, agent: Agent) -> float:
        total_rewards = 0
        observations, actions, _, done, idx = self._env.reset()

        while not all(done):
            actions_to_play = agent.parallel_play(observations, actions)
            observations, actions, rewards, done, _ = self._env.step(actions_to_play)
            total_rewards += sum(rewards)

            observations = self._remove_done(observations, done)
            actions = self._remove_done(actions, done)

        return total_rewards

    def evaluate(self, agent: Agent, episodes: int, verbose: bool = False) -> float:
        total_reward = 0

        total_iterations = int(episodes / self._run_count)

        iter_item = range(total_iterations)
        if verbose:
            iter_item = tqdm(iter_item)

        for _ in iter_item:
            total_reward += self._run(agent)
        return total_reward / (total_iterations * self._run_count)


reward = RewardsCombiner([
    OrdinaryReward(0.3),
    DurchDeclarationPenalty(-3),
    DeclaredDurchRewards(
            success_reward=5,
            failure_reward=-12,
            rival_failure_reward=6.3,
            rival_success_reward=-14.7
        ),
    DurchReward(reward=20, rival_reward=-13)
])

run_count = 10

reward_tester = RewardTester(
    env=Environment(
        reward=reward,
        rival=agent,
        run_count=run_count),
    run_count=run_count)


low_reward_tester = RewardTester(
    env=Environment(
        reward=reward,
        rival=LowPlayer(),
        run_count=run_count),
    run_count=run_count)

random_tester = RewardTester(
    env=Environment(
        reward=reward,
        rival=RandomPlayer(),
        run_count=run_count),
    run_count=run_count)

print(reward_tester.evaluate(agent, 500, verbose=True))
print(low_reward_tester.evaluate(agent, 500, verbose=True))
print(random_tester.evaluate(agent, 2000, verbose=True))

print(low_reward_tester.evaluate(LowPlayer(), 500, verbose=True))
print(reward_tester.evaluate(LowPlayer(), 500, verbose=True))

print(reward_tester.evaluate(agent2, 1000, verbose=True))


env = Environment(
    reward=reward,
    rival=agent,
    run_count=1)

atook = []
agiven = []

for i in tqdm(range(100)):
    total_rewards = 0
    observations, actions, _, done, idx = env.reset()

    while not all(done):
        actions_to_play = agent.parallel_play(observations, actions)
        observations, actions, rewards, done, _ = env.step(actions_to_play)
        total_rewards += sum(rewards)

    score = observations[0]['score']

    agiven.append(score[2] + score[1])
    atook.append(score[0])

    print(score)
    print(observations[0]['declared_durch'])
    print(observations[0]['eligible_durch'])

print(np.mean(atook))
print(np.mean(agiven))