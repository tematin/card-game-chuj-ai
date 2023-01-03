from pathlib import Path
from typing import List

import torch
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from baselines.agents import first_baseline
from baselines.baselines import RandomPlayer, LowPlayer, Agent
from game.environment import Tester, Environment, OneThreadEnvironment, \
    analyze_game_round, RewardTester
from game.rewards import OrdinaryReward, RewardsCombiner, DurchDeclarationPenalty, \
    DeclaredDurchRewards, DurchReward
from game.utils import GamePhase
from learners.explorers import EpsilonGreedy, ExplorationCombiner, Random, Softmax, ExplorationSwitcher
from learners.feature_generators import Lambda2DEmbedder, get_highest_pot_card, \
    get_pot_value, get_pot_size_indicators, generate_dataset, get_pot_card, \
    get_card_by_key, get_possible_cards, get_moved_cards, get_play_phase_action, \
    get_flat_by_key, get_declaration_phase_action, get_durch_phase_action, \
    TransformedFeatureGenerator
from learners.memory import ReplayMemory
from learners.runner import TrainRun
from learners.updaters import Step, MaximumValue
from model.model import MainNetwork, SimpleNetwork
from learners.trainers import DoubleTrainer
from learners.approximators import Torch, Buffer, SoftUpdateTorch, TargetTransformer, \
    Approximator, TransformedApproximator
from learners.transformers import MultiDimensionalScaler, SimpleScaler, ListTransformer
from debug.timer import timer


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


X, y = generate_dataset(
    env=OneThreadEnvironment(
        reward=reward,
        rival=RandomPlayer(),
    ),
    episodes=10000,
    agent=RandomPlayer(),
    feature_generator=base_embedder
)

feature_transformer = ListTransformer([
    MultiDimensionalScaler((0, 2, 3)),
    MultiDimensionalScaler((0,))]
)
feature_transformer.fit(X)

target_transformer = SimpleScaler()
target_transformer.fit(y)

#model = MainNetwork(channels=45, dense_sizes=[[256, 256], [64, 32]], depth=1).to("cuda")
model = MainNetwork(channels=60, dense_sizes=[[256, 256], [128, 128], [64, 32]], depth=2).to("cuda")
model(*[torch.tensor(x[:100]).float().to("cuda") for x in X]).mean()


approximator = TransformedApproximator(
    approximator=SoftUpdateTorch(
        tau=1e-3,
        torch_model=Torch(
            model=model,
            loss=nn.HuberLoss(delta=1.5),
            optimizer=torch.optim.Adam,
            optimizer_kwargs={'lr': 6e-4},
            scheduler=StepLR,
            scheduler_kwargs={'step_size': 100, 'gamma': 0.996}
        ),
    ),
    transformers=[
        Buffer(128),
        TargetTransformer(target_transformer)
    ]
)

feature_generator = TransformedFeatureGenerator(
    feature_transformer=feature_transformer,
    feature_generator=base_embedder
)


agent = DoubleTrainer(
    q=approximator,
    memory=ReplayMemory(
        yield_length=2,
        memory_size=400,
        extraction_count=40,
        ramp_up_size=300
    ),
    updater=Step(discount=1),
    value_calculator=MaximumValue(),
    feature_generator=feature_generator,
    explorer=ExplorationSwitcher(
        probabilities=[0.95, 0.05],
        explorations=[
            ExplorationCombiner([Random(), Softmax(0.4)], [0.1, 0.9]),
            Softmax(0.03)
        ]
    ),
    run_count=run_count
)

runner = TrainRun(
    agent=agent,
    testers={
        'low_player': Tester(80, LowPlayer()),
        'reward_low': RewardTester(reward, LowPlayer(), run_count=10, episodes=600),
        'reward_agent': RewardTester(reward, first_baseline(), run_count=10, episodes=600),
    },
    environment=Environment(
        reward=reward,
        rival=agent,
        run_count=run_count
    ),
    eval_freq=10000,
    run_count=run_count,
    checkpoint_dir=Path('runs/baseline_run_2').absolute()
)

runner.train(30000)
