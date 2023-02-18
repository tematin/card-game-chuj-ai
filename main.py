from copy import deepcopy
from pathlib import Path
from typing import List

import torch
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from baselines.baselines import RandomPlayer, LowPlayer, Agent
from game.environment import Tester, Environment, OneThreadEnvironment, RewardTester
from game.rewards import OrdinaryReward, RewardsCombiner, DurchDeclarationPenalty, \
    DeclaredDurchRewards, DurchReward
from game.utils import GamePhase
from learners.explorers import EpsilonGreedy, ExplorationCombiner, Random, Softmax, ExplorationSwitcher
from learners.feature_generators import Lambda2DEmbedder, get_highest_pot_card, \
    get_pot_value, get_pot_size_indicators, generate_dataset, get_pot_card, \
    get_card_by_key, get_possible_cards, get_moved_cards, get_play_phase_action, \
    get_flat_by_key, get_declaration_phase_action, get_durch_phase_action, \
    TransformedFeatureGenerator, get_cards_remaining
from learners.memory import ReplayMemory
from learners.representation import index_observation
from learners.runner import TrainRun, LeagueTrainRun, AgentLeague
from learners.updaters import Step, MaximumValue, UpdateStep, ClippedMaximumValue
from model.model import MainNetwork, SimpleNetwork, MainNetworkV2
from learners.trainers import DoubleTrainer, TrainedDoubleQ
from learners.approximators import Torch, Buffer, SoftUpdateTorch, TargetTransformer, \
    Approximator, TransformedApproximator
from learners.transformers import MultiDimensionalScaler, SimpleScaler, ListTransformer
from debug.timer import timer


def get_agent(path):
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
         get_cards_remaining,
         get_durch_phase_action,
         get_declaration_phase_action
         ],
    )

    feature_transformer = ListTransformer([
        MultiDimensionalScaler((0, 2, 3)),
        MultiDimensionalScaler((0,))]
    )

    target_transformer = SimpleScaler()

    model = MainNetworkV2(channels=60, dense_sizes=[[256, 256], [128, 128], [64, 32]], depth=2).to("cuda")

    X = [np.random.rand(64, 8, 4, 9), np.random.rand(64, 32)]
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
                scheduler_kwargs={'step_size': 100, 'gamma': 0.999}
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

    agent = TrainedDoubleQ(
        approximators=(approximator, approximator),
        feature_generator=feature_generator,
    )

    agent.load(path)

    return agent


reward = RewardsCombiner([
    OrdinaryReward(1 / 3),
    DeclaredDurchRewards(
            success_reward=13 + 1 / 3,
            failure_reward=-13 - 1 / 3,
            rival_failure_reward=6 + 2 / 3,
            rival_success_reward=-6 - 2 / 3
        ),
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
     get_cards_remaining,
     get_durch_phase_action,
     get_declaration_phase_action,
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
#model = MainNetworkV2(channels=60, dense_sizes=[[256, 256], [128, 128], [64, 32]], depth=2).to("cuda")
model = MainNetworkV2(channels=120, dense_sizes=[[256, 256, 256], [128, 128, 128], [64, 64, 32]],
                      depth=3, direct_channels=60).to("cuda")
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
            scheduler_kwargs={'step_size': 100, 'gamma': 0.999}
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
        memory_size=8000,
        extraction_count=40,
        ramp_up_size=3000
    ),
    updater=Step(discount=1),
    value_calculator=ClippedMaximumValue(),
    feature_generator=feature_generator,
    explorer=ExplorationCombiner([Random(), Softmax(0.4)], [0.12, 0.88]),
    run_count=run_count,
    assign_rule='both'
)


league = AgentLeague(
    seed_agents=[
        deepcopy(agent)
    ],
    game_count=80, max_agents=4
)


runner = LeagueTrainRun(
    agent=agent,
    testers={
        '5_190': Tester(30, get_agent(Path('runs/baseline_run_5/episode_190000'))),
        '8_160': Tester(30, get_agent(Path('runs/baseline_run_8/episode_190000'))),
        '8_190': Tester(30, get_agent(Path('runs/baseline_run_10/episode_190000'))),
    },
    reward=reward,
    eval_freq=10000,
    run_count=run_count,
    checkpoint_dir=Path('runs/baseline_run_13').absolute(),
    league=league
)

runner.train(18000)







