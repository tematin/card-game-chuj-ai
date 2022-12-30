from pathlib import Path
from typing import List

import torch
from torch import nn
import numpy as np

from baselines.baselines import RandomPlayer, LowPlayer, Agent
from game.environment import Tester, Environment
from game.rewards import OrdinaryReward, RewardsCombiner, DurchDeclarationPenalty
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


reward = RewardsCombiner([OrdinaryReward(0.15), DurchDeclarationPenalty(-5)])


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
    env=Environment(
        reward=reward,
        rival=RandomPlayer()
    ),
    episodes=4000,
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

yy = 10
while yy > 0.01:
    model = MainNetwork(channels=15, dense_sizes=[[150, 100], [75, 50]], depth=1).to("cuda")
    #model = SimpleNetwork().to("cuda")
    yy = model(*[torch.tensor(x[:100]).float().to("cuda") for x in X]).mean()
    yy = np.abs(yy.to("cpu").detach().numpy())



approximator = TransformedApproximator(
    approximator=SoftUpdateTorch(
        tau=1e-3,
        torch_model=Torch(
            model=model,
            loss=nn.HuberLoss(delta=1.5),
            optimizer=torch.optim.Adam,
            optimizer_kwargs={'lr': 1e-4},
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
        memory_size=20, #7000,
        extraction_count=30,
        ramp_up_size=10
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
)


runner = TrainRun(
    agent=agent,
    tester=Tester(100),
    environment=Environment(
        reward=reward,
        rival=agent
    ),
    benchmark=LowPlayer(),
    eval_freq=2000,
    #checkpoint_dir=Path('C:/Python/Repos/chuj/runs/small_whole_model')
)

runner.train(50)
timer.clear()
runner.train(100)
#runner.train(20000)
