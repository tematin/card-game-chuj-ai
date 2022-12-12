from pathlib import Path
from typing import List

import torch
from torch import nn
import numpy as np

from baselines.baselines import RandomPlayer, LowPlayer, Agent
from game.environment import Tester, Environment
from game.rewards import OrdinaryReward
from game.utils import GamePhase
from learners.explorers import EpsilonGreedy, ExplorationCombiner, Random, Softmax, ExplorationSwitcher
from learners.feature_generators import Lambda2DEmbedder, get_highest_pot_card, \
    get_pot_value, get_pot_size_indicators, FeatureGeneratorSplitter, generate_dataset
from learners.memory import ReplayMemory
from learners.runner import TrainRun
from learners.updaters import Step, MaximumValue
from model.model import MainNetwork, SimpleNetwork, SimpleMainNetwork
from learners.trainers import DoubleTrainer
from learners.approximators import Torch, Buffer, SoftUpdateTorch, TargetTransformer, \
    Approximator, ApproximatorSplitter
from learners.transformers import MultiDimensionalScaler, SimpleScaler, ListTransformer

reward = OrdinaryReward(0.3)

feature_generator_dict = {
    GamePhase.MOVING: Lambda2DEmbedder(['hand']),
    GamePhase.DURCH: Lambda2DEmbedder(
        ['hand',
         'moved_cards',
         'received_cards'],
        [],
        action_list=[0, 1]
),
    GamePhase.DECLARATION: Lambda2DEmbedder(
        ['hand',
         'moved_cards',
         'received_cards'],
    ),
    GamePhase.PLAY: Lambda2DEmbedder(
        [get_highest_pot_card,
         'hand',
         'pot',
         'possible_cards',
         'received_cards'],
        [get_pot_value,
         get_pot_size_indicators,
         'score',
         'doubled',
         'eligible_durch',
         'declared_durch'],
    ),
}

X, y = generate_dataset(
    env=Environment(
        reward=reward,
        rival=RandomPlayer()
    ),
    episodes=10000,
    agent=RandomPlayer(),
    feature_generator=FeatureGeneratorSplitter(feature_generator_dict)
)


transformers = {
    GamePhase.MOVING: ListTransformer([MultiDimensionalScaler((0, 2, 3))]),
    GamePhase.DURCH: ListTransformer([MultiDimensionalScaler((0, 2, 3)), SimpleScaler()]),
    GamePhase.DECLARATION: ListTransformer([MultiDimensionalScaler((0, 2, 3))]),
    GamePhase.PLAY: ListTransformer([MultiDimensionalScaler((0, 2, 3)), MultiDimensionalScaler((0,))])
}

transformer_dictionary = {}
approximators_dictionary = {}

models = {
    GamePhase.DURCH: MainNetwork(dense_sizes=[[200, 100], [100, 50]], depth=1),
    GamePhase.MOVING: SimpleMainNetwork(dense_sizes=[[200, 100], [100, 50]], depth=1),
    GamePhase.DECLARATION: SimpleMainNetwork(dense_sizes=[[200, 100], [100, 50]], depth=1),
    GamePhase.PLAY: MainNetwork(dense_sizes=[[200, 200], [200, 100]], depth=2)
}

for key in GamePhase:
    transformers[key].fit(X[key])

    target_transformer = SimpleScaler()
    target_transformer.fit(y[key])

    transformer = [Buffer(128), TargetTransformer(target_transformer)]
    transformer_dictionary[key] = transformer

    models[key] = SoftUpdateTorch(
        tau=1e-3,
        torch_model=Torch(
            model=models[key],
            loss=nn.HuberLoss(delta=1.5),
            optimizer=torch.optim.Adam,
            optimizer_kwargs={'lr': 1e-4},
        )
    )


feature_generator = FeatureGeneratorSplitter(feature_generator_dict, transformers)


approximator = ApproximatorSplitter(
    approximator_dictionary=models,
    transformer_dictionary=transformer_dictionary
)


agent = DoubleTrainer(
    q=approximator,
    memory=ReplayMemory(
        yield_length=2,
        memory_size=2000,
        extraction_count=2,
        ramp_up_size=1000
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
#    checkpoint_dir=Path('C:/Python/Repos/chuj/runs/double_q_baseline')
)

runner.train(16000)



took = 42 / 3
total = 42

took = 21 / 3
total = 21


alpha = 0.33333

given = total - took
print(given * alpha - took * (1 - alpha))









