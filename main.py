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
    get_pot_value, get_pot_size_indicators, FeatureGeneratorSplitter
from learners.memory import ReplayMemory
from learners.runner import TrainRun
from learners.updaters import Step, MaximumValue
from model.model import MainNetwork, SimpleNetwork
from learners.trainers import DoubleTrainer
from learners.approximators import Torch, Buffer, SoftUpdateTorch, TargetTransformer, \
    Approximator
from learners.transformers import generate_dataset, MultiDimensionalScaler, SimpleScaler

reward = OrdinaryReward(0.5)

feature_generator_dict = {
    GamePhase.MOVING: Lambda2DEmbedder(['hand'], []),
    GamePhase.DURCH: Lambda2DEmbedder(
        ['hand',
         'moved_cards',
         'received_cards'],
        []
    ),
    GamePhase.DECLARATION: Lambda2DEmbedder(
        ['hand',
         'moved_cards',
         'received_cards'],
        [],
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
    episodes=1000,
    agent=RandomPlayer(),
    feature_generator=feature_generator
)


scaler_3d = MultiDimensionalScaler((0, 2, 3))
scaler_flat = MultiDimensionalScaler((0,))
scaler_rewards = SimpleScaler()

scaler_3d.fit(X[0])
scaler_flat.fit(X[1])
scaler_rewards.fit(y)


while True:
    model = MainNetwork([[200, 200]], 1).to("cuda")
    y = model(torch.tensor(scaler_3d.transform(X[0])).float().to("cuda"),
              torch.tensor(scaler_flat.transform(X[1])).float().to("cuda")).detach().to("cpu").numpy()
    if np.abs(y.mean()) < 0.02:
        break

approx = SoftUpdateTorch(
    tau=1e-3,
    torch_model=Torch(
        model=model,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={'lr': 3e-5},
        loss=nn.HuberLoss(delta=2),
    )
)


agent = DoubleTrainer(
    q=None,
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
    feature_transformers=[scaler_3d, scaler_flat],
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
    checkpoint_dir=Path('C:/Python/Repos/chuj/runs/double_q_baseline')
)

runner.train(16000)
