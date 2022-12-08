from pathlib import Path
from typing import List

import torch
from torch import nn
import numpy as np

from baselines.baselines import RandomPlayer, LowPlayer, Agent
from game.environment import Tester, Environment
from game.rewards import OrdinaryReward
from game.stat_trackers import ScoreTracker, DurchEligibilityTracker, \
    RemainingPossibleCards
from learners.explorers import EpsilonGreedy, ExplorationCombiner, Random, Softmax, ExplorationSwitcher
from learners.feature_generators import Lambda2DEmbedder, get_highest_pot_card, \
    get_hand, get_pot_cards, get_pot_value, get_pot_size_indicators, get_eligible_durch, \
    get_current_score, get_possible_cards
from learners.memory import ReplayMemory
from learners.runner import TrainRun
from learners.updaters import Step, MaximumValue
from model.model import MainNetwork, SimpleNetwork
from learners.trainers import SimpleTrainer, DoubleTrainer
from learners.approximators import Torch, Buffer, SoftUpdateTorch, TargetTransformer, \
    Approximator
from learners.transformers import generate_dataset, MultiDimensionalScaler, SimpleScaler

reward = OrdinaryReward(0.5)
trackers = [ScoreTracker(), DurchEligibilityTracker(), RemainingPossibleCards()]

feature_generator = Lambda2DEmbedder(
    [get_highest_pot_card,
     get_hand,
     get_pot_cards,
     get_possible_cards(0),
     get_possible_cards(1)],
    [get_pot_value,
     get_pot_size_indicators,
     get_current_score,
     get_eligible_durch],
)

X, y = generate_dataset(
    env=Environment(
        reward=reward,
        trackers=trackers,
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
    q=Buffer(
        buffer_size=256,
        approximator=TargetTransformer(
            transformer=scaler_rewards,
            approximator=approx
        )
    ),
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

agent.load(Path('C:/Python/Repos/chuj/runs/double_q_baseline_extention/episode_12000'))


runner = TrainRun(
    agent=agent,
    tester=Tester(
        100,
        trackers=trackers
    ),
    environment=Environment(
        reward=reward,
        trackers=trackers,
        rival=agent
    ),
    benchmark=LowPlayer(),
    eval_freq=2000,
    checkpoint_dir=Path('C:/Python/Repos/chuj/runs/double_q_baseline_extension_2')
)

runner.train(16000)
