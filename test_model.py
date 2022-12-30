from collections import defaultdict
from pathlib import Path
from typing import List

import torch
from torch import nn
import numpy as np
from tqdm import tqdm

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


X_train, y_train = generate_dataset(
    env=Environment(
        reward=reward,
        rival=RandomPlayer()
    ),
    episodes=10000,
    agent=RandomPlayer(),
    feature_generator=base_embedder
)


X_valid, y_valid = generate_dataset(
    env=Environment(
        reward=reward,
        rival=RandomPlayer()
    ),
    episodes=1000,
    agent=RandomPlayer(),
    feature_generator=base_embedder
)



feature_transformer = ListTransformer([
    MultiDimensionalScaler((0, 2, 3)),
    MultiDimensionalScaler((0,))]
)
feature_transformer.fit(X_train)
X_train = feature_transformer.transform(X_train)
X_valid = feature_transformer.transform(X_valid)

target_transformer = SimpleScaler()
target_transformer.fit(y_train)
y_train = target_transformer.transform(y_train)
y_valid = target_transformer.transform(y_valid)

model = MainNetwork(channels=40, dense_sizes=[[150, 100], [75, 50]], depth=3).to("cuda")
model = MainNetwork(channels=15, dense_sizes=[[150, 100], [75, 50]], depth=1).to("cuda")
model = MainNetwork(channels=5, dense_sizes=[[75, 50]], depth=1).to("cuda")
approximator = TransformedApproximator(
    approximator=Torch(
        model=model,
        loss=nn.HuberLoss(delta=1.5),
        optimizer=torch.optim.Adam,
        optimizer_kwargs={'lr': 1e-3},
    ),
    transformers=[
        Buffer(256)
    ]
)

class Trainer:
    def __init__(self):
        self.train_mse = defaultdict(list)
        self.valid_mse = defaultdict(list)

    def train(self, approx, name):
        for _ in tqdm(range(10)):
            approx.update(X_train, y_train)

            y_pred = approx.batch_get(X_train, batch_size=256)
            self.train_mse[name].append(np.sqrt(np.mean((y_pred - y_train) ** 2)))

            y_pred = approx.batch_get(X_valid, batch_size=256)
            self.valid_mse[name].append(np.sqrt(np.mean((y_pred - y_valid) ** 2)))


trainer = Trainer()


trainer.train(approximator, "Slim")
trainer.train_mse["Slim"]

#'Depth 3 Baseline': [0.8513447945846687, 0.839807555773624, 0.8294982959502466, 0.8262689975794751, 0.8196809723243537, 0.817042503232328, 0.8090337819750327, 0.8030387205324323, 0.7984738184168655, 0.7935533750491895, 0.7919341036423405, 0.7921636048449003, 0.7874413459629374]})
#'Depth 3 Baseline': [0.8542634659167249, 0.8500179475603332, 0.8458036206694011, 0.850264165467206, 0.8474856935378211, 0.8489887769280412, 0.84813063540146, 0.846396200008987, 0.8489682517013718, 0.8626689998859006, 0.8651022551477606, 0.874196673517897, 0.8724717009233233]})



model = MainNetwork(channels=45, dense_sizes=[[256, 128], [64, 32]], depth=1).to("cuda")
simple = SimpleNetwork().to("cuda")

model(*X)
simple(*X)

sum([x.numel() for x in model.parameters()])
sum([x.numel() for x in simple.parameters()])
self = model
from time import time

t = time()
for _ in range(100):
    simple(*X)
print(time() - t)

t = time()
for _ in range(100):
    model(*X)
print(time() - t)
