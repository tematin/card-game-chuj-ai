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
from learners.explorers import EpsilonGreedy, ExplorationCombiner, Random, Softmax
from learners.feature_generators import Lambda2DEmbedder, get_highest_pot_card, \
    get_hand, get_pot_cards, get_pot_value, get_pot_size_indicators, get_eligible_durch, \
    get_current_score, get_possible_cards
from learners.memory import ReplayMemory
from learners.runner import TrainRun
from learners.updaters import Step, MaximumValue
from model.model import MainNetwork, SimpleNetwork, BlockConvResLayer, ResLayer, \
    SkipConnection, BlockDenseSkipConnections
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
    episodes=10,
    agent=RandomPlayer(),
    feature_generator=feature_generator,
    exclude_actions=True
)
X = [torch.tensor(x).float().to("cuda") for x in X]


class PolicyNetwork(nn.Module):
    def __init__(self, dense_sizes, depth, dropout_p=0.1, leak=0.01):
        super().__init__()

        self.conv_broad = BlockConvResLayer(
            depth=depth, channel_size=80, kernel_width=(1, 3),
            padding='same', dropout_p=dropout_p, leak=leak
        )

        conv_bridge = BlockConvResLayer(
            depth=depth, channel_size=160, kernel_width=(1, 5),
            padding='valid', dropout_p=dropout_p, leak=leak
        )

        colour_conv = ResLayer(
            channel_size=160, kernel_width=(1, 1), padding='same',
            dropout_p=dropout_p, leak=leak, activate=False
        )

        self.bridge = SkipConnection(
            main_layer=nn.Sequential(conv_bridge, colour_conv),
            downsample=nn.LazyConv2d(out_channels=160, kernel_size=(1, 9), padding='valid'),
            activation=nn.LeakyReLU(leak)
        )

        self.conv_tight = BlockConvResLayer(
            depth=depth, channel_size=40, kernel_width=(1, 1),
            padding='same', dropout_p=dropout_p, leak=leak
        )

        self.dense = nn.Sequential(
            BlockDenseSkipConnections(dense_sizes),
            nn.LazyLinear(36)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, card_encoding, rest_encoding, mask):
        broad = self.conv_broad(card_encoding)

        tight = self.bridge(broad)

        flat = torch.flatten(self.conv_tight(tight), start_dim=1)

        concatenated = torch.cat([flat, rest_encoding], dim=1)

        dense = self.dense(concatenated)

        dense[mask == 0] = -float("inf")

        return self.softmax(dense)


model = PolicyNetwork([[200, 200]], 1).to("cuda")

mask = (np.random.rand(120, 36) < 0.6)
mask = torch.tensor(mask).float()

y = model(*X, mask)

y
