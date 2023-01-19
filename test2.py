import itertools
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from tqdm import tqdm

from baselines.baselines import LowPlayer, Agent
from game.constants import PLAYERS
from game.environment import Tester, RewardTester, Environment, finish_game, \
    OneThreadEnvironment
from game.game import Hand, TrackedGameRound
from game.rewards import OrdinaryReward, RewardsCombiner, DeclaredDurchRewards, Reward
from game.utils import generate_hands
from learners.feature_generators import Lambda2DEmbedder, get_highest_pot_card, \
    get_pot_value, get_pot_size_indicators, generate_dataset, get_pot_card, \
    get_card_by_key, get_possible_cards, get_moved_cards, get_play_phase_action, \
    get_flat_by_key, get_declaration_phase_action, get_durch_phase_action, \
    TransformedFeatureGenerator, get_cards_remaining, FeatureGenerator
from learners.runner import AgentLeague

from model.model import MainNetwork, SimpleNetwork, MainNetworkV2, BlockConvResLayer, \
    SkipConnection, BlockDenseSkipConnections
from learners.trainers import DoubleTrainer, TrainedDoubleQ
from learners.approximators import Torch, Buffer, SoftUpdateTorch, TargetTransformer, \
    Approximator, TransformedApproximator
from learners.transformers import MultiDimensionalScaler, SimpleScaler, ListTransformer


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
        approximators=(approximator, deepcopy(approximator)),
        feature_generator=feature_generator,
    )

    agent.load(path)

    return agent


agent = get_agent(Path('runs/baseline_run_5/episode_190000'))


class PolicyNetwork(nn.Module):
    def __init__(self, channels, dense_sizes, depth, dropout_p=0.1, leak=0.01):
        super().__init__()

        self.conv_broad = BlockConvResLayer(
            depth=depth, channel_size=channels, kernel_width=(1, 3),
            padding='same', dropout_p=dropout_p, leak=leak
        )

        conv_bridge = nn.Sequential(
            nn.LazyConv2d(out_channels=channels,
                          kernel_size=(1, 4), padding='valid'),
            nn.Dropout2d(dropout_p),
            nn.LeakyReLU(leak),
            nn.LazyConv2d(out_channels=channels,
                          kernel_size=(1, 4), padding='valid'),
            nn.Dropout2d(dropout_p),
            nn.LeakyReLU(leak),
            nn.LazyConv2d(out_channels=channels,
                          kernel_size=(1, 3), padding='valid'),
            nn.Dropout2d(dropout_p),
        )

        self.bridge = SkipConnection(
            main_layer=conv_bridge,
            downsample=nn.LazyConv2d(out_channels=channels, kernel_size=(1, 9),
                                     padding='valid'),
            activation=nn.LeakyReLU(leak)
        )

        self.simple_conv = nn.LazyConv2d(out_channels=10, kernel_size=(1, 9),
                                         padding='valid')

        self.dense = BlockDenseSkipConnections(dense_sizes)

        self.final_dense = nn.LazyLinear(36)

        self.softmax = nn.Softmax()

    def forward(self, card_encoding, rest_encoding, action_mask):
        broad = self.conv_broad(card_encoding)
        tight = self.bridge(broad)
        flat = torch.flatten(tight, start_dim=1)

        simple = self.simple_conv(card_encoding)
        simple = torch.flatten(simple, start_dim=1)

        concatenated = torch.cat([flat, rest_encoding, simple], dim=1)

        dense = self.dense(concatenated)
        dense = self.final_dense(dense)

        ret = torch.zeros(36)
        ret[action_mask] = self.softmax(dense[action_mask])


base_embedder = Lambda2DEmbedder(
    [get_pot_card(0),
     get_pot_card(1),
     get_card_by_key('hand'),
     get_possible_cards(0),
     get_possible_cards(1),
     get_card_by_key('received_cards'),
     get_moved_cards,
     ],
    [get_pot_value,
     get_pot_size_indicators,
     get_flat_by_key('score'),
     get_flat_by_key('doubled'),
     get_flat_by_key('eligible_durch'),
     get_flat_by_key('declared_durch'),
     get_cards_remaining
     ],
)


def generate_dataset(
        env: OneThreadEnvironment,
        agent: Agent,
        episodes: int,
        feature_generator: FeatureGenerator
) -> Tuple[List[np.ndarray], np.ndarray]:
    feature_dataset = []
    reward_dataset = []

    for _ in range(episodes):
        episode_rewards = []
        observation, actions, _, done = env.reset()

        while not done:
            action = agent.play(observation, actions)

            state_feature = feature_generator.state(observation)

            observation, actions, reward, done = env.step(action)

            feature_dataset.append(state_feature)
            episode_rewards.append(reward)

        episode_rewards = np.cumsum(np.array(episode_rewards)[::-1])[::-1]
        reward_dataset.append(episode_rewards)

    feature_dataset = concatenate_feature_list(feature_dataset)
    reward_dataset = np.concatenate(reward_dataset)

    return feature_dataset, reward_dataset