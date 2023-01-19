import itertools
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from tqdm import tqdm

from baselines.baselines import LowPlayer, Agent
from game.constants import PLAYERS
from game.environment import Tester, RewardTester, Environment, finish_game
from game.game import Hand, TrackedGameRound
from game.rewards import OrdinaryReward, RewardsCombiner, DeclaredDurchRewards, Reward
from game.utils import generate_hands
from learners.feature_generators import Lambda2DEmbedder, get_highest_pot_card, \
    get_pot_value, get_pot_size_indicators, generate_dataset, get_pot_card, \
    get_card_by_key, get_possible_cards, get_moved_cards, get_play_phase_action, \
    get_flat_by_key, get_declaration_phase_action, get_durch_phase_action, \
    TransformedFeatureGenerator, get_cards_remaining
from learners.runner import AgentLeague

from model.model import MainNetwork, SimpleNetwork, MainNetworkV2
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
        approximators=[approximator, deepcopy(approximator)],
        feature_generator=feature_generator,
    )

    agent.load(path)

    return agent



league = AgentLeague(
    seed_agents=[
        get_agent(Path('runs/baseline_run_6/episode_90000')),
        get_agent(Path('runs/baseline_run_5/episode_180000')),
        get_agent(Path('runs/baseline_run_5/episode_190000')),
        get_agent(Path('runs/baseline_run_5/episode_200000'))
    ],
    game_count=100, max_agents=3
)

league._points()
league._avg_score()

t = Tester(40, get_agent(Path('runs/baseline_run_6/episode_60000')))
t.evaluate(get_agent(Path('runs/baseline_run_5/episode_60000')), verbose=2)


