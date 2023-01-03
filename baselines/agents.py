from pathlib import Path

import torch
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import StepLR

from learners.feature_generators import Lambda2DEmbedder, get_highest_pot_card, \
    get_pot_value, get_pot_size_indicators, generate_dataset, get_pot_card, \
    get_card_by_key, get_possible_cards, get_moved_cards, get_play_phase_action, \
    get_flat_by_key, get_declaration_phase_action, get_durch_phase_action, \
    TransformedFeatureGenerator, FeatureGenerator

from model.model import MainNetwork, SimpleNetwork
from learners.trainers import DoubleTrainer, TrainedDoubleQ
from learners.approximators import Torch, Buffer, SoftUpdateTorch, TargetTransformer, \
    Approximator, TransformedApproximator
from learners.transformers import MultiDimensionalScaler, SimpleScaler, ListTransformer



def first_baseline():
    target_transformer = SimpleScaler()

    model = MainNetwork(channels=45, dense_sizes=[[256, 256], [64, 32]], depth=1).to("cuda")
    X = [np.random.rand(16, 8, 4, 9), np.random.rand(16, 21)]
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
                scheduler_kwargs={'step_size': 100, 'gamma': 0.995}
            ),
        ),
        transformers=[
            Buffer(128),
            TargetTransformer(target_transformer)
        ]
    )

    feature_transformer = ListTransformer([
        MultiDimensionalScaler((0, 2, 3)),
        MultiDimensionalScaler((0,))]
    )

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

    feature_generator = TransformedFeatureGenerator(
        feature_transformer=feature_transformer,
        feature_generator=base_embedder
    )

    agent = TrainedDoubleQ(approximator, feature_generator)
    agent.load(Path('baselines/first_baseline'))

    return agent
