from training.rewards import OrdinaryReward
from training.models import QTrainer, QFunction
from training.schedulers import OrdinaryScheduler, TripleScheduler
from training.explorers import ExplorationCombiner, EpsilonGreedy, Softmax
from training.fitters import ReplayFitter, GroupedFitter, ReplayMemory
from training.updaters import Q

from baselines import LowPlayer, RandomPlayer
from copy import deepcopy
from evaluation_scripts import Tester
from object_storage import get_embedder_v2
import torch
from torch import nn




class NeuralNetwork(nn.Module):
    def __init__(self, base_conv_filters=30, conv_filters=30, dropout_p=0.2, in_channels=6):
        super(NeuralNetwork, self).__init__()
        self.by_parts = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=base_conv_filters, kernel_size=(1, 3)),
            nn.Dropout2d(dropout_p),
            nn.ReLU()
        )
        self.by_colour = nn.Sequential(
            nn.Conv2d(in_channels=base_conv_filters, out_channels=conv_filters, kernel_size=(1, 7)),
            nn.Dropout2d(dropout_p),
            nn.ReLU()
        )
        self.by_value = nn.Sequential(
            nn.Conv2d(in_channels=base_conv_filters, out_channels=conv_filters, kernel_size=(4, 3)),
            nn.Dropout2d(dropout_p),
            nn.ReLU()
        )

        self.final_dense = nn.Sequential(
            nn.LazyLinear(400),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.LazyLinear(300),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.LazyLinear(300),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.LazyLinear(200),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.LazyLinear(1)
        )

    def forward(self, card_encoding, rest_encoding):
        by_parts = self.by_parts(card_encoding)
        by_colour = torch.flatten(self.by_colour(by_parts), start_dim=1)
        by_value = torch.flatten(self.by_value(by_parts), start_dim=1)

        flat = torch.cat([by_colour, by_value, rest_encoding], dim=1)
        return self.final_dense(flat)


previous_player = None
tester = Tester(2000)

embedder = get_embedder_v2()
model = NeuralNetwork(base_conv_filters=40, conv_filters=30, dropout_p=0.2).to("cuda")
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

player = QFunction(embedder, model)
trainer = QTrainer(player, optimizer=optimizer, loss_function=loss_fn,
                   explorer=ExplorationCombiner([Softmax(2), EpsilonGreedy(1)], [0.94, 0.06]),
                   fitter=ReplayFitter(replay_memory=ReplayMemory(2000),
                                              replay_size=256 - 12 * 8,
                                              fitter=GroupedFitter(8)),
                   updater=Q(),
                   reward=OrdinaryReward(alpha=0.5))

scheduler = OrdinaryScheduler(adversary=player)


from game.game import GameRound
game = GameRound(0)
while True:
    obs = game.observe()
    player.play(obs)
    if game.end:
        break


while True:
    scheduler.train(trainer, episodes=40000)

    print(' ')
    print('----- Low')
    tester.evaluate(player, LowPlayer(), verbose=1)

    print('----- Random')
    tester.evaluate(player, RandomPlayer(), verbose=1)

    if previous_player is not None:
        print('----- Past Self')
        tester.evaluate(player, previous_player, verbose=1)
    previous_player = deepcopy(player)
