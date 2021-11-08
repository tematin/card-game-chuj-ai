from collections import defaultdict
from models import KerasQPlayer, TorchQPlayer, Sarsa, Q, EpsilonGreedy, GroupedFitter, ReplayFitter,\
    ReplayMemory, Softmax, TripleTrainer, OrdinaryTrainer, RegularFitter, ExplorationCombiner
from baselines import LowPlayer, RandomPlayer
from evaluation_scripts import (monte_carlo_evaluation, get_cached_games,
                                evaluate_on_cached_games, evaluate_on_cached_games_against,
                                analyze_game_round)
from object_storage import get_embedder_v2
from copy import deepcopy
import torch
from torch import nn


emb = get_embedder_v2()


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




model = NeuralNetwork(base_conv_filters=40, conv_filters=30, dropout_p=0.2).to("cuda")
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

player = TorchQPlayer(emb, model, optimizer, loss_fn, randomized=False)

cached_games = get_cached_games(1500)

trainer = OrdinaryTrainer(explorer=ExplorationCombiner([Softmax(2), EpsilonGreedy(1)], [0.94, 0.06]),
                          fitter=ReplayFitter(replay_memory=ReplayMemory(2000),
                                              replay_size=256 - 12 * 8,
                                              fitter=GroupedFitter(8)),
                          updater=Q())
while True:
    trainer.train(player, episodes=30000)
    ev, _ = evaluate_on_cached_games_against(cached_games, player, LowPlayer())
    print(ev)

print('---')
trainer = TripleTrainer(explorer=ExplorationCombiner([Softmax(1), EpsilonGreedy(1)], [0.95, 0.05]),
                        fitter=ReplayFitter(replay_memory=ReplayMemory(2000),
                                            replay_size=256 - 12 * 8,
                                            fitter=GroupedFitter(8)),
                        updater=Sarsa(0.3))

for i in range(40):
    trainer.train(player, episodes=7000)
    ev, _ = evaluate_on_cached_games_against(cached_games, player, LowPlayer())
    print(ev)

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
player = TorchQPlayer(emb, model, optimizer, loss_fn, randomized=False)

print('SGD ---')

for i in range(10):
    trainer.train(player, episodes=7000)
    ev, _ = evaluate_on_cached_games_against(cached_games, player, LowPlayer())
    print(ev)

print('Ord ---')
trainer = OrdinaryTrainer(explorer=ExplorationCombiner([Softmax(1), EpsilonGreedy(1)], [0.96, 0.04]),
                          fitter=ReplayFitter(replay_memory=ReplayMemory(4000),
                                              replay_size=256 - 12 * 8,
                                              fitter=GroupedFitter(8)),
                          updater=Sarsa(0.3))

for i in range(5):
    trainer.train(player, episodes=5000)
    ev, _ = evaluate_on_cached_games_against(cached_games, player, LowPlayer())
    print(ev)



for i in range(50):
    decay = 0.95 ** i
    trainer = OrdinaryTrainer(explorer=ExplorationCombiner([Softmax(1 * decay), EpsilonGreedy(1)], [1 - 0.04 * decay, 0.04 * decay]),
                              fitter=ReplayFitter(replay_memory=ReplayMemory(2000),
                                                  replay_size=256 - 12 * 8,
                                                  fitter=GroupedFitter(8)),
                              updater=Sarsa(0.3))

    trainer.train(player, episodes=7000)
    ev, _ = evaluate_on_cached_games_against(cached_games, player, LowPlayer())
    print(ev)

torch.save(model, 'sarsa_run_3')



trainer = OrdinaryTrainer(explorer=ExplorationCombiner([Softmax(1), EpsilonGreedy(1)], [0.96, 0.04]),
                          fitter=GroupedFitter(8),
                          updater=Sarsa(0.3))


X, y = MonteCarlo().get_update_data(episode)

X.X

