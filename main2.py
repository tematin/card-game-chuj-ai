from training.rewards import OrdinaryReward
from training.models import QTrainer, QFunction
from training.schedulers import OrdinaryScheduler, TripleScheduler
from training.explorers import ExplorationCombiner, EpsilonGreedy, Softmax
from training.fitters import ReplayFitter, GroupedFitter, ReplayMemory
from training.updaters import Q, Sarsa
from encoders import Lambda2DEmbedder, get_hand, get_pot_cards, get_first_pot_card, get_pot_value, get_card_took_flag
from object_storage import get_model

from baselines import LowPlayer, RandomPlayer
from copy import deepcopy
from evaluation_scripts import Tester
import torch
from torch import nn
import dill
from functools import partial


class DropoutNeuralNetwork(nn.Module):
    def __init__(self, base_conv_filters=10, conv_filters=10, dropout_p=0.2, in_channels=6):
        super().__init__()
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
            nn.LazyLinear(200),
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


class BatchNormNeuralNetwork(nn.Module):
    def __init__(self, base_conv_filters=30, conv_filters=30, dropout_p=0.2, in_channels=6):
        super().__init__()
        self.by_parts = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=base_conv_filters, kernel_size=(1, 3)),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.by_colour = nn.Sequential(
            nn.Conv2d(in_channels=base_conv_filters, out_channels=conv_filters, kernel_size=(1, 7)),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.by_value = nn.Sequential(
            nn.Conv2d(in_channels=base_conv_filters, out_channels=conv_filters, kernel_size=(4, 3)),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )

        self.final_dense = nn.Sequential(
            nn.LazyLinear(200),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.LazyLinear(200),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.ReLU(),
            nn.LazyLinear(1)
        )

    def forward(self, card_encoding, rest_encoding):
        by_parts = self.by_parts(card_encoding)
        by_colour = torch.flatten(self.by_colour(by_parts), start_dim=1)
        by_value = torch.flatten(self.by_value(by_parts), start_dim=1)

        flat = torch.cat([by_colour, by_value, rest_encoding], dim=1)
        return self.final_dense(flat)


embedder = Lambda2DEmbedder([get_hand,
                             get_pot_cards,
                             get_first_pot_card,
                             lambda x: x.right_hand,
                             lambda x: x.left_hand],
                             [get_pot_value,
                              get_card_took_flag])

model = DropoutNeuralNetwork(base_conv_filters=30, conv_filters=30).to("cuda")

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam
player = QFunction(embedder, model)

class StaticQTrainer:
    def __init__(self, q, optimizer, loss_function, explorer, fitter, updater, reward):
        self.q = q
        self.q.set_optimizer(optimizer)
        self.q.set_loss_function(loss_function)
        self.explorer = explorer
        self.fitter = fitter
        self.updater = updater
        self.reward = reward
        self.episodes = {}

    def play(self, observation):
        self.q.play(observation)

    def start_episode(self, player):
        id = uuid.uuid4().hex
        self.episodes[id] = Episode(player)
        return id

    def trainable_play(self, observation, episode_id):
        embeddings, values = self.q.get_embedding_value_pair(observation)
        p = self.explorer.get_probabilities(values)
        idx = np.random.choice(len(values), p=p)

        reward = self.reward.get_reward(observation)

        values = values.flatten()
        self.episodes[episode_id].add(embeddings=embeddings[[idx]],
                                      reward=reward,
                                      played_value=values[idx],
                                      expected_value=np.sum(values * p),
                                      greedy_value=np.max(values))

        return observation.eligible_choices[idx]

    def clear_game(self, observation, episode_id):
        pass

    def finalize_episode(self, game, episode_id):
        episode = self.episodes[episode_id]
        reward = self.reward.finalize_reward(game, episode.player)
        episode.finalize(reward)

        X = episode.embeddings
        y = self.updater.get_update_data(episode)
        data = self.fitter.get_data(X, y)
        if data is not None:
            self.q.fit(*data)
        del self.episodes[episode_id]


trainer = QTrainer(player, optimizer=optimizer, loss_function=loss_fn,
                   explorer=ExplorationCombiner([Softmax(2), EpsilonGreedy(1)], [0.8, 0.2]),
                   fitter=ReplayFitter(replay_memory=ReplayMemory(2000),
                                       replay_size=256 - 12 * 8,
                                       fitter=GroupedFitter(8)),
                   updater=Q(),
                   reward=OrdinaryReward(alpha=0.5))

scores = []
scheduler = OrdinaryScheduler(adversary=player)


for i in range(10):
    scheduler.train(trainer, episodes=30000)

    print(' ')
    print('----- Low')
    score = tester.evaluate(player, LowPlayer(), verbose=1)
    scores.append(score)





