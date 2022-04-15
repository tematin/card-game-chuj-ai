import numpy as np
from tqdm import tqdm

from training.rewards import OrdinaryReward
from training.models import QTrainer, QFunction, to_cuda_list
from training.schedulers import OrdinaryScheduler, TripleScheduler
from training.explorers import ExplorationCombiner, EpsilonGreedy, Softmax
from training.fitters import ReplayFitter, GroupedFitter, ReplayMemory
from training.updaters import Q, Sarsa
from encoders import Lambda2DEmbedder, get_hand, get_pot_cards, get_first_pot_card, get_pot_value, get_card_took_flag, \
    concatenate_embeddings, get_current_score, LambdaEmbedder
from object_storage import get_model
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from baselines import LowPlayer, RandomPlayer
from copy import deepcopy
from evaluation_scripts import Tester
import torch
from torch import nn
import dill
from functools import partial
from game.game import GameRound


def generate_episode(player, reward, embedder):
    game = GameRound(0)
    embeddings = [[], [], []]

    while not game.end:
        if game.trick_end:
            game.play()
            continue
        obs = game.observe()
        emb = embedder.get_state_embedding(obs)
        embeddings[obs.phasing_player].append(emb)
        card = player.play(obs)
        game.play(card)

    rewards = [reward.finalize_reward(game, i) for i in range(3)]
    X = concatenate_embeddings(embeddings[0] + embeddings[1] + embeddings[2])
    y = np.repeat(rewards, 12)

    return X, y


def generate_datapoints(player, reward, embedder, episodes):
    ret_X = []
    ret_y = []
    for _ in tqdm(range(episodes)):
        X, y = generate_episode(player, reward, embedder)
        ret_X.append(X)
        ret_y.append(y)

    return concatenate_embeddings(ret_X), np.concatenate(ret_y)


def generate_indexes(size, batch_size):
    idx = np.arange(size)
    np.random.shuffle(idx)
    i = 0
    while i < size:
        yield idx[i:(i + batch_size)]
        i += batch_size


def train(X, y, model, optimizer, loss_fn, epochs=1, batch_size=128):
    model.train(True)
    size = y.shape[0]

    X = to_cuda_list(X)
    y = torch.tensor(y).float().reshape(-1, 1).to("cuda")

    for _ in tqdm(range(epochs)):
        for idx in generate_indexes(size, batch_size):
            batch_X = [x[idx] for x in X]
            batch_y = y[idx]

            pred = model(*batch_X)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate(X, y, model, loss_fn):
    size = y.shape[0]
    model.train(False)

    X = to_cuda_list(X)
    y = torch.tensor(y).float().to("cuda").reshape(-1, 1)

    losses = 0
    for idx in generate_indexes(size, 2**9):
        batch_X = [x[idx] for x in X]
        batch_y = y[idx]

        pred = model(*batch_X)
        loss = loss_fn(pred, batch_y)
        losses += loss.item() * len(idx)
    losses /= size
    return losses


class SimplifiedReward:
    def __init__(self, alpha):
        self.alpha = alpha

    def get_reward(self, observation):
        score = observation.tracker.score.score
        return self._reward_from_score(score, observation.phasing_player)

    def finalize_reward(self, game, player):
        score = game.get_points()
        return self._reward_from_score(score, player)

    def _reward_from_score(self, score, player):
        took = score[player]
        given = sum(score) - score[player]
        if took == -10:
            took = 21
            given = 0
        elif given == -10:
            took = 0
            given = 21
        return self.alpha * given - (1 - self.alpha) * took



player = LowPlayer()
reward = SimplifiedReward(0)
loss_fn = nn.MSELoss()

simple_embedder = LambdaEmbedder([get_hand,
                                  lambda x: x.right_hand,
                                  lambda x: x.left_hand],
                                 [])

np.random.seed(1)
train_X, train_y = generate_datapoints(player, reward, simple_embedder, episodes=300000)
np.random.seed(10)
val_X, val_y = generate_datapoints(player, reward, simple_embedder, episodes=10000)



mask = train_X.X.sum(1) == 36
train_X = train_X[mask]
train_y = train_y[mask]

mask = val_X.X.sum(1) == 36
val_X = val_X[mask]
val_y = val_y[mask]

sc_X = StandardScaler()
sc_X.fit(train_X.X)

train_X.X = sc_X.transform(train_X.X)
val_X.X = sc_X.transform(val_X.X)

std = train_y.std()
train_y = (train_y / std)

mean = train_y.mean()
train_y = train_y - train_y.mean()

val_y = (val_y / std) - mean

class DropoutNeuralNetwork(nn.Module):
    def __init__(self, base_conv_filters=10, conv_filters=10, dropout_p=0.2, in_channels=3):
        super().__init__()
        self.by_parts = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=base_conv_filters,
                      kernel_size=(1, 5), padding='same'),
            nn.Dropout2d(dropout_p),
            nn.ReLU()
        )
        self.by_colour = nn.Sequential(
            nn.Conv2d(in_channels=base_conv_filters, out_channels=conv_filters,
                      kernel_size=(1, 9), padding='valid'),
            nn.Dropout2d(dropout_p),
            nn.ReLU()
        )
        self.by_value = nn.Sequential(
            nn.Conv2d(in_channels=base_conv_filters, out_channels=conv_filters,
                      kernel_size=(4, 3), padding='valid'),
            nn.Dropout2d(dropout_p),
            nn.ReLU()
        )

        self.by_pure_colour = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=conv_filters,
                      kernel_size=(1, 9), padding='valid'),
            nn.Dropout2d(dropout_p),
            nn.ReLU()
        )

        self.final_dense = nn.Sequential(
            nn.LazyLinear(400),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.LazyLinear(200),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.LazyLinear(200),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.LazyLinear(200),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.LazyLinear(200),
            nn.Dropout(dropout_p),
            nn.ReLU(),

            nn.LazyLinear(1)
        )

    def forward(self, card_encoding):
        by_parts = self.by_parts(card_encoding)
        by_colour = torch.flatten(self.by_colour(by_parts), start_dim=1)
        by_value = torch.flatten(self.by_value(by_parts), start_dim=1)
        by_pure_colour = torch.flatten(self.by_pure_colour(card_encoding), start_dim=1)

        flat = torch.cat([by_colour, by_value, by_pure_colour], dim=1)
        return self.final_dense(flat)


train_X.X = train_X.X.reshape(-1, 3, 4, 9)
val_X.X = val_X.X.reshape(-1, 3, 4, 9)


np.save('train_X.npz', train_X.X)
np.save('train_y.npz', train_y)
np.save('valid_X.npz', val_X.X)
np.save('valid_y.npz', val_y)

with open('scaler.pickle', 'wb') as f:
    pickle.dump(sc_X, f)


model = DropoutNeuralNetwork(base_conv_filters=15, conv_filters=60, dropout_p=0.4).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.04)
optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.01)
optimizer = torch.optim.RMSProp(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.01)


size = train_y.shape[0]
X = to_cuda_list(train_X)
y = torch.tensor(train_y).float().reshape(-1, 1).to("cuda")

for idx in generate_indexes(size, 32):
    batch_X = [x[idx] for x in X]
    batch_y = y[idx]

    pred = model(*batch_X)
    loss = loss_fn(pred, batch_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



while True:
    train(train_X, train_y, model, optimizer, loss_fn, batch_size=2**10, epochs=1)
    print(evaluate(train_X, train_y, model, loss_fn))
    print(evaluate(val_X, val_y, model, loss_fn))


X = to_cuda_list(val_X)
pred = model(*X)

plt.scatter(pred.cpu().detach().numpy().flatten(), val_y, alpha=0.01)

train_y.mean()



np.random.seed(10)
Xr, yr, gamer = generate_episode(player, reward, embedder)

mask = Xr.X.sum(1) == 36
Xr = Xr[mask]
yr = yr[mask]

one_emb = Xr.X.reshape(3, 4, 9)


def generate_episode(player, reward, embedder):
    game = GameRound(0)
    embeddings = [[], [], []]

    while not game.end:
        if game.trick_end:
            game.play()
            continue
        obs = game.observe()
        emb = embedder.get_state_embedding(obs)
        embeddings[obs.phasing_player].append(emb)
        card = player.play(obs)
        game.play(card)

    rewards = [reward.finalize_reward(game, i) for i in range(3)]
    X = concatenate_embeddings(embeddings[0] + embeddings[1] + embeddings[2])
    y = np.repeat(rewards, 12)

    return X, y, game


gamer.tracker.score.score
gamer.tracker.history.history[0]._cards

