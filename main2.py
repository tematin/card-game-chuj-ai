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


class DropoutNeuralNetwork(nn.Module):
    def __init__(self, base_conv_filters=10, conv_filters=10, dropout_p=0.2, in_channels=5):
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
    def __init__(self, base_conv_filters=30, conv_filters=30, dropout_p=0.2, in_channels=5):
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
            nn.LazyLinear(1)
        )

    def forward(self, card_encoding, rest_encoding):
        by_parts = self.by_parts(card_encoding)
        by_colour = torch.flatten(self.by_colour(by_parts), start_dim=1)
        by_value = torch.flatten(self.by_value(by_parts), start_dim=1)

        flat = torch.cat([by_colour, by_value, rest_encoding], dim=1)
        return self.final_dense(flat)


class NeuralNetwork(nn.Module):
    def __init__(self, base_conv_filters=30, conv_filters=30, dropout_p=0.2, in_channels=5):
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
    return losses,



embedder = Lambda2DEmbedder([get_hand,
                             get_pot_cards,
                             get_first_pot_card,
                             lambda x: x.right_hand,
                             lambda x: x.left_hand],
                            [get_pot_value,
                             get_card_took_flag,
                             get_current_score])

simple_embedder = LambdaEmbedder([get_hand,
                                  get_pot_cards,
                                  get_first_pot_card,
                                  lambda x: x.right_hand,
                                  lambda x: x.left_hand],
                                 [get_pot_value,
                                  get_card_took_flag,
                                  get_current_score])

player = LowPlayer()
reward = OrdinaryReward(0.5)

np.random.seed(10)
val_X, val_y = generate_datapoints(player, reward, embedder, episodes=10000)

model = NeuralNetwork(base_conv_filters=60, conv_filters=60).to("cuda")
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

while True:
    train_X, train_y = generate_datapoints(player, reward, embedder, episodes=20000)
    train(train_X, train_y, model, optimizer, loss_fn, batch_size=2**10, epochs=1)
    print(evaluate(train_X, train_y, model, loss_fn))
    print(evaluate(val_X, val_y, model, loss_fn))

None

np.random.seed(1)
eye_X, eye_y = generate_datapoints(player, reward, embedder, episodes=50)
eye_X2, eye_y2 = generate_datapoints(player, reward, embedder, episodes=50)

X = to_cuda_list(eye_X)
y = torch.tensor(eye_y).float().reshape(-1, 1).to("cuda")
pred = model(*X)

pred = pred.to("cpu").detach().numpy().flatten()
true = eye_y


cards_left = eye_X.X[0].sum((2, 3))[:, 0]
points_left = eye_X.X[1][:, -3:].sum(1)

res = (pred - true)

plt.scatter(cards_left, res, alpha=0.2)
plt.scatter(points_left, res, alpha=0.2)

idx = np.where(res > 8)[0]

eye_X[idx[0]].X

from evaluation_scripts import finish_game


simple_embedder = LambdaEmbedder([get_hand,
                                  lambda x: x.right_hand,
                                  lambda x: x.left_hand],
                                 [])

np.random.seed(1)
train_X, train_y = generate_datapoints(player, reward, simple_embedder, episodes=100000)
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

class FlatNeuralNetwork(nn.Module):
    def __init__(self, dropout_p=0.2):
        super().__init__()
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

    def forward(self, X):
        return self.final_dense(X)


model = FlatNeuralNetwork(0.4).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.01)
optimizer = torch.optim.RMSProp(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.01)


while True:
    train(train_X, train_y, model, optimizer, loss_fn, batch_size=2**10, epochs=1)
    print(evaluate(train_X, train_y, model, loss_fn))
    print(evaluate(val_X, val_y, model, loss_fn))


X = to_cuda_list(val_X)
pred = model(*X)

plt.scatter(pred.cpu().detach().numpy().flatten(), val_y, alpha=0.1)

train_y.mean()

