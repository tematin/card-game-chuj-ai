from baselines import LowPlayer
from evaluation_scripts import (get_cached_games,
                                evaluate_on_cached_games_against)
from object_storage import get_embedder_v2
from copy import deepcopy
import torch
from torch import nn
from game import GameRound
from training.models import get_reward, Episode, MonteCarlo, concatenate_embeddings
import numpy as np
from tqdm import tqdm


class Episode:
    def __init__(self):
        self.embeddings = []
        self.values = []
        self.p = []
        self.idx = []
        self.rewards = []
        self.observations = []
        self.ends = []

    def add(self, embeddings, values, p, idx, rewards, observation=None, end=None):
        self.embeddings.append(embeddings)
        self.values.append(values)
        self.p.append(p)
        self.idx.append(idx)
        self.rewards.append(rewards)
        self.observations.append(deepcopy(observation))
        self.ends.append(end)

    def finalize(self, end_reward):
        self.rewards.append(end_reward)
        self.rewards = self.rewards[1:]
        self.embeddings = concatenate_embeddings(self.embeddings)
        self.length = len(self.values)
        self.rewards = np.array(self.rewards)

    @property
    def expected_values(self):
        return np.array([np.sum(p * val) for p, val in zip(self.p, self.values)])

    @property
    def played_values(self):
        return np.array([val[i] for i, val in zip(self.idx, self.values)])

    @property
    def greedy_values(self):
        return np.array([np.max(val) for val in self.values])


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
            nn.LazyLinear(300),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.LazyLinear(200),
            nn.Dropout(dropout_p),
            nn.ReLU()
        )

        self.end = nn.LazyLinear(36)

    def forward(self, card_encoding, rest_encoding):
        by_parts = self.by_parts(card_encoding)
        by_colour = torch.flatten(self.by_colour(by_parts), start_dim=1)
        by_value = torch.flatten(self.by_value(by_parts), start_dim=1)

        flat = torch.cat([by_colour, by_value, rest_encoding], dim=1)
        return self.end(self.final_dense(flat))


def get_reward(game, player=0):
    return (sum(game.get_points()) - 2 * game.get_points()[player])


def _to_cuda_list(embeddings):
    X = embeddings.X
    if not isinstance(X, list):
        X = [X]
    return [torch.tensor(x).float().to("cuda") for x in X]

emb = get_embedder_v2()
model = NeuralNetwork(base_conv_filters=40, conv_filters=30, dropout_p=0.2).to("cuda")
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

def get_episodes():
    game = GameRound(0)
    episodes = [Episode(), Episode(), Episode()]
    previous_scores = [0, 0, 0]
    while True:
        i = game.phasing_player
        observation = game.observe()

        current_score = get_reward(game, i)
        current_reward = current_score - previous_scores[i]
        previous_scores[i] = current_score

        embs = emb.get_state_embedding(observation)
        res = model(*_to_cuda_list(embs))
        p = res.detach().to("cpu").numpy().flatten()
        re_idx = [int(c) for c in observation.eligible_choices]
        p = p[re_idx]
        p -= p.max()
        p = np.exp(p)
        p /= p.sum()

        idx = np.random.choice(len(p), p=p)

        episodes[i].add(embs, re_idx, p, idx, current_reward, observation=observation, end=res)
        game.play(observation.eligible_choices[idx])

        if game.end:
            for i in range(3):
                current_score = get_reward(game, i)
                episodes[i].finalize(current_score - previous_scores[i])
            break
    return episodes


def update_episode(episode):
    embeddings, y = MonteCarlo().get_update_data(episode)

    for i in range(12):
        grad = np.zeros(36)
        grad[episode.values[i]] -= episode.p[i]
        grad[episode.idx[i]] += 1

        _, baseline = base_player.get_embedding_value_pair(episode.observations[i])
        baseline = np.max(baseline)

        grad *= (y[i] - baseline)

        pred = emb.get_state_embedding(episode.observations[i])
        res = model(*_to_cuda_list(pred))

        optimizer.zero_grad()
        res.backward(torch.tensor(grad.reshape(1, -1)).to("cuda"))
        optimizer.step()


while True:
    for i in tqdm(range(7000)):
        for episode in get_episodes():
            update_episode(episode)

    class P:
        def __init__(self, embedder, model):
            self.embedder = embedder
            self.model = model

        def play(self, observation):
            emb = self.embedder.get_state_embedding(observation)
            res = model(*_to_cuda_list(emb))
            re_idx = [int(c) for c in observation.eligible_choices]
            return observation.eligible_choices[np.argmax(res.detach().to("cpu").numpy().flatten()[re_idx])]

    cached_games = get_cached_games(700)
    ev, _ = evaluate_on_cached_games_against(cached_games, P(emb, model), LowPlayer())
    print(ev)
