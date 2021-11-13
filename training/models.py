import numpy as np

from encoders import concatenate_embeddings
import torch
import uuid


def to_cuda_list(embeddings):
    X = embeddings.X
    if not isinstance(X, list):
        X = [X]
    return [torch.tensor(x).float().to("cuda") for x in X]


class QFunction:
    def __init__(self, embedder, q_function):
        self.embedder = embedder
        self.q_function = q_function

    def get_embedding_value_pair(self, observation):
        self.train_mode(False)
        embeddings = self.embedder.get_state_actions_embedding(observation)
        X = to_cuda_list(embeddings)
        return embeddings, self.q_function(*X).detach().to("cpu").numpy().flatten()

    def play(self, observation):
        _, values = self.get_embedding_value_pair(observation)
        idx = np.argmax(values.flatten())
        return observation.eligible_choices[idx]

    def train_mode(self, value):
        self.q_function.train(value)

class QTrainer:
    def __init__(self, q, optimizer, loss_function, explorer, fitter, updater, reward):
        self.q = q
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.explorer = explorer
        self.fitter = fitter
        self.updater = updater
        self.reward = reward
        self.episodes = {}

    def play(self, observation):
        self.q.play(observation)

    def start_episode(self, player):
        id = uuid.uuid4()
        self.episodes[id] = Episode(player)
        return id

    def trainable_play(self, observation, episode_id):
        embeddings, values = self.q.get_embedding_value_pair(observation)
        p = self.explorer.get_probabilities(values)
        idx = np.random.choice(len(values), p=p)

        reward = self.reward.get_reward(observation)

        self.episodes[episode_id].add(embeddings[[idx]], values.flatten(), p, idx, reward, observation)

        return observation.eligible_choices[idx]

    def finalize_episode(self, game, episode_id):
        reward = self.reward.finalize_reward(game, self.episodes[episode_id].player)
        self.episodes[episode_id].finalize(reward)
        X, y = self.updater.get_update_data(self.episodes[episode_id])
        data = self.fitter.get_data(X, y)
        if data is not None:
            self.fit(*data)

    def fit(self, X, y):
        self.q.train_mode(True)
        X = to_cuda_list(X)
        y = torch.tensor(y).float().to("cuda")
        pred = self.q.q_function(*X)
        loss = self.loss_function(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Episode:
    def __init__(self, player):
        self.player = player
        self.embeddings = []
        self.values = []
        self.p = []
        self.idx = []
        self.rewards = []
        self.observations = []

    def add(self, embeddings, values, p, idx, reward, observation=None):
        self.embeddings.append(embeddings)
        self.values.append(values)
        self.p.append(p)
        self.idx.append(idx)
        self.observations.append(observation)

        if self.rewards:
            self.rewards.append(reward - self.rewards[-1])
        else:
            self.rewards.append(reward)

    def finalize(self, end_reward):
        self.rewards.append(end_reward - self.rewards[-1])
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

