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
        self.optimizer = None
        self.loss_function = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer(self.q_function.parameters())

    def set_loss_function(self, func):
        self.loss_function = func

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

    def fit(self, X, y):
        self.train_mode(True)
        X = to_cuda_list(X)
        y = torch.tensor(y).float().to("cuda")
        pred = self.q_function(*X)
        loss = self.loss_function(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class QTrainer:
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


class Episode:
    def __init__(self, player):
        self.player = player
        self.embeddings = []
        self.played_values = []
        self.expected_values = []
        self.greedy_values = []
        self.rewards = []
        self.total_rewards = []

    def add(self, embeddings, reward, played_value=None, expected_value=None, greedy_value=None):
        self.embeddings.append(embeddings)
        self.played_values.append(played_value)
        self.expected_values.append(expected_value)
        self.greedy_values.append(greedy_value)
        self.total_rewards.append(reward)

    def finalize(self, end_reward):
        self.total_rewards.append(end_reward)
        self.total_rewards = np.array(self.total_rewards)
        self.rewards = self.total_rewards[1:] - self.total_rewards[:-1]

        self.length = len(self.embeddings)
        self.embeddings = concatenate_embeddings(self.embeddings)


