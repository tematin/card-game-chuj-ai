from tqdm import tqdm
import numpy as np
from abc import abstractmethod, ABC

from game import GameRound, CARDS_PER_PLAYER
from encoders import concatenate_embeddings, Embeddings
import torch


class SuperQPlayer:
    def __init__(self, embedder, q_function):
        self.embedder = embedder
        self.q_function = q_function

    def play(self, observation):
        _, values = self.get_embedding_value_pair(observation)
        idx = np.argmax(values.flatten())
        return observation.eligible_choices[idx]

    @abstractmethod
    def get_embedding_value_pair(self, observation):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass


class KerasQPlayer(SuperQPlayer):
    def get_embedding_value_pair(self, observation):
        embeddings = self.embedder.get_state_actions_embedding(observation)
        values = self.q_function.predict(embeddings.X, batch_size=12)
        return embeddings, values.flatten()

    def fit(self, X, y):
        self.q_function.fit(X.X, y, verbose=0, batch_size=256)


class TorchQPlayer(SuperQPlayer):
    def __init__(self, embedder, q_function, optimizer, loss_function, randomized=False):
        super().__init__(embedder, q_function)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.randomized = randomized

    def fit(self, X, y):
        self.q_function.train()
        X = self._to_cuda_list(X)
        y = torch.tensor(y).float().to("cuda")
        pred = self.q_function(*X)
        loss = self.loss_function(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_embedding_value_pair(self, observation):
        self.q_function.train(self.randomized)
        embeddings = self.embedder.get_state_actions_embedding(observation)
        X = self._to_cuda_list(embeddings)
        return embeddings, self.q_function(*X).detach().to("cpu").numpy().flatten()

    def _to_cuda_list(self, embeddings):
        X = embeddings.X
        if not isinstance(X, list):
            X = [X]
        return [torch.tensor(x).float().to("cuda") for x in X]


class Episode:
    def __init__(self):
        self.embeddings = []
        self.values = []
        self.p = []
        self.idx = []
        self.rewards = []
        self.observations = []

    def add(self, embeddings, values, p, idx, rewards, observation=None):
        self.embeddings.append(embeddings)
        self.values.append(values)
        self.p.append(p)
        self.idx.append(idx)
        self.rewards.append(rewards)
        self.observations.append(observation)

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


class OrdinaryTrainer:
    def __init__(self, explorer, fitter, updater):
        self.explorer = explorer
        self.fitter = fitter
        self.updater = updater

    def train(self, q_player, episodes=1, adversary=None, secondary_adversary=None):
        if adversary is None:
            adversary = q_player
        if secondary_adversary is None:
            secondary_adversary = adversary

        for i in tqdm(range(episodes), ascii=True):
            starting_player = i % 3
            if i % 2 == 0:
                episode = self.play_episode(q_player, adversary,
                                            secondary_adversary, starting_player)
            else:
                episode = self.play_episode(q_player, secondary_adversary,
                                            adversary, starting_player)
            X, y = self.updater.get_update_data(episode)
            self.fitter.fit(q_player, X, y)

    def play_episode(self, q_player, right_adversary, left_adversary, starting_player):
        game = GameRound(starting_player)
        previous_score = 0
        episode = Episode()
        while True:
            observation = game.observe()
            if game.phasing_player == 0:
                current_score = get_reward(game)
                current_reward = current_score - previous_score
                previous_score = current_score

                embs, vals = q_player.get_embedding_value_pair(observation)
                p = self.explorer.get_probabilities(vals)
                idx = np.random.choice(len(vals), p=p)
                episode.add(embs[[idx]], vals.flatten(), p, idx, current_reward)
                game.play(observation.eligible_choices[idx])
            elif game.phasing_player == 1:
                game.play(right_adversary.play(observation))
            elif game.phasing_player == 2:
                game.play(left_adversary.play(observation))
            if game.end:
                current_score = get_reward(game)
                episode.finalize(current_score - previous_score)
                break
        return episode


class TripleTrainer:
    def __init__(self, explorer, fitter, updater, reward_function=None):
        self.explorer = explorer
        self.fitter = fitter
        self.updater = updater
        if reward_function is None:
            self.reward_function = get_reward
        else:
            self.reward_function = reward_function

    def train(self, q_player, episodes=1):
        for i in tqdm(range(episodes), ascii=True):
            starting_player = i % 3
            episodes = self.play_episode(q_player, starting_player)
            for episode in episodes:
                X, y = self.updater.get_update_data(episode)
                self.fitter.fit(q_player, X, y)

    def play_episode(self, q_player, starting_player):
        game = GameRound(starting_player)
        episodes = [Episode(), Episode(), Episode()]
        previous_scores = [0, 0, 0]
        while True:
            i = game.phasing_player
            observation = game.observe()

            current_score = get_reward(game, i)
            current_reward = current_score - previous_scores[i]
            previous_scores[i] = current_score

            embs, vals = q_player.get_embedding_value_pair(observation)

            p = self.explorer.get_probabilities(vals)
            idx = np.random.choice(len(vals), p=p)
            episodes[i].add(embs[[idx]], vals.flatten(), p, idx, current_reward)
            game.play(observation.eligible_choices[idx])

            if game.end:
                for i in range(3):
                    current_score = get_reward(game, i)
                    episodes[i].finalize(current_score - previous_scores[i])
                break
        return episodes


class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.replay_memory_X = None
        self.replay_memory_y = None

    def add_to_replay(self, X, y):
        if self.replay_memory_X is None:
            self.replay_memory_X = X
            self.replay_memory_y = y
        else:
            self.replay_memory_X.append(X)
            self.replay_memory_y = np.vstack([self.replay_memory_y, y])
        self._truncate_memory()

    def _truncate_memory(self):
        self.replay_memory_X = self.replay_memory_X[-self.size:]
        self.replay_memory_y = self.replay_memory_y[-self.size:]

    def get_from_replay(self, size):
        idx = np.random.choice(np.arange(self.replay_memory_X.data_count),
                               size=size)
        return self.replay_memory_X[idx], self.replay_memory_y[idx]

    def add_and_extract(self, X, y, size):
        if self.replay_memory_X is None:
            self.add_to_replay(X, y)
            return X, y
        else:
            replay_X, replay_y = self.get_from_replay(size)
            self.add_to_replay(X, y)
            return concatenate_embeddings([X, replay_X]), np.vstack([y, replay_y])


class BaseFitter(ABC):
    @abstractmethod
    def get_data(self, X, y):
        pass

    def fit(self, model, X, y):
        data = self.get_data(X, y)
        if data is None:
            return
        X, y = data
        model.fit(X, y)


class RegularFitter(BaseFitter):
    def get_data(self, X, y):
        return X, y


class GroupedFitter(BaseFitter):
    def __init__(self, episodes_in_batch=1):
        self.episodes_in_batch = episodes_in_batch
        self.X = []
        self.y = []

    def get_data(self, X, y):
        self.X.append(X)
        self.y.append(y)
        if len(self.X) >= self.episodes_in_batch:
            X = concatenate_embeddings(self.X)
            y = np.concatenate(self.y).reshape(-1, 1)
            self.X = []
            self.y = []
            return X, y
        else:
            return None


class ReplayFitter(BaseFitter):
    def __init__(self, replay_memory, fitter, replay_size):
        self.replay_memory = replay_memory
        self.fitter = fitter
        self.replay_size = replay_size

    def get_data(self, X ,y):
        data = self.fitter.get_data(X, y)
        if data is None:
            return None
        X, y = data
        replay_X, replay_y = self.replay_memory.add_and_extract(X, y, self.replay_size)
        return concatenate_embeddings([X, replay_X]), np.vstack([y, replay_y])


def get_reward(game, player=0):
    return (sum(game.get_points()) - 2 * game.get_points()[player])


class MonteCarlo:
    def get_update_data(self, episode):
        X = episode.embeddings
        y = np.cumsum(episode.rewards).reshape(-1, 1)
        return X, y


class Sarsa:
    def __init__(self, lmbda, expected=True):
        self.lmbda = lmbda
        self.expected = expected
        i = np.arange(CARDS_PER_PLAYER)
        self.value_discounts = (1 - lmbda) * (lmbda ** i)
        self.reward_discounts = 1 - np.cumsum(self.value_discounts)
        self.reward_discounts = np.insert(self.reward_discounts, 0, 1)

    def get_update_data(self, episode):
        y = []
        length = episode.length
        if self.expected:
            vals = episode.expected_values
        else:
            vals = episode.played_values
        for i in range(length):
            reward = np.sum(episode.rewards[i:] * self.reward_discounts[:(length - i)])
            value = np.sum(vals[(i + 1):] * self.value_discounts[:(length - i - 1)])
            y.append(reward + value)
        return episode.embeddings, np.array(y).reshape(-1, 1)


class Q:
    def get_update_data(self, episode):
        y = episode.rewards + np.append(episode.greedy_values[1:], 0)
        return episode.embeddings, y.reshape(-1, 1)


class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = float(epsilon)

    def get_probabilities(self, values):
        p = np.full_like(values, self.epsilon / len(values))
        p[np.argmax(values)] += 1 - self.epsilon
        return p


class Softmax:
    def __init__(self, temperature):
        self.t = temperature

    def get_probabilities(self, values):
        p = np.exp((values - values.max()) / self.t)
        p /= p.sum()
        return p


class ExplorationCombiner:
    def __init__(self, explorations, probabilities):
        self.explorations = explorations
        self.probabilities = np.array(probabilities).reshape(-1, 1)

    def get_probabilities(self, values):
        probs = []
        for exp in self.explorations:
            probs.append(exp.get_probabilities(values))
        p = (self.probabilities * np.stack(probs)).sum(0).flatten()
        return p / p.sum()
