from object_storage import NeuralNetwork, get_embedder_v2
from training.encoders import Lambda2DEmbedder, get_hand, get_card_took_flag
from training.models import to_cuda_list, Episode, QFunction
from training.schedulers import OrdinaryScheduler
from training.explorers import ExplorationCombiner, EpsilonGreedy, Softmax
from training.fitters import ReplayFitter, GroupedFitter, ReplayMemory
from training.updaters import Sarsa
from training.rewards import OrdinaryReward

from evaluation.core import Tester

from baselines.baselines import LowPlayer, RandomPlayer
from copy import deepcopy

import numpy as np
import uuid
import torch
from torch import nn


class VFunction:
    def __init__(self, embedder, v_function):
        self.embedder = embedder
        self.v_function = v_function
        self.optimizer = None
        self.loss_function = None

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer(self.v_function.parameters())

    def set_loss_function(self, func):
        self.loss_function = func

    def get_embedding_value_pair(self, observation):
        self.train_mode(False)
        embeddings = self.embedder.get_state_embedding(observation)
        X = to_cuda_list(embeddings)
        return embeddings, self.v_function(*X).detach().to("cpu").numpy().flatten()

    def train_mode(self, value):
        self.v_function.train(value)

    def fit(self, X, y):
        self.train_mode(True)
        X = to_cuda_list(X)
        y = torch.tensor(y).float().to("cuda")
        pred = self.v_function(*X)
        loss = self.loss_function(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class QVTrainer:
    def __init__(self, q, v, optimizer, loss_function, explorer, q_fitter, v_fitter, v_updater, q_updater, reward):
        self.q = q
        self.v = v

        self.q.set_optimizer(optimizer)
        self.v.set_optimizer(optimizer)

        self.q.set_loss_function(loss_function)
        self.v.set_loss_function(loss_function)

        self.explorer = explorer

        self.q_fitter = q_fitter
        self.v_fitter = v_fitter

        self.q_updater = q_updater
        self.v_updater = v_updater
        self.reward = reward
        self.episodes = {}

    def play(self, observation):
        self.q.play(observation)

    def start_episode(self, player):
        id = uuid.uuid4().hex
        self.episodes[id] = (Episode(player), Episode(player))
        return id

    def trainable_play(self, observation, episode_id):
        action_embeddings, action_values = self.q.get_embedding_value_pair(observation)

        p = self.explorer.get_probabilities(action_values)
        idx = np.random.choice(len(action_values), p=p)

        self.last_reward = self.reward.get_reward(observation)
        self.last_embedding = action_embeddings[[idx]]

        return observation.eligible_choices[idx]

    def clear_game(self, observation, episode_id):
        state_embedding, state_value = self.v.get_embedding_value_pair(observation)

        self.episodes[episode_id][0].add(embeddings=self.last_embedding,
                                         reward=self.last_reward,
                                         greedy_value=state_value,
                                         expected_value=state_value,
                                         played_value=state_value)

        self.episodes[episode_id][1].add(embeddings=state_embedding,
                                         reward=self.last_reward,
                                         greedy_value=state_value,
                                         expected_value=state_value,
                                         played_value=state_value)

    def finalize_episode(self, game, episode_id):
        action_episode, state_episode = self.episodes[episode_id]

        reward = self.reward.finalize_reward(game, state_episode.player)

        state_episode.finalize(reward)
        action_episode.finalize(reward)

        X = action_episode.embeddings
        y = self.q_updater.get_update_data(action_episode)
        data = self.q_fitter.get_data(X, y)
        if data is not None:
            self.q.fit(*data)

        X = state_episode.embeddings
        y = self.v_updater.get_update_data(state_episode)
        data = self.v_fitter.get_data(X, y)
        if data is not None:
            self.v.fit(*data)

        del self.episodes[episode_id]


action_embedder = get_embedder_v2()
state_embedder = Lambda2DEmbedder([get_hand,
                                   get_state_possible_cards(1),
                                   get_state_possible_cards(2)],
                                   [get_card_took_flag])

q = QFunction(action_embedder, NeuralNetwork().to("cuda"))
v = VFunction(state_embedder, NeuralNetwork(in_channels=3).to("cuda"))

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam

trainer = QVTrainer(q, v, optimizer=optimizer, loss_function=loss_fn,
                    explorer=ExplorationCombiner([Softmax(1), EpsilonGreedy(1)], [0.98, 0.02]),
                    q_fitter=ReplayFitter(replay_memory=ReplayMemory(2000),
                                          replay_size=256 - 12 * 8,
                                          fitter=GroupedFitter(8)),
                    v_fitter=ReplayFitter(replay_memory=ReplayMemory(2000),
                                          replay_size=256 - 12 * 8,
                                          fitter=GroupedFitter(8)),
                    q_updater=Sarsa(0.4),
                    v_updater=Sarsa(0.4),
                    reward=OrdinaryReward(alpha=0.5))

scheduler = OrdinaryScheduler(adversary=q)

previous_player = None
tester = Tester(2000)

from game.game import GameRound

starting_player = 0
episode_id = trainer.start_episode(0)
game = GameRound(starting_player)
while True:
    observation = game.observe()
    if game.phasing_player == -1:
        trainer.clear_game(observation, episode_id)
        game.clear()
    if game.phasing_player == 0:
        card = trainer.trainable_play(observation, episode_id)
    else:
        card = q.play(observation)
    game.play(card)
    if game.end:
        break

action_episode, state_episode = trainer.episodes[episode_id]

reward = trainer.reward.finalize_reward(game, state_episode.player)

state_episode.finalize(reward)
action_episode.finalize(reward)

state_episode.rewards.sum()
action_episode.rewards.sum()

Sarsa(0.3).get_update_data(state_episode)
Sarsa(0.3).get_update_data(action_episode)





while True:
    scheduler.train(trainer, episodes=20000)

    print(' ')
    print('----- Low')
    tester.evaluate(q, LowPlayer(), verbose=1)

    print('----- Random')
    tester.evaluate(q, RandomPlayer(), verbose=1)

    if previous_player is not None:
        print('----- Past Self')
        tester.evaluate(q, previous_player, verbose=1)
    previous_player = deepcopy(q)

