from copy import deepcopy
from pprint import pprint
from typing import List, Any

import numpy as np
from tqdm import tqdm

from baselines.agents import phase_one
from baselines.baselines import LowPlayer, Agent
from game.constants import CARDS_PER_PLAYER, PLAYERS
from game.environment import Tester
from game.game import TrackedGameRound, Hand
from game.rewards import OrdinaryReward, DeclaredDurchRewards, RewardsCombiner, Reward
from game.utils import generate_hands, GamePhase
from learners.explorers import Explorer, Softmax, ExplorationCombiner, Random
from learners.trainers import ValueAgent


class TreeSearch(Agent):
    def __init__(self, agent: ValueAgent, reward: Reward,
                 prior_explorer: Explorer, rollout_explorer: Explorer,
                 batch_size: int, disadvantage_cutoff: float, sigmas: List[float]) -> None:
        self._agent = agent
        self._reward = reward
        self._prior_explorer = prior_explorer
        self._rollout_explorer = rollout_explorer
        self._batch_size = batch_size
        self._disadvantage_cutoff = disadvantage_cutoff
        self._sigmas = sigmas

    def _create_simulated_game(self, obs):
        observation = deepcopy(obs)
        fixed_cards = deepcopy(observation['played_cards'])
        fixed_cards[0].extend(observation['hand'])

        possible = [set(observation['possible_cards'][i]) for i in range(2)]
        fixed_cards[1].extend(possible[0] - possible[1])
        fixed_cards[2].extend(possible[1] - possible[0])

        remaining_cards = list(possible[0] & possible[1])

        np.random.shuffle(remaining_cards)
        to_fill = CARDS_PER_PLAYER - len(fixed_cards[1])
        fixed_cards[1].extend(remaining_cards[:to_fill])
        fixed_cards[2].extend(remaining_cards[to_fill:])

        moved_cards = [
            observation['moved_cards'],
            tuple(np.random.choice(fixed_cards[2], 2, replace=False)),
            observation['received_cards']
        ]

        for i in range(PLAYERS):
            fixed_cards[i].extend(moved_cards[i])
            for card in moved_cards[i]:
                fixed_cards[(i + 1) % PLAYERS].remove(card)

        simulated_game = TrackedGameRound(
            hands=fixed_cards,
            starting_player=observation['starting_player']
        )

        while simulated_game.phase == GamePhase.MOVING:
            simulated_game.play(moved_cards[simulated_game.phasing_player])

        return simulated_game

    def _evaluate_history(self, games, history):
        game_count = len(games)

        rewards = [deepcopy(self._reward) for _ in range(game_count)]
        for game, reward in zip(games, rewards):
            reward.reset(game.observe()[0])

        aggregate_log_probs = np.zeros(game_count)

        for _, historical_action in history:
            observations = []
            actions = []
            action_idx = []

            for game in games:
                o, a = game.observe()
                observations.append(o)
                actions.append(a)
                action_idx.append(a.index(historical_action))

            values = self._agent.parallel_values(observations, actions)
            probs = [self._prior_explorer.get_probabilities(x) for x in values]
            probs = np.array([x[i] for x, i in zip(probs, action_idx)])

            aggregate_log_probs += np.log(probs)

            for game, reward in zip(games, rewards):
                game.play(historical_action)

                if game.phasing_player == 0:
                    reward.step(game.observe()[0])

        return games, rewards, aggregate_log_probs

    def _play_actions(self, games, action):
        games = [deepcopy(g) for g in games]

        for game in games:
            game.play(action)

        return games

    def _advance_games(self, games):
        while True:
            to_move = [x.phasing_player != 0 and not x.end for x in games]

            if sum(to_move) == 0:
                return games

            relevant_games = [x for x, m in zip(games, to_move) if m]

            observation_list = []
            action_list = []

            for game in relevant_games:
                o, a = game.observe()
                observation_list.append(o)
                action_list.append(a)

            values = self._agent.parallel_values(observation_list, action_list)

            for game, value, actions in zip(relevant_games, values, action_list):
                probabilities = self._rollout_explorer.get_probabilities(value)
                action = np.random.choice(actions, p=probabilities)
                game.play(action)

    def _game_reward(self, games, rewards):
        rewards = deepcopy(rewards)

        ret = []
        for g, r in zip(games, rewards):
            ret.append(r.step(g.observe()[0]))

        return np.array(ret)

    def _game_value(self, games):
        observations = []
        actions = []

        for g in games:
            o, a = g.observe()
            observations.append(o)
            actions.append(a)

        values = self._agent.parallel_values(observations, actions)

        return np.array([np.max(x) for x in values])

    def _simulate_games(self, observation, actions):
        history = (observation['durch_history']
                   + observation['declaration_history']
                   + observation['play_history'])

        batch_results = []

        games = [self._create_simulated_game(observation)
                 for _ in range(self._batch_size)]
        games, rewards, log_probs = self._evaluate_history(games, history)

        for action in actions:
            copied_games = self._play_actions(games, action)
            copied_games = self._advance_games(copied_games)

            step_rewards = self._game_reward(copied_games, rewards)
            value = self._game_value(copied_games)

            value = value + step_rewards
            batch_results.append(value)

        return np.stack(batch_results).T, log_probs

    def _filter_actions(self, results, log_probs, sigma):
        probs = log_probs - log_probs.max()
        probs = np.exp(probs)

        value_mean = (probs.reshape(-1, 1) * results).mean(0)
        value_std = np.sqrt((probs ** 2).sum()) * results.std(0) * (1 / results.shape[0])

        lower_range = value_mean - sigma * value_std
        upper_range = value_mean + sigma * value_std

        lowest_best_estimate = lower_range.max()
        return upper_range >= lowest_best_estimate

    def play(self, observation, actions):
        if observation['phase'] != GamePhase.PLAY:
            return self._agent.play(observation, actions)

        if len(actions) == 1:
            return actions[0]

        values = self._agent.values(observation, actions)
        disadvantage = values - np.max(values)
        actions = [x for x, d in zip(actions, disadvantage)
                   if d > self._disadvantage_cutoff]

        results = np.empty((0, len(actions)))
        log_probs = np.empty(0)

        for sigma in self._sigmas:
            if len(actions) == 1:
                return actions[0]

            batch_results, batch_log_probs = self._simulate_games(observation, actions)

            results = np.vstack([results, batch_results])
            log_probs = np.concatenate([log_probs, batch_log_probs])

            mask = self._filter_actions(results, log_probs, sigma)
            results = results[:, mask]
            actions = [x for x, m in zip(actions, mask) if m]

        else:
            if len(actions) == 1:
                return actions[0]
            raise RuntimeError("Should not reach")


class Ensamble(ValueAgent):
    def __init__(self, agents: List[ValueAgent]):
        self._agents = agents

    def values(self, observation: dict, actions: List[Any]) -> np.ndarray:
        values = [agent.values(observation, actions) for agent in self._agents]
        return np.stack(values).mean(0)


agent = phase_one('10_190')
other_agent = phase_one('8_190')

tester = Tester(25, other_agent)

reward = RewardsCombiner([
    OrdinaryReward(1 / 3),
    DeclaredDurchRewards(
            success_reward=13 + 1 / 3,
            failure_reward=-13 - 1 / 3,
            rival_failure_reward=6 + 2 / 3,
            rival_success_reward=-6 - 2 / 3
        ),
])

prior_explorer = ExplorationCombiner(
    [Softmax(1), Random()],
    [0.9, 0.1])

rollout_explorer = Softmax(0.6)


ts_agent = TreeSearch(agent=agent, reward=reward,
                      prior_explorer=prior_explorer,
                      rollout_explorer=rollout_explorer,
                      batch_size=10,
                      disadvantage_cutoff=-5,
                      sigmas=[3, 3, 3, 3, 3, 2.5, 2, 2, 2, 0]
                      )

tester.evaluate(ts_agent, verbose=2)

self = ts_agent

hands = deepcopy(tester._hands_list[27])

game = TrackedGameRound(
    hands=deepcopy(hands), #generate_hands(),
    starting_player=np.random.randint(3)
)


while not game.end:
    observation, actions = game.observe()
    if game.phase == GamePhase.PLAY:
        observation, actions = game.observe()
        action = ts_agent.play(observation, actions)
    else:
        action = other_agent.play(*game.observe())

    print(action)
    game.play(action)



while not game.end:
    if game.phasing_player == 0 and game.phase == GamePhase.PLAY:
        observation, actions = game.observe()
        action = ts_agent.play(observation, actions)
    else:
        action = agent.play(*game.observe())
    print(action)
    game.play(action)
