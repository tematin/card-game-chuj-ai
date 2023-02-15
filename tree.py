from copy import deepcopy
from pprint import pprint

import numpy as np
from tqdm import tqdm

from baselines.agents import phase_one
from baselines.baselines import LowPlayer, Agent
from game.constants import CARDS_PER_PLAYER, PLAYERS
from game.game import TrackedGameRound, Hand
from game.rewards import OrdinaryReward, DeclaredDurchRewards, RewardsCombiner, Reward
from game.utils import generate_hands, GamePhase
from learners.trainers import ValueAgent


class TreeSearch(Agent):
    def __init__(self, agent: ValueAgent, reward: Reward) -> None:
        self._agent = agent
        self._reward = reward

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

        reward = deepcopy(self._reward)
        reward.reset(simulated_game.observe(player=0)[0])

        while simulated_game.phase == GamePhase.MOVING:
            simulated_game.play(moved_cards[simulated_game.phasing_player])
            if simulated_game.phasing_player == 0:
                reward.step(simulated_game.observe()[0])

        history = (observation['durch_history']
                   + observation['declaration_history']
                   + observation['play_history'])

        for _, a in history:
            simulated_game.play(a)
            if simulated_game.phasing_player == 0:
                reward.step(simulated_game.observe()[0])

        assert simulated_game.phasing_player == 0

        return simulated_game, reward

    def _yield_simulated_games(self, observation, actions, samples, size):
        games = []
        rewards = []

        for _ in range(samples):
            if len(games) > size:
                yield games, rewards
                games = []
                rewards = []

            game, processed_reward = self._create_simulated_game(observation)
            for action in actions:
                seed_game = deepcopy(game)
                seed_game.play(action)
                games.append(seed_game)
                rewards.append(deepcopy(processed_reward))

        yield games, rewards

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

            actions_to_play = self._agent.parallel_play(observation_list, action_list)

            for game, action in zip(relevant_games, actions_to_play):
                game.play(action)

    def _game_value(self, games):
        observations = []
        actions = []

        for g in games:
            o, a = g.observe()
            observations.append(o)
            actions.append(a)

        values = self._agent.parallel_values(observations, actions)

        return np.array([np.max(x) for x in values])

    def play(self, observation, actions):
        if observation['phase'] != GamePhase.PLAY:
            return self._agent.play(observation, actions)

        if len(actions) == 1:
            return actions[0]

        results = []
        for games, rewards in self._yield_simulated_games(observation, actions, 100, 50):
            games = self._advance_games(games)
            value = self._game_value(games)
            step_rewards = [r.step(g.observe()[0]) for r, g in zip(rewards, games)]
            value += step_rewards

            results.append(value.reshape(-1, len(actions)))

        results = np.concatenate(results, axis=0)
        return actions[results.mean(0).argmax()]


agent = phase_one('8_190')

reward = RewardsCombiner([
    OrdinaryReward(1 / 3),
    DeclaredDurchRewards(
            success_reward=13 + 1 / 3,
            failure_reward=-13 - 1 / 3,
            rival_failure_reward=6 + 2 / 3,
            rival_success_reward=-6 - 2 / 3
        ),
])
ts_agent = TreeSearch(agent=agent, reward=reward)


pp = []

for _ in tqdm(range(300)):
    game = TrackedGameRound(
        hands=generate_hands(),
        starting_player=np.random.randint(3)
    )

    while not game.end:
        if game.phasing_player == 0:
            action = ts_agent.play(*game.observe())
        else:
            action = agent.play(*game.observe())
        game.play(action)

    pp.append(game.points)

np.stack(pp).mean(0)
