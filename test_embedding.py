import numpy as np

from baselines.baselines import RandomPlayer
from game.environment import Environment
from game.rewards import OrdinaryReward, RewardsCombiner, DurchDeclarationPenalty
from learners.feature_generators import *
from pprint import pprint


reward = RewardsCombiner([OrdinaryReward(0.15), DurchDeclarationPenalty(-5)])


embedder = Lambda2DEmbedder(
    [get_pot_card(0),
     get_pot_card(1),
     get_card_by_key('hand'),
     get_possible_cards(0),
     get_possible_cards(1),
     get_card_by_key('received_cards'),
     get_moved_cards,
     get_play_phase_action
     ],
    [get_pot_value,
     get_pot_size_indicators,
     get_flat_by_key('score'),
     get_flat_by_key('doubled'),
     get_flat_by_key('eligible_durch'),
     get_flat_by_key('declared_durch'),
     get_durch_phase_action,
     get_declaration_phase_action
     ],
)

agent = RandomPlayer()


env = Environment(
    reward=reward,
    rival=RandomPlayer()
)


feature_dataset = []
reward_dataset = []

for _ in range(100):
    episode_rewards = []
    observation, actions = env.reset()

    done = False
    while not done:
        action = agent.play(observation, actions)
        idx = actions.index(action)

        action_feature = embedder.state_action(observation, actions)
        feature = index_features(action_feature, idx)

        observation, actions, reward, done = env.step(action)

        feature_dataset.append(feature)
        episode_rewards.append(reward)

    episode_rewards = np.cumsum(np.array(episode_rewards)[::-1])[::-1]
    reward_dataset.append(episode_rewards)


feature_dataset = concatenate_feature_list(feature_dataset)
reward_dataset = np.concatenate(reward_dataset)
