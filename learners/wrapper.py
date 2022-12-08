import numpy as np

from .approximators import Approximator
from .feature_generators import FeatureGenerator


class CityLearnWrapper:
    def __init__(self, q: Approximator, feature_generator: FeatureGenerator):
        self._q = q
        self._feature_generator = feature_generator

    def register_reset(self, observation, action_space, agent_id):
        return self.compute_action(observation, agent_id)

    def compute_action(self, observation, agent_id):
        features, actions = self._feature_generator.state_action(observation)

        q_vals = self._q.get(features)
        idx = np.argmax(q_vals)

        return actions[idx]
