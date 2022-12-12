class Reward:
    def reset(self, observation):
        pass

    def step(self, observation):
        pass


class OrdinaryReward(Reward):
    def __init__(self, alpha):
        self.alpha = alpha

    def reset(self, observation):
        self._total_took = 0
        self._total_given = 0

    def step(self, observation):
        score = observation.features['score']

        took = score[0]
        given = sum(score) - took

        delta_took = took - self._total_took
        delta_given = given - self._total_given

        self._total_took = took
        self._total_given = given

        return self.alpha * delta_given - (1 - self.alpha) * delta_took


class EndReward(Reward):
    def __init__(self, alpha):
        self.alpha = alpha

    def step(self, observation):
        if len(observation['hand']) > 0:
            return 0

        score = observation['score']

        took = score[0]
        given = sum(score) - took

        return self.alpha * given - (1 - self.alpha) * took
