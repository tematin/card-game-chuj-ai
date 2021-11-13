
class OrdinaryReward:
    def __init__(self, alpha):
        self.alpha = alpha

    def get_reward(self, observation):
        score = observation.tracker.score.score
        return self._reward_from_score(score, observation.phasing_player)

    def finalize_reward(self, game, player):
        score = game.get_points()
        return self._reward_from_score(score, player)

    def _reward_from_score(self, score, player):
        took = score[player]
        given = sum(score) - score[player]
        return self.alpha * given + (1 - self.alpha) * took