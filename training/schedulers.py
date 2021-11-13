from tqdm import tqdm

from game.game import GameRound


class OrdinaryScheduler:
    def __init__(self, adversary):
        self.adversary = adversary

    def train(self, trainer, episodes=1):
        for i in tqdm(range(episodes), ascii=True):
            print('--------')
            starting_player = i % 3
            episode_id = trainer.start_episode(0)
            game = GameRound(starting_player)
            while True:
                observation = game.observe()
                if game.phasing_player == 0:
                    card = trainer.trainable_play(observation, episode_id)
                else:
                    card = self.adversary.play(observation)
                print(card)
                game.play(card)
                if game.end:
                    trainer.finalize_episode(game, episode_id)


class TripleScheduler:
    def train(self, trainer, episodes=1):
        for i in tqdm(range(episodes), ascii=True):
            starting_player = i % 3
            episode_ids = [trainer.start_episode(0),
                           trainer.start_episode(1),
                           trainer.start_episode(2)]
            game = GameRound(starting_player)
            while True:
                observation = game.observe()
                card = trainer.trainable_play(observation, episode_ids[game.phasing_player])
                game.play(card)
                if game.end:
                    for j in range(3):
                        trainer.finalize_episode(game, episode_ids[j])