from training.models import KerasQPlayer, Sarsa, EpsilonGreedy, GroupedFitter, ReplayFitter,\
    ReplayMemory, Softmax, TripleTrainer, OrdinaryTrainer, ExplorationCombiner
from baselines import LowPlayer
from evaluation_scripts import (get_cached_games,
                                evaluate_on_cached_games_against)
from object_storage import get_model, get_embedder
import keras

emb = get_embedder()
old_model = get_model()
model = get_model()

old_model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(learning_rate=0.0001))
model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(learning_rate=0.0001))

model.load_weights('sarsa_run_2_checkpoint_138k')
old_model.load_weights('sarsa_run_2_checkpoint_138k')

old_player = KerasQPlayer(emb, old_model)
player = KerasQPlayer(emb, model)

cached_games = get_cached_games(700)

trainer = OrdinaryTrainer(explorer=Softmax(1),
                          fitter=ReplayFitter(replay_memory=ReplayMemory(2000),
                                              replay_size=256 - 12 * 8,
                                              fitter=GroupedFitter(8)),
                          updater=Sarsa(0.3))

trainer = TripleTrainer(explorer=ExplorationCombiner([Softmax(1), EpsilonGreedy(1)], [0.96, 0.04]),
                        fitter=ReplayFitter(replay_memory=ReplayMemory(2000),
                                            replay_size=256 - 12 * 8,
                                            fitter=GroupedFitter(8)),
                        updater=Sarsa(0.3))


for i in range(8):
    trainer.train(player, episodes=2000)
    ev, _ = evaluate_on_cached_games_against(cached_games, player, LowPlayer())
    print(ev)
    ev, _ = evaluate_on_cached_games_against(cached_games, player, old_player)
    print(ev)

