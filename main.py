from training.rewards import OrdinaryReward
from training.models import QTrainer, QFunction
from training.schedulers import OrdinaryScheduler, TripleScheduler
from training.explorers import ExplorationCombiner, EpsilonGreedy, Softmax
from training.fitters import ReplayFitter, GroupedFitter, ReplayMemory
from training.updaters import Q, Sarsa

from baselines import LowPlayer, RandomPlayer
from copy import deepcopy
from evaluation_scripts import Tester
from object_storage import get_embedder_v2, get_model
import torch
from torch import nn
import dill
from functools import partial


tester = Tester(2000)

embedder = get_embedder_v2()
model = get_model()
loss_fn = nn.MSELoss()
#optimizer = partial(torch.optim.Adam, lr=1e-4)
optimizer = torch.optim.Adam

player = QFunction(embedder, model)
with open('models/baseline.dill', 'rb') as f:
    player = dill.load(f)
player.embedder = embedder

trainer = QTrainer(player, optimizer=optimizer, loss_function=loss_fn,
                   explorer=ExplorationCombiner([Softmax(2), EpsilonGreedy(1)], [0.94, 0.06]),
                   fitter=ReplayFitter(replay_memory=ReplayMemory(2000),
                                              replay_size=256 - 12 * 8,
                                              fitter=GroupedFitter(8)),
                   updater=Q(),
                   reward=OrdinaryReward(alpha=0.5))

previous_player = deepcopy(player)

scores = []
scheduler = OrdinaryScheduler(adversary=player)

for i in range(10):
    scheduler.train(trainer, episodes=30000)

    print(' ')
    print('----- Low')
    score = tester.evaluate(player, LowPlayer(), verbose=1)
    scores.append(score)

    #if i >= 300:
    #    print('----- Past Self')
    #    tester.evaluate(player, previous_player, verbose=1)
    #previous_player = deepcopy(player)

player.q_function.by_value[1].requires_grad_(False)
player.q_function.by_colour[1].requires_grad_(False)
player.q_function.by_parts[1].requires_grad_(False)
