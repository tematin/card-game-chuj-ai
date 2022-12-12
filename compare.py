from pathlib import Path
from typing import List

import torch
from torch import nn
import numpy as np

from baselines.baselines import RandomPlayer, LowPlayer, Agent
from game.constants import PLAYERS
from game.environment import Tester, Environment
from game.game import TrackedGameRound
from game.rewards import OrdinaryReward
from game.stat_trackers import ScoreTracker, DurchEligibilityTracker, \
    RemainingPossibleCards
from game.utils import generate_hands, Card
from learners.explorers import EpsilonGreedy, ExplorationCombiner, Random, Softmax, ExplorationSwitcher
from learners.feature_generators import Lambda2DEmbedder, get_highest_pot_card, \
    get_hand, get_pot_cards, get_pot_value, get_pot_size_indicators, get_eligible_durch, \
    get_current_score, get_possible_cards, generate_dataset
from learners.memory import ReplayMemory
from learners.runner import TrainRun
from learners.updaters import Step, MaximumValue
from model.model import MainNetwork, SimpleNetwork
from learners.trainers import SimpleTrainer, DoubleTrainer
from learners.approximators import Torch, Buffer, SoftUpdateTorch, TargetTransformer, \
    Approximator
from learners.transformers import MultiDimensionalScaler, SimpleScaler


reward = OrdinaryReward(0.5)
trackers = [ScoreTracker(), DurchEligibilityTracker(), RemainingPossibleCards()]

feature_generator = Lambda2DEmbedder(
    [get_highest_pot_card,
     get_hand,
     get_pot_cards,
     get_possible_cards(0),
     get_possible_cards(1)],
    [get_pot_value,
     get_pot_size_indicators,
     get_current_score,
     get_eligible_durch],
)

X, y = generate_dataset(
    env=Environment(
        reward=reward,
        trackers=trackers,
        rival=RandomPlayer()
    ),
    episodes=1000,
    agent=RandomPlayer(),
    feature_generator=feature_generator
)


scaler_3d = MultiDimensionalScaler((0, 2, 3))
scaler_flat = MultiDimensionalScaler((0,))
scaler_rewards = SimpleScaler()

scaler_3d.fit(X[0])
scaler_flat.fit(X[1])
scaler_rewards.fit(y)


while True:
    model = MainNetwork([[200, 200]], 1).to("cuda")
    y = model(torch.tensor(scaler_3d.transform(X[0])).float().to("cuda"),
              torch.tensor(scaler_flat.transform(X[1])).float().to("cuda")).detach().to("cpu").numpy()
    if np.abs(y.mean()) < 0.02:
        break

approx = SoftUpdateTorch(
    tau=1e-3,
    torch_model=Torch(
        model=model,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={'lr': 1e-4},
        loss=nn.HuberLoss(delta=2),
    )
)

def get_agent():
    return DoubleTrainer(
        q=Buffer(
            buffer_size=256,
            approximator=TargetTransformer(
                transformer=scaler_rewards,
                approximator=approx
            )
        ),
        memory=ReplayMemory(
            yield_length=2,
            memory_size=2000,
            extraction_count=2,
            ramp_up_size=1000
        ),
        updater=Step(discount=1),
        value_calculator=MaximumValue(),
        feature_generator=feature_generator,
        explorer=ExplorationCombiner([Random(), Softmax(0.4)], [0.1, 0.9]),
        feature_transformers=[scaler_3d, scaler_flat],
    )


tester = Tester(
    100,
    trackers=trackers
)

agent = get_agent()
adversary = get_agent()


agent.load(Path('C:\Python\Repos\chuj\\runs\double_q_baseline_extension_2\episode_4000'))
adversary.load(Path('C:/Python/Repos/chuj/runs/double_q_baseline_extention/episode_22000'))
tester.evaluate(agent, adversary, verbose=1)


agent.load(Path('C:\Python\Repos\chuj\\runs\double_q_baseline_extension_2\episode_12000'))
adversary.load(Path('C:/Python/Repos/chuj/runs/double_q_baseline_extention/episode_22000'))
tester.evaluate(agent, adversary, verbose=1)

agent.load(Path('C:\Python\Repos\chuj\\runs\double_q_baseline_extension_2\episode_12000'))
adversary.load(Path('C:\Python\Repos\chuj\\runs\double_q_baseline_extension_2\episode_4000'))
tester.evaluate(agent, adversary, verbose=1)



agents = [get_agent(), get_agent(), get_agent()]
agents[0].load(Path('C:\Python\Repos\chuj\\runs\double_q_baseline_extension_2\episode_4000'))
agents[1].load(Path('C:\Python\Repos\chuj\\runs\double_q_baseline_extension_2\episode_12000'))
agents[2].load(Path('C:/Python/Repos/chuj/runs/double_q_baseline_extention/episode_22000'))


class Comp(Agent):
    def __init__(self, aa):
        self.aa = aa

    def play(self, observation: dict):
        q = []
        for a in self.aa:
            d = a.debug(observation)
            q.append(d['q_avg'])
        q = np.array(q).sum(0)
        return d['actions'][np.argmax(q)]


tester.evaluate(agents[0], LowPlayer(), verbose=1)
tester.evaluate(agents[1], LowPlayer(), verbose=1)
tester.evaluate(agents[2], LowPlayer(), verbose=1)
tester.evaluate(Comp([agents[0], agents[0]]), RandomPlayer(), verbose=1)



def analyze_game_round(agent, trackers):
    hands = generate_hands()
    game = TrackedGameRound(
        starting_player=np.random.choice(np.arange(PLAYERS)),
        hands=hands,
        trackers=trackers
    )
    while True:
        observation = game.observe()
        card = agent.step(observation)
        print('')
        print("Choice Made", card)
        if game.phasing_player == 0:
            print('')
            for k, v in observation.items():
                print(k)
                print(v)

            print('------------------')

            debug_samples = agent.debug(observation)
            for k, v in debug_samples.items():
                print(k)
                print(v)

            print('------------------')

        game.play(card)

        if game.end:
            print(game.points)
            break


trackers = [ScoreTracker(), DurchEligibilityTracker(), RemainingPossibleCards()]
analyze_game_round(agents[0], trackers)
