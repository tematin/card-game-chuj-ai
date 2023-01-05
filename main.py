from pathlib import Path
from typing import List

import torch
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from baselines.agents import first_baseline
from baselines.baselines import RandomPlayer, LowPlayer, Agent
from game.environment import Tester, Environment, OneThreadEnvironment, \
    analyze_game_round, RewardTester
from game.rewards import OrdinaryReward, RewardsCombiner, DurchDeclarationPenalty, \
    DeclaredDurchRewards, DurchReward
from game.utils import GamePhase
from learners.explorers import EpsilonGreedy, ExplorationCombiner, Random, Softmax, ExplorationSwitcher
from learners.feature_generators import Lambda2DEmbedder, get_highest_pot_card, \
    get_pot_value, get_pot_size_indicators, generate_dataset, get_pot_card, \
    get_card_by_key, get_possible_cards, get_moved_cards, get_play_phase_action, \
    get_flat_by_key, get_declaration_phase_action, get_durch_phase_action, \
    TransformedFeatureGenerator, get_cards_remaining
from learners.memory import ReplayMemory
from learners.representation import index_observation
from learners.runner import TrainRun
from learners.updaters import Step, MaximumValue, UpdateStep
from model.model import MainNetwork, SimpleNetwork, MainNetworkV2
from learners.trainers import DoubleTrainer
from learners.approximators import Torch, Buffer, SoftUpdateTorch, TargetTransformer, \
    Approximator, TransformedApproximator
from learners.transformers import MultiDimensionalScaler, SimpleScaler, ListTransformer
from debug.timer import timer


reward = RewardsCombiner([
    OrdinaryReward(0.3),
    DeclaredDurchRewards(
            success_reward=15,
            failure_reward=-15.5,
            rival_failure_reward=6,
            rival_success_reward=-6
        ),
])

run_count = 10


base_embedder = Lambda2DEmbedder(
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
     get_cards_remaining,
     get_durch_phase_action,
     get_declaration_phase_action
     ],
)


X, y = generate_dataset(
    env=OneThreadEnvironment(
        reward=reward,
        rival=RandomPlayer(),
    ),
    episodes=10000,
    agent=RandomPlayer(),
    feature_generator=base_embedder
)

feature_transformer = ListTransformer([
    MultiDimensionalScaler((0, 2, 3)),
    MultiDimensionalScaler((0,))]
)
feature_transformer.fit(X)

target_transformer = SimpleScaler()
target_transformer.fit(y)

#model = MainNetwork(channels=45, dense_sizes=[[256, 256], [64, 32]], depth=1).to("cuda")
model = MainNetworkV2(channels=60, dense_sizes=[[256, 256], [128, 128], [64, 32]], depth=2).to("cuda")
model(*[torch.tensor(x[:100]).float().to("cuda") for x in X]).mean()


approximator = TransformedApproximator(
    approximator=SoftUpdateTorch(
        tau=1e-3,
        torch_model=Torch(
            model=model,
            loss=nn.HuberLoss(delta=1.5),
            optimizer=torch.optim.Adam,
            optimizer_kwargs={'lr': 6e-4},
            scheduler=StepLR,
            scheduler_kwargs={'step_size': 100, 'gamma': 0.996}
        ),
    ),
    transformers=[
        Buffer(128),
        TargetTransformer(target_transformer)
    ]
)

feature_generator = TransformedFeatureGenerator(
    feature_transformer=feature_transformer,
    feature_generator=base_embedder
)


agent = DoubleTrainer(
    q=approximator,
    memory=ReplayMemory(
        yield_length=2,
        memory_size=400,
        extraction_count=40,
        ramp_up_size=300
    ),
    updater=Step(discount=1),
    value_calculator=MaximumValue(),
    feature_generator=feature_generator,
    explorer=ExplorationSwitcher(
        probabilities=[0.95, 0.05],
        explorations=[
            ExplorationCombiner([Random(), Softmax(0.4)], [0.1, 0.9]),
            Softmax(0.03)
        ]
    ),
    run_count=run_count
)

runner = TrainRun(
    agent=agent,
    testers={
        'low_player': Tester(80, LowPlayer()),
        'reward_low': RewardTester(reward, LowPlayer(), run_count=10, episodes=600),
        'reward_agent': RewardTester(reward, first_baseline(), run_count=10, episodes=600),
    },
    environment=Environment(
        reward=reward,
        rival=agent,
        run_count=run_count
    ),
    eval_freq=10000,
    run_count=run_count,
    checkpoint_dir=Path('runs/baseline_run_4').absolute()
)

runner.train(15000)




raise










reward = RewardsCombiner([
    OrdinaryReward(0.3),
    DurchDeclarationPenalty(-3),
    DeclaredDurchRewards(
            success_reward=5,
            failure_reward=-12,
            rival_failure_reward=6.3,
            rival_success_reward=-14.7
        ),
    DurchReward(reward=20, rival_reward=-13)
])


env = Environment(
    reward=reward,
    rival=agent,
    run_count=1
)

total_rewards = 0
observations, actions, _, done, idx = env.reset()

while not all(done):
    actions_to_play = agent.parallel_play(observations, actions)
    observations, actions, rewards, done, _ = env.step(actions_to_play)
    print(observations[0]['declared_durch'])
    print(observations[0]['eligible_durch'])
    print(rewards)
    total_rewards += sum(rewards)


rr = [x[-1].reward for x in agent._memories[0][0]._steps]

while True:
    x = agent._memories[0][1].get()
    if x[0][1].reward == -11.8:
        break

x[0][1]
feature_transformer.inverse_transform(x[0][0].features)[1]

feats = x[0][0].features
feats = feature_transformer.inverse_transform(feats)
feats[1][:, 9] = 1
feats[1][:, 10] = 0
feats = feature_transformer.transform(feats)

agent._q[0].get(x[0][0].features)
agent._q[0].get(feats)


memory_steps_list = x


#hands = generate_hands()
starting = 1
game = TrackedGameRound(
    starting_player=starting,
    hands=deepcopy(hands)
)
act_list_1 = []

for _ in range(1):
    print('############################################')

while True:
    observation, actions = game.observe()
    action = agent.play(observation, actions)

    act_list_1.append(action)

    print('### ')
    print(game.phasing_player)
    print("Choice Made", action)
    print('')
    for k, v in observation.items():
        print(k)
        print(v)

    print('------------------')

    debug_samples = agent.debug(observation, actions)
    for k, v in debug_samples.items():
        print(k)
        print(v)

    print('------------------')

    game.play(action)

    if game.end:
        print(game.points)
        print(starting)
        break


hands2 = deepcopy(hands)

game = TrackedGameRound(
    starting_player=np.random.choice(np.arange(PLAYERS)),
    hands=hands2
)

while True:
    observation, actions = game.observe()
    action = agent.play(observation, actions)
    if isinstance(action, int):
        action = 0
    print('')
    print("Choice Made", action)
    if game.phasing_player == 0:
        print('')
        for k, v in observation.items():
            print(k)
            print(v)

        print('------------------')

        debug_samples = agent.debug(observation, actions)
        for k, v in debug_samples.items():
            print(k)
            print(v)

        print('------------------')

    game.play(action)

    if game.end:
        print(game.points)
        break














def get_data(self):
    memory_steps_list = []
    for run_memories in self._memories:
        memory_steps_list.extend(run_memories[0].get())

    features = []
    for memory_list in memory_steps_list:
        for memory in memory_list:
            if memory.features:
                features.append(memory.features)

    if not features:
        return

    main_q_vals = a.get_from_list(features, update_mode=True)
    other_q_vals = b.get_from_list(features, update_mode=True)

    for memory_list in memory_steps_list:
        update_steps = []

        for memory in memory_list:

            if memory.features:
                main_q_val = main_q_vals.pop(0)
                other_q_val = other_q_vals.pop(0)

                step = UpdateStep(
                    features=index_observation(memory.features,
                                               memory.action_took),
                    value=self._value_calculator.double(
                        action_values=main_q_val,
                        action_took=memory.action_took,
                        action_probs=self._explorer.get_probabilities(main_q_val),
                        reference_values=other_q_val
                    ),
                    reward=memory.reward
                )

            else:
                step = UpdateStep(
                    features=None,
                    value=0,
                    reward=memory.reward
                )
            update_steps.append(step)

        data = self._updater.get_updates(update_steps)
        yield data

a = deepcopy(agent._q[0])
b = deepcopy(agent._q[1])

def get_datas(e):
    datas = []
    for _ in range(e):
        datas.extend(list(get_data(agent)))

    y = [i[1] for i in datas]
    x = [i[0] for i in datas]

    x = concatenate_feature_list(x)
    y = np.array(y)

    return x, y

for _ in range(1000):
    x, y = get_datas(100)

    xx = feature_transformer.inverse_transform(x)

    eligible = xx[1][:, 9:12]
    hand_size = xx[0][:, 2].sum((1, 2))

    mask = (eligible.sum(1) == 1) & (eligible[:, 2] == 1) & (hand_size == 1)

    y_fil = y[mask]

    xxx = [xx[0][mask], xx[1][mask]]
    print(xxx[0][:, 3:5].mean())
    xxx = feature_transformer.transform(xxx)
    print(a.get(xxx, update_mode=True).mean())

    a.update(x, y)

mask = (eligible.sum(1) == 1) & (eligible[:, 1] == 1) & (hand_size == 1)

xxx = [xx[0][mask], xx[1][mask]]
xxx = feature_transformer.transform(xxx)

#xxx[0][:, 2] = 0
#xxx[0][:, 4] = 0

print(a.get(xxx, update_mode=True).mean())

