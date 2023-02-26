
hands = generate_hands()


full_game = TrackedGameRound(
    starting_player=0,
    hands=deepcopy(hands)
)

partial_game = PartialGameRound(
    starting_player=2,
    hand=deepcopy(hands[1])
)


def compare(o, p):
    assert set(o.keys()) == set(p.keys())

    for key in o:
        if key in ['pot', 'hand']:
            matches = set(o[key]) == set(p[key])
        else:
            matches = o[key] == p[key]
        if not matches:
            print(key)
            print(o[key])
            print(p[key])
            assert False


agent = LowPlayer()

observation, actions = full_game.observe()
received = agent.play(observation, actions)
full_game.play(received)

observation, actions = full_game.observe()
moved = agent.play(observation, actions)
full_game.play(moved)

observation, actions = full_game.observe()
action = agent.play(observation, actions)
full_game.play(action)

partial_game.play(moved)
partial_game.play(received)


while not full_game.end:
    print('---')
    print(full_game.phasing_player, partial_game._phasing_player)
    print(partial_game._phase)
    observation, actions = full_game.observe()
    action = agent.play(observation, actions)
    print(action)

    if full_game.phasing_player == 1:
        partial_observation = partial_game.observe()
        compare(observation, partial_observation)

    partial_game.play(action)
    full_game.play(action)
