import numpy as np
from tqdm import tqdm
from training.encoders import Lambda2DEmbedder, get_hand, get_pot_cards, get_possible_cards, \
    get_highest_pot_card, get_card_took_flag, \
    concatenate_embeddings, get_value_cards_took, get_value_cards_received, \
    get_split_pot_value
from game.game import GameRound
from baselines.baselines import LowPlayer
import pickle


def generate_episode(player, embedder):
    game = GameRound(0)
    embeddings = [[], [], []]

    while not game.end:
        if game.trick_end:
            game.play()
            continue
        obs = game.observe()
        emb = embedder.get_state_embedding(obs)
        embeddings[obs.phasing_player].append(emb)
        card = player.play(obs)
        game.play(card)

    embeddings = [concatenate_embeddings(x) for x in embeddings]
    return embeddings, game


def generate_dataset(player, embedder, games):
    X = []
    y = []

    for _ in tqdm(range(games)):
        embeddings, finished_game = generate_episode(player, embedder)

        value_tracker = finished_game.tracker.value
        played_durch = finished_game.tracker.durch.played_durch()
        for i in range(3):
            X.append(embeddings[i])
            red_cards = np.zeros(9)
            red_cards[:value_tracker.red_cards[i]] = 1
            target = np.array(
                [value_tracker.yellow_cards[i],
                 value_tracker.green_cards[i],
                 int(played_durch[i]),
                 played_durch[(i + 1) % 3] + played_durch[(i + 1) % 3]]
            )
            target = np.concatenate([red_cards, target])
            y.append(np.tile(target, (12, 1)))

    X = concatenate_embeddings(X)
    y = np.vstack(y)

    return X, y


embedder = Lambda2DEmbedder(
    [get_hand,
     get_pot_cards,
     get_highest_pot_card,
     get_possible_cards(1),
     get_possible_cards(2)],
    [get_split_pot_value,
     get_card_took_flag,
     get_value_cards_took,
     get_value_cards_received],
    include_values=True
)

player = LowPlayer()

np.random.seed(10)
X_train, y_train = generate_dataset(player, embedder, 30000)

np.random.seed(100)
X_valid, y_valid = generate_dataset(player, embedder, 10000)

with open('q_redesign/whole_dataset.pkl', 'wb') as f:
    pickle.dump((X_train, y_train, X_valid, y_valid), f)
