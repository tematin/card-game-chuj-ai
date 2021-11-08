from encoders import Lambda2DEmbedder, get_hand, get_pot_cards, get_possible_cards, \
    get_highest_pot_card, get_pot_value, get_card_took_flag, get_historically_played_cards
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def get_embedder_v1():
    return Lambda2DEmbedder([get_highest_pot_card,
                             get_hand,
                             get_historically_played_cards,
                             get_pot_cards],
                            [get_pot_value,
                             get_card_took_flag])


def get_keras_model():
    l2 = 0.01
    dropout_rate = 0.3
    filters = 30
    card_inp = keras.Input(shape=(5, 4, 9, 1))
    rest_inp = keras.Input(shape=(4))

    by_parts = layers.Conv3D(filters=filters, kernel_size=(5, 1, 3), padding='valid',
                             activation="relu")(card_inp)
    by_colour = layers.Conv3D(filters=filters, kernel_size=(1, 1, 7), padding='valid',
                              activation="relu")(by_parts)
    by_colour = layers.Flatten()(by_colour)

    by_value = layers.Conv3D(filters=filters, kernel_size=(1, 4, 3), padding='valid',
                             activation="relu")(by_parts)
    by_value = layers.Flatten()(by_value)

    x = layers.concatenate([by_colour, by_value, rest_inp])
    x = layers.Dense(300, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(200, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout_rate)(x)
    val = layers.Dense(1)(x)

    model = keras.Model([card_inp, rest_inp], val)
    return model


def get_torch_model():
    pass


def get_embedder_v2():
    return Lambda2DEmbedder([get_hand,
                             get_pot_cards,
                             get_highest_pot_card,
                             get_possible_cards(1),
                             get_possible_cards(2)],
                            [get_pot_value,
                             get_card_took_flag])


