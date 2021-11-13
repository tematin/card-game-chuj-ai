from encoders import Lambda2DEmbedder, get_hand, get_pot_cards, get_possible_cards, \
    get_highest_pot_card, get_pot_value, get_card_took_flag, get_historically_played_cards


def get_embedder_v1():
    return Lambda2DEmbedder([get_highest_pot_card,
                             get_hand,
                             get_historically_played_cards,
                             get_pot_cards],
                            [get_pot_value,
                             get_card_took_flag])


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

