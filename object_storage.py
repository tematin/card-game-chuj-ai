from encoders import Lambda2DEmbedder, get_hand, get_pot_cards, get_possible_cards, \
    get_highest_pot_card, get_pot_value, get_card_took_flag, get_historically_played_cards
import torch
from torch import nn


def get_embedder_v1():
    return Lambda2DEmbedder([get_highest_pot_card,
                             get_hand,
                             get_historically_played_cards,
                             get_pot_cards],
                            [get_pot_value,
                             get_card_took_flag])


def get_embedder_v2():
    return Lambda2DEmbedder([get_hand,
                             get_pot_cards,
                             get_highest_pot_card,
                             get_possible_cards(1),
                             get_possible_cards(2)],
                            [get_pot_value,
                             get_card_took_flag])


class NeuralNetwork(nn.Module):
    def __init__(self, base_conv_filters=30, conv_filters=30, dropout_p=0.2, in_channels=7):
        super(NeuralNetwork, self).__init__()
        self.by_parts = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=base_conv_filters, kernel_size=(1, 3)),
            nn.Dropout2d(dropout_p),
            nn.ReLU()
        )
        self.by_colour = nn.Sequential(
            nn.Conv2d(in_channels=base_conv_filters, out_channels=conv_filters, kernel_size=(1, 7)),
            nn.Dropout2d(dropout_p),
            nn.ReLU()
        )
        self.by_value = nn.Sequential(
            nn.Conv2d(in_channels=base_conv_filters, out_channels=conv_filters, kernel_size=(4, 3)),
            nn.Dropout2d(dropout_p),
            nn.ReLU()
        )

        self.final_dense = nn.Sequential(
            nn.LazyLinear(400),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.LazyLinear(300),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.LazyLinear(300),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.LazyLinear(200),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.LazyLinear(1)
        )

    def forward(self, card_encoding, rest_encoding):
        by_parts = self.by_parts(card_encoding)
        by_colour = torch.flatten(self.by_colour(by_parts), start_dim=1)
        by_value = torch.flatten(self.by_value(by_parts), start_dim=1)

        flat = torch.cat([by_colour, by_value, rest_encoding], dim=1)
        return self.final_dense(flat)


def get_model():
    return NeuralNetwork(base_conv_filters=40, conv_filters=30, dropout_p=0.2).to("cuda")
