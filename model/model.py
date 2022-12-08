import torch
from torch import nn


class SkipConnection(nn.Module):
    def __init__(self, main_layer, downsample, activation=None):
        super().__init__()
        self.main_layer = main_layer
        self.downsample = downsample
        self.activation = activation

    def forward(self, x):
        out = self.main_layer(x) + self.downsample(x)
        if self.activation is None:
            return out
        else:
            return self.activation(out)


class ResLayer(nn.Module):
    def __init__(self, channel_size, kernel_width, padding,
                 dropout_p, leak, activate=True):
        super().__init__()
        if activate:
            activation = nn.LeakyReLU(leak)
        else:
            activation = None

        main_layer = nn.Sequential(
            nn.LazyConv2d(out_channels=channel_size,
                          kernel_size=kernel_width, padding=padding),
            nn.Dropout2d(dropout_p),
            nn.LeakyReLU(leak),
            nn.LazyConv2d(out_channels=channel_size,
                          kernel_size=kernel_width, padding=padding),
            nn.Dropout2d(dropout_p),
        )

        downsample = nn.LazyConv2d(out_channels=channel_size,
                                   kernel_size=1, bias=False)

        self.layer = SkipConnection(main_layer=main_layer,
                                    downsample=downsample,
                                    activation=activation)

    def forward(self, x):
        return self.layer(x)


class DenseSkipConnection(nn.Module):
    def __init__(self, layer_densities, dropout_p=0.2, leak=0.01):
        super().__init__()
        main_layer = []
        for layer_density in layer_densities:
            main_layer.extend([
                nn.LazyLinear(layer_density),
                nn.Dropout(dropout_p),
                nn.LeakyReLU(leak)
            ])
        main_layer = nn.Sequential(*main_layer)
        activation = nn.LeakyReLU(leak)
        downsample = nn.LazyLinear(layer_densities[-1], bias=False)

        self.layer = SkipConnection(main_layer=main_layer,
                                    downsample=downsample,
                                    activation=activation)

    def forward(self, x):
        return self.layer(x)


class BlockConvResLayer(nn.Module):
    def __init__(self, depth, channel_size, kernel_width, padding, dropout_p, leak):
        super().__init__()

        convolution_layer = []
        for _ in range(depth):
            layer = ResLayer(channel_size=channel_size,
                             kernel_width=kernel_width, padding=padding,
                             dropout_p=dropout_p, leak=leak)
            convolution_layer.append(layer)
        self.convolution_layer = nn.Sequential(*convolution_layer)

    def forward(self, x):
        return self.convolution_layer(x)


class BlockDenseSkipConnections(nn.Module):
    def __init__(self, dense_sizes):
        super().__init__()
        dense_layer = []
        for size in dense_sizes:
            layer = DenseSkipConnection(size)
            dense_layer.append(layer)

        self.dense = nn.Sequential(*dense_layer)

    def forward(self, X):
        return self.dense(X)


class MainNetwork(nn.Module):
    def __init__(self, dense_sizes, depth, dropout_p=0.1, leak=0.01):
        super().__init__()

        self.conv_broad = BlockConvResLayer(
            depth=depth, channel_size=80, kernel_width=(1, 3),
            padding='same', dropout_p=dropout_p, leak=leak
        )

        conv_bridge = BlockConvResLayer(
            depth=depth, channel_size=160, kernel_width=(1, 5),
            padding='valid', dropout_p=dropout_p, leak=leak
        )

        colour_conv = ResLayer(
            channel_size=160, kernel_width=(1, 1), padding='same',
            dropout_p=dropout_p, leak=leak, activate=False
        )

        self.bridge = SkipConnection(
            main_layer=nn.Sequential(conv_bridge, colour_conv),
            downsample=nn.LazyConv2d(out_channels=160, kernel_size=(1, 9), padding='valid'),
            activation=nn.LeakyReLU(leak)
        )

        self.conv_tight = BlockConvResLayer(
            depth=depth, channel_size=40, kernel_width=(1, 1),
            padding='same', dropout_p=dropout_p, leak=leak
        )

        self.dense = BlockDenseSkipConnections(dense_sizes)

        self.final_dense = nn.LazyLinear(1)

    def forward(self, card_encoding, rest_encoding):
        broad = self.conv_broad(card_encoding)

        tight = self.bridge(broad)

        flat = torch.flatten(self.conv_tight(tight), start_dim=1)

        concatenated = torch.cat([flat, rest_encoding], dim=1)

        dense = self.dense(concatenated)

        return self.final_dense(dense)


class SimpleNetwork(nn.Module):
    def __init__(self, base_conv_filters=30, conv_filters=30, dropout_p=0.2):
        super().__init__()

        self.by_parts = nn.Sequential(
            nn.LazyConv2d(out_channels=base_conv_filters, kernel_size=(1, 3)),
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


