import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt

import torch
from torch import nn


def generate_indexes(size, batch_size):
    idx = np.arange(size)
    np.random.shuffle(idx)
    i = 0
    while i < size:
        yield idx[i:(i + batch_size)]
        i += batch_size


def train(X, y, model, optimizer, loss_fn, batch_size=128):
    model.train(True)
    size = y.shape[0]

    X = torch.tensor(X).float().to("cuda")
    y = torch.tensor(y).float().reshape(-1, 1).to("cuda")

    for idx in generate_indexes(size, batch_size):
        batch_X = X[idx]
        batch_y = y[idx]

        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(X, y, model, loss_fn):
    size = y.shape[0]
    model.train(False)

    X = torch.tensor(X).float().to("cuda")
    y = torch.tensor(y).float().to("cuda").reshape(-1, 1)

    losses = 0
    for idx in generate_indexes(size, 2**9):
        batch_X = X[idx]
        batch_y = y[idx]

        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)
        losses += loss.item() * len(idx)
    losses /= size
    return losses


def load_data(scale_X=True, scale_y=True, add_value_layer=False):
    train_X = np.load('train_X.npz.npy')
    train_y = np.load('train_y.npz.npy')
    val_X = np.load('valid_X.npz.npy')
    val_y = np.load('valid_y.npz.npy')

    if scale_X:
        mean = 1/3
        std = np.sqrt(1/3 * ((1 - mean) ** 2) + 2/3 * ((0 - mean) ** 2))

        train_X = (train_X - mean) / std
        val_X = (val_X - mean) / std

    if scale_y:
        mean = train_y.mean()
        std = train_y.std()

        train_y = (train_y - mean) / std
        val_y = (val_y - mean) / std

    if add_value_layer:
        value_matrix = np.zeros((1, 1, 4, 9))
        value_matrix[0, 0, 0, :] = 1
        value_matrix[0, 0, 1, -3] = 4
        value_matrix[0, 0, 2, -3] = 8
        if scale_X:
            mean = value_matrix.mean()
            std = value_matrix.std()
            value_matrix = (value_matrix - mean) / std
        stack_train = np.repeat(value_matrix, repeats=train_X.shape[0], axis=0)
        stack_val = np.repeat(value_matrix, repeats=val_X.shape[0], axis=0)
        train_X = np.concatenate([train_X, stack_train], axis=1)
        val_X = np.concatenate([val_X, stack_val], axis=1)

    return Dataset(train_X, train_y, val_X, val_y)


class Dataset:
    def __init__(self, train_X, train_y, valid_X, valid_y):
        self.train_X = train_X
        self.train_y = train_y
        self.valid_X = valid_X
        self.valid_y = valid_y


class TrainSuite:
    def __init__(self, model, optimizer, loss_fn, optimizer_kwargs, dataset):
        self.model = model
        self.optimizer = optimizer(model.parameters(), **optimizer_kwargs)
        self.loss_fn = loss_fn
        self.train_accuracy = []
        self.test_accuracy = []
        self.dataset = dataset

    def train(self, epochs=1):
        for _ in tqdm(range(epochs)):
            train(dataset.train_X, dataset.train_y, self.model, self.optimizer, loss_fn, batch_size=2**10)
            self.train_accuracy.append(evaluate(dataset.train_X, dataset.train_y, self.model, loss_fn))
            self.test_accuracy.append(evaluate(dataset.valid_X, dataset.valid_y, self.model, loss_fn))

    def plot_perf(self):
        plt.plot(self.train_accuracy)
        plt.plot(self.test_accuracy)
        print(np.min(self.test_accuracy))

    @property
    def length(self):
        return len(self.train_accuracy)


class SuiteComparator:
    def __init__(self):
        self.suites = []
        self.names = []

    def add(self, suite, name=None):
        self.suites.append(suite)
        if name is None:
            name = f'model_{len(self.suites)}'
        self.names.append(name)

    def train_to_epochs(self, epochs):
        for suite in self.suites:
            while suite.length < epochs:
                suite.train()

    def plot_compare(self):
        for suite in self.suites:
            plt.plot(suite.test_accuracy)
        plt.legend(self.names)


class ConvResNet(nn.Module):
    def __init__(self, channel_size, kernel_width, padding, dropout_p=0.2, leak=0.01):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.LazyConv2d(out_channels=channel_size,
                          kernel_size=(1, kernel_width), padding=padding),
            nn.Dropout2d(dropout_p),
            nn.LeakyReLU(leak),
            nn.LazyConv2d(out_channels=channel_size,
                          kernel_size=(1, kernel_width), padding=padding),
            nn.Dropout2d(dropout_p),
        )
        self.final_activation = nn.LeakyReLU(leak)
        self.downsample = nn.LazyConv2d(out_channels=channel_size, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.first_layer(x)
        out = out + self.downsample(x)
        return self.final_activation(out)


class DenseResNet(nn.Module):
    def __init__(self, layer_densities, dropout_p=0.2, leak=0.01):
        super().__init__()
        main_layer = []
        for layer_density in layer_densities:
            main_layer.extend([
                nn.LazyLinear(layer_density),
                nn.Dropout2d(dropout_p),
                nn.LeakyReLU(leak)
            ])
        self.first_layer = nn.Sequential(*main_layer)
        self.final_activation = nn.LeakyReLU(leak)
        self.downsample = nn.LazyLinear(layer_densities[-1], bias=False)

    def forward(self, x):
        out = self.first_layer(x)
        out = out + self.downsample(x)
        return self.final_activation(out)


class ResNeuralNetwork(nn.Module):
    def __init__(self, dense_sizes, depth, dropout_p=0.1, leak=0.01):
        super().__init__()
        convolution_layer = []

        for _ in range(depth):
            layer = ConvResNet(channel_size=40, kernel_width=3, padding='same',
                           dropout_p=dropout_p, leak=leak)
            convolution_layer.append(layer)

        layer = nn.Sequential(
            nn.LazyConv2d(out_channels=20,
                          kernel_size=(1, 9), padding='valid'),
            nn.Dropout2d(dropout_p),
            nn.LeakyReLU(leak)
        )
        convolution_layer.append(layer)

        self.convolution_layer = nn.Sequential(*convolution_layer)

        dense_layer = []
        for size in dense_sizes:
            layer = nn.Sequential(
                nn.LazyLinear(size),
                nn.Dropout(dropout_p),
                nn.LeakyReLU(leak)
            )
            dense_layer.append(layer)
        dense_layer.append(nn.LazyLinear(1))

        self.final_dense = nn.Sequential(*dense_layer)

    def forward(self, card_encoding):
        conv = torch.flatten(self.convolution_layer(card_encoding), start_dim=1)
        return self.final_dense(conv)


class FullResNeuralNetwork(nn.Module):
    def __init__(self, dense_sizes, depth, dropout_p=0.1, leak=0.01):
        super().__init__()
        convolution_layer = []

        for _ in range(depth):
            layer = ConvResNet(channel_size=40, kernel_width=3, padding='same',
                           dropout_p=dropout_p, leak=leak)
            convolution_layer.append(layer)

        layer = nn.Sequential(
            nn.LazyConv2d(out_channels=20,
                          kernel_size=(1, 9), padding='valid'),
            nn.Dropout2d(dropout_p),
            nn.LeakyReLU(leak)
        )
        convolution_layer.append(layer)

        self.convolution_layer = nn.Sequential(*convolution_layer)

        dense_layer = []
        for size in dense_sizes:
            layer = DenseResNet(size)
            dense_layer.append(layer)
        dense_layer.append(nn.LazyLinear(1))

        self.final_dense = nn.Sequential(*dense_layer)

    def forward(self, card_encoding):
        conv = torch.flatten(self.convolution_layer(card_encoding), start_dim=1)
        return self.final_dense(conv)


dataset = load_data(add_value_layer=True)

loss_fn = nn.MSELoss()

comparator = SuiteComparator()

comparator.add(
    suite=TrainSuite(
        model=ResNeuralNetwork(dense_sizes=[200, 200, 200, 200], depth=4).to("cuda"),
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={'weight_decay': 0.04},
        loss_fn=loss_fn,
        dataset=dataset
    ),
    name="baseline4+3"
)

comparator.add(
    suite=TrainSuite(
        model=FullResNeuralNetwork(dense_sizes=[[200, 200], [200, 200]], depth=4).to("cuda"),
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={'weight_decay': 0.04},
        loss_fn=loss_fn,
        dataset=dataset
    ),
    name="longer_model_6+6"
)

comparator.train_to_epochs(3)
comparator.plot_compare()

comparator.train_to_epochs(30)
comparator.plot_compare()


#model = NeuralNetwork(base_conv_filters=20, conv_filters=15, dropout_p=0.1,
#                      depth=10, girth=150, in_channels=4, leak=0.01).to("cuda")

#model = NewNeuralNetwork(dense_sizes=[200, 200, 200],
#                         channel_sizes=[20, 20, 20, 20, 20, 20, 20],
#                         kernel_widths=[3, 3, 3, 3, 3, 3, 9],
#                         paddings=['same', 'same', 'same', 'same', 'same', 'same', 'valid'],
#                         dropout_p=0.1, leak=0.01).to("cuda")

#model = ResNeuralNetwork(dense_sizes=[200, 200, 200],
#                         depth=4).to("cuda")

#optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.04)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.003, weight_decay=0.02)
#optimizer = torch.optim.RMSProp(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.01)





