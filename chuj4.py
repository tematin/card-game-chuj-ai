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
    def __init__(self, channel_size, kernel_width, padding, dropout_p=0.2, leak=0.01, activate=True):
        super().__init__()
        self.activate = activate

        self.first_layer = nn.Sequential(
            nn.LazyConv2d(out_channels=channel_size,
                          kernel_size=kernel_width, padding=padding),
            nn.Dropout2d(dropout_p),
            nn.LeakyReLU(leak),
            nn.LazyConv2d(out_channels=channel_size,
                          kernel_size=kernel_width, padding=padding),
            nn.Dropout2d(dropout_p),
        )
        self.final_activation = nn.LeakyReLU(leak)
        self.downsample = nn.LazyConv2d(out_channels=channel_size, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.first_layer(x)
        out = out + self.downsample(x)
        if self.activate:
            return self.final_activation(out)
        else:
            return out


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


class ThreePhaseResNeuralNetwork(nn.Module):
    def __init__(self, dense_sizes, depth, dropout_p=0.1, leak=0.01):
        super().__init__()
        convolution_layer = []

        for _ in range(depth):
            layer = ConvResNet(channel_size=40, kernel_width=(1, 3), padding='same',
                               dropout_p=dropout_p, leak=leak)
            convolution_layer.append(layer)

        layer = nn.Sequential(
            nn.LazyConv2d(out_channels=200,
                          kernel_size=(1, 9), padding='valid'),
            nn.Dropout2d(dropout_p),
            nn.LeakyReLU(leak)
        )
        convolution_layer.append(layer)

        for _ in range(depth):
            layer = ConvResNet(channel_size=200, kernel_width=(1, 1), padding='same',
                               dropout_p=dropout_p, leak=leak)
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


class ThreePhaseResNeuralNetworkPlus(nn.Module):
    def __init__(self, dense_sizes, depth, dropout_p=0.1, leak=0.01):
        super().__init__()

        convolution_layer = []
        for _ in range(depth):
            layer = ConvResNet(channel_size=40, kernel_width=(1, 3), padding='same',
                               dropout_p=dropout_p, leak=leak)
            convolution_layer.append(layer)
        self.conv_broad = nn.Sequential(*convolution_layer)

        self.conv_bridge = nn.Sequential(
            nn.LazyConv2d(out_channels=200,
                          kernel_size=(1, 5), padding='valid'),
            nn.Dropout2d(dropout_p),
            nn.LeakyReLU(leak),
            nn.LazyConv2d(out_channels=200,
                          kernel_size=(1, 5), padding='valid'),
            nn.Dropout2d(dropout_p),
            nn.LeakyReLU(leak)
        )

        self.conv_tight_ramp = ConvResNet(
            channel_size=200, kernel_width=(1, 1), padding='same',
            dropout_p=dropout_p, leak=leak, activate=False
        )

        self.skip_connection = nn.LazyConv2d(out_channels=200, kernel_size=(1, 9), padding='valid')
        self.skip_activation = nn.LeakyReLU(leak)

        convolution_layer = []
        for i in range(depth - 1):
            layer = ConvResNet(channel_size=200, kernel_width=(1, 1), padding='same',
                               dropout_p=dropout_p, leak=leak)
            convolution_layer.append(layer)

        self.conv_tight = nn.Sequential(*convolution_layer)

        dense_layer = []
        for size in dense_sizes:
            layer = DenseResNet(size)
            dense_layer.append(layer)
        dense_layer.append(nn.LazyLinear(1))

        self.final_dense = nn.Sequential(*dense_layer)


    def forward(self, card_encoding):
        broad = self.conv_broad(card_encoding)
        tight = self.conv_bridge(broad)
        tight_ramp = self.conv_tight_ramp(tight)
        activated_ramp = self.skip_activation(tight_ramp + self.skip_connection(broad))
        flat = torch.flatten(self.conv_tight(activated_ramp), start_dim=1)
        return self.final_dense(flat)



dataset = load_data(add_value_layer=True)

loss_fn = nn.MSELoss()

comparator = SuiteComparator()

comparator.add(
    suite=TrainSuite(
        model=ThreePhaseResNeuralNetworkPlus(dense_sizes=[[200, 200], [200, 200]], depth=2).to("cuda"),
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={'weight_decay': 0.04},
        loss_fn=loss_fn,
        dataset=dataset
    ),
    name="plusplus"
)

comparator.add(
    suite=TrainSuite(
        model=ThreePhaseResNeuralNetwork(dense_sizes=[[200, 200], [200, 200]], depth=2).to("cuda"),
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={'weight_decay': 0.04},
        loss_fn=loss_fn,
        dataset=dataset
    ),
    name="three_phase"
)

comparator.train_to_epochs(10)
comparator.plot_compare()

comparator.train_to_epochs(25)
comparator.plot_compare()


plt.plot(comparator.suites[-1].train_accuracy)
plt.plot(comparator.suites[-1].test_accuracy)

plt.plot(comparator.suites[-2].train_accuracy)
plt.plot(comparator.suites[-2].test_accuracy)

plt.plot(comparator.suites[-3].train_accuracy)
plt.plot(comparator.suites[-3].test_accuracy)


model = comparator.suites[0].model


unscaled_dataset = load_data(scale_X=False, scale_y=False)

X = dataset.valid_X
X_uns = unscaled_dataset.valid_X
y_true = dataset.valid_y
y_true = -unscaled_dataset.valid_y

model.train(False)

cuda_X = torch.tensor(X).float().to("cuda")
y_pred = model(cuda_X).detach().to("cpu").numpy().flatten()

def descale(x):
    ma = 1.2288641831570022
    mi = -2.567583087675631
    x -= mi
    x *= 21 / (ma - mi)
    return 21 - x

y_pred = descale(y_pred)

y_true = descale(y_true)

plt.scatter(y_true, y_pred, alpha=0.05)

res = y_pred - y_true
plt.hist(res, 50)


idx = np.where(res > 10)[0]

hard = X_uns[idx]
y_pred[idx]
y_true[idx]


from game.game import GameRound, Card, Hand, generate_hands, get_deck
from baselines import LowPlayer

plt.hist(res, 50)

player = LowPlayer()

b = 3
sample_avg = []
for b in tqdm(range(len(X_uns))):
    samples = []
    for _ in range(100):
        one_emb = X_uns[b]

        game = GameRound(0)

        hands = [[], [], []]
        for player_int, colour, value in zip(*np.where(one_emb)):
            hands[player_int].append(Card(colour=int(colour), value=int(value)))

        #hands[0] = list(np.sort(hands[0]))
        #hands[1] = list(np.sort(hands[1]))
        #hands[2] = list(np.sort(hands[2]))

        np.random.shuffle(hands[0])
        np.random.shuffle(hands[1])
        np.random.shuffle(hands[2])

        game.hands = [Hand(hands[0]),
                      Hand(hands[1]),
                      Hand(hands[2])]

        while not game.end:
            if game.trick_end:
                game.play()
                continue

            obs = game.observe()
            card = player.play(obs)
            game.play(card)

        samples.append(game.tracker.score.score[0])
    sample_avg.append(np.mean(samples))

sample_avg = np.array(sample_avg)

print(y_pred[idx[b]])
print(y_true[idx[b]])


plt.scatter(y_true, y_pred, alpha=0.05)
plt.scatter(sample_avg, y_pred, alpha=0.05)
plt.scatter(sample_avg, y_true, alpha=0.05)


plt.hist(res, 50, alpha=0.5)
plt.hist(y_pred - sample_avg, 50, alpha=0.5)


np.sqrt(np.mean(res ** 2))
np.sqrt(np.mean((y_pred - sample_avg) ** 2))
