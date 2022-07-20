import pickle
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from time import time


from training.models import to_cuda_list


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


class ConvResNet(nn.Module):
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
        main_layer = nn.Sequential(*main_layer)
        activation = nn.LeakyReLU(leak)
        downsample = nn.LazyLinear(layer_densities[-1], bias=False)

        self.layer = SkipConnection(main_layer=main_layer,
                                    downsample=downsample,
                                    activation=activation)

    def forward(self, x):
        return self.layer(x)


class BlockConvResNet(nn.Module):
    def __init__(self, depth, channel_size, kernel_width, padding, dropout_p, leak):
        super().__init__()

        convolution_layer = []
        for _ in range(depth):
            layer = ConvResNet(channel_size=channel_size,
                               kernel_width=kernel_width, padding=padding,
                               dropout_p=dropout_p, leak=leak)
            convolution_layer.append(layer)
        self.convolution_layer = nn.Sequential(*convolution_layer)

    def forward(self, x):
        return self.convolution_layer(x)


class BlockDenseResNet(nn.Module):
    def __init__(self, dense_sizes):
        super().__init__()
        dense_layer = []
        for size in dense_sizes:
            layer = DenseResNet(size)
            dense_layer.append(layer)
        dense_layer.append(nn.LazyLinear(1))

        self.dense = nn.Sequential(*dense_layer)

    def forward(self, X):
        return self.dense(X)


class MainNetwork(nn.Module):
    def __init__(self, dense_sizes, depth, dropout_p=0.1, leak=0.01):
        super().__init__()

        self.conv_broad = BlockConvResNet(
            depth=depth, channel_size=40, kernel_width=(1, 3),
            padding='same', dropout_p=dropout_p, leak=leak
        )

        conv_bridge = BlockConvResNet(
            depth=depth, channel_size=200, kernel_width=(1, 5),
            padding='valid', dropout_p=dropout_p, leak=leak
        )

        colour_conv = ConvResNet(
            channel_size=200, kernel_width=(1, 1), padding='same',
            dropout_p=dropout_p, leak=leak, activate=False
        )

        self.bridge = SkipConnection(
            main_layer=nn.Sequential(conv_bridge, colour_conv),
            downsample=nn.LazyConv2d(out_channels=200, kernel_size=(1, 9), padding='valid'),
            activation=nn.LeakyReLU(leak)
        )

        self.conv_tight = BlockConvResNet(
            depth=depth, channel_size=200, kernel_width=(1, 1),
            padding='same', dropout_p=dropout_p, leak=leak
        )

        self.dense = BlockDenseResNet(dense_sizes)

        self.final_dense = nn.Sequential(
            nn.LazyLinear(13),
            nn.Sigmoid()
        )

    def forward(self, card_encoding, rest_encoding):
        broad = self.conv_broad(card_encoding)

        tight = self.bridge(broad)

        flat = torch.flatten(self.conv_tight(tight), start_dim=1)

        concatenated = torch.cat([flat, rest_encoding], dim=1)

        return self.final_dense(concatenated)


class Dataset:
    def __init__(self, train_X, train_y, valid_X, valid_y, scaler_X=None, scaler_y=None):
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        if self.scaler_X is not None:
            train_X = self.scaler_X.fit_transform(train_X)
            valid_X = self.scaler_X.transform(valid_X)

        if self.scaler_y is not None:
            train_y = self.scaler_y.fit_transform(train_y)
            valid_y = self.scaler_y.transform(valid_y)

        self.train_X = train_X
        self.train_y = train_y
        self.valid_X = valid_X
        self.valid_y = valid_y


class EmbeddingScaler:
    def __init__(self, scalers):
        self.scalers = scalers

    def fit(self, X):
        if X.length == 0:
            self.scalers.fit(X.X)
        else:
            for i in range(X.length):
                self.scalers[i].fit(X.X[i])

    def transform(self, X):
        if X.length == 0:
            self.scalers.transform(X.X)
        else:
            for i in range(X.length):
                X.X[i] = self.scalers[i].transform(X.X[i])
        return X

    def inverse_transform(self, X):
        if X.length == 0:
            self.scalers.inverse_transform(X.X)
        else:
            for i in range(X.length):
                X.X[i] = self.scalers[i].inverse_transform(X.X[i])
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class MultiDimensionalScaler:
    def __init__(self, collapse_axis):
        self.collapse_axis = collapse_axis

    def fit(self, X):
        self.mean = X.mean(self.collapse_axis, keepdims=True)
        self.std = X.std(self.collapse_axis, keepdims=True)

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return (X * self.std) + self.mean

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def info():
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


def generate_indexes(size, batch_size, randomize=True):
    idx = np.arange(size)
    if randomize:
        np.random.shuffle(idx)
    i = 0
    while i < size:
        yield idx[i:(i + batch_size)]
        i += batch_size


def train_epoch(X, y, model, optimizer, loss_fn, batch_size=128):
    model.train(True)
    size = y.shape[0]

    y = torch.tensor(y).float().to("cuda")

    for idx in tqdm(generate_indexes(size, batch_size),
                    total=X.data_count // batch_size, ascii=True):
        batch_X = X[idx]
        batch_y = y[idx]

        batch_X = to_cuda_list(batch_X)
        batch_y = torch.tensor(batch_y).float().to("cuda")

        pred = model(*batch_X)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del batch_X
        del batch_y


def batch_predict(X, model, batch_size=128):
    model.train(False)
    size = X.data_count

    ret = []

    for idx in tqdm(generate_indexes(size, batch_size, randomize=False),
                    total=X.data_count // batch_size, ascii=True):

        batch_X = X[idx]
        batch_X = to_cuda_list(batch_X)

        pred = model(*batch_X).to("cpu").detach()
        ret.append(pred)

        del batch_X

    return torch.cat(ret, dim=0)


class TrainSuite:
    def __init__(self, model, optimizer, dataset, optimizer_kwargs):
        self.model = model
        self.optimizer = optimizer(model.parameters(), **optimizer_kwargs)

        self.train_bce = []
        self.valid_bce = []

        self.train_mse = []
        self.valid_mse = []

        self.dataset = dataset

    def train(self, epochs=1):
        for _ in range(epochs):
            t = time()
            info()
            train_epoch(self.dataset.train_X, self.dataset.train_y, self.model,
                        self.optimizer, nn.BCELoss(), batch_size=2**11)

            pred = batch_predict(self.dataset.train_X, self.model, 2**11)
            self.train_bce.append(self.bce_evaluate(pred, self.dataset.train_y))
            self.train_mse.append(self.point_loss(pred, self.dataset.train_y))

            pred = batch_predict(self.dataset.valid_X, self.model, 2**11)
            self.valid_bce.append(self.bce_evaluate(pred, self.dataset.valid_y))
            self.valid_mse.append(self.point_loss(pred, self.dataset.valid_y))
            print(time() - t)

    def bce_evaluate(self, pred, true):
        true = torch.tensor(true, dtype=torch.float)
        loss = nn.BCELoss()(pred, true).item()
        del true
        return loss

    def point_loss(self, pred, true):
        pred = pred.detach().to("cpu").numpy()

        value = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 8, 0, 0])

        true = (true * value).sum(1)
        pred = (pred * value).sum(1)

        return np.mean((true - pred) ** 2)


    @property
    def length(self):
        return len(self.train_bce)


def get_dataset():
    with open('q_redesign/whole_dataset.pkl', 'rb') as f:
        loaded = pickle.load(f)


    scaler_X = EmbeddingScaler([MultiDimensionalScaler((0, 2, 3)),
                                StandardScaler()])
    np.random.seed(0)

    a, b, c, d = loaded

    #mask = np.random.random(a.data_count) < 0.1
    #a = a[mask]
    #b = b[mask]

    #mask = np.random.random(c.data_count) < 0.1
    #c = c[mask]
    #d = d[mask]

    return Dataset(a, b, c, d, scaler_X=scaler_X)


dataset = get_dataset()

model = MainNetwork([[200, 200], [200, 200]], 3).to("cuda")


suite = TrainSuite(
    model=model,
    optimizer=torch.optim.AdamW,
    optimizer_kwargs={'weight_decay': 0.04},
    dataset=dataset
)

suite.train(30)

plt.plot(suite.train_bce)
plt.plot(suite.valid_bce)
plt.show()

plt.plot(suite.train_mse)
plt.plot(suite.valid_mse)
plt.show()


halt

descaled_features = dataset.scaler_X.inverse_transform(dataset.valid_X)
left = descaled_features.X[0][:, 0].sum((1, 2))


pred = batch_predict(dataset.valid_X, model, 2 ** 10)
pred = pred.numpy()

true = dataset.valid_y

value = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 8, 0, 0])
pred_agg = np.sum(pred * value, axis=1)
true_agg = np.sum(true * value, axis=1)

residual = pred_agg - true_agg


plt.scatter(pred_agg, true_agg, alpha=0.001)
plt.hist(residual, 50)

plt.scatter(left, residual, alpha=0.01)
plt.boxplot([residual[left == (x + 1)] for x in range(12)], whis=3)


idx = 1
mask = true[:, idx] == 1
plt.hist(pred[mask, idx], 50, alpha=0.4)
plt.hist(pred[~mask, idx], 50, alpha=0.4)

plt.show()

plt.scatter(pred[:, 4], pred[:, 5], alpha=0.1)


mask = (left == 1) & (np.abs(residual) > 1.2)

dx = dataset.valid_X[mask]
dy = dataset.valid_y[mask]


short_value = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 8])
in_the_pot = (dx.X[1][:, :4] * np.array([1, 1, 4, 8])).sum(1)
took = (dx.X[1][:, 7:18] * short_value).sum(1)
received = (dx.X[1][:, 18:29] * short_value).sum(1)

value_left = 21 - (took + received)

idx = 1

dx.X[1][idx, 7:18]
dx.X[1][idx, 18:29]


dy[idx]
np.round(pred[mask][idx], 3)


dx[idx].X[0]




