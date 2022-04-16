import matplotlib.pyplot as plt

from game.game import GameRound
from encoders import Lambda2DEmbedder, get_hand, get_pot_cards, get_first_pot_card, get_pot_value, get_card_took_flag, \
    concatenate_embeddings, get_current_score, LambdaEmbedder
from baselines import LowPlayer, RandomPlayer
from evaluation_scripts import finish_game
import torch
from torch import nn

from tqdm import tqdm
import numpy as np

player = LowPlayer()
simple_embedder = LambdaEmbedder([get_hand,
                                  lambda x: x.right_hand,
                                  lambda x: x.left_hand],
                                 [])
X = []
y = []

for _ in tqdm(range(300000)):
    game = GameRound(0)

    X.append(simple_embedder.get_state_embedding(game.observe()).X.flatten())
    finish_game(game, [player, player, player])

    cards_received = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for pot in game.tracker.history.history:
        if pot.get_pot_owner() != 0:
            continue
        if pot.get_point_value() == 0:
            continue

        for card in pot._cards:
            if card.get_point_value() == 1:
                cards_received[card.value] += 1
            elif card.get_point_value() == 4:
                cards_received[-2] += 1
            elif card.get_point_value() == 8:
                cards_received[-1] += 1

    y.append(cards_received)

X = np.stack(X).reshape(-1, 3, 4, 9)
y = np.array(y)

mean = 1/3
std = np.sqrt(1/3 * ((1 - mean) ** 2) + 2/3 * ((0 - mean) ** 2))

X = (X - mean) / std

value_matrix = np.zeros((1, 1, 4, 9))
value_matrix[0, 0, 0, :] = 1
value_matrix[0, 0, 1, -3] = 4
value_matrix[0, 0, 2, -3] = 8
mean = value_matrix.mean()
std = value_matrix.std()
value_matrix = (value_matrix - mean) / std

stack_train = np.repeat(value_matrix, repeats=X.shape[0], axis=0)
X = np.concatenate([X, stack_train], axis=1)


X = np.load('sep_X.npy')
y = np.load('sep_y.npy')

y_orig = np.stack([y[:, :9].sum(1), y[:, 9], y[:, 10]]).T

y_ord = np.zeros(y.shape)
y_ord[:, -1] = y[:, -1]
y_ord[:, -2] = y[:, -2]
for i, val in enumerate(y_orig[:, 0]):
    y_ord[i, :val] = 1

rows = y.shape[0]
y_b = np.zeros((rows, 2))
y_s = np.zeros((rows, 10))

y_b[:, 0] = y[:, -2]
y_b[:, 1] = y[:, -1]

for i, val in enumerate(y_orig[:, 0]):
    y_s[i, val] = 1

y_s = y_orig[:, :1]

from sklearn.preprocessing import StandardScaler

#target_scaler = StandardScaler()
#y = target_scaler.fit_transform(y)

class DimThreePhaseResNeuralNetwork(nn.Module):
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

        self.final_dense = nn.Sequential(*dense_layer)

        self.big_cards = nn.Sequential(
            nn.LazyLinear(2),
            nn.Sigmoid()
        )

        self.small_cards = nn.Sequential(
            nn.LazyLinear(9),
            nn.Sigmoid()
        )

    def forward(self, card_encoding):
        conv = torch.flatten(self.convolution_layer(card_encoding), start_dim=1)
        last_dense = self.final_dense(conv)
        return self.small_cards(last_dense), self.big_cards(last_dense)


dmodel = DimThreePhaseResNeuralNetwork(dense_sizes=[[200, 200], [200, 200]], depth=2).to("cuda")

optimizer = torch.optim.AdamW(dmodel.parameters(), **{'weight_decay': 0.04})


def BinomialLoss(input, target):
    return - (target * torch.log(input) + (9 - target) * torch.log(1 - input)).mean()


def CustomLoss(input, target):
    success_mask = target > torch.arange(0, 9).to("cuda")
    equal_mask = target == torch.arange(0, 9).to("cuda")

    return - (torch.log(input[success_mask]).sum() + torch.log(1 - input[equal_mask]).sum()) / input.shape[0]


def dtrain(X, y_s, y_b, dmodel, optimizer, batch_size=128):
    dmodel.train(True)
    size = y.shape[0]

    X_cude = torch.tensor(X).float().to("cuda")
    y_s_cuda = torch.tensor(y_s).float().to("cuda").long()
    y_b_cuda = torch.tensor(y_b).float().to("cuda")

    for idx in generate_indexes(size, batch_size):
        pass
        batch_X = X_cude[idx]
        batch_y_s = y_s_cuda[idx]
        batch_y_b = y_b_cuda[idx]

        pred_s, pred_b = dmodel(batch_X)
        loss = nn.BCELoss()(pred_b, batch_y_b) + CustomLoss(pred_s, batch_y_s)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


losses_4 = []
for _ in tqdm(range(11)):
    dtrain(X, y_s, y_b, dmodel, optimizer, batch_size=128)

    dmodel.train(False)

    valid_X = dataset.valid_X
    cuda_X = torch.tensor(valid_X).float().to("cuda")
    y_pred_s, y_pred_b = dmodel(cuda_X)

    y_pred_s = y_pred_s.detach().to("cpu").numpy()
    y_pred_b = y_pred_b.detach().to("cpu").numpy()

    #y_pred = target_scaler.inverse_transform(y_pred)
    #y_pred = y_pred * np.array([1, 4, 8])
    y_pred_b_sum = (y_pred_b * np.array([4, 8])).sum(1)

    positive = np.cumprod(y_pred_s, axis=1)
    negative = (1 - y_pred_s)[:, 1:]
    negative = np.hstack([negative, np.ones((negative.shape[0], 1))])

    y_pred_s_sum = ((positive * negative) * (np.arange(9) + 1)).sum(1)

    #y_pred_s_sum = (y_pred_s * 9).sum(1)

    y_pred_new = y_pred_s_sum + y_pred_b_sum

    res_new = y_pred_new - y_true
    losses_4.append(np.sqrt((res_new**2).mean()))


plt.hist(res, 50, alpha=0.3)
plt.hist(res_new, 50, alpha=0.3)
plt.plot(losses)
plt.plot(losses_2)
plt.plot(losses_4)



input = torch.rand((10, 9)).to("cuda")
target = torch.randint(0, 10, size=(10, 1)).to("cuda")

CustomLoss(input, target)