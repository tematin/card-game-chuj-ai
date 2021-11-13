import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.X)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 200),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


target_f = lambda x: (((2 * x[:, 1]) ** (x[:, 0] * 4).astype(int))
                      - np.log(x[:, 2]) * np.sin(x[:, 3]) + x[:, 4] * 4
                      + np.cos(10 * x[:, 0]) * 5
                      - (x[:, 1] + x[:, 2] + 2 * x[:, 4]) ** 2).reshape(-1, 1)

train_X = np.random.rand(10000, 5)
valid_X = np.random.rand(10000, 5)
offset_X = np.random.rand(10000, 5)
offset_X[:, 0] += 1


train_dataset = NumpyDataset(train_X, target_f(train_X) + 20)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)

valid_dataset = NumpyDataset(valid_X, target_f(valid_X))
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=64)

offset_dataset = NumpyDataset(offset_X, target_f(offset_X))
offset_dataloader = DataLoader(offset_dataset, shuffle=False, batch_size=64)


model = NeuralNetwork().to("cuda")
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

def train(dataset, epochs=1):
    model.train()
    for _ in range(epochs):
        for X, y in dataset:
            X, y = X.float().to("cuda"), y.float().to("cuda")
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate(dataset):
    model.train(False)
    test_loss = 0
    for X, y in dataset:
        X, y = X.float().to("cuda"), y.float().to("cuda")
        pred = model(X)
        loss = loss_fn(pred, y)
        test_loss += loss.item() * len(X)
    return test_loss / len(dataset)



train(train_dataloader, 40)
print(evaluate(train_dataloader))
print(evaluate(valid_dataloader))
print(evaluate(offset_dataloader))
print(get_std(train_dataloader))
print(get_std(offset_dataloader))

def get_std(dataset):
    for X, y in dataset:
        break

    X, y = X.float().to("cuda"), y.float().to("cuda")
    model.train()
    a = []
    for _ in range(100):
        a.append(model(X).to("cpu").detach().numpy())

    print(np.hstack(a).std(1).mean())
