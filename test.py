import torch
from torch import nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.dense = nn.Sequential(
            nn.LazyLinear(10),
            nn.ReLU(),
            nn.LazyLinear(10),
            nn.ReLU(),
            nn.LazyLinear(1)
        )

    def forward(self, data):
        return self.dense(data)


model = NeuralNetwork().to("cuda")
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

endpoints = []

for i in range(12):
    X = np.random.rand(30, 10)
    X = torch.tensor(X).float().to("cuda")
    y = model(X)
    endpoints.append(y[0])

pred = torch.cat(endpoints)
y = torch.tensor(np.random.rand(12)).to("cuda").float()

loss = loss_fn(pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()


