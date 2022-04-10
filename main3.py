
np.random.seed(10)
val_X, val_y = generate_datapoints(player, reward, embedder, episodes=10000)

model = NeuralNetwork(base_conv_filters=60, conv_filters=60).to("cuda")
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

while True:
    train_X, train_y = generate_datapoints(player, reward, embedder, episodes=20000)
    train(train_X, train_y, model, optimizer, loss_fn, batch_size=2**10, epochs=1)
    print(evaluate(train_X, train_y, model, loss_fn))
    print(evaluate(val_X, val_y, model, loss_fn))

None

np.random.seed(1)
eye_X, eye_y = generate_datapoints(player, reward, embedder, episodes=50)
eye_X2, eye_y2 = generate_datapoints(player, reward, embedder, episodes=50)

X = to_cuda_list(eye_X)
y = torch.tensor(eye_y).float().reshape(-1, 1).to("cuda")
pred = model(*X)

pred = pred.to("cpu").detach().numpy().flatten()
true = eye_y


cards_left = eye_X.X[0].sum((2, 3))[:, 0]
points_left = eye_X.X[1][:, -3:].sum(1)

res = (pred - true)

plt.scatter(cards_left, res, alpha=0.2)
plt.scatter(points_left, res, alpha=0.2)

idx = np.where(res > 8)[0]

eye_X[idx[0]].X

from evaluation_scripts import finish_game


embedder = Lambda2DEmbedder([get_hand,
                             get_pot_cards,
                             get_first_pot_card,
                             lambda x: x.right_hand,
                             lambda x: x.left_hand],
                            [get_pot_value,
                             get_card_took_flag,
                             get_current_score])

simple_embedder = LambdaEmbedder([get_hand,
                                  get_pot_cards,
                                  get_first_pot_card,
                                  lambda x: x.right_hand,
                                  lambda x: x.left_hand],
                                 [get_pot_value,
                                  get_card_took_flag,
                                  get_current_score])

