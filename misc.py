import matplotlib.pyplot as plt

valid_states = generate_stateaction_dataset(BaselineEncoder(), RandomPlayer(), 2000)
valid_states = np.vstack(valid_states)

original = keras.Input(shape=(145))
x = layers.Dense(400, activation="relu")(original)
x = layers.Dense(300, activation="relu")(x)
x = layers.Dense(300, activation="relu")(x)
encoded = layers.Dense(10)(x)
x = layers.Dense(400, activation="relu")(encoded)
x = layers.Dense(300, activation="relu")(x)
x = layers.Dense(300, activation="relu")(x)
reconstructed = layers.Dense(145, activation="relu")(x)

encoder = keras.Model(original, encoded)
autoencoder = keras.Model(original, reconstructed)

autoencoder.compile(loss='mse', optimizer='adam')
autoencoder.summary()

for i in range(5):
    states = generate_stateaction_dataset(BaselineEncoder(), RandomPlayer(), 5000)
    states = np.vstack(states)

    epochs = 25 if i == 0 else 5
    autoencoder.fit(states, states, verbose=True, batch_size=64, epochs=epochs)

    predicted = autoencoder.predict(valid_states)
    print("Validation loss:", np.mean((predicted - valid_states)**2))

encoder.predict(valid_states)

states = generate_stateaction_dataset(BaselineEncoder(), LowPlayer(), 2000)
predicted = autoencoder.predict(states)
print("Validation loss:", np.mean((predicted - states) ** 2))



def train(self, iters):
    starter = 0
    for i in range(iters):
        print(i)
        embeddings = []
        starter = (starter + 1) % 3
        game = GameRound(starter)
        while True:
            if game.phasing_player == 0:
                if np.random.rand() < 0.05:
                    card = RandomPlayer().play(*game.observe())
                    emb = self.embedder.encode(*game.observe(), card)
                else:
                    card, emb = self.play(*game.observe(), return_embedding=True)
                embeddings.append(emb)
            else:
                card = self.play(*game.observe())
            game.play(card)
            if game.end:
                break
        X = np.vstack(embeddings)
        y = np.full((len(embeddings), 1), game.get_points()[0])
        self.q_function.fit(X, y)



def get_points(p):
    ret = np.array([0, 0, 0])
    idx = np.random.choice(3, p=p, size=11)
    points = np.array([8, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    for i, j in zip(idx, points):
        ret[i] += j
    return ret

def get_loss_prob(a):
    h = []
    pr = a/21
    for i in range(2000):
        p = np.array([pr, (1-pr)/2, (1-pr)/2])
        score = np.array([0, 0, 0])
        while max(score) < 100:
            score += get_points(p)
        h.append(score[0])
    h = np.array(h)
    return (h >= 100).mean()

X = np.linspace(4, 8, 50)
y = [get_loss_prob(x) for x in X]


X = np.array([1, 2, 3, 3.5, 3.8, 4, 4.1, 4.5, 4.6, 4.7]).reshape(-1, 1)
y = np.sin(X * 2) +
y = np.array([2, 1.5, 1.2, 1.8, 2.1, 0.8, 3, 1, 4, 2.2]).reshape(-1, 1)

kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
model = GaussianProcessRegressor(alpha=0.001, kernel=kernel, normalize_y=True, n_restarts_optimizer=9)
model.fit(X, y)

pred_X = np.linspace(1, 5, 20).reshape(-1, 1)

y_pred, sigma = model.predict(pred_X, return_std=True)
y_pred = y_pred.flatten()

plt.plot(pred_X, y_pred + 2 * sigma)
plt.plot(pred_X, y_pred - 2 * sigma)
plt.scatter(X, y)





