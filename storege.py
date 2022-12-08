
class Tracker:
    def __init__(self, ref_dataset: List[np.ndarray],
                 models: List[Approximator]) -> None:
        self._dataset = ref_dataset
        self._models = models

        self._data = [[] for _ in self._models]

    def register(self) -> None:
        for i, model in enumerate(self._models):
            self._data[i].append(model.get(self._dataset))


tracker_instance = Tracker(
    ref_dataset=[scaler_3d.transform(X[0]), scaler_flat.transform(X[1])],
    models=[
        agent._q[0]._approximator._approximator._model,
        agent._q[0]._approximator._approximator._final_model,
        agent._q[1]._approximator._approximator._model,
        agent._q[1]._approximator._approximator._final_model,
    ]
)
