from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List
import numpy as np


class Transformer(ABC):
    @abstractmethod
    def fit(self, x: Union[np.ndarray, float]) -> None:
        pass

    @abstractmethod
    def transform(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        pass

    @abstractmethod
    def inverse_transform(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        pass

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


class MultiDimensionalScaler(Transformer):
    def __init__(self, collapse_axis):
        self.collapse_axis = collapse_axis

    def fit(self, x: np.ndarray) -> None:
        self._mean = x.mean(self.collapse_axis, keepdims=True)
        self._std = x.std(self.collapse_axis, keepdims=True)

    def transform(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return (x - self._mean) / self._std

    def inverse_transform(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return (x * self._std) + self._mean

    @property
    def params(self) -> dict:
        return {
            'mean': self._mean,
            'std': self._std
        }

    def save(self, path: Path) -> None:
        path.mkdir(exist_ok=True)
        np.savez(path / 'scaler.npz', mean=self._mean, std=self._std)

    def load(self, path: Path) -> None:
        arrays = np.load(str(path / 'scaler.npz'))
        self._mean = arrays['mean']
        self._std = arrays['std']


class SimpleScaler(Transformer):
    def fit(self, x: np.ndarray) -> None:
        self._mean = x.mean()
        self._std = x.std()

    def transform(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return (x - self._mean) / self._std

    def inverse_transform(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return (x * self._std) + self._mean

    @property
    def params(self) -> dict:
        return {
            'mean': self._mean,
            'std': self._std
        }

    def save(self, path: Path) -> None:
        path.mkdir(exist_ok=True)
        np.savez(path / 'scaler.npz', mean=self._mean, std=self._std)

    def load(self, path: Path) -> None:
        arrays = np.load(str(path / 'scaler.npz'))
        self._mean = arrays['mean']
        self._std = arrays['std']


class ListTransformer:
    def __init__(self, transformers: List[Transformer]) -> None:
        self._transformers = transformers

    def fit(self, x: List[np.ndarray]) -> None:
        for transformer, arr in zip(self._transformers, x):
            transformer.fit(arr)

    def transform(self, x: List[np.ndarray]) -> List[np.ndarray]:
        ret = []
        for transformer, arr in zip(self._transformers, x):
            ret.append(transformer.transform(arr))
        return ret

    def inverse_transform(self, x: List[np.ndarray]) -> List[np.ndarray]:
        ret = []
        for transformer, arr in zip(self._transformers, x):
            ret.append(transformer.inverse_transform(arr))
        return ret

    def save(self, path: Path) -> None:
        path.mkdir(exist_ok=True)
        for i, transformer in enumerate(self._transformers):
            transformer.save(path / str(i))

    def load(self, path: Path) -> None:
        for i, transformer in enumerate(self._transformers):
            transformer.load(path / str(i))


class DummyListTransformer:
    def fit(self, x: List[np.ndarray]) -> None:
        pass

    def transform(self, x: List[np.ndarray]) -> List[np.ndarray]:
        return x

    def inverse_transform(self, x: List[np.ndarray]) -> List[np.ndarray]:
        return x

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass
