from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
from typing import List, Union, Type, Optional, Any, Dict, Tuple
from abc import abstractmethod, ABC
import torch
from torch import nn

from game.utils import GamePhase
from .representation import concatenate_feature_list, slice_features
from .transformers import Transformer
from debug.timer import timer


class Approximator(ABC):

    @abstractmethod
    def get(self, x: List[np.ndarray], update_mode: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, x: List[np.ndarray], y: np.ndarray) -> None:
        pass

    def save(self, directory: Path) -> None:
        pass

    def load(self, directory: Path) -> None:
        pass

    def decay(self) -> None:
        pass

    def get_from_list(self, x: List[List[np.ndarray]],
                      update_mode: bool = False) -> List[np.ndarray]:
        lengths = [xx[0].shape[0] for xx in x]
        joined_x = concatenate_feature_list(x)
        y = self.get(joined_x, update_mode=update_mode)
        return np.split(y, np.cumsum(lengths[:-1]))

    def batch_get(self, x: List[np.ndarray], batch_size: int,
                  update_mode: bool = False) -> np.ndarray:
        ret = []
        idx = 0
        while idx < len(x[0]):
            ret.append(self.get(slice_features(x, idx, idx + batch_size),
                                update_mode=update_mode))
            idx += batch_size
        return np.concatenate(ret, axis=0)


class DataTransformer(ABC):

    def transform_input(self, x: List[np.ndarray]) -> List[np.ndarray]:
        return x

    def transform_output(self, y: np.ndarray) -> np.ndarray:
        return y

    def process_update(self, x: List[np.ndarray], y: np.ndarray
                       ) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
        yield x, y

    def save(self, directory: Path) -> None:
        pass

    def load(self, directory: Path) -> None:
        pass


class TransformedApproximator(Approximator):

    def __init__(
            self,
            approximator: Approximator,
            transformers: Union[List[DataTransformer], DataTransformer],
    ) -> None:
        self._approximator = approximator
        self._transformers = (transformers if isinstance(transformers, list)
                              else [transformers])

    def get(self, features: List[np.ndarray], update_mode: bool = False) -> np.ndarray:
        for transformer in self._transformers:
            features = transformer.transform_input(features)

        y = self._approximator.get(features, update_mode=update_mode)

        for transformer in self._transformers:
            y = transformer.transform_output(y)

        return y

    def update(self, x: List[np.ndarray], y: np.ndarray) -> None:
        y = np.array(y).reshape(-1, 1)
        self._apply_update(x, y, self._transformers)

    def _apply_update(self, x: List[np.ndarray], y: np.ndarray,
                      transformers: List[DataTransformer]) -> None:
        for trans_x, trans_y in transformers[0].process_update(x, y):
            if len(transformers) == 1:
                self._approximator.update(trans_x, trans_y)
            else:
                self._apply_update(trans_x, trans_y, transformers[1:])

    def decay(self) -> None:
        self._approximator.decay()

    def save(self, path: Path) -> None:
        path.mkdir(exist_ok=True)
        self._approximator.save(path / 'approximator')
        for i, transformer in enumerate(self._transformers):
            transformer.save(path / f'transformer_{i}')

    def load(self, path: Path) -> None:
        self._approximator.load(path / 'approximator')
        for i, transformer in enumerate(self._transformers):
            transformer.load(path / f'transformer_{i}')


class Buffer(DataTransformer):

    def __init__(self, buffer_size: int) -> None:
        self._max_buffer_size = buffer_size

        self._staged_features = []
        self._staged_targets = []

        self._buffer_size = 0

    def process_update(self, x: List[np.ndarray], y: np.ndarray
                       ) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
        self._staged_features.append(x)
        self._staged_targets.append(y)
        self._buffer_size += len(y)

        while self._buffer_size >= self._max_buffer_size:
            concat_x = concatenate_feature_list(self._staged_features)
            concat_y = np.concatenate(self._staged_targets, axis=0)

            split_concat_x = [np.split(x, [self._max_buffer_size]) for x in concat_x]
            ret_x = [x[0] for x in split_concat_x]
            rest_x = [x[1] for x in split_concat_x]

            ret_y, rest_y = np.split(concat_y, [self._max_buffer_size])

            self._staged_features = [rest_x]
            self._staged_targets = [rest_y]
            self._buffer_size = len(rest_y)

            yield ret_x, ret_y

    @property
    def params(self) -> dict:
        return {
            'buffer_size': self._buffer_size,
        }


class TargetTransformer(DataTransformer):

    def __init__(self, transformer: Transformer) -> None:
        self._transformer = transformer

    def process_update(self, x: List[np.ndarray], y: np.ndarray
                       ) -> Tuple[List[np.ndarray], np.ndarray]:
        y = self._transformer.transform(y)
        yield x, y

    def transform_output(self, y: np.ndarray) -> np.ndarray:
        return self._transformer.inverse_transform(y)

    @property
    def params(self) -> dict:
        return {
            'transformer': self._transformer,
        }

    def save(self, directory: Path) -> None:
        self._transformer.save(directory)

    def load(self, directory: Path) -> None:
        self._transformer.load(directory)


class Torch(Approximator):

    def __init__(self, model: nn.Module,
                 loss: nn.Module,
                 optimizer: Type[torch.optim.Optimizer],
                 optimizer_kwargs: dict,
                 scheduler: Any = None,
                 scheduler_kwargs: Optional[dict] = None) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self._device)
        self._optimizer = optimizer(model.parameters(), **optimizer_kwargs)
        self._loss = loss
        self._optimizer_kwargs = optimizer_kwargs

        if scheduler is not None:
            self._scheduler = scheduler(self._optimizer, **scheduler_kwargs)
        else:
            self._scheduler = None

    @timer.trace("Hard Get")
    def get(self, x: List[np.ndarray], update_mode: bool = False) -> np.ndarray:
        self.model.train(False)
        x = [torch.tensor(xx).float().to(self._device) for xx in x]
        pred = self.model(*x).to("cpu").detach().numpy()
        return pred.flatten()

    @timer.trace("Hard Update")
    def update(self, x: List[np.ndarray], y: np.ndarray) -> None:
        self.model.train(True)
        x = [torch.tensor(xx).float().to(self._device) for xx in x]
        y = torch.tensor(y).float().to(self._device)

        pred = self.model(*x)
        loss = self._loss(pred, y)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    @property
    def params(self) -> dict:
        return {
            'model': str(self.model),
            'loss': str(self._loss),
            'optimizer': str(self._optimizer)
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), directory / 'model.pt')

    def load(self, directory: Path) -> None:
        self.model.load_state_dict(torch.load(directory / 'model.pt'))

    def decay(self) -> None:
        if self._scheduler is not None:
            self._scheduler.step()


class SoftUpdateTorch(Approximator):

    def __init__(self, torch_model: Torch, tau: float, tau_decay: float = 1) -> None:
        self._model = torch_model
        self._final_model = deepcopy(torch_model)
        self._tau = tau
        self._orig_tau = tau
        self._tau_decay = tau_decay

        for target_param, param in zip(self._final_model.model.parameters(),
                                       self._model.model.parameters()):
            target_param.data.copy_(param.data)

    @timer.trace("Soft Get")
    def get(self, x: List[np.ndarray], update_mode: bool = False) -> np.ndarray:
        if update_mode:
            return self._model.get(x)
        else:
            return self._final_model.get(x)

    @timer.trace("Soft Update")
    def update(self, x: List[np.ndarray], y: np.ndarray) -> None:
        self._model.update(x, y)

        for target_param, param in zip(self._final_model.model.parameters(),
                                       self._model.model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self._tau) + param.data * self._tau
            )

    def decay(self) -> None:
        self._tau *= self._tau_decay
        self._model.decay()
        self._final_model.decay()

    def save(self, directory: Path) -> None:
        directory.mkdir(exist_ok=True)
        self._model.save(directory / 'inter')
        self._final_model.save(directory / 'final')

    def load(self, directory: Path) -> None:
        self._model.load(directory / 'inter')
        self._final_model.load(directory / 'final')

    @property
    def params(self) -> dict:
        return {
            'model': self._model,
            'tau': self._orig_tau,
            'tau_decay': self._tau_decay
        }
