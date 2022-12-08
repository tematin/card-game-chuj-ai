from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
from typing import List, Union, Type, Optional, Any
from abc import abstractmethod, ABC
import torch
from torch import nn

from .representation import concatenate_feature_list
from .transformers import Transformer


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

    def train_mode(self, val: bool) -> None:
        pass

    @property
    def update_mode_dependent(self) -> bool:
        return False

    def decay(self) -> None:
        pass


class Buffer(Approximator):

    def __init__(self, approximator: Approximator,
                 buffer_size: int) -> None:
        self._max_buffer_size = buffer_size
        self._approximator = approximator

        self._staged_features = []
        self._staged_targets = []

        self._buffer_size = 0

    def update(self, x: List[np.ndarray], y: np.ndarray) -> None:
        self._staged_features.append(x)
        self._staged_targets.append(y)
        self._buffer_size += len(x)

        if self._buffer_size >= self._max_buffer_size:
            concat_x = concatenate_feature_list(self._staged_features)
            concat_y = np.concatenate(self._staged_targets, axis=0)

            split_concat_x = [np.split(x, [self._max_buffer_size]) for x in concat_x]
            ret_x = [x[0] for x in split_concat_x]
            rest_x = [x[1] for x in split_concat_x]

            ret_y, rest_y = np.split(concat_y, [self._max_buffer_size])

            self._approximator.update(ret_x, ret_y)

            self._staged_features = [rest_x]
            self._staged_targets = [rest_y]
            self._buffer_size = len(rest_x)

    def get(self, x: List[np.ndarray], update_mode: bool = False) -> np.ndarray:
        return self._approximator.get(x, update_mode)

    @property
    def params(self) -> dict:
        return {
            'buffer_size': self._buffer_size,
            'approximator': self._approximator
        }

    def save(self, directory: Path) -> None:
        self._approximator.save(directory)

    def load(self, directory: Path) -> None:
        self._approximator.load(directory)

    def update_mode_dependent(self) -> bool:
        return self._approximator.update_mode_dependent

    def decay(self) -> None:
        self._approximator.decay()


class TargetTransformer(Approximator):

    def __init__(self, approximator: Approximator,
                 transformer: Transformer) -> None:
        self._approximator = approximator
        self._transformer = transformer

    def update(self, x: List[np.ndarray], y: np.ndarray) -> None:
        y = self._transformer.transform(y)
        self._approximator.update(x, y)

    def get(self, x: List[np.ndarray], update_mode: bool = False) -> np.ndarray:
        val = self._approximator.get(x, update_mode)
        return self._transformer.inverse_transform(val)

    @property
    def params(self) -> dict:
        return {
            'transformer': self._transformer,
            'approximator': self._approximator
        }

    def save(self, directory: Path) -> None:
        self._approximator.save(directory)

    def load(self, directory: Path) -> None:
        self._approximator.load(directory)

    def update_mode_dependent(self) -> bool:
        return self._approximator.update_mode_dependent

    def decay(self) -> None:
        self._approximator.decay()


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

    def get(self, x: List[np.ndarray], update_mode: bool = False) -> np.ndarray:
        self.model.train(False)
        x = [torch.tensor(xx).float().to(self._device) for xx in x]
        pred = self.model(*x).to("cpu").detach().numpy()
        return pred.flatten()

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

    def train_mode(self, val: bool) -> None:
        self.model.train(val)

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

    def get(self, x: List[np.ndarray], update_mode: bool = False) -> np.ndarray:
        if update_mode:
            return self._model.get(x)
        else:
            return self._final_model.get(x)

    def update(self, x: List[np.ndarray], y: np.ndarray) -> None:
        self._model.update(x, y)

        for target_param, param in zip(self._final_model.model.parameters(),
                                       self._model.model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self._tau) + param.data * self._tau
            )

    def update_mode_dependent(self) -> bool:
        return True

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