from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from torch import nn


class Policy(ABC):
    @abstractmethod
    def get(self, x: List[np.ndarray]) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, x: List[np.ndarray], ) -> None:
        pass
