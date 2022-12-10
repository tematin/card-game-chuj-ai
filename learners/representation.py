from __future__ import annotations

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from game.utils import Observation


@dataclass
class TrainTuple:
    observation: Observation
    target: float


def concatenate_feature_list(feature_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    if len(feature_list) == 0:
        return []

    ret = []
    for i in range(len(feature_list[0])):
        ret.append(np.concatenate([x[i] for x in feature_list], axis=0))

    return ret


def index_observation(x: Observation, idx: int):
    x.features = [xx[idx:idx+1] for xx in x.features]
    return x
