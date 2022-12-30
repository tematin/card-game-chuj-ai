from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from game.utils import GamePhase


def concatenate_feature_list(feature_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    if len(feature_list) == 0:
        return []

    ret = []
    for i in range(len(feature_list[0])):
        ret.append(np.concatenate([x[i] for x in feature_list], axis=0))

    return ret


def index_features(x: List[np.ndarray], idx: int):
    return [xx[idx:idx+1] for xx in x]


def slice_features(x: List[np.ndarray], start: int, end: int):
    return [xx[start:end] for xx in x]


def index_observation(features: List[np.ndarray], idx: int):
    features = [xx[idx:idx+1] for xx in features]
    return features
