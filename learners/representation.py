from __future__ import annotations

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TrainTuple:
    features: List[np.ndarray]
    target: float


def unpack_train_tuple_list(train_data: List[TrainTuple]
                            ) -> Tuple[List[np.ndarray], np.ndarray]:
    features = concatenate_feature_list([x.features for x in train_data])
    targets = np.array([x.target for x in train_data])
    return features, targets


def concatenate_feature_list(feature_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    if len(feature_list) == 0:
        return []

    ret = []
    for i in range(len(feature_list[0])):
        ret.append(np.concatenate([x[i] for x in feature_list], axis=0))

    return ret


def index_array_list(x: List[np.ndarray], idx: int):
    return [xx[idx:idx+1] for xx in x]
