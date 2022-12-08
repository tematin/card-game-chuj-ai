from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any, Optional, Tuple, Union, TypeVar
import numpy as np


T = TypeVar('T')


@dataclass
class MemoryStep:
    features: List[np.ndarray]
    action_took: Optional[int]
    reward: float


class Memory(ABC):

    @abstractmethod
    def reset_episode(self) -> None:
        pass

    @abstractmethod
    def set(self, step: T, skip: bool = False) -> None:
        pass

    @abstractmethod
    def get(self) -> List[List[T]]:
        pass

    @property
    @abstractmethod
    def yield_length(self) -> int:
        pass


class _InnerListStorage:
    def __init__(self, items: Optional[List[List[Any]]] = None):
        self._items = items or []

    def append(self, item: Any) -> None:
        if not self._items:
            self._items.append([])

        self._items[-1].append(item)

    def append_outer(self) -> None:
        self._items.append([])

    def __getitem__(self, idx: Union[Tuple[int, int], int]) -> Any:
        if isinstance(idx, int):
            return self._items[idx]
        else:
            return self._items[idx[0]][idx[1]]

    def length(self, idx: Optional[int] = None) -> int:
        if idx is None:
            return len(self._items)
        else:
            return len(self._items[idx])

    def total_length(self) -> int:
        return sum([self.length(i) for i in range(self.length())])

    def remove(self, idx: Optional[int] = None, count: int = 1) -> None:
        if idx is None:
            self._items = self._items[count:]
        else:
            self._items[idx] = self._items[idx][count:]


class PassthroughMemory(Memory):

    def __init__(self, yield_length: int) -> None:
        self._steps = _InnerListStorage()
        self._keep = _InnerListStorage()
        self._yield_length = yield_length

    def reset_episode(self) -> None:
        self._steps.append_outer()
        self._keep.append_outer()

    def set(self, step: T, skip: bool = False) -> None:
        self._steps.append(step)
        self._keep.append(not skip)

    def get(self) -> List[List[T]]:
        ret = []
        while (self._steps.length() > 1
                or self._steps.length(0) >= self._yield_length):

            if ((sum(self._keep[0]) == 0
                    or self._keep.length(0) < self._yield_length)):
                self._steps.remove()
                self._keep.remove()
                break

            if self._keep[0, 0]:
                ret.append(self._steps[0, :self._yield_length])

            self._steps.remove(0)
            self._keep.remove(0)

        return ret

    @property
    def yield_length(self) -> int:
        return self._yield_length

    @property
    def params(self) -> dict:
        return {
            'type': 'passthrough',
            'yield_length': self._yield_length,
        }


class EmptyMemory(Memory):

    def reset_episode(self) -> None:
        pass

    def set(self, step: T, skip: bool = False) -> None:
        pass

    def get(self) -> List[List[T]]:
        return []

    @property
    def yield_length(self) -> int:
        return 0

    @property
    def params(self) -> dict:
        return {
            'type': 'empty',
        }


class ReplayMemory(Memory):

    def __init__(self, yield_length: int, memory_size: int,
                 extraction_count: int, ramp_up_size: int) -> None:
        self._steps = _InnerListStorage()
        self._keep = _InnerListStorage()
        self._memory_size = memory_size + yield_length - 1
        self._yield_length = yield_length
        self._extraction_count = extraction_count
        self._ramp_up_size = ramp_up_size + yield_length - 1
        self._length = 0

    def reset_episode(self) -> None:
        self._steps.append_outer()
        self._keep.append_outer()

    def set(self, step: T, skip: bool = False) -> None:
        self._length += int(not skip)
        self._steps.append(step)
        self._keep.append(not skip)

        while self._length > self._memory_size:
            while sum(self._keep[0]) == 0:
                self._steps.remove()
                self._keep.remove()

            keep = self._keep[0, 0]
            self._steps.remove(0)
            self._keep.remove(0)

            if keep:
                self._length -= 1
                break

    def get(self) -> List[List[T]]:
        if self._length < self._yield_length:
            return []

        if self._length < self._ramp_up_size:
            return []

        keep_idx = {i: np.where(self._keep[i, :-(self._yield_length - 1)])[0]
                    for i in range(self._keep.length())}
        episode_prob = np.array([len(keep_idx[i]) for i in range(self._keep.length())])
        episode_prob = episode_prob / episode_prob.sum()

        ret = []
        for _ in range(self._extraction_count):
            idx = np.random.choice(np.arange(len(episode_prob)), p=episode_prob)
            start = np.random.choice(keep_idx[idx])
            end = start + self._yield_length

            ret.append(self._steps[idx, start:end])
        return ret

    @property
    def yield_length(self) -> int:
        return self._yield_length

    @property
    def params(self) -> dict:
        return {
            'type': 'replay',
            'memory_size': self._memory_size,
            'yield_length': self._yield_length,
            'extraction_count': self._extraction_count,
            'ramp_up_size': self._ramp_up_size
        }


class MemoryCombiner(Memory):

    def __init__(self, memories: List[Memory]) -> None:
        self._memories = memories

    def reset_episode(self) -> None:
        for m in self._memories:
            m.reset_episode()

    def set(self, step: T, skip: bool = False) -> None:
        for m in self._memories:
            m.set(step, skip)

    def get(self) -> List[List[T]]:
        ret = []
        for m in self._memories:
            ret.extend(m.get())
        return ret

    @property
    def yield_length(self) -> int:
        return max([m.yield_length for m in self._memories])
