from collections import defaultdict
from time import time
from typing import Optional
import numpy as np
from pprint import pprint


class Timer:
    def __init__(self):
        self.clear()

    def trace(self, name: Optional[str] = None):
        def wrapper(func):
            resolved_name = name or func.__name__

            def decorated_method(*args, **kwargs):
                t = time()
                ret = func(*args, **kwargs)
                self._times[resolved_name].append(time() - t)
                return ret

            return decorated_method
        return wrapper

    def means(self):
        return {k: np.mean(v) for k, v in self._times.items()}

    def print_means(self):
        pprint(self.means())

    def totals(self):
        return {k: np.sum(v) for k, v in self._times.items()}

    def print_totals(self):
        pprint(self.totals())

    def call_count(self):
        return {k: len(v) for k, v in self._times.items()}

    def print_call_count(self):
        pprint(self.call_count())

    def clear(self):
        self._times = defaultdict(list)


timer = Timer()
