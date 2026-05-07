"""Dataset & dataloader — minimal map-style + iter-style support."""
from __future__ import annotations
from typing import Sequence, Callable, Iterator, Any, Tuple, List
import numpy as np


class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int):
        raise NotImplementedError


class TensorDataset(Dataset):
    """Pairs aligned arrays (e.g. X, y). Returns tuple per index."""
    def __init__(self, *arrays):
        n = len(arrays[0])
        for a in arrays:
            if len(a) != n:
                raise ValueError("all arrays must have the same length")
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, i):
        return tuple(a[i] for a in self.arrays)


def default_collate(batch: Sequence) -> Any:
    """Stack a list of samples into batched arrays."""
    elem = batch[0]
    if isinstance(elem, (tuple, list)):
        return tuple(default_collate([b[i] for b in batch]) for i in range(len(elem)))
    if isinstance(elem, np.ndarray):
        return np.stack(batch)
    if isinstance(elem, (int, float, np.integer, np.floating)):
        return np.asarray(batch)
    return batch
