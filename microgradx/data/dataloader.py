"""DataLoader — shuffles + batches + (optional) augments."""
from __future__ import annotations
from typing import Iterator, Callable, Optional, List, Any
import numpy as np

from microgradx.data.dataset import Dataset, default_collate


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate
        self.transform = transform
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        n = len(self.dataset)
        idx = np.arange(n)
        if self.shuffle:
            self._rng.shuffle(idx)
        for start in range(0, n, self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            if len(batch_idx) < self.batch_size and self.drop_last:
                continue
            samples = [self.dataset[int(i)] for i in batch_idx]
            if self.transform is not None:
                samples = [self.transform(s) for s in samples]
            yield self.collate_fn(samples)
