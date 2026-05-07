"""Image-style augmentations. Operate on (C, H, W) NumPy arrays."""
from __future__ import annotations
import numpy as np
from typing import Callable, List, Tuple


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class Normalize:
    """(x - mean) / std, broadcast across spatial dims."""
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)

    def __call__(self, x):
        return (x.astype(np.float32) - self.mean) / self.std


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            return x[..., ::-1].copy()
        return x


class RandomCrop:
    """Pad then crop to size — standard CIFAR-style augmentation."""
    def __init__(self, size: Tuple[int, int], padding: int = 0):
        self.size = size
        self.padding = padding

    def __call__(self, x):
        if self.padding:
            x = np.pad(x, ((0, 0), (self.padding, self.padding),
                           (self.padding, self.padding)))
        _, H, W = x.shape
        h, w = self.size
        top = np.random.randint(0, H - h + 1)
        left = np.random.randint(0, W - w + 1)
        return x[:, top:top + h, left:left + w]


class ToFloat:
    """Convert uint8 [0,255] to float32 [0,1]."""
    def __call__(self, x):
        return x.astype(np.float32) / 255.0
