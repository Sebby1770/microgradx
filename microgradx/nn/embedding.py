"""Embedding lookup table — autograd via the GetItem op (uses np.add.at)."""
import numpy as np

from microgradx.tensor import Tensor
from microgradx.nn.module import Module


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Standard small init
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02,
            requires_grad=True,
        )

    def forward(self, idx) -> Tensor:
        # idx may be a Tensor of ints or a numpy array of ints
        if isinstance(idx, Tensor):
            key = idx.data.astype(np.int64)
        else:
            key = np.asarray(idx, dtype=np.int64)
        return self.weight[key]
