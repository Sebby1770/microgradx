"""Top-k accuracy metrics."""
import numpy as np
import microgradx as mg
from microgradx.metrics import accuracy


def test_top1_accuracy_perfect():
    # logits such that argmax matches targets
    logits = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 5.0, 1.0],
            [1.0, 2.0, 9.0],
        ],
        dtype=np.float32,
    )
    targets = np.array([0, 1, 2], dtype=np.int64)
    acc = accuracy(logits, targets, topk=1)
    assert np.isclose(acc, 1.0)


def test_top1_accuracy_half():
    logits = np.array(
        [
            [10.0, 0.0],
            [0.0, 5.0],
            [3.0, 1.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    targets = np.array([0, 0, 0, 0], dtype=np.int64)  # hits on rows 0 and 2
    acc = accuracy(logits, targets, topk=(1,))
    assert np.isclose(acc, 0.5)


def test_topk_tuple():
    logits = np.array(
        [
            [0.1, 0.2, 0.9],  # true=0 is rank 3 → miss top1, miss top2
            [0.9, 0.5, 0.1],  # true=0 is rank 1 → hit top1
            [0.1, 0.9, 0.5],  # true=0 is rank 3 → miss
        ],
        dtype=np.float32,
    )
    targets = np.array([0, 0, 0], dtype=np.int64)
    top1, top2 = accuracy(logits, targets, topk=(1, 2))
    assert np.isclose(top1, 1.0 / 3.0)
    # top2: row0 still miss (0 is 3rd), row1 hit, row2 hit (0 is 3rd? scores: 0.1,0.9,0.5 → ranks 1,2,0 for classes 1,2,0 — class0 is rank 3)
    # row2: class order by score: 1 (0.9), 2 (0.5), 0 (0.1) → top2 = {1,2}, miss
    assert np.isclose(top2, 1.0 / 3.0)


def test_accuracy_with_tensor():
    logits = mg.Tensor(np.eye(3, dtype=np.float32) * 5)
    targets = mg.Tensor(np.array([0, 1, 2], dtype=np.int64))
    acc = accuracy(logits, targets, topk=(1,))
    assert np.isclose(acc, 1.0)


def test_accuracy_export():
    import microgradx as mgx
    assert hasattr(mgx, "accuracy")
    assert hasattr(mgx, "metrics")
