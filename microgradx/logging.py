"""Lightweight experiment loggers.

Currently ships a CSV metrics logger; TensorBoard-style writers are on the
roadmap.
"""
from __future__ import annotations

import csv
from typing import Any, Dict, Optional, TextIO


class CSVLogger:
    """Append named scalar metrics to a CSV file.

    The header row is written on the first :meth:`log` call from the keyword
    argument names. Subsequent rows must supply the same keys (extra keys are
    ignored; missing keys become empty cells).

        from microgradx import CSVLogger
        log = CSVLogger("run.csv")
        log.log(epoch=1, loss=0.5, acc=0.9)
        log.close()

    Also works as a context manager::

        with CSVLogger("run.csv") as log:
            log.log(epoch=1, loss=0.5)
    """

    def __init__(self, path: str):
        self.path = path
        self._file: TextIO = open(path, "w", newline="")
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[list] = None
        self._closed = False

    def log(self, **metrics: Any) -> None:
        if self._closed:
            raise ValueError("CSVLogger is closed")
        if not metrics:
            raise ValueError("log() requires at least one metric keyword")
        if self._writer is None:
            self._fieldnames = list(metrics.keys())
            self._writer = csv.DictWriter(
                self._file, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            self._writer.writeheader()
        # Fill missing keys so DictWriter doesn't error on incomplete rows.
        row: Dict[str, Any] = {k: metrics.get(k, "") for k in self._fieldnames}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
