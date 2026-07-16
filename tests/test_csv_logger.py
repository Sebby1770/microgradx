"""CSVLogger — write metrics rows and close cleanly."""
import csv
from pathlib import Path

import microgradx as mg
from microgradx import CSVLogger


def test_csv_logger_writes_header_and_rows(tmp_path):
    path = tmp_path / "run.csv"
    log = CSVLogger(str(path))
    log.log(epoch=1, loss=0.5, acc=0.9)
    log.log(epoch=2, loss=0.3, acc=0.95)
    log.close()

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["epoch"] == "1"
    assert rows[0]["loss"] == "0.5"
    assert rows[0]["acc"] == "0.9"
    assert rows[1]["epoch"] == "2"
    assert float(rows[1]["loss"]) == 0.3


def test_csv_logger_context_manager(tmp_path):
    path = tmp_path / "ctx.csv"
    with CSVLogger(str(path)) as log:
        log.log(step=0, lr=1e-3)
        log.log(step=1, lr=5e-4)
    assert path.exists()
    text = path.read_text()
    assert "step" in text and "lr" in text
    assert "0.001" in text or "0.0005" in text or "5e-04" in text or "0.0005" in text


def test_csv_logger_import_from_package():
    assert CSVLogger is mg.CSVLogger


def test_csv_logger_creates_file(tmp_path):
    path = Path(tmp_path) / "metrics.csv"
    log = CSVLogger(str(path))
    log.log(a=1)
    log.close()
    assert path.is_file()
    assert path.stat().st_size > 0
