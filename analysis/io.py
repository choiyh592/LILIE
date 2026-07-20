"""Small IO helpers shared across modules.

Tabular intermediates are written as Parquet when a Parquet engine is
available (the spec names ``qeeg.parquet``); otherwise we transparently fall
back to CSV so the pipeline still runs on a minimal stack. ``read_table``
reads whichever exists.
"""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def write_table(df: pd.DataFrame, path_no_ext: str) -> str:
    """Write ``df`` to ``<path_no_ext>.parquet`` if possible else ``.csv``.

    Returns the actual path written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path_no_ext)), exist_ok=True)
    if _parquet_available():
        path = path_no_ext + ".parquet"
        df.to_parquet(path, index=False)
    else:
        path = path_no_ext + ".csv"
        df.to_csv(path, index=False)
    return path


def read_table(path_no_ext: str) -> pd.DataFrame:
    for ext, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = path_no_ext + ext
        if os.path.exists(p):
            return reader(p)
    # allow a full path with extension already
    if os.path.exists(path_no_ext):
        if path_no_ext.endswith(".parquet"):
            return pd.read_parquet(path_no_ext)
        return pd.read_csv(path_no_ext)
    raise FileNotFoundError(f"No .parquet or .csv found for {path_no_ext}")


def write_json(obj: Any, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
