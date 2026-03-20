from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_CURVE_COLUMNS = {
    "cell_id",
    "cycle_index",
    "step_type",
    "time_s",
    "voltage_v",
    "current_a",
    "capacity_ah",
    "temperature_c",
}


OPTIONAL_CURVE_COLUMNS = {
    "internal_resistance_ohm",
    "soc",
}


REQUIRED_LABEL_COLUMNS = {"cell_id", "eol_cycle"}


class SchemaError(ValueError):
    """Raised when an input dataset does not match the expected schema."""



def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use CSV or Parquet.")



def _check_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise SchemaError(f"{name} is missing required columns: {missing}")



def load_curve_data(path: str | Path) -> pd.DataFrame:
    df = _read_table(path)
    _check_columns(df, REQUIRED_CURVE_COLUMNS, "Curve dataset")

    df = df.copy()
    df["step_type"] = df["step_type"].astype(str).str.lower().str.strip()
    df["cell_id"] = df["cell_id"].astype(str)
    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce").astype("Int64")

    numeric_cols = [
        "time_s",
        "voltage_v",
        "current_a",
        "capacity_ah",
        "temperature_c",
        "internal_resistance_ohm",
        "soc",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["cell_id", "cycle_index", "step_type", "time_s", "voltage_v", "capacity_ah"])
    df = df.sort_values(["cell_id", "cycle_index", "step_type", "time_s"]).reset_index(drop=True)
    return df



def load_label_data(path: str | Path) -> pd.DataFrame:
    df = _read_table(path)
    _check_columns(df, REQUIRED_LABEL_COLUMNS, "Label dataset")
    df = df.copy()
    df["cell_id"] = df["cell_id"].astype(str)
    df["eol_cycle"] = pd.to_numeric(df["eol_cycle"], errors="coerce")
    df = df.dropna(subset=["cell_id", "eol_cycle"]).drop_duplicates(subset=["cell_id"])
    return df.reset_index(drop=True)
