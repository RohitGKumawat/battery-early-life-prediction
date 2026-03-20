from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    early_cycles: int = 30
    voltage_grid_points: int = 32
    soc_grid_points: int = 25
    include_curve_features: bool = True
    include_delta_features: bool = True



def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])



def _resample_q_of_v(curve: pd.DataFrame, grid_points: int) -> tuple[np.ndarray, np.ndarray]:
    temp = curve[["voltage_v", "capacity_ah"]].dropna().copy()
    if temp.empty or temp["voltage_v"].nunique() < 3:
        return np.array([]), np.array([])

    temp = temp.sort_values("voltage_v")
    temp = temp.groupby("voltage_v", as_index=False)["capacity_ah"].mean()
    v = temp["voltage_v"].to_numpy(dtype=float)
    q = temp["capacity_ah"].to_numpy(dtype=float)

    if np.nanmin(v) == np.nanmax(v):
        return np.array([]), np.array([])

    grid = np.linspace(np.nanmin(v), np.nanmax(v), grid_points)
    q_interp = np.interp(grid, v, q)
    return grid, q_interp



def _resample_v_of_soc(curve: pd.DataFrame, grid_points: int) -> tuple[np.ndarray, np.ndarray]:
    temp = curve[["capacity_ah", "voltage_v"]].dropna().copy()
    if temp.empty or temp["capacity_ah"].nunique() < 3:
        return np.array([]), np.array([])

    q = temp["capacity_ah"].to_numpy(dtype=float)
    q_min, q_max = float(np.nanmin(q)), float(np.nanmax(q))
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_max <= q_min:
        return np.array([]), np.array([])

    temp["soc"] = (temp["capacity_ah"] - q_min) / (q_max - q_min)
    temp = temp.sort_values("soc")
    temp = temp.groupby("soc", as_index=False)["voltage_v"].mean()

    soc = temp["soc"].to_numpy(dtype=float)
    v = temp["voltage_v"].to_numpy(dtype=float)
    if len(np.unique(soc)) < 3:
        return np.array([]), np.array([])

    grid = np.linspace(0.0, 1.0, grid_points)
    v_interp = np.interp(grid, soc, v)
    return grid, v_interp



def _curve_summary_features(prefix: str, values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    feats: dict[str, float] = {
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_std": float(np.nanstd(values)),
        f"{prefix}_min": float(np.nanmin(values)),
        f"{prefix}_max": float(np.nanmax(values)),
        f"{prefix}_range": float(np.nanmax(values) - np.nanmin(values)),
        f"{prefix}_area": float(np.trapz(values)),
    }
    for i, val in enumerate(values):
        feats[f"{prefix}_p{i:02d}"] = float(val)
    return feats



def _basic_cycle_features(cycle_df: pd.DataFrame, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    out[f"{prefix}_capacity_ah"] = float(cycle_df["capacity_ah"].max() - cycle_df["capacity_ah"].min())
    out[f"{prefix}_voltage_mean"] = float(cycle_df["voltage_v"].mean())
    out[f"{prefix}_voltage_std"] = float(cycle_df["voltage_v"].std(ddof=0))
    out[f"{prefix}_current_mean"] = float(cycle_df["current_a"].mean())
    out[f"{prefix}_temp_mean_c"] = float(cycle_df["temperature_c"].mean())
    out[f"{prefix}_temp_max_c"] = float(cycle_df["temperature_c"].max())
    out[f"{prefix}_duration_s"] = float(cycle_df["time_s"].max() - cycle_df["time_s"].min())
    if "internal_resistance_ohm" in cycle_df.columns and cycle_df["internal_resistance_ohm"].notna().any():
        out[f"{prefix}_ir_mean_ohm"] = float(cycle_df["internal_resistance_ohm"].mean())
        out[f"{prefix}_ir_max_ohm"] = float(cycle_df["internal_resistance_ohm"].max())
    return out



def build_feature_matrix(curves: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    curves = curves[curves["cycle_index"] <= config.early_cycles].copy()
    if curves.empty:
        raise ValueError("No rows remain after filtering by early_cycles.")

    feature_rows: list[dict[str, float | str]] = []

    for cell_id, cell_df in curves.groupby("cell_id"):
        row: dict[str, float | str] = {"cell_id": cell_id}
        cycle_features = []
        discharge_cycles = cell_df[cell_df["step_type"] == "discharge"]
        charge_cycles = cell_df[cell_df["step_type"] == "charge"]

        available_cycles = sorted(discharge_cycles["cycle_index"].dropna().unique().tolist())
        if not available_cycles:
            continue

        first_cycle = int(available_cycles[0])
        last_cycle = int(available_cycles[-1])
        row["first_cycle_observed"] = first_cycle
        row["last_cycle_observed"] = last_cycle
        row["n_early_cycles"] = len(available_cycles)

        per_cycle_summary = []
        for cyc in available_cycles:
            dcyc = discharge_cycles[discharge_cycles["cycle_index"] == cyc]
            if dcyc.empty:
                continue
            feats = _basic_cycle_features(dcyc, prefix="discharge")
            feats["cycle_index"] = cyc
            per_cycle_summary.append(feats)

        if per_cycle_summary:
            cycle_df = pd.DataFrame(per_cycle_summary)
            numeric_cols = [c for c in cycle_df.columns if c != "cycle_index"]
            for col in numeric_cols:
                values = cycle_df[col].to_numpy(dtype=float)
                row[f"{col}_mean_over_cycles"] = float(np.nanmean(values))
                row[f"{col}_std_over_cycles"] = float(np.nanstd(values))
                row[f"{col}_slope_over_cycles"] = _safe_slope(
                    cycle_df["cycle_index"].to_numpy(dtype=float), values
                )
            for col in numeric_cols:
                row[f"{col}_first"] = float(cycle_df.iloc[0][col])
                row[f"{col}_last"] = float(cycle_df.iloc[-1][col])
                row[f"{col}_delta_last_first"] = float(cycle_df.iloc[-1][col] - cycle_df.iloc[0][col])

        first_discharge = discharge_cycles[discharge_cycles["cycle_index"] == first_cycle]
        last_discharge = discharge_cycles[discharge_cycles["cycle_index"] == last_cycle]

        if config.include_curve_features:
            first_grid_v, first_qv = _resample_q_of_v(first_discharge, config.voltage_grid_points)
            last_grid_v, last_qv = _resample_q_of_v(last_discharge, config.voltage_grid_points)

            if first_qv.size:
                row.update(_curve_summary_features("q_of_v_first", first_qv))
                dq_dv_first = np.gradient(first_qv, first_grid_v)
                row.update(_curve_summary_features("dq_dv_first", dq_dv_first))
            if last_qv.size:
                row.update(_curve_summary_features("q_of_v_last", last_qv))
                dq_dv_last = np.gradient(last_qv, last_grid_v)
                row.update(_curve_summary_features("dq_dv_last", dq_dv_last))
            if first_qv.size and last_qv.size:
                delta_qv = last_qv - first_qv
                row.update(_curve_summary_features("delta_q_of_v_last_first", delta_qv))

        if config.include_delta_features:
            first_charge = charge_cycles[charge_cycles["cycle_index"] == first_cycle]
            last_charge = charge_cycles[charge_cycles["cycle_index"] == last_cycle]
            if not first_charge.empty and not first_discharge.empty:
                soc_grid, first_v_charge = _resample_v_of_soc(first_charge, config.soc_grid_points)
                _, first_v_discharge = _resample_v_of_soc(first_discharge, config.soc_grid_points)
                if first_v_charge.size and first_v_discharge.size:
                    delta_v_soc_first = first_v_charge - first_v_discharge
                    row.update(_curve_summary_features("delta_v_soc_first", delta_v_soc_first))
            if not last_charge.empty and not last_discharge.empty:
                soc_grid, last_v_charge = _resample_v_of_soc(last_charge, config.soc_grid_points)
                _, last_v_discharge = _resample_v_of_soc(last_discharge, config.soc_grid_points)
                if last_v_charge.size and last_v_discharge.size:
                    delta_v_soc_last = last_v_charge - last_v_discharge
                    row.update(_curve_summary_features("delta_v_soc_last", delta_v_soc_last))
            if (
                "delta_v_soc_first_mean" in row
                and "delta_v_soc_last_mean" in row
            ):
                # Build difference curve when both are available.
                _, first_v_charge = _resample_v_of_soc(first_charge, config.soc_grid_points)
                _, first_v_discharge = _resample_v_of_soc(first_discharge, config.soc_grid_points)
                _, last_v_charge = _resample_v_of_soc(last_charge, config.soc_grid_points)
                _, last_v_discharge = _resample_v_of_soc(last_discharge, config.soc_grid_points)
                if (
                    first_v_charge.size
                    and first_v_discharge.size
                    and last_v_charge.size
                    and last_v_discharge.size
                ):
                    delta_delta_v_soc = (last_v_charge - last_v_discharge) - (first_v_charge - first_v_discharge)
                    row.update(_curve_summary_features("delta_delta_v_soc_last_first", delta_delta_v_soc))

        feature_rows.append(row)

    feature_df = pd.DataFrame(feature_rows)
    if feature_df.empty:
        raise ValueError("No features could be built. Check dataset contents and schema.")
    return feature_df
