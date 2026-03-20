from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from battery_ml.data_io import load_curve_data, load_label_data
from battery_ml.features import FeatureConfig, build_feature_matrix
from battery_ml.modeling import train_classification, train_regression



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an early battery life prediction model.")
    parser.add_argument("--curves", type=str, required=True, help="Path to raw per-point curve data (CSV/Parquet).")
    parser.add_argument("--labels", type=str, required=True, help="Path to per-cell labels (CSV/Parquet).")
    parser.add_argument("--output-dir", type=str, default="models/run_01", help="Directory to save artifacts.")
    parser.add_argument("--early-cycles", type=int, default=30, help="Number of early cycles to use.")
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
        help="Predict numeric EOL cycle or binary short/long life class.",
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=None,
        help="Threshold on eol_cycle for classification. If omitted, median eol_cycle is used.",
    )
    parser.add_argument("--use-xgboost", action="store_true", help="Include XGBoost in model comparison.")
    parser.add_argument("--use-lightgbm", action="store_true", help="Include LightGBM in model comparison.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    curves = load_curve_data(args.curves)
    labels = load_label_data(args.labels)

    config = FeatureConfig(early_cycles=args.early_cycles)
    features = build_feature_matrix(curves, config)
    data = features.merge(labels, on="cell_id", how="inner")
    if data.empty:
        raise ValueError("No overlapping cell_id values between feature table and label table.")

    if args.task == "classification":
        threshold = args.classification_threshold
        if threshold is None:
            threshold = float(data["eol_cycle"].median())
        data["life_class"] = (data["eol_cycle"] >= threshold).astype(int)
        target_col = "life_class"
    else:
        target_col = "eol_cycle"
        threshold = None

    exclude_cols = ["cell_id", "eol_cycle", "life_class"]
    feature_columns = [c for c in data.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(data[c])]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_dir / "feature_table.csv", index=False)

    run_info = {
        "curves_path": args.curves,
        "labels_path": args.labels,
        "task": args.task,
        "early_cycles": args.early_cycles,
        "classification_threshold": threshold,
        "n_cells": int(data["cell_id"].nunique()),
        "n_features": int(len(feature_columns)),
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    if args.task == "classification":
        metrics = train_classification(
            data,
            feature_columns,
            target_col,
            out_dir,
            include_xgboost=args.use_xgboost,
            include_lightgbm=args.use_lightgbm,
        )
    else:
        metrics = train_regression(
            data,
            feature_columns,
            target_col,
            out_dir,
            include_xgboost=args.use_xgboost,
            include_lightgbm=args.use_lightgbm,
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
