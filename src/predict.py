from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from battery_ml.data_io import load_curve_data
from battery_ml.features import FeatureConfig, build_feature_matrix



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict battery life for new cells using a trained model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model.joblib")
    parser.add_argument("--curves", type=str, required=True, help="Path to new raw curve data")
    parser.add_argument("--output-path", type=str, default="reports/inference_predictions.csv")
    parser.add_argument("--early-cycles", type=int, default=30)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    model = joblib.load(args.model_path)
    curves = load_curve_data(args.curves)
    features = build_feature_matrix(curves, FeatureConfig(early_cycles=args.early_cycles))
    feature_cols = [c for c in features.columns if c != "cell_id" and pd.api.types.is_numeric_dtype(features[c])]

    preds = model.predict(features[feature_cols])
    out = pd.DataFrame({"cell_id": features["cell_id"], "prediction": preds})
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(out.head())


if __name__ == "__main__":
    main()
