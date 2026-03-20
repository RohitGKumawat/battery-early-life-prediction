# Early Battery Life Prediction with Machine Learning

Predict whether a lithium-ion cell will have **short** or **long** life, or directly predict **end-of-life cycle count**, using only the **first few charge/discharge cycles**.

This project is built for the battery-analytics case study where you want to use:
- early-cycle voltage, current, temperature, and capacity curves
- derived features such as **Q(V)**, **dQ/dV**, **ΔV(SOC)**, and internal-resistance trends
- targets such as **end-of-life (EOL) cycle count** or a binary **short-life vs long-life** label

The idea follows the well-known early-prediction literature: Severson et al. showed that cycle life can be predicted from early-cycle discharge voltage curves, using a dataset of 124 commercial LFP/graphite cells with cycle lives from about 150 to 2,300 cycles. Their best models reported 9.1% test error for cycle-life regression using the first 100 cycles, and 4.9% test error for binary classification using the first 5 cycles. More recent NREL work emphasizes that interpretable features from **Q(V)**, **dQ/dV**, and **ΔV(SOC)** can help link early prediction to degradation physics such as lithium loss, stoichiometry shifts, diffusivity changes, and resistance evolution. citeturn1view1turn1view0

---

## What this repo does

This repo gives you a complete starter pipeline for:
1. loading battery early-cycle curve data
2. engineering interpretable features from the first `N` cycles
3. training **regression** models for EOL cycle count
4. training **classification** models for short-life vs long-life cells
5. saving model artifacts, metrics, predictions, and feature importance tables
6. running the project immediately using a **synthetic demo dataset**

---

## Project structure

```text
battery_early_life_prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── sample/
├── models/
├── reports/
└── src/
    ├── generate_demo_data.py
    ├── train.py
    ├── predict.py
    └── battery_ml/
        ├── __init__.py
        ├── data_io.py
        ├── features.py
        └── modeling.py
```

---

## Dataset format expected

### 1) Curve data file
Use a CSV or Parquet file with one row per measurement point.

Required columns:

| column | meaning |
|---|---|
| `cell_id` | unique battery/cell ID |
| `cycle_index` | cycle number starting from 1 |
| `step_type` | `charge` or `discharge` |
| `time_s` | time within the step |
| `voltage_v` | voltage |
| `current_a` | current |
| `capacity_ah` | cumulative capacity during the step |
| `temperature_c` | temperature |

Optional columns:

| column | meaning |
|---|---|
| `internal_resistance_ohm` | internal resistance estimate |
| `soc` | state of charge |

### 2) Label data file
One row per cell.

Required columns:

| column | meaning |
|---|---|
| `cell_id` | unique battery/cell ID |
| `eol_cycle` | cycle count at end-of-life |

---

## Features engineered

The feature pipeline extracts information from the first `N` cycles only.

### Per-cycle summary features
- discharge capacity
- mean/std voltage
- mean current
- mean/max temperature
- step duration
- optional internal resistance statistics

### Trend features across early cycles
- mean over cycles
- standard deviation over cycles
- slope over cycles
- first-cycle value
- last-cycle value
- last-minus-first delta

### Curve features
- **Q(V)** samples from the first discharge curve and the last early discharge curve
- **dQ/dV** derived from resampled Q(V)
- **ΔQ(V)** between late and early cycles
- **ΔV(SOC)** from charge vs discharge voltage curves
- **Δ(ΔV(SOC))** between the last and first early cycle

These are intended to be interpretable and close to the kinds of voltage-curve signatures discussed in early battery-life prediction research. citeturn1view0turn1view1

---

## Models included

### Baselines
- Linear Regression
- Elastic Net
- Logistic Regression (for classification)

### Stronger tree models
- Random Forest
- XGBoost *(optional via command flag)*
- LightGBM *(optional via command flag)*

The training script automatically compares candidate models and saves the best one based on cross-validation:
- **MAE** for regression
- **F1 score** for classification

---

## Installation

```bash
git clone https://github.com/your-username/battery_early_life_prediction.git
cd battery_early_life_prediction
python -m venv .venv
```

### Windows
```bash
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=src
```

### macOS / Linux
```bash
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

---

## Quick start with the demo dataset

### Step 1: generate synthetic battery data

```bash
python src/generate_demo_data.py --n-cells 80 --early-cycles 40 --output-dir data/sample
```

This creates:
- `data/sample/battery_curves.csv`
- `data/sample/battery_labels.csv`

### Step 2: train a regression model for EOL cycle count

```bash
python src/train.py \
  --curves data/sample/battery_curves.csv \
  --labels data/sample/battery_labels.csv \
  --task regression \
  --early-cycles 30 \
  --output-dir models/regression_run
```

### Step 3: train a binary short-life vs long-life classifier

```bash
python src/train.py \
  --curves data/sample/battery_curves.csv \
  --labels data/sample/battery_labels.csv \
  --task classification \
  --early-cycles 30 \
  --output-dir models/classification_run
```

By default, tree-boosting libraries are **not** included in the model sweep, so the project runs cleanly on a standard scikit-learn setup. If you want to include them, add `--use-xgboost` and/or `--use-lightgbm`.

Example:

```bash
python src/train.py \
  --curves data/sample/battery_curves.csv \
  --labels data/sample/battery_labels.csv \
  --task regression \
  --early-cycles 30 \
  --use-xgboost \
  --use-lightgbm \
  --output-dir models/regression_boosted_run
```

By default, the classification threshold is the **median** EOL cycle count of the training data. You can override that with:

```bash
python src/train.py \
  --curves data/sample/battery_curves.csv \
  --labels data/sample/battery_labels.csv \
  --task classification \
  --classification-threshold 800 \
  --early-cycles 30 \
  --output-dir models/classification_run_800
```

### Step 4: predict on new cells

```bash
python src/predict.py \
  --model-path models/regression_run/model.joblib \
  --curves data/sample/battery_curves.csv \
  --early-cycles 30 \
  --output-path reports/inference_predictions.csv
```

---

## Output files

Each training run saves:

- `feature_table.csv` → engineered features per cell
- `run_info.json` → run configuration
- `metrics.json` → final model metrics
- `predictions.csv` → actual vs predicted values on the test split
- `feature_importance.csv` → ranked feature importance or coefficient magnitude
- `model.joblib` → trained pipeline
- `classification_report.txt` → only for classification runs

---

## License

MIT
