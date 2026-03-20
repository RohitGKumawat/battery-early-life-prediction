from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None


RANDOM_STATE = 42



def _numeric_feature_columns(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]



def make_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(transformers=[("num", numeric_pipe, feature_columns)], remainder="drop")



def build_regressors(include_xgboost: bool = False, include_lightgbm: bool = False) -> dict[str, Any]:
    models: dict[str, Any] = {
        "linear_regression": LinearRegression(),
        "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=50000),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if include_xgboost and XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
    if include_lightgbm and LGBMRegressor is not None:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            min_data_in_leaf=1,
            verbosity=-1,
            random_state=RANDOM_STATE,
        )
    return models



def build_classifiers(include_xgboost: bool = False, include_lightgbm: bool = False) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if include_xgboost and XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
    if include_lightgbm and LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            min_data_in_leaf=1,
            verbosity=-1,
            random_state=RANDOM_STATE,
        )
    return models



def _feature_importances(pipe: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        values = np.abs(coef.ravel()) if np.ndim(coef) > 1 else np.abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return pd.DataFrame({"feature": feature_columns, "importance": values}).sort_values(
        "importance", ascending=False
    )



def train_regression(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    out_dir: str | Path,
    test_size: float = 0.2,
    include_xgboost: bool = False,
    include_lightgbm: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = data[feature_columns].copy()
    y = data[target_col].astype(float).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    cv = KFold(n_splits=min(5, max(3, len(X_train) // 8)), shuffle=True, random_state=RANDOM_STATE)

    preprocessor = make_preprocessor(feature_columns)
    candidates = build_regressors(include_xgboost=include_xgboost, include_lightgbm=include_lightgbm)

    cv_scores: dict[str, float] = {}
    fitted_pipelines: dict[str, Pipeline] = {}

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
        cv_scores[name] = float(-scores.mean())
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

    best_name = min(cv_scores, key=cv_scores.get)
    best_pipe = fitted_pipelines[best_name]
    preds = best_pipe.predict(X_test)

    metrics = {
        "task": "regression",
        "best_model": best_name,
        "cv_mae": cv_scores[best_name],
        "test_mae": float(mean_absolute_error(y_test, preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "test_r2": float(r2_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    pred_df = pd.DataFrame({"actual": y_test.to_numpy(), "predicted": preds})
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    fi = _feature_importances(best_pipe, feature_columns)
    fi.to_csv(out_dir / "feature_importance.csv", index=False)
    joblib.dump(best_pipe, out_dir / "model.joblib")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics



def train_classification(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    out_dir: str | Path,
    test_size: float = 0.2,
    include_xgboost: bool = False,
    include_lightgbm: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = data[feature_columns].copy()
    y = data[target_col].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    n_splits = min(5, max(3, int(y_train.value_counts().min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    preprocessor = make_preprocessor(feature_columns)
    candidates = build_classifiers(include_xgboost=include_xgboost, include_lightgbm=include_lightgbm)

    cv_scores: dict[str, float] = {}
    fitted_pipelines: dict[str, Pipeline] = {}

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        cv_scores[name] = float(scores.mean())
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

    best_name = max(cv_scores, key=cv_scores.get)
    best_pipe = fitted_pipelines[best_name]
    preds = best_pipe.predict(X_test)

    metrics = {
        "task": "classification",
        "best_model": best_name,
        "cv_f1": cv_scores[best_name],
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_f1": float(f1_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    model = best_pipe.named_steps["model"]
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
        probas = best_pipe.predict_proba(X_test)[:, 1]
        metrics["test_roc_auc"] = float(roc_auc_score(y_test, probas))
        pred_df = pd.DataFrame({"actual": y_test.to_numpy(), "predicted": preds, "prob_1": probas})
    else:
        pred_df = pd.DataFrame({"actual": y_test.to_numpy(), "predicted": preds})

    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    fi = _feature_importances(best_pipe, feature_columns)
    fi.to_csv(out_dir / "feature_importance.csv", index=False)
    joblib.dump(best_pipe, out_dir / "model.joblib")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, preds))

    return metrics
