#!/usr/bin/env python3
"""
train.py — Campus Placement Prediction (Classification Only)

Trains 6 classification models (target = 'placed') on the Campus Placement dataset
and saves models + comparison metrics.

Expected input file: ./data/dataset.csv
Outputs saved to ./models/:
 - feature_names_classification.pkl
 - scaler_classification.pkl
 - logistic_regression.pkl
 - decision_tree.pkl
 - knn.pkl
 - naive_bayes.pkl
 - random_forest.pkl
 - xgboost.pkl
 - model_comparison_classification.csv
"""

import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# XGBoost (install if missing: pip install xgboost)
from xgboost import XGBClassifier


class CFG:
    """Configuration"""
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    DATASET_FILE = "dataset.csv"
    TARGET_CLASS = "placed"

    TEST_SIZE = 0.30
    RANDOM_STATE = 42

    # Scale features for these models
    MODELS_NEED_SCALING = ["Logistic Regression", "kNN"]

    # Categorical columns (from your schema)
    CAT_COLS = [
        "gender", "city_tier", "ssc_board", "hsc_board",
        "hsc_stream", "degree_field", "specialization"
    ]
    # ID columns to drop
    ID_COLS = ["student_id"]

    # Drop leakage columns from classification features
    DROP_FROM_CLASS_X = ["salary_lpa"]


def load_data() -> pd.DataFrame:
    """Read dataset from CSV"""
    path = CFG.DATA_DIR / CFG.DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    print(f"✓ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def _map_binary(series: pd.Series) -> pd.Series:
    """Map binary-like values ('Yes'/'No', True/False, 1/0) to 0/1."""
    def m(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().lower()
        if s in {"1", "yes", "y", "true", "t", "1.0"}: return 1
        if s in {"0", "no", "n", "false", "f", "0.0"}: return 0
        try:
            v = float(s)
            return 1 if v == 1.0 else 0 if v == 0.0 else np.nan
        except Exception:
            return np.nan
    return series.map(m)


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Preprocess dataset:
      - Drop ID columns
      - Cast target 'placed' to int
      - One-hot encode categorical columns
      - Fill missing values
      - Drop leakage columns (salary_lpa) from X
    """
    df = df.copy()

    # Drop IDs
    for col in CFG.ID_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Ensure target is numeric
    if CFG.TARGET_CLASS not in df.columns:
        raise KeyError(f"Target '{CFG.TARGET_CLASS}' not found in dataset")
    df[CFG.TARGET_CLASS] = pd.to_numeric(df[CFG.TARGET_CLASS], errors="coerce").fillna(0).astype(int)

    # If binary-like fields exist as strings, normalize them (optional, based on your CSV)
    for bcol in ["leadership_roles", "extracurricular_activities"]:
        if bcol in df.columns and df[bcol].dtype == object:
            df[bcol] = _map_binary(df[bcol]).fillna(0).astype(int)

    # One-hot encode categorical columns
    present_cat = [c for c in CFG.CAT_COLS if c in df.columns]
    for c in present_cat:
        df[c] = df[c].fillna("Unknown")
    if present_cat:
        df = pd.get_dummies(df, columns=present_cat, drop_first=True)

    # Fill missing numeric with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # Build X and y (drop target + leakage cols)
    y = df[CFG.TARGET_CLASS].astype(int)
    drop_cols = [CFG.TARGET_CLASS] + [c for c in CFG.DROP_FROM_CLASS_X if c in df.columns]
    X = df.drop(columns=drop_cols)

    feature_names = list(X.columns)
    print(f"✓ Preprocessed: X={X.shape}, y={y.shape}")
    return X, y, feature_names


def train_classification(X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> None:
    """Train and evaluate classification models; save models and comparison CSV"""
    print("\n[Classification] Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG.TEST_SIZE, stratify=y, random_state=CFG.RANDOM_STATE
    )

    # Scale features for selected models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save artifacts
    with open(CFG.MODEL_DIR / "feature_names_classification.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    with open(CFG.MODEL_DIR / "scaler_classification.pkl", "wb") as f:
        pickle.dump(scaler, f)

    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=CFG.RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=CFG.RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=CFG.RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, random_state=CFG.RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
    }

    rows = []
    print("\n[Classification] Training models...")
    for name, model in models.items():
        print(f" -> {name}")
        if name in CFG.MODELS_NEED_SCALING:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics (handle AUC edge cases)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = float("nan")

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": auc,
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }
        rows.append(row)

        # Save model
        model_file = CFG.MODEL_DIR / f"{name.replace(' ', '_').lower()}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        # Confusion matrix (printed)
        cm = confusion_matrix(y_test, y_pred)
        print(f"    Acc={row['Accuracy']:.4f} AUC={row['AUC']:.4f} Prec={row['Precision']:.4f} Rec={row['Recall']:.4f} F1={row['F1']:.4f} MCC={row['MCC']:.4f}")
        print(f"    Confusion Matrix:\n{cm}")

    # Save comparison CSV
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(CFG.MODEL_DIR / "model_comparison_classification.csv", index=False)
    print("✓ Saved classification comparison: model_comparison_classification.csv")


if __name__ == "__main__":
    # Load
    df = load_data()

    # Classification task
    Xc, yc, feat_c = preprocess(df)
    train_classification(Xc, yc, feat_c)

    print("\nTraining complete. Models and comparison files are in ./models/")