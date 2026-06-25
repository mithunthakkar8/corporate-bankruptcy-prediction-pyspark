#!/usr/bin/env python3
"""
Corporate Bankruptcy Forecasting – Local sklearn Version
AIML427 Big Data – Assignment 3, Question 2 (Individual Project)

Replicates the four Spark ML experiments locally using pandas + sklearn.
Results should be comparable (not identical) to the cluster run due to
differences in Random Forest implementation between Spark ML and sklearn.

Four experiments:
  1. Baseline      – no scaling, no PCA
  2. Scaled        – StandardScaler only
  3. PCA only      – internal StandardScaler + PCA (k=50)
  4. Scaled + PCA  – StandardScaler + PCA (k=50)

Split: Temporal — train <= 2017, test > 2017

Usage:
    python q2_bankruptcy_local.py
    python q2_bankruptcy_local.py --data path/to/company_years_h1.parquet
    python q2_bankruptcy_local.py --sample 0.2   # use 20% of data for quick test
"""

import time
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    average_precision_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
LABEL_COL           = "main_label"
YEAR_COL            = "year"
TEMPORAL_SPLIT_YEAR = 2017          # train: 2006-2017, test: 2018-2020
SEED                = 42
NUM_TREES           = 100
MAX_DEPTH           = 10
MIN_SAMPLES_LEAF    = 10
PCA_K               = 50

# Columns to drop — identifiers with no predictive value
ID_COLS = {"num", "company", "industry", "link", "emis_id"}

# Near-null threshold — drop features with > 95% missing
NULL_THRESHOLD = 0.95

# Asymmetric cost matrix (credit risk literature: FN costs 10x more than FP)
C_TP, C_TN, C_FP, C_FN = 1, 0, -1, -10


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(path: str, sample_fraction: float = 1.0) -> pd.DataFrame:
    print(f"\n  Loading: {path}")
    df = pd.read_parquet(path)
    print(f"  Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Drop ID columns
    drop_ids = [c for c in ID_COLS if c in df.columns]
    df = df.drop(columns=drop_ids)

    # Drop string columns
    str_cols = df.select_dtypes(include="object").columns.tolist()
    df = df.drop(columns=str_cols)

    # Drop bool — convert to int
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    for c in bool_cols:
        df[c] = df[c].astype(float)

    # Cast everything to float
    for c in df.columns:
        if c not in (LABEL_COL, YEAR_COL):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if 0 < sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=SEED)
        print(f"  Sampled {sample_fraction*100:.0f}%: {len(df):,} rows")

    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────
def temporal_split(df: pd.DataFrame):
    train = df[df[YEAR_COL] <= TEMPORAL_SPLIT_YEAR].drop(columns=[YEAR_COL])
    test  = df[df[YEAR_COL] >  TEMPORAL_SPLIT_YEAR].drop(columns=[YEAR_COL])

    print(f"\n  Temporal split at year {TEMPORAL_SPLIT_YEAR}:")
    year_counts = df[YEAR_COL].value_counts().sort_index()
    for yr, cnt in year_counts.items():
        label = "TRAIN" if yr <= TEMPORAL_SPLIT_YEAR else "TEST "
        print(f"    {label} | {yr} : {cnt:>8,} rows")

    return train, test


def drop_near_null_features(train: pd.DataFrame, feature_cols: list) -> list:
    null_pct = train[feature_cols].isnull().mean()
    valid    = null_pct[null_pct <= NULL_THRESHOLD].index.tolist()
    dropped  = [c for c in feature_cols if c not in valid]
    if dropped:
        print(f"\n  Dropped {len(dropped)} near-null features (>{NULL_THRESHOLD*100:.0f}% null):")
        for c in dropped:
            print(f"    {c}  ({null_pct[c]*100:.1f}% null)")
    print(f"  Using {len(valid)} features")
    return valid


def compute_class_weights(y_train: pd.Series) -> dict:
    counts  = y_train.value_counts()
    total   = len(y_train)
    w_neg   = total / (2.0 * counts.get(0, 1))
    w_pos   = total / (2.0 * counts.get(1, 1))
    print(f"\n  Class distribution: 0={counts.get(0,0):,}  |  1={counts.get(1,0):,}")
    print(f"  Class weights     : w_neg={w_neg:.4f}  |  w_pos={w_pos:.4f}")
    weights = {0: w_neg, 1: w_pos}
    return weights


def impute_medians(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list):
    """Fit medians on train only, apply to both — prevents data leakage."""
    medians = train[feature_cols].median()
    train[feature_cols] = train[feature_cols].fillna(medians)
    test[feature_cols]  = test[feature_cols].fillna(medians)
    return train, test


# ── Evaluation ────────────────────────────────────────────────────────────────
def compute_utility(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tp + tn + fp + fn
    return (C_TP*tp + C_TN*tn + C_FP*fp + C_FN*fn) / total if total > 0 else 0.0


def evaluate(y_true, y_pred, y_prob, split_name: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc     = accuracy_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    from sklearn.metrics import fbeta_score
    f2      = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    mcc     = matthews_corrcoef(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    pr_auc  = average_precision_score(y_true, y_prob)
    utility = compute_utility(y_true, y_pred)

    print(f"    {split_name:5s}: Acc={acc:.4f}  AUC-ROC={auc_roc:.4f}  PR-AUC={pr_auc:.4f}  "
          f"F1={f1:.4f}  F2={f2:.4f}  Prec={prec:.4f}  Recall={rec:.4f}  "
          f"MCC={mcc:.4f}  Utility={utility:.4f}")
    print(f"           Confusion matrix -> TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    return dict(accuracy=acc, auc_roc=auc_roc, pr_auc=pr_auc,
                f1=f1, f2=f2, precision=prec, recall=rec,
                mcc=mcc, utility=utility,
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


# ── Build pipeline ────────────────────────────────────────────────────────────
def build_pipeline(use_scaler: bool, use_pca: bool, class_weights: dict) -> Pipeline:
    steps = []

    if use_pca:
        # PCA always needs scaling — add scaler even if use_scaler is False
        steps.append(("scaler", StandardScaler()))
        steps.append(("pca", PCA(n_components=PCA_K, random_state=SEED)))
    elif use_scaler:
        steps.append(("scaler", StandardScaler()))

    steps.append(("rf", RandomForestClassifier(
        n_estimators     = NUM_TREES,
        max_depth        = MAX_DEPTH,
        min_samples_leaf = MIN_SAMPLES_LEAF,
        max_features     = "sqrt",
        class_weight     = class_weights,
        random_state     = SEED,
        n_jobs           = -1,       # use all CPU cores
    )))

    return Pipeline(steps)


# ── Single experiment ─────────────────────────────────────────────────────────
def run_experiment(name, X_train, y_train, X_test, y_test,
                   use_scaler, use_pca, class_weights) -> dict:
    sep = "-" * 70
    print(f"\n{sep}")
    print(f"  Experiment : {name}")
    print(f"  Scaler={use_scaler}  PCA={use_pca}  Trees={NUM_TREES}  "
          f"MaxDepth={MAX_DEPTH}  PCA_K={PCA_K}")
    print(sep)

    pipeline = build_pipeline(use_scaler, use_pca, class_weights)

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time : {train_time:.1f} s")

    train_pred = pipeline.predict(X_train)
    train_prob = pipeline.predict_proba(X_train)[:, 1]
    test_pred  = pipeline.predict(X_test)
    test_prob  = pipeline.predict_proba(X_test)[:, 1]

    train_metrics = evaluate(y_train, train_pred, train_prob, "Train")
    test_metrics  = evaluate(y_test,  test_pred,  test_prob,  "Test")

    total_time = time.time() - t0

    return {
        "name":            name,
        "use_scaler":      use_scaler,
        "use_pca":         use_pca,
        "train_time_s":    train_time,
        "total_time_s":    total_time,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"test_{k}":  v for k, v in test_metrics.items()},
    }


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(results: list):
    W = 150
    print(f"\n{'='*W}")
    print("RESULTS SUMMARY")
    print(f"{'='*W}")
    hdr = (f"{'Experiment':<45} {'TestAcc':>8} {'AUC-ROC':>8} {'PR-AUC':>8} "
           f"{'Prec':>8} {'Recall':>8} {'F1':>8} {'F2':>8} "
           f"{'MCC':>8} {'Utility':>9} {'Time(s)':>8}")
    print(hdr)
    print("-" * W)
    for r in results:
        print(f"{r['name']:<45} {r['test_accuracy']:>8.4f} {r['test_auc_roc']:>8.4f} "
              f"{r['test_pr_auc']:>8.4f} {r['test_precision']:>8.4f} "
              f"{r['test_recall']:>8.4f} {r['test_f1']:>8.4f} {r['test_f2']:>8.4f} "
              f"{r['test_mcc']:>8.4f} {r['test_utility']:>9.4f} "
              f"{r['train_time_s']:>8.1f}")
    print("=" * W)

    if len(results) >= 2:
        print("\n--- Effect of Scaling (no PCA) ---")
        r0, r1 = results[0], results[1]
        for m in ["auc_roc", "pr_auc", "f2", "mcc"]:
            delta = r1[f"test_{m}"] - r0[f"test_{m}"]
            print(f"  {m.upper():8s}: {r0[f'test_{m}']:.4f} -> {r1[f'test_{m}']:.4f}  "
                  f"(delta = {delta:+.4f})")

    if len(results) >= 4:
        print("\n--- Effect of PCA (Scaled vs Scaled+PCA) ---")
        r1, r3 = results[1], results[3]
        for m in ["auc_roc", "pr_auc", "f2", "mcc"]:
            delta = r3[f"test_{m}"] - r1[f"test_{m}"]
            print(f"  {m.upper():8s}: {r1[f'test_{m}']:.4f} -> {r3[f'test_{m}']:.4f}  "
                  f"(delta = {delta:+.4f})")


# ── Save results ──────────────────────────────────────────────────────────────
def save_results(results: list, elapsed: float, output_path: str = "q2_local_results.csv"):
    rows = []
    for r in results:
        rows.append({
            "Experiment":     r["name"],
            "TrainAcc":       r["train_accuracy"],
            "TestAcc":        r["test_accuracy"],
            "TrainAUC":       r["train_auc_roc"],
            "TestAUC":        r["test_auc_roc"],
            "TrainPRAUC":     r["train_pr_auc"],
            "TestPRAUC":      r["test_pr_auc"],
            "TrainPrec":      r["train_precision"],
            "TestPrec":       r["test_precision"],
            "TrainRecall":    r["train_recall"],
            "TestRecall":     r["test_recall"],
            "TrainF1":        r["train_f1"],
            "TestF1":         r["test_f1"],
            "TrainF2":        r["train_f2"],
            "TestF2":         r["test_f2"],
            "TrainMCC":       r["train_mcc"],
            "TestMCC":        r["test_mcc"],
            "TrainUtility":   r["train_utility"],
            "TestUtility":    r["test_utility"],
            "Test_TP":        r["test_tp"],
            "Test_TN":        r["test_tn"],
            "Test_FP":        r["test_fp"],
            "Test_FN":        r["test_fn"],
            "TrainTime_s":    r["train_time_s"],
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")
    print(f"  Total wall-clock time: {elapsed:.1f} s")


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Corporate Bankruptcy Forecasting – Local sklearn Version"
    )
    parser.add_argument("--data",   default="company_years_h1.parquet",
                        help="Path to parquet file")
    parser.add_argument("--sample", type=float, default=1.0,
                        help="Fraction of data to use (0-1]. Use 0.1 for quick test.")
    parser.add_argument("--output", default="q2_local_results.csv",
                        help="CSV file to save results")
    return parser.parse_args()


def main():
    args = parse_args()
    wall_start = time.time()

    print(f"\n{'='*70}")
    print("  Corporate Bankruptcy Forecasting – Local sklearn Version")
    print(f"{'='*70}")
    print(f"  Data   : {args.data}")
    print(f"  Sample : {args.sample*100:.0f}%  |  Seed={SEED}")
    print(f"  Model  : RandomForest  trees={NUM_TREES}  depth={MAX_DEPTH}")
    print(f"  PCA K  : {PCA_K}")
    print(f"  Split  : Temporal — train <= {TEMPORAL_SPLIT_YEAR}, test > {TEMPORAL_SPLIT_YEAR}")

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_data(args.data, sample_fraction=args.sample)

    # ── Temporal split ────────────────────────────────────────────────────────
    train_df, test_df = temporal_split(df)
    print(f"\n  Train rows : {len(train_df):,}  |  Test rows : {len(test_df):,}")

    # ── Feature cols ──────────────────────────────────────────────────────────
    all_cols     = [c for c in train_df.columns if c != LABEL_COL]
    feature_cols = drop_near_null_features(train_df, all_cols)

    # ── Class weights ─────────────────────────────────────────────────────────
    y_train = train_df[LABEL_COL].astype(int)
    y_test  = test_df[LABEL_COL].astype(int)
    class_weights = compute_class_weights(y_train)

    # ── Impute ────────────────────────────────────────────────────────────────
    print("\n  Imputing missing values with training medians...")
    train_df, test_df = impute_medians(train_df, test_df, feature_cols)

    X_train = train_df[feature_cols].values
    X_test  = test_df[feature_cols].values

    # ── Four experiments ──────────────────────────────────────────────────────
    experiments = [
        ("Baseline (no scaling, no PCA)",       False, False),
        ("StandardScaler only (no PCA)",        True,  False),
        ("PCA only (scaled internally)",        False, True),
        ("StandardScaler + PCA",                True,  True),
    ]

    results = []
    for name, use_scaler, use_pca in experiments:
        r = run_experiment(
            name, X_train, y_train, X_test, y_test,
            use_scaler, use_pca, class_weights,
        )
        results.append(r)

    # ── Summary & save ────────────────────────────────────────────────────────
    print_summary(results)
    elapsed = time.time() - wall_start
    save_results(results, elapsed, output_path=args.output)


if __name__ == "__main__":
    main()
