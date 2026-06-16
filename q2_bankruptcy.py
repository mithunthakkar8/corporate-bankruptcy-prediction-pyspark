#!/usr/bin/env python3
"""
Corporate Bankruptcy Prediction using Apache Spark ML
AIML427 Big Data – Assignment 3, Question 2 (Individual Project)

Dataset : V4 Group Corporate Bankruptcy Dataset  (company_years_h1.parquet)
Task    : Binary classification – predict 1-year-ahead financial distress
          (main_label = 1: distressed, 0: healthy)
Model   : Random Forest Classifier

Four experiments are run to satisfy the report requirements:
  1. Baseline  – no scaling, no PCA
  2. Scaled    – StandardScaler only
  3. PCA-only  – PCA (StandardScaler applied internally, as PCA requires it)
  4. Scaled+PCA – StandardScaler followed by PCA

Usage (local):
    python q2_bankruptcy.py [DATA_PATH] [--sample FRACTION]

Usage (Hadoop cluster via spark-submit):
    spark-submit --master yarn --deploy-mode cluster \
        q2_bankruptcy.py hdfs:///user/<you>/data/company_years_h1.parquet
"""

import sys
import time
import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler, PCA
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

# ─── Hyper-parameters ────────────────────────────────────────────────────────
LABEL_COL   = "main_label"
TRAIN_RATIO = 0.8
TEST_RATIO  = 0.2
SEED        = 42
NUM_TREES   = 100
MAX_DEPTH   = 10
MIN_INST    = 10        # min instances per leaf – reduces overfitting
PCA_K       = 50        # number of principal components to retain

# Columns that carry no predictive information (IDs / URLs)
_ID_COLS = {"num", "company", "industry", "link", "emis_id"}

# ─── Dataset metadata (V4 Group Corporate Bankruptcy, h1 file) ────────────────
# Source: EMIS database — Czech Republic, Hungary, Poland, Slovakia
# Instances: 1,000,087 company-year observations (1-year prediction horizon)
# Raw columns: 143 (131 financial features + 6 near-null + 5 ID cols + 1 label)
# After preprocessing: 131 features used for modelling
_COUNTRY_MAP = {
    0: "CZ (Czech Republic)",   # 554,751 observations (55.5%)
    1: "HU (Hungary)",          #  55,736 observations ( 5.6%)
    2: "PL (Poland)",           #  53,916 observations ( 5.4%)
    3: "SK (Slovakia)",         # 335,684 observations (33.6%)
}
_SECTOR_MAP = {
    1: "Construction (NAICS 23)",              # 257,073 observations (25.7%)
    2: "Manufacturing (NAICS 31-33)",          #  84,300 observations ( 8.4%)
    3: "Retail Trade (NAICS 44)",              # 173,066 observations (17.3%)
    4: "Wholesale Trade (NAICS 42)",           # 372,319 observations (37.2%)
    5: "Transportation & Warehousing (NAICS 48)",  #  99,261 observations ( 9.9%)
    6: "Utilities (NAICS 22)",                 #  14,068 observations ( 1.4%)
}

# ─── Cost matrix (10:1 FN/FP ratio, credit-risk literature) ──────────────────
# Rationale: missing a bankruptcy (FN) costs ~10x more than a false alarm (FP)
# because a lender loses 40-80% of loan principal on default vs. only analyst
# time for an unnecessary credit review. (Altman 1968; Zmijewski 1984)
C_TP =   1   # benefit : correctly flagged distressed company
C_TN =   0   # neutral : correctly identified healthy company
C_FP =  -1   # cost    : false alarm
C_FN = -10   # cost    : missed bankruptcy


# ─── Spark session ────────────────────────────────────────────────────────────
def build_spark(app_name: str = "CorporateBankruptcyPrediction") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


# ─── Data loading & preprocessing ────────────────────────────────────────────
def load_data(spark: SparkSession, path: str, sample_fraction: float = 1.0):
    """
    Load the parquet file, drop identifier columns, cast all feature columns
    to DoubleType, and optionally sample a fraction for local testing.

    Returns
    -------
    df           : preprocessed Spark DataFrame
    feature_cols : list of feature column names (label excluded)
    """
    df = spark.read.parquet(path)

    # 1. Drop pure-identifier columns (not predictive)
    id_cols_present = [c for c in _ID_COLS if c in df.columns]
    if id_cols_present:
        df = df.drop(*id_cols_present)

    # 2. Drop any remaining string columns (company name text, URL, etc.)
    str_cols = [c for c, dtype in df.dtypes if dtype == "string"]
    if str_cols:
        df = df.drop(*str_cols)

    # 3. Cast label to DoubleType (may arrive as int or boolean)
    df = df.withColumn(LABEL_COL, F.col(LABEL_COL).cast(DoubleType()))

    # 4. Cast all non-label columns to DoubleType
    for col, dtype in df.dtypes:
        if col == LABEL_COL:
            continue
        if dtype in ("boolean", "bool", "int", "integer", "bigint", "long", "short"):
            df = df.withColumn(col, F.col(col).cast(DoubleType()))

    # 5. Optional sampling (for local smoke-tests)
    if 0.0 < sample_fraction < 1.0:
        df = df.sample(fraction=sample_fraction, seed=SEED)

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    return df, feature_cols


def filter_valid_features(df, feature_cols: list, min_non_null_fraction: float = 0.05):
    """
    Drop feature columns where fewer than min_non_null_fraction rows are non-null.
    Spark's Imputer cannot compute a surrogate (median/mean) for an all-null column.
    Runs a single aggregation pass – call on the training DataFrame only.
    """
    n = df.count()
    if n == 0:
        return feature_cols

    non_null_counts = (
        df.agg(*[F.count(c).alias(c) for c in feature_cols])
        .collect()[0]
        .asDict()
    )
    valid = [c for c in feature_cols
             if non_null_counts.get(c, 0) / n >= min_non_null_fraction]
    dropped = [c for c in feature_cols if c not in set(valid)]
    if dropped:
        print(f"  Dropping {len(dropped)} feature(s) with <{min_non_null_fraction*100:.0f}% "
              f"non-null values: {dropped}")
    print(f"  Using {len(valid)} features for modelling.")
    return valid


# ─── Class-weight computation ─────────────────────────────────────────────────
def add_class_weights(train_df, label_col: str = LABEL_COL):
    """
    Compute inverse-frequency class weights and add them as 'classWeight'.
    This counteracts the severe positive-class imbalance (~0.4 % distressed).
    Weights are: w_k = N / (2 * n_k)  where n_k = count of class k.
    """
    counts = (
        train_df.groupBy(label_col)
        .count()
        .collect()
    )
    count_map = {int(row[label_col]): row["count"] for row in counts}
    total = sum(count_map.values())
    n_neg = count_map.get(0, 1)
    n_pos = count_map.get(1, 1)
    w_neg = total / (2.0 * n_neg)
    w_pos = total / (2.0 * n_pos)

    print(f"  Class distribution: 0 (healthy)={n_neg:,}  |  1 (distressed)={n_pos:,}")
    print(f"  Class weights: w_neg={w_neg:.4f}  |  w_pos={w_pos:.4f}")

    train_df = train_df.withColumn(
        "classWeight",
        F.when(F.col(label_col) == 1.0, w_pos).otherwise(w_neg),
    )
    return train_df


# ─── ML Pipeline builder ──────────────────────────────────────────────────────
def build_pipeline(
    feature_cols: list,
    use_scaler: bool,
    use_pca: bool,
) -> Pipeline:
    """
    Construct a Spark ML Pipeline with:
      - Median imputation for all numeric features
      - VectorAssembler
      - (optional) StandardScaler
      - (optional) PCA  [always scales internally if use_scaler is False]
      - RandomForestClassifier with class-weight support
    """
    stages = []

    # Stage 1: Impute missing values with column-wise median
    imputed_cols = [f"{c}__imp" for c in feature_cols]
    imputer = Imputer(
        inputCols=feature_cols,
        outputCols=imputed_cols,
        strategy="median",
    )
    stages.append(imputer)

    # Stage 2: Assemble all imputed features into a single vector
    assembler = VectorAssembler(
        inputCols=imputed_cols,
        outputCol="raw_features",
        handleInvalid="skip",   # drop any row that still has NaN after imputation
    )
    stages.append(assembler)
    current_col = "raw_features"

    # Stage 3 (optional): StandardScaler
    if use_scaler:
        scaler = StandardScaler(
            inputCol=current_col,
            outputCol="scaled_features",
            withMean=True,
            withStd=True,
        )
        stages.append(scaler)
        current_col = "scaled_features"

    # Stage 4 (optional): PCA
    if use_pca:
        # PCA requires zero-mean, unit-variance data; scale internally if not done yet
        if not use_scaler:
            pre_scaler = StandardScaler(
                inputCol=current_col,
                outputCol="pre_pca_features",
                withMean=True,
                withStd=True,
            )
            stages.append(pre_scaler)
            current_col = "pre_pca_features"

        pca = PCA(k=PCA_K, inputCol=current_col, outputCol="pca_features")
        stages.append(pca)
        current_col = "pca_features"

    # Stage 5: Random Forest Classifier
    rf = RandomForestClassifier(
        featuresCol=current_col,
        labelCol=LABEL_COL,
        weightCol="classWeight",
        numTrees=NUM_TREES,
        maxDepth=MAX_DEPTH,
        minInstancesPerNode=MIN_INST,
        featureSubsetStrategy="sqrt",   # sqrt(n_features) per split – standard for clf
        seed=SEED,
    )
    stages.append(rf)

    return Pipeline(stages=stages)


# ─── Evaluation ───────────────────────────────────────────────────────────────
def _confusion_matrix(predictions):
    """Compute TP, TN, FP, FN in a single aggregation pass."""
    row = predictions.agg(
        F.sum(F.when(
            (F.col(LABEL_COL)==1) & (F.col("prediction")==1), 1).otherwise(0)
        ).alias("tp"),
        F.sum(F.when(
            (F.col(LABEL_COL)==0) & (F.col("prediction")==0), 1).otherwise(0)
        ).alias("tn"),
        F.sum(F.when(
            (F.col(LABEL_COL)==0) & (F.col("prediction")==1), 1).otherwise(0)
        ).alias("fp"),
        F.sum(F.when(
            (F.col(LABEL_COL)==1) & (F.col("prediction")==0), 1).otherwise(0)
        ).alias("fn"),
    ).collect()[0]
    return int(row.tp), int(row.tn), int(row.fp), int(row.fn)


def _precision(tp, fp) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def _recall(tp, fn) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def _mcc(tp, tn, fp, fn) -> float:
    """Matthews Correlation Coefficient — robust single metric for imbalanced data."""
    denom = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    return (tp*tn - fp*fn) / denom if denom > 0 else 0.0


def _utility(tp, tn, fp, fn) -> float:
    """
    Cost-weighted utility per instance using the asymmetric cost matrix.
    Higher is better. Negative means the model costs more than it saves.
    Baseline (always predict healthy): utility = C_FN * P/N ≈ -0.037
    """
    total = tp + tn + fp + fn
    raw = C_TP*tp + C_TN*tn + C_FP*fp + C_FN*fn
    return raw / total if total > 0 else 0.0


def evaluate(predictions, split_name: str) -> dict:
    """Return accuracy, AUC-ROC, PR-AUC, F1, F2, MCC, and cost-weighted utility."""

    auc_roc = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, metricName="areaUnderROC"
    ).evaluate(predictions)

    pr_auc = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, metricName="areaUnderPR"
    ).evaluate(predictions)

    acc = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, metricName="accuracy"
    ).evaluate(predictions)

    f1 = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, metricName="f1"
    ).evaluate(predictions)

    f2 = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, metricName="fMeasureByLabel",
        beta=2.0, metricLabel=1.0
    ).evaluate(predictions)

    # Single-pass confusion matrix → precision, recall, MCC, utility
    tp, tn, fp, fn = _confusion_matrix(predictions)
    prec    = _precision(tp, fp)
    rec     = _recall(tp, fn)
    mcc     = _mcc(tp, tn, fp, fn)
    utility = _utility(tp, tn, fp, fn)

    print(f"    {split_name:5s}: Acc={acc:.4f}  AUC-ROC={auc_roc:.4f}  "
          f"PR-AUC={pr_auc:.4f}  F1={f1:.4f}  F2={f2:.4f}  "
          f"Prec={prec:.4f}  Recall={rec:.4f}  MCC={mcc:.4f}  Utility={utility:.4f}")
    print(f"           Confusion matrix -> TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    return {
        "accuracy": acc, "auc_roc": auc_roc, "pr_auc": pr_auc,
        "f1": f1, "f2": f2, "precision": prec, "recall": rec,
        "mcc": mcc, "utility": utility,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ─── Single experiment runner ─────────────────────────────────────────────────
def run_experiment(
    name: str,
    train_df,
    test_df,
    feature_cols: list,
    use_scaler: bool,
    use_pca: bool,
) -> dict:
    sep = "-" * 65
    print(f"\n{sep}")
    print(f"  Experiment : {name}")
    print(f"  Scaler={use_scaler}  PCA={use_pca}  "
          f"Trees={NUM_TREES}  MaxDepth={MAX_DEPTH}  PCA_K={PCA_K}")
    print(sep)

    pipeline = build_pipeline(feature_cols, use_scaler=use_scaler, use_pca=use_pca)

    t0 = time.time()
    model = pipeline.fit(train_df)
    train_time = time.time() - t0
    print(f"  Training time : {train_time:.1f} s")

    train_preds = model.transform(train_df)
    test_preds  = model.transform(test_df)

    train_metrics = evaluate(train_preds, "Train")
    test_metrics  = evaluate(test_preds,  "Test")

    train_preds.unpersist()
    test_preds.unpersist()

    total_time = time.time() - t0

    return {
        "name":           name,
        "use_scaler":     use_scaler,
        "use_pca":        use_pca,
        "train_acc":      train_metrics["accuracy"],
        "test_acc":       test_metrics["accuracy"],
        "train_auc":      train_metrics["auc_roc"],
        "test_auc":       test_metrics["auc_roc"],
        "train_pr_auc":   train_metrics["pr_auc"],
        "test_pr_auc":    test_metrics["pr_auc"],
        "train_f1":       train_metrics["f1"],
        "test_f1":        test_metrics["f1"],
        "train_f2":       train_metrics["f2"],
        "test_f2":        test_metrics["f2"],
        "train_precision": train_metrics["precision"],
        "test_precision":  test_metrics["precision"],
        "train_recall":    train_metrics["recall"],
        "test_recall":     test_metrics["recall"],
        "train_mcc":      train_metrics["mcc"],
        "test_mcc":       test_metrics["mcc"],
        "train_utility":  train_metrics["utility"],
        "test_utility":   test_metrics["utility"],
        "train_tp": train_metrics["tp"], "train_tn": train_metrics["tn"],
        "train_fp": train_metrics["fp"], "train_fn": train_metrics["fn"],
        "test_tp":  test_metrics["tp"],  "test_tn":  test_metrics["tn"],
        "test_fp":  test_metrics["fp"],  "test_fn":  test_metrics["fn"],
        "train_time_s":   train_time,
        "total_time_s":   total_time,
    }


# ─── Summary printer ──────────────────────────────────────────────────────────
def print_summary(results: list) -> None:
    W = 140
    print("\n" + "=" * W)
    print("RESULTS SUMMARY")
    print("=" * W)
    hdr = (f"{'Experiment':<40} {'TestAcc':>8} {'AUC-ROC':>8} {'PR-AUC':>8} "
           f"{'Prec':>8} {'Recall':>8} {'F1':>8} {'F2':>8} "
           f"{'MCC':>8} {'Utility':>9} {'Time(s)':>8}")
    print(hdr)
    print("-" * W)
    for r in results:
        print(
            f"{r['name']:<40} {r['test_acc']:>8.4f} {r['test_auc']:>8.4f} "
            f"{r['test_pr_auc']:>8.4f} {r['test_precision']:>8.4f} {r['test_recall']:>8.4f} "
            f"{r['test_f1']:>8.4f} {r['test_f2']:>8.4f} "
            f"{r['test_mcc']:>8.4f} {r['test_utility']:>9.4f} {r['train_time_s']:>8.1f}"
        )
    print("=" * W)

    # Scaling comparison (experiments 0 vs 1)
    if len(results) >= 2:
        print("\n--- Effect of Scaling (no PCA) ---")
        r0, r1 = results[0], results[1]
        print(f"  AUC  : {r0['test_auc']:.4f} -> {r1['test_auc']:.4f}  "
              f"(delta = {r1['test_auc'] - r0['test_auc']:+.4f})")
        print(f"  F1   : {r0['test_f1']:.4f} -> {r1['test_f1']:.4f}  "
              f"(delta = {r1['test_f1'] - r0['test_f1']:+.4f})")

    # PCA comparison (experiments 1 vs 3)
    if len(results) >= 4:
        print("\n--- Effect of PCA (with scaling) ---")
        r1, r3 = results[1], results[3]
        print(f"  AUC  : {r1['test_auc']:.4f} -> {r3['test_auc']:.4f}  "
              f"(delta = {r3['test_auc'] - r1['test_auc']:+.4f})")
        print(f"  F1   : {r1['test_f1']:.4f} -> {r3['test_f1']:.4f}  "
              f"(delta = {r3['test_f1'] - r1['test_f1']:+.4f})")


# ─── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Corporate Bankruptcy Prediction – AIML427 A3 Q2"
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default="data/company_years_h1.parquet",
        help="Path to company_years_h1.parquet (local or HDFS)",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        metavar="FRACTION",
        help="Fraction of data to sample (0, 1] – use < 1 for local testing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="HDFS_PATH",
        help="HDFS path to write results text file (e.g. /user/you/q2_results)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    wall_start = time.time()

    # ── Load & inspect ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  AIML427 A3 Q2 - Corporate Bankruptcy Prediction")
    print(f"{'='*65}")
    print(f"  Data path : {args.data_path}")
    print(f"  Sample    : {args.sample * 100:.0f}%  |  Seed={SEED}")
    print(f"  Model     : RandomForest  trees={NUM_TREES}  depth={MAX_DEPTH}")
    print(f"  PCA K     : {PCA_K}")
    print()

    df, feature_cols = load_data(spark, args.data_path, sample_fraction=args.sample)

    n_rows = df.count()
    print(f"  Rows    : {n_rows:,}")
    print(f"  Features: {len(feature_cols)}")
    print()

    # Label distribution
    label_dist = (
        df.groupBy(LABEL_COL).count().orderBy(LABEL_COL).collect()
    )
    print("  Label distribution:")
    for row in label_dist:
        pct = row["count"] / n_rows * 100
        label_name = "distressed" if int(row[LABEL_COL]) == 1 else "healthy"
        print(f"    {int(row[LABEL_COL])} ({label_name}): {row['count']:>8,}  ({pct:.2f}%)")

    # Country distribution
    country_dist = (
        df.groupBy("country").count().orderBy("country").collect()
    )
    print("  Country distribution:")
    for row in country_dist:
        name = _COUNTRY_MAP.get(int(row["country"]), str(int(row["country"])))
        pct = row["count"] / n_rows * 100
        print(f"    {name}: {row['count']:>8,}  ({pct:.1f}%)")

    # Sector distribution
    sector_dist = (
        df.groupBy("sector_1").count().orderBy("sector_1").collect()
    )
    print("  Sector distribution (sector_1):")
    for row in sector_dist:
        name = _SECTOR_MAP.get(int(row["sector_1"]), str(int(row["sector_1"])))
        pct = row["count"] / n_rows * 100
        print(f"    {name}: {row['count']:>8,}  ({pct:.1f}%)")

    # ── Train / test split ────────────────────────────────────────────────────
    train_df, test_df = df.randomSplit([TRAIN_RATIO, TEST_RATIO], seed=SEED)

    # Add class weights to the training set
    print("\n  Computing class weights ...")
    train_df = add_class_weights(train_df)

    # Cache in memory only — no disk spill (avoids "No space left" on shared cluster)
    train_df.persist(StorageLevel.MEMORY_ONLY)
    test_df.persist(StorageLevel.MEMORY_ONLY)

    # Force caching (also counts rows)
    n_train = train_df.count()
    n_test  = test_df.count()
    print(f"\n  Train rows : {n_train:,}  |  Test rows : {n_test:,}")

    # Drop features that are entirely (or nearly entirely) null in training data.
    # Spark's Imputer cannot compute a median/mean for an all-null column.
    print("\n  Checking feature coverage on training data ...")
    feature_cols = filter_valid_features(train_df, feature_cols)

    # ── Four experiments ──────────────────────────────────────────────────────
    experiments = [
        # (display name,                        use_scaler, use_pca)
        ("Baseline (no scaling, no PCA)",        False,      False),
        ("StandardScaler only (no PCA)",         True,       False),
        ("PCA only (scaled internally)",         False,      True),
        ("StandardScaler + PCA",                 True,       True),
    ]

    results = []
    for name, use_scaler, use_pca in experiments:
        r = run_experiment(
            name, train_df, test_df, feature_cols,
            use_scaler=use_scaler, use_pca=use_pca,
        )
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(results)
    elapsed = time.time() - wall_start
    print(f"\n  Total wall-clock time: {elapsed:.1f} s")

    # ── Save results to HDFS (useful in cluster deploy-mode where stdout ──────
    # ── goes to YARN logs rather than the terminal)                       ──────
    if args.output:
        lines = []
        lines.append(
            "Experiment,TrainAcc,TestAcc,TrainAUC,TestAUC,"
            "TrainPRAUC,TestPRAUC,"
            "TrainPrec,TestPrec,TrainRecall,TestRecall,"
            "TrainF1,TestF1,TrainF2,TestF2,"
            "TrainMCC,TestMCC,TrainUtility,TestUtility,"
            "TP,TN,FP,FN,TrainTime_s"
        )
        for r in results:
            lines.append(
                f"{r['name']},{r['train_acc']:.4f},{r['test_acc']:.4f},"
                f"{r['train_auc']:.4f},{r['test_auc']:.4f},"
                f"{r['train_pr_auc']:.4f},{r['test_pr_auc']:.4f},"
                f"{r['train_precision']:.4f},{r['test_precision']:.4f},"
                f"{r['train_recall']:.4f},{r['test_recall']:.4f},"
                f"{r['train_f1']:.4f},{r['test_f1']:.4f},"
                f"{r['train_f2']:.4f},{r['test_f2']:.4f},"
                f"{r['train_mcc']:.4f},{r['test_mcc']:.4f},"
                f"{r['train_utility']:.4f},{r['test_utility']:.4f},"
                f"{r['test_tp']},{r['test_tn']},{r['test_fp']},{r['test_fn']},"
                f"{r['train_time_s']:.1f}"
            )
        lines.append(f"\nTotal wall-clock time: {elapsed:.1f} s")
        # Write as a single-partition text file to HDFS
        spark.createDataFrame([(l,) for l in lines], ["line"]) \
             .coalesce(1) \
             .write.mode("overwrite").text(args.output)
        print(f"\n  Results saved to HDFS: {args.output}")

    spark.stop()


if __name__ == "__main__":
    main()
