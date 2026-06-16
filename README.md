# Bankruptcy Prediction using Apache Spark on YARN Cluster

A distributed machine learning pipeline for corporate bankruptcy prediction, built with **PySpark** and designed to run on a **Hadoop YARN cluster**. This project was developed as part of **AIML427 Big Data – Assignment 3** and tackles binary classification on a large-scale financial dataset of over 1 million company-year observations.

---

## Overview

Corporate bankruptcy prediction is a critical task in credit risk management. Missing a bankruptcy (false negative) can cost a lender 40–80% of loan principal, whereas a false alarm (false positive) costs only analyst time. This pipeline uses a **Random Forest Classifier** with inverse-frequency class weighting to address the severe class imbalance (~0.4% distressed companies) inherent in real-world financial data.

Four experiments are run to evaluate the impact of feature scaling and dimensionality reduction:

| # | Experiment | Scaler | PCA |
|---|---|---|---|
| 1 | Baseline | ✗ | ✗ |
| 2 | StandardScaler only | ✓ | ✗ |
| 3 | PCA only (scaled internally) | ✗ | ✓ |
| 4 | StandardScaler + PCA | ✓ | ✓ |

---

## Dataset

**V4 Group Corporate Bankruptcy Dataset** (`company_years_h1.parquet`) sourced from the EMIS database.

| Property | Detail |
|---|---|
| Instances | 1,000,087 company-year observations |
| Raw columns | 143 (131 financial features + 6 near-null + 5 ID cols + 1 label) |
| Label | `main_label` — 1: financially distressed, 0: healthy |
| Prediction horizon | 1-year-ahead financial distress |

**Country breakdown:**

| Country | Observations | Share |
|---|---|---|
| Czech Republic | 554,751 | 55.5% |
| Slovakia | 335,684 | 33.6% |
| Hungary | 55,736 | 5.6% |
| Poland | 53,916 | 5.4% |

**Sector breakdown (NAICS codes):**

| Sector | Observations | Share |
|---|---|---|
| Wholesale Trade (42) | 372,319 | 37.2% |
| Construction (23) | 257,073 | 25.7% |
| Retail Trade (44) | 173,066 | 17.3% |
| Transportation & Warehousing (48) | 99,261 | 9.9% |
| Manufacturing (31–33) | 84,300 | 8.4% |
| Utilities (22) | 14,068 | 1.4% |

---

## Architecture

```
Raw Parquet (HDFS)
        │
        ▼
┌───────────────────┐
│  Data Loading &   │
│  Preprocessing    │  Drop ID cols, cast to Double, optional sampling
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Train/Test Split │  80% train / 20% test (stratified by seed)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Class Weighting  │  Inverse-frequency weights to handle ~0.4% minority class
└────────┬──────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│              Spark ML Pipeline             │
│  1. Median Imputer (per-column)            │
│  2. VectorAssembler                        │
│  3. StandardScaler (optional)              │
│  4. PCA – k=50 components (optional)       │
│  5. RandomForestClassifier                 │
│     trees=100, maxDepth=10, sqrt features  │
└────────────────────────────────────────────┘
         │
         ▼
┌───────────────────┐
│    Evaluation     │  AUC-ROC, PR-AUC, F1, F2, MCC, Utility Score
└───────────────────┘
```

---

## Evaluation Metrics

The model is evaluated using a comprehensive set of metrics suited to imbalanced binary classification:

- **AUC-ROC** — overall discriminative ability
- **PR-AUC** — precision-recall tradeoff (more informative under class imbalance)
- **F1 / F2** — harmonic mean of precision & recall (F2 weights recall 2×)
- **MCC** — Matthews Correlation Coefficient, robust single metric for imbalanced data
- **Cost-Weighted Utility** — asymmetric cost matrix reflecting real-world credit risk:

| Outcome | Cost |
|---|---|
| True Positive (caught bankruptcy) | +1 |
| True Negative (correctly healthy) | 0 |
| False Positive (false alarm) | −1 |
| False Negative (missed bankruptcy) | **−10** |

---

## Requirements

- Python 3.7+
- Apache Spark 3.x
- Hadoop YARN cluster (or local mode for testing)
- PySpark

```bash
pip install pyspark
```

---

## Usage

### Local Mode (with optional sampling for quick tests)

```bash
python q2_bankruptcy.py data/company_years_h1.parquet --sample 0.1
```

### YARN Cluster (spark-submit)

```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  q2_bankruptcy.py \
  hdfs:///user/<your-username>/data/company_years_h1.parquet
```

### Save Results to HDFS

```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  q2_bankruptcy.py \
  hdfs:///user/<your-username>/data/company_years_h1.parquet \
  --output hdfs:///user/<your-username>/q2_results
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `data_path` | `data/company_years_h1.parquet` | Path to the Parquet file (local or HDFS) |
| `--sample FRACTION` | `1.0` | Fraction of data to use (0–1]; useful for local smoke tests |
| `--output HDFS_PATH` | `None` | HDFS path to write CSV results (useful in cluster deploy-mode) |

---

## Model Configuration

| Hyperparameter | Value |
|---|---|
| Number of trees | 100 |
| Max depth | 10 |
| Min instances per leaf | 10 |
| Feature subset strategy | `sqrt` (standard for classification) |
| PCA components (k) | 50 |
| Train/test split | 80% / 20% |
| Random seed | 42 |

---

## Project Structure

```
├── q2_bankruptcy.py       # Main PySpark script
└── 300681732_Report.pdf   # Project report
```

---

## Key Design Decisions

**Class imbalance handling:** Inverse-frequency class weights (`w_k = N / (2 * n_k)`) are computed from the training set and passed to the Random Forest via `weightCol`. This avoids the need for oversampling/undersampling while preserving the full dataset.

**Null-feature filtering:** Features with fewer than 5% non-null values are dropped before modelling. Spark's `Imputer` cannot compute a valid median for near-empty columns.

**PCA pre-scaling:** PCA requires zero-mean, unit-variance data. The pipeline automatically inserts a `StandardScaler` before PCA even when `use_scaler=False` is selected (the "PCA only" experiment).

**Memory management:** DataFrames are persisted with `MEMORY_ONLY` storage level to avoid disk spill on shared YARN clusters with limited local storage.

**Adaptive query execution:** Spark's adaptive query execution (`spark.sql.adaptive.enabled`) and automatic partition coalescing are enabled to optimise shuffle performance on the cluster.

---

## References

- Altman, E.I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *Journal of Finance*, 23(4), 589–609.
- Zmijewski, M.E. (1984). Methodological issues related to the estimation of financial distress prediction models. *Journal of Accounting Research*, 22, 59–82.
- EMIS Database — V4 Group Corporate Bankruptcy Dataset
