"""
Train and compare credit-risk classification models on the UCI Statlog
German Credit dataset.

Pipeline:
  1. Load raw space-separated data, assign column names, recode target.
  2. Quick EDA (class distribution -> highlights class imbalance).
  3. Build a ColumnTransformer (one-hot encode categoricals, scale numerics)
     wrapped in a Pipeline so preprocessing travels with the saved model.
  4. Train 4 variants:
       - Logistic Regression (baseline, class_weight='balanced')
       - Logistic Regression + SMOTE (on training fold only)
       - Random Forest (baseline, class_weight='balanced')
       - Random Forest + SMOTE (on training fold only)
  5. Evaluate all variants on a held-out stratified test set:
       Accuracy, Precision, Recall, F1, ROC-AUC.
  6. Pick the best variant by ROC-AUC, refit is already done, save it.
  7. Persist metrics.json, confusion_matrix.png, roc_curve.png.
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # never try to open a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mappings import CATEGORICAL_COLUMNS, COLUMN_NAMES, NUMERIC_COLUMNS

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
df = pd.read_csv("data/german.data", sep=r"\s+", header=None, names=COLUMN_NAMES)

# Raw target: 1 = good (no default), 2 = bad (default).
# Recode -> 0 = good/no-default, 1 = bad/default (the "positive" class we
# care about detecting for risk purposes).
df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})

print("=" * 60)
print("Loaded German Credit dataset:", df.shape)
print("=" * 60)

# --------------------------------------------------------------------------
# 2. EDA: class distribution
# --------------------------------------------------------------------------
class_counts = df["credit_risk"].value_counts().sort_index()
print("\nClass distribution (0 = good/no-default, 1 = bad/default):")
print(class_counts.to_string())
print(f"\nGood: {class_counts.get(0, 0)}  |  Bad: {class_counts.get(1, 0)}")
print(f"Imbalance ratio (good:bad) = "
      f"{class_counts.get(0, 0) / class_counts.get(1, 0):.2f} : 1")

# --------------------------------------------------------------------------
# 3. Train/test split (stratified, held-out test set never touched by SMOTE)
# --------------------------------------------------------------------------
X = df.drop(columns=["credit_risk"])
y = df["credit_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# --------------------------------------------------------------------------
# 4. Preprocessing: one-hot encode categoricals, scale numerics
# --------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ]
)

# --------------------------------------------------------------------------
# 5. Build the 4 model variants
# --------------------------------------------------------------------------
variants = {
    "LogisticRegression (class_weight=balanced)": Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    ),
    "LogisticRegression + SMOTE": ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    ),
    "RandomForest (class_weight=balanced)": Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "RandomForest + SMOTE": ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "model",
                RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
            ),
        ]
    ),
}

# --------------------------------------------------------------------------
# 6. Train + evaluate each variant
# --------------------------------------------------------------------------
results = []
fitted_pipelines = {}

for name, pipe in variants.items():
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "variant": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    results.append(metrics)

print("\n" + "=" * 90)
print("MODEL COMPARISON (held-out test set)")
print("=" * 90)
results_df = pd.DataFrame(results).set_index("variant")
print(results_df.to_string())
print("=" * 90)

# --------------------------------------------------------------------------
# 7. Pick the best variant by ROC-AUC
# --------------------------------------------------------------------------
best_result = max(results, key=lambda r: r["roc_auc"])
best_name = best_result["variant"]
best_pipeline = fitted_pipelines[best_name]

print(f"\nBest variant by ROC-AUC: {best_name}  (ROC-AUC = {best_result['roc_auc']})")

# Mark the winner in the saved metrics for the README / app to highlight
for r in results:
    r["is_best"] = r["variant"] == best_name

with open("reports/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved reports/metrics.json")

# --------------------------------------------------------------------------
# 8. Save the final model
# --------------------------------------------------------------------------
joblib.dump(best_pipeline, "models/model.joblib")
print("Saved models/model.joblib")

# --------------------------------------------------------------------------
# 9. Confusion matrix + ROC curve plots for the final model
# --------------------------------------------------------------------------
y_pred_best = best_pipeline.predict(X_test)
y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(5, 4.5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Good (0)", "Bad (1)"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix — {best_name}")
fig.tight_layout()
fig.savefig("reports/confusion_matrix.png", dpi=150)
plt.close(fig)
print("Saved reports/confusion_matrix.png")

fpr, tpr, _ = roc_curve(y_test, y_proba_best)
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.plot(fpr, tpr, label=f"ROC-AUC = {best_result['roc_auc']:.3f}", color="#1f77b4", lw=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve — {best_name}")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("reports/roc_curve.png", dpi=150)
plt.close(fig)
print("Saved reports/roc_curve.png")

print("\nDone.")
