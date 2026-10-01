import json
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from data_utils import load_flat


def compute(y, pred, score):
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
    }


X_train, y_train = load_flat("train")
X_test, y_test = load_flat("test")

# Majority-class baseline
majority = np.bincount(y_train).argmax()
majority_row = compute(
    y_test,
    np.full(len(y_test), majority),
    np.full(len(y_test), 0.5),
)

# Logistic regression baseline
logreg = joblib.load("results/logreg_baseline.joblib")
logreg_row = compute(
    y_test,
    logreg.predict(X_test),
    logreg.predict_proba(X_test)[:, 1],
)

# CNN (from step 31)
with open("results/cnn_test_metrics.json") as f:
    cnn_row = json.load(f)

rows = {
    "Majority class": majority_row,
    "Logistic regression": logreg_row,
    "Small CNN": cnn_row,
}

cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
lines = ["| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
         "|---|---|---|---|---|---|"]
for name, m in rows.items():
    lines.append(f"| {name} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")

table = "\n".join(lines)
print(table)

with open("results/results_table.md", "w") as f:
    f.write(table + "\n")