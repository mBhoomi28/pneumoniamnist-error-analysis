import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

data = np.load("results/cnn_test_outputs.npz")
probs, labels = data["probs"], data["labels"]
preds = (probs > 0.5).astype(int)

metrics = {
    "accuracy": accuracy_score(labels, preds),
    "precision": precision_score(labels, preds),
    "recall": recall_score(labels, preds),
    "f1": f1_score(labels, preds),
    "roc_auc": roc_auc_score(labels, probs),
}
metrics = {k: float(v) for k, v in metrics.items()}

for name, value in metrics.items():
    print(f"{name:10s} {value:.3f}")

with open("results/cnn_test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)