import json
import joblib
from sklearn.metrics import accuracy_score, recall_score
from data_utils import load_flat

X_val, y_val = load_flat("val")
model = joblib.load("results/logreg_baseline.joblib")
pred = model.predict(X_val)

metrics = {
    "val_accuracy": accuracy_score(y_val, pred),
    "val_pneumonia_recall": recall_score(y_val, pred, pos_label=1),
}

print(f"Validation accuracy: {metrics['val_accuracy']:.3f}")
print(f"Pneumonia recall:    {metrics['val_pneumonia_recall']:.3f}")

with open("results/baseline_val_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)