import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from data_utils import load_flat

X_val, y_val = load_flat("val")
model = joblib.load("results/logreg_baseline.joblib")
pred = model.predict(X_val)

ConfusionMatrixDisplay.from_predictions(
    y_val, pred, display_labels=["normal", "pneumonia"], cmap="Blues"
)
plt.title("Logistic regression baseline (validation)")
plt.tight_layout()
plt.savefig("figures/baseline_confusion.png", dpi=150)
plt.show()