import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

data = np.load("results/cnn_test_outputs.npz")
probs, labels = data["probs"], data["labels"]
preds = (probs > 0.5).astype(int)

ConfusionMatrixDisplay.from_predictions(
    labels, preds, display_labels=["normal", "pneumonia"], cmap="Blues"
)
plt.title("CNN confusion matrix (test)")
plt.tight_layout()
plt.savefig("figures/cnn_test_confusion.png", dpi=150)
plt.show()