import numpy as np
import pandas as pd

data = np.load("results/cnn_test_outputs.npz")
probs, labels = data["probs"], data["labels"]

df = pd.DataFrame({
    "index": np.arange(len(labels)),
    "true_label": labels.astype(int),
    "pred_label": (probs > 0.5).astype(int),
    "score": probs,  # model's probability of pneumonia
})
df.to_csv("results/test_predictions.csv", index=False)
print(df.head())
print("Saved", len(df), "rows to results/test_predictions.csv")