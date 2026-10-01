import numpy as np
import pandas as pd

df = pd.read_csv("results/test_predictions.csv")

t, p = df["true_label"], df["pred_label"]
df["group"] = np.select(
    [(t == 1) & (p == 1), (t == 0) & (p == 0), (t == 0) & (p == 1), (t == 1) & (p == 0)],
    ["correct_pneumonia", "correct_normal", "false_positive", "false_negative"],
    default="unknown",
)
# Confidence = how sure the model was about the class it predicted
df["confidence"] = np.where(p == 1, df["score"], 1 - df["score"])

df.to_csv("results/test_predictions.csv", index=False)
print(df["group"].value_counts())