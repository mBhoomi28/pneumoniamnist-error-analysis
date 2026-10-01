import pandas as pd

df = pd.read_csv("results/test_predictions.csv")

fp = df[df["group"] == "false_positive"].sort_values("confidence", ascending=False)
fn = df[df["group"] == "false_negative"].sort_values("confidence", ascending=False)

fp.to_csv("results/false_positives_ranked.csv", index=False)
fn.to_csv("results/false_negatives_ranked.csv", index=False)

print("Most confident false positives:")
print(fp[["index", "score", "confidence"]].head(5).to_string(index=False))
print("\nMost confident false negatives:")
print(fn[["index", "score", "confidence"]].head(5).to_string(index=False))