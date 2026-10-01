import pandas as pd
from gallery_utils import plot_grid

df = pd.read_csv("results/test_predictions.csv")

for group, title, path in [
    ("correct_pneumonia", "Correct: pneumonia (random sample)", "figures/gallery_correct_pneumonia.png"),
    ("correct_normal", "Correct: normal (random sample)", "figures/gallery_correct_normal.png"),
]:
    subset = df[df["group"] == group].sample(16, random_state=0)
    plot_grid(subset, title, path)