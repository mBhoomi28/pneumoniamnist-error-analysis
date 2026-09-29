import numpy as np
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST, INFO

label_names = INFO["pneumoniamnist"]["label"]
splits = ["train", "val", "test"]

counts = {}
for split in splits:
    ds = PneumoniaMNIST(split=split, download=True, root="data")
    counts[split] = np.bincount(ds.labels.flatten(), minlength=2)

x = np.arange(len(splits))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - width / 2, [counts[s][0] for s in splits], width, label=label_names["0"])
ax.bar(x + width / 2, [counts[s][1] for s in splits], width, label=label_names["1"])
ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.set_ylabel("Number of images")
ax.set_title("Class counts per split")
ax.legend()

plt.tight_layout()
plt.savefig("figures/class_counts.png", dpi=150)
plt.show()