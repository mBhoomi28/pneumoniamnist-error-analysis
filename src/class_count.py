import numpy as np
from medmnist import PneumoniaMNIST, INFO

label_names = INFO["pneumoniamnist"]["label"]

for split in ["train", "val", "test"]:
    ds = PneumoniaMNIST(split=split, download=True, root="data")
    labels = ds.labels.flatten()
    values, counts = np.unique(labels, return_counts=True)
    print(split)
    for v, c in zip(values, counts):
        print(f"  {label_names[str(v)]}: {c} ({c / len(labels):.1%})")