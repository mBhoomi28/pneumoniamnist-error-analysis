import numpy as np
from data_utils import load_flat

_, y_train = load_flat("train")
_, y_val = load_flat("val")

majority = np.bincount(y_train).argmax()
accuracy = (y_val == majority).mean()

print("Most common class in train:", majority, "(0 = normal, 1 = pneumonia)")
print(f"Always-predict-majority validation accuracy: {accuracy:.3f}")