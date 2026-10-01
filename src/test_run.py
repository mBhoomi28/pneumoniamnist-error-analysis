import numpy as np
import torch
from model import SmallCNN
from cnn_data import get_loaders

_, _, test_loader = get_loaders()

model = SmallCNN()
model.load_state_dict(torch.load("results/best_cnn.pt", map_location="cpu"))
model.eval()

all_probs, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        probs = torch.sigmoid(model(images)).flatten()
        all_probs.append(probs.numpy())
        all_labels.append(labels.flatten().numpy())

probs = np.concatenate(all_probs)
labels = np.concatenate(all_labels)

np.savez("results/cnn_test_outputs.npz", probs=probs, labels=labels)
print("Saved", len(probs), "test predictions to results/cnn_test_outputs.npz")