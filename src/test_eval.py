import torch
import torch.nn as nn
from model import SmallCNN
from cnn_data import get_loaders
from train_utils import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
_, val_loader, _ = get_loaders()
model = SmallCNN().to(device)
criterion = nn.BCEWithLogitsLoss()

val_loss, val_acc = evaluate(model, val_loader, criterion, device)
print(f"Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.3f}")