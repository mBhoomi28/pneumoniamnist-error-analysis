import json
import torch
import torch.nn as nn
from model import SmallCNN
from cnn_data import get_loaders
from train_utils import train_one_epoch, evaluate

EPOCHS = 20
LR = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, val_loader, _ = get_loaders()
model = SmallCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

history = {"train_loss": [], "val_loss": [], "val_acc": []}
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "results/best_cnn.pt")
        print("  -> saved new best model")

with open("results/history.json", "w") as f:
    json.dump(history, f, indent=2)