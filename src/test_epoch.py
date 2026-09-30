import torch
import torch.nn as nn
from model import SmallCNN
from cnn_data import get_loaders
from train_utils import train_one_epoch

device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, _, _ = get_loaders()
model = SmallCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
print(f"Train loss after 1 epoch: {loss:.4f}")