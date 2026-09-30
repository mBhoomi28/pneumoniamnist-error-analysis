import torch
from model import SmallCNN
from cnn_data import get_loaders

_, val_loader, _ = get_loaders()

model = SmallCNN()
model.load_state_dict(torch.load("results/best_cnn.pt", map_location="cpu"))
model.eval()

images, labels = next(iter(val_loader))
with torch.no_grad():
    probs = torch.sigmoid(model(images[:10]))
preds = (probs > 0.5).int().flatten().tolist()

print("True:     ", labels[:10].flatten().tolist())
print("Predicted:", preds)