import json
import matplotlib.pyplot as plt

with open("results/history.json") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(7, 5))
plt.plot(epochs, history["train_loss"], label="Train loss")
plt.plot(epochs, history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN training curves")
plt.legend()
plt.tight_layout()
plt.savefig("figures/loss_curves.png", dpi=150)
plt.show()