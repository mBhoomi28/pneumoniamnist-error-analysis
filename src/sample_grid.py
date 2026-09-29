import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST, INFO

label_names = INFO["pneumoniamnist"]["label"]
train = PneumoniaMNIST(split="train", download=True, root="data")

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(train.imgs[i], cmap="gray")
    label = int(train.labels[i][0])
    ax.set_title(label_names[str(label)])
    ax.axis("off")

plt.tight_layout()
plt.savefig("figures/sample_grid.png", dpi=150)
plt.show()