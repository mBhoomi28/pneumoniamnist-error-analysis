import math
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST


def plot_grid(df, title, path, n=16, cols=4):
    images = PneumoniaMNIST(split="test", download=True, root="data").imgs
    df = df.head(n)
    rows = max(1, math.ceil(len(df) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, df.iterrows()):
        ax.imshow(images[int(r["index"])], cmap="gray")
        ax.set_title(f"#{int(r['index'])}  score {r['score']:.2f}", fontsize=9)

    fig.suptitle(title + "\n(score = model's probability of pneumonia)", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()