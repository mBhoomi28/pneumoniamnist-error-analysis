from medmnist import PneumoniaMNIST


def load_flat(split):
    """Return flattened images (scaled 0 to 1) and labels for a split."""
    ds = PneumoniaMNIST(split=split, download=True, root="data")
    X = ds.imgs.reshape(len(ds.imgs), -1) / 255.0
    y = ds.labels.flatten()
    return X, y