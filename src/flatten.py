from data_utils import load_flat

X_train, y_train = load_flat("train")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Pixel range:", X_train.min(), "to", X_train.max())