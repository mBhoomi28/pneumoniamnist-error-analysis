from medmnist import PneumoniaMNIST, INFO

info = INFO["pneumoniamnist"]
print("Labels:", info["label"])

for split in ["train", "val", "test"]:
    ds = PneumoniaMNIST(split=split, download=True, root="data")
    print(split, "| images:", ds.imgs.shape, "| labels:", ds.labels.shape)