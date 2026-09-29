from medmnist import PneumoniaMNIST
import os
os.makedirs("data", exist_ok=True)

train = PneumoniaMNIST(split="train", download=True, root="data")
val = PneumoniaMNIST(split="val", download=True, root="data")
test = PneumoniaMNIST(split="test", download=True, root="data")

print(len(train), len(val), len(test))