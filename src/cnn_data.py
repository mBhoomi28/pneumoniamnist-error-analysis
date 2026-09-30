from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from medmnist import PneumoniaMNIST


def get_loaders(batch_size=64):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    train = PneumoniaMNIST(split="train", transform=tf, download=True, root="data")
    val = PneumoniaMNIST(split="val", transform=tf, download=True, root="data")
    test = PneumoniaMNIST(split="test", transform=tf, download=True, root="data")

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, _, _ = get_loaders()
    images, labels = next(iter(train_loader))
    print("Batch images:", images.shape)
    print("Batch labels:", labels.shape)