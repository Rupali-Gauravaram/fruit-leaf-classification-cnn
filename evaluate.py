"""
Evaluation entry point.

The full comparison (CNN vs Random Forest, train-test gap, leakage check,
confusion matrices, augmentation experiment) lives in `eda.ipynb` and is the
recommended way to read the results.

This script reproduces just the CNN test-accuracy step, loading weights saved
by train.py.

Usage:
    1. python train.py            # produces small_cnn.pth
    2. python evaluate.py
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model.small_cnn import SmallCNN

DATA_DIR = "data/fruit_leaves"
BATCH_SIZE = 32
WEIGHTS = "small_cnn.pth"


def main():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SmallCNN(n_classes=len(full_dataset.classes))
    model.load_state_dict(torch.load(WEIGHTS))
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy on {total} unseen leaf images: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
