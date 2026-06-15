"""
Training entry point.

The full, narrated workflow for this project lives in `eda.ipynb` (load data ->
build CNN -> Random Forest baseline -> diagnosis -> augmentation experiment),
because the project is an *investigation* and is best read cell by cell.

This script reproduces just the core CNN training step from that notebook, for
anyone who prefers a plain Python file.

Usage:
    1. pip install -r requirements.txt
    2. Download the dataset (see README) into data/fruit_leaves/
    3. python train.py
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model.small_cnn import SmallCNN

DATA_DIR = "data/fruit_leaves"
EPOCHS = 8
BATCH_SIZE = 32


def main():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, _ = random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SmallCNN(n_classes=len(full_dataset.classes))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{EPOCHS} - Average Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "small_cnn.pth")
    print("Done training. Weights saved to small_cnn.pth")


if __name__ == "__main__":
    main()
