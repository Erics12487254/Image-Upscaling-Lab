from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import TinyImageDataset


def main():
    dataset = TinyImageDataset()
    # Send two labeled images to the model at a time.
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Flatten four pixels per image, then make one score per class.
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(4, 2),
    )
    # Compare each image's two class scores with its correct label.
    loss_fn = nn.CrossEntropyLoss()
    # SGD uses the calculated gradients to update the model parameters.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(5):
        # An epoch is one pass through every batch in the loader.
        model.train()
        total_loss = 0.0

        for images, labels in loader:
            # Predict, measure error, calculate gradients, and update parameters.
            optimizer.zero_grad()
            scores = model(images)
            loss = loss_fn(scores, labels)
            loss.backward()
            optimizer.step()

            # Convert this batch's mean loss into a total for averaging later.
            total_loss += loss.item() * len(labels)

        # Divide by the number of images to get mean loss for the epoch.
        average_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}: loss = {average_loss:.4f}")

    # Save the trained parameters beside this script for the loading example.
    weights_file = Path(__file__).with_name("tiny_model.pth")
    torch.save(model.state_dict(), weights_file)
    print(f"Saved model weights to {weights_file.name}")


if __name__ == "__main__":
    main()

