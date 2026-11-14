import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from dataset import TinyImageDataset


def main():
    dataset = TinyImageDataset()

    # These tiny subsets only demonstrate the workflow, not reliable evaluation.
    train_data = Subset(dataset, [0, 1])
    val_data = Subset(dataset, [2, 3])
    train_loader = DataLoader(train_data, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=2, shuffle=False)

    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    # Measure how far the class scores are from the correct labels.
    loss_fn = nn.CrossEntropyLoss()
    # Only the training loop uses this optimizer to change parameters.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_losses = []
    val_losses = []

    for epoch in range(5):
        # Training batches update the model parameters.
        model.train()
        train_total = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(labels)

        average_train_loss = train_total / len(train_data)
        train_losses.append(average_train_loss)

        # Validation measures performance without updating the model.
        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                loss = loss_fn(model(images), labels)
                # Weight each batch mean by its number of examples.
                val_total += loss.item() * len(labels)

        # Keep one average from each set for every epoch.
        average_val_loss = val_total / len(val_data)
        val_losses.append(average_val_loss)
        print(
            f"Epoch {epoch + 1}: "
            f"train loss = {average_train_loss:.4f}, "
            f"validation loss = {average_val_loss:.4f}"
        )


if __name__ == "__main__":
    main()
