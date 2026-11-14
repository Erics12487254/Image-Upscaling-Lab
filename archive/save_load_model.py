from pathlib import Path

import torch
from torch import nn


def main():
    # Run train_classifier.py first to create this weights file.
    weights_file = Path(__file__).with_name("tiny_model.pth")

    # Recreate the same layers and sizes used when saving the weights.
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(4, 2),
    )
    # Read the saved parameter tensors and copy them into this model.
    weights = torch.load(weights_file, weights_only=True)
    model.load_state_dict(weights)
    # Put the model in prediction mode.
    model.eval()

    print(f"Loaded model weights from {weights_file.name}")


if __name__ == "__main__":
    main()
