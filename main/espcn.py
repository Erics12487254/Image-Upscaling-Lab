"""ESPCN model loading and inference for the x2 color checkpoint."""

from pathlib import Path

import numpy as np


def load_model(checkpoint: Path):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("ESPCN requires PyTorch. Install dependencies with: py -m pip install -r requirements.txt") from exc

    class ESPCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.conv3 = nn.Conv2d(32, 12, kernel_size=3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        def forward(self, image):
            image = self.relu1(self.conv1(image))
            image = self.relu2(self.conv2(image))
            return self.pixel_shuffle(self.conv3(image))

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ESPCN checkpoint not found: {checkpoint}")
    model = ESPCN()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return torch, model


def upscale(image: np.ndarray, torch, model) -> np.ndarray:
    image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        output = model(image_tensor)
    output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    return (output * 255).round().astype(np.uint8)
