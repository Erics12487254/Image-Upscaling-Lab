
from pathlib import Path

import cv2
import torch
import torch.nn as nn


class ESPCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # 12 channels become 3 colour channels at 2× resolution.
        self.conv3 = nn.Conv2d(32, 12, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x


def load_image_pair(filename, low_folder, high_folder):
    """Load one colour LR/HR pair and return matching tensors."""
    low_path = low_folder / filename
    high_path = high_folder / filename

    low_image = cv2.imread(str(low_path), cv2.IMREAD_COLOR)
    high_image = cv2.imread(str(high_path), cv2.IMREAD_COLOR)

    if low_image is None:
        raise FileNotFoundError(f"Could not read image: {low_path}")
    if high_image is None:
        raise FileNotFoundError(f"Could not read image: {high_path}")

    # Convert pixel values from 0–255 to 0–1.
    low_image = low_image.astype("float32") / 255.0
    high_image = high_image.astype("float32") / 255.0

    # Convert HWC images to CHW tensors, then add batch dimension.
    low_tensor = torch.from_numpy(low_image).permute(2, 0, 1).unsqueeze(0)
    high_tensor = torch.from_numpy(high_image).permute(2, 0, 1).unsqueeze(0)

    # Match the target dimensions to the model's exact 2× output size.
    height, width = low_tensor.shape[-2:]
    high_tensor = high_tensor[:, :, :height * 2, :width * 2]

    return low_tensor, high_tensor


# The script is in test/Stage 7.
test_folder = Path(__file__).resolve().parent.parent
low_folder = test_folder / "data" / "low_resolution"
high_folder = test_folder / "data" / "high_resolution"

# Keep this photo aside so it can be evaluated after training.
held_out_name = "Alexei_Leonov.jpg"

training_names = [
    "Abbey_Lincoln_in_1966.jpg",
    "Agfa_Click_BW_1.jpg",
    "American_bison_k5680-1.jpg",
    "President_Barack_Obama.jpg",
]

training_pairs = [
    load_image_pair(name, low_folder, high_folder)
    for name in training_names
]

model = ESPCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()

patch_size = 64
training_steps = 100

for step in range(1, training_steps + 1):
    # Choose an image and a random LR crop.
    pair_index = torch.randint(0, len(training_pairs), (1,)).item()
    low_image, high_image = training_pairs[pair_index]

    height, width = low_image.shape[-2:]
    top = torch.randint(0, height - patch_size + 1, (1,)).item()
    left = torch.randint(0, width - patch_size + 1, (1,)).item()

    low_patch = low_image[
        :, :, top:top + patch_size, left:left + patch_size
    ]

    # The matching HR crop starts at 2× the LR coordinates.
    high_patch = high_image[
        :,
        :,
        top * 2:(top + patch_size) * 2,
        left * 2:(left + patch_size) * 2,
    ]

    optimizer.zero_grad()

    prediction = model(low_patch)
    loss = loss_function(prediction, high_patch)

    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}/{training_steps}, loss: {loss.item():.6f}")


# Save the colour model weights separately from the grayscale model.
model_path = Path(__file__).resolve().parent / "espcn_color_x2.pth"
torch.save(model.state_dict(), model_path)

print(f"Model saved to: {model_path}")
print(f"Reserved for later evaluation: {held_out_name}")


# Evaluate on the held-out image.
model.eval()

low_test_path = low_folder / held_out_name
high_test_path = high_folder / held_out_name

low_test_image = cv2.imread(str(low_test_path), cv2.IMREAD_COLOR)
high_test_image = cv2.imread(str(high_test_path), cv2.IMREAD_COLOR)

if low_test_image is None:
    raise FileNotFoundError(f"Could not read image: {low_test_path}")
if high_test_image is None:
    raise FileNotFoundError(f"Could not read image: {high_test_path}")

low_test_image = low_test_image.astype("float32") / 255.0

test_input = (
    torch.from_numpy(low_test_image)
    .permute(2, 0, 1)
    .unsqueeze(0)
)

with torch.no_grad():
    sr_output = model(test_input)


# Crop to the reference dimensions in case its width or height is odd.
target_height, target_width = high_test_image.shape[:2]
sr_output = sr_output[:, :, :target_height, :target_width]

# Convert CHW tensor back to HWC image format.
sr_image = sr_output.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
sr_image = (sr_image * 255).round().astype("uint8")

results_folder = test_folder / "SR_results"
results_folder.mkdir(parents=True, exist_ok=True)

result_path = results_folder / "Alexei_Leonov_ESPCN_color_x2.png"

if not cv2.imwrite(str(result_path), sr_image):
    raise OSError(f"Could not save image: {result_path}")

print("SR output:", sr_image.shape)
print(f"Saved result to: {result_path}")
