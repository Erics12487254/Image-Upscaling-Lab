"""Dataset preparation helpers for paired high- and low-resolution images."""

from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def downscale_image(image, scale: int = 2):
    """Create an LR image by shrinking each dimension by ``scale``."""
    if scale < 2:
        raise ValueError("scale must be an integer of at least 2")
    height, width = image.shape[:2]
    target = (width // scale, height // scale)
    if min(target) < 1:
        raise ValueError("Image is too small for the requested scale")
    return cv2.resize(image, target, interpolation=cv2.INTER_CUBIC)


def prepare_dataset(high_resolution_dir: Path, low_resolution_dir: Path, scale: int = 2):
    """Downscale every supported image and save it under the same filename."""
    high_resolution_dir = Path(high_resolution_dir)
    low_resolution_dir = Path(low_resolution_dir)
    low_resolution_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path for path in high_resolution_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images in {high_resolution_dir}")
    outputs = []
    for source_path in image_paths:
        output_path = low_resolution_dir / source_path.name
        if not cv2.imwrite(str(output_path), downscale_image(read_image(source_path), scale)):
            raise OSError(f"Could not save image: {output_path}")
        outputs.append(output_path)
    return outputs


def prepare_image(source_path: Path, low_resolution_dir: Path, scale: int = 2):
    """Downscale one source image and save it under its original filename."""
    source_path = Path(source_path)
    if source_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image format: {source_path.suffix}")
    low_resolution_dir = Path(low_resolution_dir)
    low_resolution_dir.mkdir(parents=True, exist_ok=True)
    output_path = low_resolution_dir / source_path.name
    image = downscale_image(read_image(source_path), scale)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not save image: {output_path}")
    return output_path
