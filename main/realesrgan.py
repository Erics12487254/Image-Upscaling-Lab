"""Real-ESRGAN x2plus model loading and inference."""

from pathlib import Path

import numpy as np


def load_model(weights: Path):
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as exc:
        raise RuntimeError(
            "Real-ESRGAN dependencies are missing. Install them with: "
            "py -m pip install -r requirements.txt"
        ) from exc

    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Real-ESRGAN x2plus weights not found: {weights}")
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    return RealESRGANer(
        scale=2,
        model_path=str(weights),
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None,
    )


def upscale(image: np.ndarray, upsampler) -> np.ndarray:
    output, _ = upsampler.enhance(image, outscale=2)
    return output
