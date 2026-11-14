"""Interactive one-image super-resolution pipeline."""

import sys
import time
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]

# Keep this file runnable both directly (editor Run button) and as a module.
if __package__ in (None, ""):
    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [
        entry for entry in sys.path
        if Path(entry or ".").resolve() != script_dir
    ]
    sys.path.insert(0, str(ROOT))
    from main.data import prepare_image, read_image
    from main.evaluation import calculate_metrics, save_results
    from main.bicubic import upscale as upscale_bicubic
    from main.espcn import load_model as load_espcn, upscale as upscale_espcn
    from main.realesrgan import load_model as load_realesrgan, upscale as upscale_realesrgan
else:
    from .data import prepare_image, read_image
    from .evaluation import calculate_metrics, save_results
    from .bicubic import upscale as upscale_bicubic
    from .espcn import load_model as load_espcn, upscale as upscale_espcn
    from .realesrgan import load_model as load_realesrgan, upscale as upscale_realesrgan


def resize_to_reference(image, reference):
    """Match model output dimensions to the reference before evaluation."""
    if image.shape[:2] != reference.shape[:2]:
        image = cv2.resize(
            image,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    return image


def process_image(
    image_path,
    *,
    low_dir=ROOT / "data/low_resolution",
    output_dir=ROOT / "data/output",
    database=ROOT / "evaluation.db",
    espcn_checkpoint=ROOT / "models/espcn_color_x2.pth",
    realesrgan_weights=ROOT / "models/RealESRGAN_x2plus.pth",
):
    """Run all three upscalers on one image and replace the evaluation rows."""
    image_path = Path(image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Clear the previous operation before preparing or running any models.
    save_results(database, [], reset=True)
    reference = read_image(image_path)
    low_path = prepare_image(image_path, low_dir, scale=2)
    low_image = read_image(low_path)

    torch, espcn_model = load_espcn(espcn_checkpoint)
    realesrgan_model = load_realesrgan(realesrgan_weights)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_outputs = []
    start = time.perf_counter()
    method_outputs.append(("Bicubic", upscale_bicubic(low_image, 2), time.perf_counter() - start))

    start = time.perf_counter()
    image = upscale_espcn(low_image, torch, espcn_model)
    method_outputs.append(("ESPCN", image, time.perf_counter() - start))

    start = time.perf_counter()
    image = upscale_realesrgan(low_image, realesrgan_model)
    method_outputs.append(("RealESRGAN", image, time.perf_counter() - start))

    results = []
    for method, image, runtime in method_outputs:
        image = resize_to_reference(image, reference)
        output_path = output_dir / f"{image_path.stem}_{method}_x2.png"
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Could not save output image: {output_path}")

        metrics = calculate_metrics(reference, image)
        results.append((
            method,
            str(image_path.resolve()),
            str(low_path.resolve()),
            str(output_path.resolve()),
            metrics["psnr"],
            metrics["ssim"],
            runtime,
            image.shape[1],
            image.shape[0],
        ))
        print(
            f"{method}: PSNR={metrics['psnr']:.4f} dB | "
            f"SSIM={metrics['ssim']:.4f} | runtime={runtime:.4f}s | "
            f"output={output_path}"
        )

    save_results(database, results)
    print(f"\nSaved results to {database}")


def run_pipeline_command(args):
    """Compatibility wrapper for callers that pass paths as attributes."""
    image_path = input("Please enter an image path: ").strip().strip('"')
    process_image(
        image_path,
        low_dir=args.low_dir,
        output_dir=args.output_dir,
        database=args.database,
        espcn_checkpoint=args.espcn_checkpoint,
        realesrgan_weights=args.realesrgan_weights,
    )


def main():
    image_path = input("Please enter an image path: ").strip().strip('"')
    process_image(image_path)


if __name__ == "__main__":
    main()
