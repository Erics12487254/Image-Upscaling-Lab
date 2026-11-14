import sqlite3
from types import SimpleNamespace

import cv2
import numpy as np

from main import cli, espcn, realesrgan


def test_pipeline_compares_one_image_and_replaces_results(tmp_path, monkeypatch):
    high = tmp_path / "high"
    low = tmp_path / "low"
    output = tmp_path / "output"
    high.mkdir()
    reference = np.full((16, 16, 3), 80, dtype=np.uint8)
    assert cv2.imwrite(str(high / "sample.png"), reference)

    def resize(image, *_args):
        height, width = image.shape[:2]
        return cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

    monkeypatch.setattr(cli, "load_espcn", lambda _path: (object(), object()))
    monkeypatch.setattr(cli, "upscale_espcn", resize)
    monkeypatch.setattr(cli, "load_realesrgan", lambda _path: object())
    monkeypatch.setattr(cli, "upscale_realesrgan", resize)
    database = tmp_path / "evaluation.db"
    args = SimpleNamespace(
        low_dir=low,
        output_dir=output,
        database=database,
        espcn_checkpoint=tmp_path / "fake_espcn.pth",
        realesrgan_weights=tmp_path / "fake_realesrgan.pth",
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: str(high / "sample.png"))
    cli.run_pipeline_command(args)

    with sqlite3.connect(args.database) as connection:
        methods = [row[0] for row in connection.execute(
            "SELECT method FROM evaluation_results ORDER BY id"
        )]
    assert methods == ["Bicubic", "ESPCN", "RealESRGAN"]
    assert len(list(output.glob("*.png"))) == 3

    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO evaluation_results (method, reference_path, low_reso_path, output_path) VALUES ('Old', 'old', 'old', 'old')")
    cli.run_pipeline_command(args)
    with sqlite3.connect(database) as connection:
        methods = [row[0] for row in connection.execute("SELECT method FROM evaluation_results ORDER BY id")]
    assert methods == ["Bicubic", "ESPCN", "RealESRGAN"]
