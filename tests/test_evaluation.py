import sqlite3

import numpy as np
import pytest

from main.evaluation import calculate_metrics, save_results


def test_identical_images_have_perfect_similarity():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    metrics = calculate_metrics(image, image.copy())
    assert metrics["psnr"] == float("inf")
    assert metrics["ssim"] == pytest.approx(1.0)


def test_metrics_require_matching_dimensions():
    with pytest.raises(ValueError, match="shapes differ"):
        calculate_metrics(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8))


def test_results_save_all_methods_for_an_operation(tmp_path):
    database = tmp_path / "results.db"
    rows = [
        (method, "ref.png", "lr.png", f"{method}.png", 30.0, 0.9, 0.01, 8, 8)
        for method in ("Bicubic", "ESPCN", "RealESRGAN")
    ]
    save_results(database, rows, reset=True)
    with sqlite3.connect(database) as connection:
        methods = [row[0] for row in connection.execute("SELECT method FROM evaluation_results ORDER BY id")]
    assert methods == ["Bicubic", "ESPCN", "RealESRGAN"]


def test_reset_replaces_previous_results(tmp_path):
    database = tmp_path / "results.db"
    row = ("Bicubic", "ref.png", "lr.png", "out.png", 30.0, 0.9, 0.01, 8, 8)
    save_results(database, [("Old", *row[1:])])
    save_results(database, [row], reset=True)
    with sqlite3.connect(database) as connection:
        methods = [row[0] for row in connection.execute("SELECT method FROM evaluation_results")]
    assert methods == ["Bicubic"]
