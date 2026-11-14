import cv2
import numpy as np
import pytest

from main.data import downscale_image, prepare_dataset, prepare_image


def test_downscale_image_halves_dimensions():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    assert downscale_image(image, 2).shape == (10, 15, 3)


def test_downscale_rejects_invalid_scale():
    with pytest.raises(ValueError):
        downscale_image(np.zeros((4, 4, 3), dtype=np.uint8), 1)


def test_prepare_dataset_writes_matching_filename(tmp_path):
    high = tmp_path / "high"
    low = tmp_path / "low"
    high.mkdir()
    assert cv2.imwrite(str(high / "sample.png"), np.zeros((20, 30, 3), dtype=np.uint8))
    paths = prepare_dataset(high, low, 2)
    assert [path.name for path in paths] == ["sample.png"]
    assert cv2.imread(str(paths[0])).shape == (10, 15, 3)


def test_prepare_image_processes_one_image(tmp_path):
    source = tmp_path / "source.png"
    low = tmp_path / "low"
    assert cv2.imwrite(str(source), np.zeros((20, 30, 3), dtype=np.uint8))
    output = prepare_image(source, low, 2)
    assert output == low / "source.png"
    assert cv2.imread(str(output)).shape == (10, 15, 3)
