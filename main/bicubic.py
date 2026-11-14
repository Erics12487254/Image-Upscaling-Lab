"""Bicubic interpolation baseline."""

import cv2


def upscale(image, scale: int = 2):
    if scale < 1:
        raise ValueError("scale must be at least 1")
    height, width = image.shape[:2]
    return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
