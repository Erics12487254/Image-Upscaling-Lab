"""Image comparison metrics and SQLite result persistence."""

import sqlite3
from pathlib import Path

import cv2
from skimage.metrics import structural_similarity


def calculate_metrics(reference, output):
    if reference is None or output is None:
        raise ValueError("Reference and output images must be loaded")
    if reference.shape != output.shape:
        raise ValueError(f"Image shapes differ: {reference.shape} vs {output.shape}")
    psnr = float("inf") if cv2.norm(reference, output, cv2.NORM_L2) == 0 else float(cv2.PSNR(reference, output))
    return {
        "psnr": psnr,
        "ssim": float(structural_similarity(reference, output, channel_axis=2, data_range=255)),
    }


def save_results(database_path: Path, results, *, reset=False):
    """Save results to SQLite, optionally replacing the previous run."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                method TEXT NOT NULL,
                reference_path TEXT NOT NULL,
                low_reso_path TEXT NOT NULL,
                output_path TEXT NOT NULL,
                psnr REAL, ssim REAL, runtime REAL,
                width INTEGER, height INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        if reset:
            connection.execute("DELETE FROM evaluation_results")
        connection.executemany("""
            INSERT INTO evaluation_results (
                method, reference_path, low_reso_path, output_path,
                psnr, ssim, runtime, width, height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, results)
