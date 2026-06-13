import logging
import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def list_available_classes(train_dir: str) -> list[str]:
    p = Path(train_dir)
    if not p.exists():
        logger.warning("Training directory not found: %s", train_dir)
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir())


def find_class_folder(train_dir: str, fruit_name: str) -> Path | None:
    p = Path(train_dir)
    target = fruit_name.lower()
    for d in p.iterdir():
        if d.is_dir() and d.name.lower() == target:
            return d
        # Partial match: folder starts with the fruit name
        if d.is_dir() and d.name.lower().startswith(target):
            return d
    return None


def load_image_gray(image_path: str | Path, size: int) -> np.ndarray | None:
    img = cv2.imread(str(image_path))
    if img is None:
        logger.debug("Could not read: %s", image_path)
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def make_foreground_mask(gray: np.ndarray, threshold: int) -> np.ndarray:
  
    mask = (gray < threshold).astype(np.uint8)
    return mask


def iter_class_images(
    class_dir: Path,
    image_size: int,
    max_images: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
  
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    count = 0
    for fpath in sorted(class_dir.iterdir()):
        if fpath.suffix.lower() not in extensions:
            continue
        gray = load_image_gray(fpath, image_size)
        if gray is None:
            continue
        yield gray, fpath
        count += 1
        if max_images and count >= max_images:
            break
