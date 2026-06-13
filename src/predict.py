"""
Prediction: classify an image using trained prototypes.

For each block of the image:
  1. Build Laplacian Pyramid.
  2. Extract features per block per level.
  3. Compare to stored prototypes with Euclidean / normalized distance.
  4. Assign the nearest class (or background if activity too low).
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from .data_utils import load_image_gray, make_foreground_mask
from .pyramid import build_laplacian_pyramid
from .features import (
    extract_level_features,
    cv2_resize_mask,
    N_FEATURES_PER_LEVEL,
)

logger = logging.getLogger(__name__)


def _distance(feat: np.ndarray, proto_mean: np.ndarray, proto_std: np.ndarray, metric: str) -> float:
    """Compute distance between a feature vector and a prototype."""
    diff = feat - proto_mean
    if metric == "normalized":
        diff = diff / proto_std
    return float(np.sqrt(np.sum(diff ** 2)))


def predict_image(
    image_path: str,
    prototypes: dict,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify an image into fruit classes.

    Returns
    -------
    label_map   : (H, W) int array — class index per pixel (-1 = background)
    color_mask  : (H, W, 3) BGR array — colored segmentation for visualization
    orig_bgr    : (H, W, 3) BGR array — resized original image
    """
    size = cfg["image_size"]
    bg_thresh = cfg["background_threshold"]
    levels = cfg["pyramid_levels"]
    block_sizes = cfg["block_sizes"]
    metric = cfg["distance_metric"]
    min_fg = cfg["min_foreground_fraction"]

    # Load and preprocess
    orig_bgr = cv2.imread(str(image_path))
    if orig_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig_bgr = cv2.resize(orig_bgr, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    fg_mask = make_foreground_mask(gray, bg_thresh)

    # Build pyramid
    lap_pyr = build_laplacian_pyramid(gray, levels)

    # Build per-block label map at each level then upsample to full size
    class_names = list(prototypes.keys())
    n_classes = len(class_names)

    # We aggregate block predictions at the finest level's grid
    finest_bsize = block_sizes[0]
    h, w = gray.shape
    n_rows = max(1, h // finest_bsize)
    n_cols = max(1, w // finest_bsize)

    # For each block position, build a multi-level feature vector
    block_labels = np.full((n_rows, n_cols), -1, dtype=np.int32)
    block_dists = np.full((n_rows, n_cols, n_classes), np.inf, dtype=np.float32)

    # Extract feature grids at each level
    level_grids = []
    for lv_idx, (lap_level, bsize) in enumerate(zip(lap_pyr, block_sizes[:levels])):
        fg_at_level = cv2_resize_mask(fg_mask, *lap_level.shape[:2])
        feat_grid, fg_grid = extract_level_features(lap_level, bsize, fg_at_level)
        level_grids.append((feat_grid, fg_grid, bsize, lap_level.shape[:2]))

    # For each block in the finest grid, collect features from all levels
    for r in range(n_rows):
        for c in range(n_cols):
            # Check foreground at finest level
            fg_frac = level_grids[0][1][
                min(r, level_grids[0][1].shape[0] - 1),
                min(c, level_grids[0][1].shape[1] - 1),
            ]
            if fg_frac < 0.3:
                continue  # background block

            # Build multi-level feature vector for this block
            feat_parts = []
            for lv_idx, (feat_grid, fg_grid, bsize, lv_shape) in enumerate(level_grids):
                # Map finest-level block (r,c) to this level's block index
                scale = 2 ** lv_idx
                lr = min(r // scale, feat_grid.shape[0] - 1)
                lc = min(c // scale, feat_grid.shape[1] - 1)
                feat_parts.append(feat_grid[lr, lc])

            feat_vec = np.concatenate(feat_parts, axis=0).astype(np.float32)

            # Classify: nearest prototype
            best_cls = -1
            best_dist = np.inf
            for cls_idx, cls_name in enumerate(class_names):
                proto = prototypes[cls_name]
                d = _distance(feat_vec, proto["mean"], proto["std"], metric)
                block_dists[r, c, cls_idx] = d
                if d < best_dist:
                    best_dist = d
                    best_cls = cls_idx

            block_labels[r, c] = best_cls

    # Upsample block labels to pixel map
    label_map = _upsample_labels(block_labels, h, w, finest_bsize)

    # Build colored mask
    color_mask = _colorize(label_map, class_names)

    return label_map, color_mask, orig_bgr


def predict_class(
    image_path: str,
    prototypes: dict,
    cfg: dict,
) -> tuple[str, float]:
    """
    Predict image-level class using the same feature pipeline as training:
    extract_pyramid_features on the whole image, compare to per-class prototypes.
    Returns (predicted_class_name, confidence_as_inverse_relative_distance).
    """
    from .data_utils import load_image_gray, make_foreground_mask
    from .pyramid import build_laplacian_pyramid
    from .features import extract_pyramid_features

    size = cfg["image_size"]
    bg_thresh = cfg["background_threshold"]
    levels = cfg["pyramid_levels"]
    block_sizes = cfg["block_sizes"]
    metric = cfg["distance_metric"]

    gray = load_image_gray(image_path, size)
    if gray is None:
        return "unknown", 0.0
    fg_mask = make_foreground_mask(gray, bg_thresh)
    lap_pyr = build_laplacian_pyramid(gray, levels)
    feat_vec, fg_frac = extract_pyramid_features(lap_pyr, block_sizes[:levels], fg_mask)

    if fg_frac < cfg.get("min_foreground_fraction", 0.05):
        return "unknown", 0.0

    class_names = list(prototypes.keys())
    best_cls = "unknown"
    best_dist = np.inf
    distances = []
    for cls_name in class_names:
        proto = prototypes[cls_name]
        d = _distance(feat_vec, proto["mean"], proto["std"], metric)
        distances.append(d)
        if d < best_dist:
            best_dist = d
            best_cls = cls_name

    # Confidence: how much closer best is vs second-best
    distances_sorted = sorted(distances)
    if len(distances_sorted) > 1 and distances_sorted[1] > 0:
        confidence = float(1.0 - distances_sorted[0] / distances_sorted[1])
    else:
        confidence = 1.0
    return best_cls, confidence


def _upsample_labels(block_labels: np.ndarray, h: int, w: int, block_size: int) -> np.ndarray:
    """Expand block-level label grid to full image size."""
    label_map = np.full((h, w), -1, dtype=np.int32)
    n_rows, n_cols = block_labels.shape
    for r in range(n_rows):
        for c in range(n_cols):
            lbl = block_labels[r, c]
            r0, c0 = r * block_size, c * block_size
            r1, c1 = min(r0 + block_size, h), min(c0 + block_size, w)
            label_map[r0:r1, c0:c1] = lbl
    return label_map


# Default colors (BGR) for up to 10 classes
_PALETTE = [
    (0, 255, 255),    # yellow  — Banana
    (0, 0, 255),      # red     — Strawberry
    (0, 165, 255),    # orange  — Pineapple
    (0, 255, 0),      # green   — Pear
    (180, 105, 255),  # pink    — Kiwi
    (255, 0, 0),      # blue
    (255, 255, 0),    # cyan
    (128, 0, 128),    # purple
    (0, 128, 128),    # teal
    (128, 128, 0),    # olive
]


def _colorize(label_map: np.ndarray, class_names: list[str]) -> np.ndarray:
    """Convert integer label map to BGR color image."""
    h, w = label_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, name in enumerate(class_names):
        if cls_idx >= len(_PALETTE):
            break
        color[label_map == cls_idx] = _PALETTE[cls_idx]
    return color


def get_class_colors(class_names: list[str]) -> dict[str, tuple]:
    """Return {class_name: BGR_color} mapping for legend building."""
    return {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(class_names)}
