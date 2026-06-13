"""
Block-based feature extraction from Laplacian Pyramid levels.

For each pyramid level we divide the image into non-overlapping blocks
and compute texture/activity statistics in each block.

Features per block:
  - mean absolute response (activity)
  - standard deviation
  - variance
  - min-max range
  - energy (mean squared response)
  - mean (DC component)
  - second moment

OWN IMPLEMENTATION: all block iteration and statistics are computed with NumPy.
"""

from typing import NamedTuple

import numpy as np


class BlockFeatures(NamedTuple):
    """Feature vector names for documentation / debugging."""
    mean_abs: float
    std: float
    variance: float
    range_val: float
    energy: float
    mean: float
    kurtosis: float


FEATURE_NAMES = list(BlockFeatures._fields)
N_FEATURES_PER_LEVEL = len(FEATURE_NAMES)


def _block_stats(block: np.ndarray) -> np.ndarray:
    """Compute feature vector for a single block. Returns float32 array of length 7."""
    flat = block.ravel().astype(np.float32)
    mean_abs = float(np.mean(np.abs(flat)))
    std = float(np.std(flat))
    variance = float(np.var(flat))
    range_val = float(np.max(flat) - np.min(flat))
    energy = float(np.mean(flat ** 2))
    mean = float(np.mean(flat))
    # Kurtosis: 4th standardised moment — measures peakedness of the distribution.
    # High kurtosis → sharp edges/impulses in the Laplacian band (busy texture).
    # Low kurtosis → flat, uniform region.
    denom = std ** 2 if std > 1e-6 else 1.0
    kurtosis = float(np.mean((flat - mean) ** 4) / (denom ** 2 + 1e-12))
    return np.array([mean_abs, std, variance, range_val, energy, mean, kurtosis], dtype=np.float32)


def extract_level_features(
    lap_level: np.ndarray,
    block_size: int,
    foreground_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract block features from one Laplacian pyramid level.

    Parameters
    ----------
    lap_level : 2D float array
    block_size : block side length in pixels (for this level)
    foreground_mask : optional binary mask (1=foreground, 0=background) at the same resolution

    Returns
    -------
    feature_grid : shape (n_row_blocks, n_col_blocks, N_FEATURES_PER_LEVEL)
    fg_grid      : shape (n_row_blocks, n_col_blocks) — fraction of foreground pixels per block
    """
    h, w = lap_level.shape[:2]
    n_rows = max(1, h // block_size)
    n_cols = max(1, w // block_size)

    feature_grid = np.zeros((n_rows, n_cols, N_FEATURES_PER_LEVEL), dtype=np.float32)
    fg_grid = np.ones((n_rows, n_cols), dtype=np.float32)  # default: all foreground

    for r in range(n_rows):
        for c in range(n_cols):
            r0, c0 = r * block_size, c * block_size
            r1, c1 = min(r0 + block_size, h), min(c0 + block_size, w)
            block = lap_level[r0:r1, c0:c1]
            feature_grid[r, c] = _block_stats(block)

            if foreground_mask is not None:
                mask_block = foreground_mask[r0:r1, c0:c1]
                fg_grid[r, c] = float(np.mean(mask_block))

    return feature_grid, fg_grid


def extract_pyramid_features(
    laplacian_pyramid: list[np.ndarray],
    block_sizes: list[int],
    foreground_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract and concatenate features across all pyramid levels.

    Returns
    -------
    feature_vector : 1D float32 array — concatenated mean features over all foreground blocks
    fg_fraction    : overall fraction of foreground blocks
    """
    level_vectors = []
    fg_fractions = []

    for level_idx, (lap_level, bsize) in enumerate(zip(laplacian_pyramid, block_sizes)):
        # Resize foreground mask to match this level's resolution
        fg_mask_at_level = None
        if foreground_mask is not None:
            h, w = lap_level.shape[:2]
            fg_mask_at_level = (
                cv2_resize_mask(foreground_mask, h, w)
            )

        feat_grid, fg_grid = extract_level_features(lap_level, bsize, fg_mask_at_level)

        # Aggregate: mean over foreground blocks (or all blocks if no mask)
        fg_mask_flat = fg_grid.ravel() > 0.5
        feats_flat = feat_grid.reshape(-1, N_FEATURES_PER_LEVEL)

        if fg_mask_flat.any():
            level_vec = feats_flat[fg_mask_flat].mean(axis=0)
            fg_fractions.append(float(fg_mask_flat.mean()))
        else:
            level_vec = feats_flat.mean(axis=0)
            fg_fractions.append(0.0)

        level_vectors.append(level_vec)

    feature_vector = np.concatenate(level_vectors, axis=0)
    fg_fraction = float(np.mean(fg_fractions)) if fg_fractions else 0.0
    return feature_vector.astype(np.float32), fg_fraction


def cv2_resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a binary mask to (target_h, target_w) using nearest-neighbour."""
    import cv2
    resized = cv2.resize(mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0.5).astype(np.uint8)
