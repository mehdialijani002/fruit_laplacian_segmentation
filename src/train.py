import logging
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import load_config
from .data_utils import (
    find_class_folder,
    load_image_gray,
    make_foreground_mask,
    list_available_classes,
)
from .pyramid import build_laplacian_pyramid
from .features import extract_pyramid_features

logger = logging.getLogger(__name__)


def train(config_path: str | None = None) -> dict:
  
    cfg = load_config(config_path)
    selected = cfg["selected_fruits"]
    train_dir = cfg["data"]["fruits360_train"]
    image_size = cfg["image_size"]
    bg_thresh = cfg["background_threshold"]
    levels = cfg["pyramid_levels"]
    block_sizes = cfg["block_sizes"]
    out_dir = Path(cfg["outputs"]["models"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / cfg["model_filename"]

    available = list_available_classes(train_dir)
    logger.info("Available classes (%d): %s", len(available), available[:10])

    prototypes: dict[str, dict] = {}

    for fruit in selected:
        folder = find_class_folder(train_dir, fruit)
        if folder is None:
            logger.warning("Folder not found for '%s' in %s — skipping.", fruit, train_dir)
            continue
        logger.info("Training '%s' from %s ...", fruit, folder.name)

        all_features: list[np.ndarray] = []
        images = sorted(folder.iterdir())
        images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

        for img_path in tqdm(images, desc=fruit, leave=False):
            gray = load_image_gray(img_path, image_size)
            if gray is None:
                continue
            fg_mask = make_foreground_mask(gray, bg_thresh)
            lap_pyr = build_laplacian_pyramid(gray, levels)
            feat_vec, fg_frac = extract_pyramid_features(lap_pyr, block_sizes[:levels], fg_mask)
            if fg_frac < 0.01:
                continue  # skip nearly-all-background images
            all_features.append(feat_vec)

        if not all_features:
            logger.warning("No valid features extracted for '%s'.", fruit)
            continue

        feat_matrix = np.stack(all_features, axis=0)
        proto_mean = feat_matrix.mean(axis=0)
        proto_std = feat_matrix.std(axis=0) + 1e-6  # avoid division by zero

        prototypes[fruit] = {
            "mean": proto_mean,
            "std": proto_std,
            "folder_name": folder.name,
            "n_images": len(all_features),
        }
        logger.info(
            "  Prototype for '%s': %d images, feature dim=%d",
            fruit, len(all_features), proto_mean.shape[0],
        )

    with open(model_path, "wb") as f:
        pickle.dump({"prototypes": prototypes, "config": cfg}, f)
    logger.info("Prototypes saved to %s", model_path)
    return prototypes


def load_prototypes(model_path: str) -> tuple[dict, dict]:
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["prototypes"], data["config"]
