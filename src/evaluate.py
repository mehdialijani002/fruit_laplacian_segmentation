import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import load_config
from .data_utils import find_class_folder, list_available_classes
from .predict import predict_class
from .train import load_prototypes
from .metrics import compute_metrics, save_metrics
from .visualize import save_example, save_confusion_matrix_plot

logger = logging.getLogger(__name__)


def evaluate(config_path: str | None = None, model_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    prototypes, train_cfg = load_prototypes(
        model_path or str(Path(cfg["outputs"]["models"]) / cfg["model_filename"])
    )
    test_dir = cfg["data"]["fruits360_test"]
    reports_dir = Path(cfg["outputs"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    examples_correct = Path(cfg["outputs"]["examples"]) / "correct"
    examples_wrong = Path(cfg["outputs"]["examples"]) / "wrong"
    examples_correct.mkdir(parents=True, exist_ok=True)
    examples_wrong.mkdir(parents=True, exist_ok=True)

    class_names = list(prototypes.keys())
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # Clear example folders so each run produces a clean set (Windows-safe)
    import os
    for d in [examples_correct, examples_wrong]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            try:
                os.remove(f)
            except OSError:
                pass

    y_true, y_pred = [], []
    # Per-class counters: save up to MAX_PER_CLASS correct and wrong per class
    MAX_PER_CLASS = 2
    correct_counts: dict[str, int] = {c: 0 for c in class_names}
    wrong_counts: dict[str, int] = {c: 0 for c in class_names}

    for true_fruit in class_names:
        folder = find_class_folder(test_dir, true_fruit)
        if folder is None:
            logger.warning("Test folder not found for '%s'", true_fruit)
            continue
        images = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in extensions]
        logger.info("Evaluating '%s' (%d images)...", true_fruit, len(images))

        for img_path in tqdm(images, desc=true_fruit, leave=False):
            try:
                pred_fruit, conf = predict_class(str(img_path), prototypes, cfg)
            except Exception as e:
                logger.debug("Error on %s: %s", img_path, e)
                continue

            y_true.append(true_fruit)
            y_pred.append(pred_fruit)

            # Save up to MAX_PER_CLASS examples per class for both correct and wrong
            correct = pred_fruit == true_fruit
            if correct and correct_counts[true_fruit] < MAX_PER_CLASS:
                save_example(str(img_path), true_fruit, pred_fruit, conf,
                             examples_correct, cfg, prototypes)
                correct_counts[true_fruit] += 1
            elif not correct and wrong_counts[true_fruit] < MAX_PER_CLASS:
                save_example(str(img_path), true_fruit, pred_fruit, conf,
                             examples_wrong, cfg, prototypes)
                wrong_counts[true_fruit] += 1

    metrics = compute_metrics(y_true, y_pred, class_names)
    save_metrics(metrics, reports_dir, tag="5_fruits")
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names,
        reports_dir / "confusion_matrix_5_fruits.png",
    )
    logger.info("Accuracy: %.3f", metrics["accuracy"])
    return metrics
