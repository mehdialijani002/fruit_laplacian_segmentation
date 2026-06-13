import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_metrics(y_true: list[str], y_pred: list[str], class_names: list[str]) -> dict:
    if not y_true:
        return {"accuracy": 0.0, "confusion_matrix": [], "per_class": {}}

    n = len(class_names)
    idx = {name: i for i, name in enumerate(class_names)}

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        ti = idx.get(t, -1)
        pi = idx.get(p, -1)
        if ti >= 0 and pi >= 0:
            cm[ti, pi] += 1

    accuracy = float(np.trace(cm)) / max(len(y_true), 1)

    per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class[name] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    return {
        "accuracy": round(accuracy, 4),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "per_class": per_class,
        "n_samples": len(y_true),
    }


def save_metrics(metrics: dict, out_dir: Path, tag: str = "5_fruits") -> None:
    """Save metrics JSON and CSV."""
    import csv

    json_path = out_dir / f"metrics_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", json_path)

    csv_path = out_dir / f"evaluation_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1"])
        for name, vals in metrics["per_class"].items():
            writer.writerow([name, vals["precision"], vals["recall"], vals["f1"]])
        writer.writerow(["OVERALL_ACCURACY", metrics["accuracy"], "", ""])
    logger.info("CSV saved to %s", csv_path)
