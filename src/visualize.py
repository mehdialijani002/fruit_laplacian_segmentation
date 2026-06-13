import logging
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .predict import predict_image, get_class_colors, _colorize

logger = logging.getLogger(__name__)


def save_segmentation(image_path: str, prototypes: dict, cfg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    label_map, color_mask, orig_bgr = predict_image(image_path, prototypes, cfg)

    cv2.imwrite(str(out_dir / f"{stem}_original.jpg"), orig_bgr)
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), color_mask)

    overlay = cv2.addWeighted(orig_bgr, 0.6, color_mask, 0.4, 0)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"), overlay)

    # Save legend
    _save_legend(prototypes, out_dir / f"{stem}_legend.png")
    logger.info("Saved segmentation results for %s to %s", stem, out_dir)


def save_example(
    image_path: str,
    true_label: str,
    pred_label: str,
    confidence: float,
    out_dir: Path,
    cfg: dict,
    prototypes: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    label_map, color_mask, orig_bgr = predict_image(image_path, prototypes, cfg)
    overlay = cv2.addWeighted(orig_bgr, 0.6, color_mask, 0.4, 0)

    title = f"True: {true_label} | Pred: {pred_label} ({confidence:.0%})"
    cv2.putText(overlay, title, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    fname = f"{stem}_true-{true_label}_pred-{pred_label}.jpg"
    cv2.imwrite(str(out_dir / fname), overlay)


def save_confusion_matrix_plot(cm: list, class_names: list[str], out_path: Path) -> None:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(max(5, len(class_names)), max(4, len(class_names))))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", out_path)


def _save_legend(prototypes: dict, out_path: Path) -> None:
    colors = get_class_colors(list(prototypes.keys()))
    fig, ax = plt.subplots(figsize=(3, 0.4 * len(prototypes) + 0.5))
    for i, (name, bgr) in enumerate(colors.items()):
        rgb = (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=rgb))
        ax.text(1.2, i + 0.5, name, va="center", fontsize=10)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(colors))
    ax.axis("off")
    ax.set_title("Class Colors")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=100)
    plt.close(fig)
