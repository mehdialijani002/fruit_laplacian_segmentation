"""
Demo: run prediction on multiple Fruits-262 or custom images.
Saves original, mask, and overlay for each image.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.train import load_prototypes
from src.visualize import save_segmentation
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Demo: run prediction on a folder of images.")
    parser.add_argument("image_dir", help="Folder containing images")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max", type=int, default=20, help="Max images to process")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    model_path = args.model or str(Path(cfg["outputs"]["models"]) / cfg["model_filename"])
    prototypes, _ = load_prototypes(model_path)

    out_dir = Path(args.out_dir or cfg["outputs"]["segmentations"]) / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in sorted(Path(args.image_dir).rglob("*")) if p.suffix.lower() in extensions]
    images = images[: args.max]

    print(f"Processing {len(images)} image(s)...")
    for img_path in images:
        try:
            save_segmentation(str(img_path), prototypes, cfg, out_dir)
            print(f"  OK: {img_path.name}")
        except Exception as e:
            print(f"  FAIL: {img_path.name} — {e}")

    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
