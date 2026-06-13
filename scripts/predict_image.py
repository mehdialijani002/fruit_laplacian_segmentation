"""Run prediction on a single image and save segmentation output."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.train import load_prototypes
from src.visualize import save_segmentation
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Predict fruit class for a single image.")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-dir", default=None, help="Output directory (default: outputs/segmentations)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    model_path = args.model or str(Path(cfg["outputs"]["models"]) / cfg["model_filename"])
    prototypes, train_cfg = load_prototypes(model_path)

    out_dir = Path(args.out_dir or cfg["outputs"]["segmentations"])
    save_segmentation(args.image, prototypes, cfg, out_dir)

    from src.predict import predict_class
    pred, conf = predict_class(args.image, prototypes, cfg)
    print(f"\nPredicted class: {pred}  (confidence: {conf:.1%})")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
