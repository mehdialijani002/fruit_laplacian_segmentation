import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate fruit classification on Fruits-360 Test set.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None, help="Path to .pkl model file")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    metrics = evaluate(args.config, args.model)
    print(f"\nAccuracy: {metrics['accuracy']:.1%}")
    print("Per-class:")
    for cls, vals in metrics["per_class"].items():
        print(f"  {cls:15s}  P={vals['precision']:.2f}  R={vals['recall']:.2f}  F1={vals['f1']:.2f}")


if __name__ == "__main__":
    main()
