"""Train prototypes for the 5 selected fruit classes."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import train
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train Laplacian Pyramid prototypes for fruit classification.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    prototypes = train(args.config)
    print(f"\nTrained {len(prototypes)} class prototype(s): {list(prototypes.keys())}")


if __name__ == "__main__":
    main()
