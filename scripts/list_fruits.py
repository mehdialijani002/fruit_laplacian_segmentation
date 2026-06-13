"""List all available fruit classes in the Training folder."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data_utils import list_available_classes
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="List available fruit classes in the Training folder.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--filter", default=None, help="Filter by prefix (case-insensitive)")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    classes = list_available_classes(cfg["data"]["fruits360_train"])

    if args.filter:
        classes = [c for c in classes if c.lower().startswith(args.filter.lower())]

    print(f"\nFound {len(classes)} class(es):")
    for c in classes:
        print(f"  {c}")


if __name__ == "__main__":
    main()
