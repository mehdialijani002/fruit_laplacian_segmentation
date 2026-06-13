"""
prepare_report_assets.py
========================
Copies output assets into docs/figures/ for LaTeX / Overleaf compilation.

Copies per-class correct examples (correct_example_banana.jpg, etc.),
one wrong example, confusion matrix, segmentation overlay, and Fruits-262 demo.

Usage
-----
    python scripts/prepare_report_assets.py
    python scripts/prepare_report_assets.py --project-root /path/to/project
"""

import argparse
import shutil
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def find_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def find_first_image(directory: Path) -> Path | None:
    imgs = find_images(directory)
    return imgs[0] if imgs else None


def find_image_for_class(directory: Path, class_name: str) -> Path | None:
    """Find first image in directory whose filename contains class_name."""
    safe = class_name.lower().replace(" ", "").replace("_", "")
    for p in find_images(directory):
        fname = p.name.lower().replace(" ", "").replace("_", "")
        if safe in fname:
            return p
    return None


def copy_asset(src: Path | None, dst: Path, label: str, required: bool = True) -> bool:
    tag = "[REQUIRED]" if required else "[optional]"
    if src is None or not src.exists():
        print(f"  [MISSING] {tag} {label}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  [OK]      {label}")
    print(f"            {src.name}  ->  docs/figures/{dst.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy report assets into docs/figures/ for LaTeX compilation."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args()

    if args.project_root is not None:
        root = args.project_root.resolve()
    else:
        root = Path(__file__).resolve().parent.parent

    if not root.is_dir():
        print(f"ERROR: Project root not found: {root}", file=sys.stderr)
        return 1

    print(f"Project root: {root}\n")

    reports_dir  = root / "outputs" / "reports"
    correct_dir  = root / "outputs" / "examples" / "correct"
    wrong_dir    = root / "outputs" / "examples" / "wrong"
    seg_dir      = root / "outputs" / "segmentations"
    demo_dir     = root / "outputs" / "segmentations" / "demo"
    figures_dir  = root / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    print("Copying assets:")
    print("-" * 60)

    # Confusion matrix
    results["confusion_matrix"] = copy_asset(
        reports_dir / "confusion_matrix_5_fruits.png",
        figures_dir / "confusion_matrix_5_fruits.png",
        "Confusion matrix", required=True
    )

    # Per-class correct examples — one per class
    class_labels = ["Banana 1", "Kiwi 1", "Pear 1", "Mango 1", "Orange 1"]
    correct_found = {}
    for cls in class_labels:
        safe_name = cls.lower().replace(" ", "_")
        src = find_image_for_class(correct_dir, cls.replace(" ", "").replace("1", "1"))
        # fallback: search by class token
        if src is None:
            token = cls.split()[0].lower()  # "banana", "kiwi", etc.
            for p in find_images(correct_dir):
                if token in p.name.lower():
                    src = p
                    break
        dst = figures_dir / f"correct_example_{safe_name.replace(' ','_')}.jpg"
        ok = copy_asset(src, dst, f"Correct example — {cls}", required=False)
        results[f"correct_{safe_name}"] = ok
        if ok:
            correct_found[cls] = dst

    # Ensure at least one correct example exists for LaTeX \includegraphics
    if correct_found:
        first_correct = list(correct_found.values())[0]
        generic_dst = figures_dir / "correct_example.jpg"
        if not generic_dst.exists():
            shutil.copy2(first_correct, generic_dst)
            print(f"  [OK]      Generic correct_example.jpg -> {first_correct.name}")
        results["correct_example"] = True
    else:
        results["correct_example"] = copy_asset(
            find_first_image(correct_dir),
            figures_dir / "correct_example.jpg",
            "Correct example (generic fallback)", required=True
        )

    # Wrong example
    results["wrong_example"] = copy_asset(
        find_first_image(wrong_dir),
        figures_dir / "wrong_example.jpg",
        "Wrong classification example", required=True
    )

    # Segmentation overlay — prefer _overlay file
    seg_src = None
    for p in find_images(seg_dir):
        if "overlay" in p.name.lower():
            seg_src = p
            break
    if seg_src is None:
        seg_src = find_first_image(seg_dir)
    results["segmentation_overlay"] = copy_asset(
        seg_src, figures_dir / "segmentation_overlay.png",
        "Segmentation overlay", required=False
    )

    # Fruits-262 demo
    demo_src = None
    for p in find_images(demo_dir):
        if "overlay" in p.name.lower():
            demo_src = p
            break
    if demo_src is None:
        demo_src = find_first_image(demo_dir)
    results["fruits262_demo"] = copy_asset(
        demo_src, figures_dir / "fruits262_demo.png",
        "Fruits-262 demo overlay", required=False
    )

    # Summary
    required_keys = {"confusion_matrix", "correct_example", "wrong_example"}
    n_ok = sum(results.values())
    n_total = len(results)
    print()
    print("=" * 60)
    print(f"  Copied: {n_ok}/{n_total}  |  Missing: {n_total - n_ok}/{n_total}")
    print(f"  Output: {figures_dir}")

    missing_required = [k for k in required_keys if not results.get(k)]
    if missing_required:
        print("\n  WARNING — required assets missing:")
        for k in missing_required:
            print(f"    {k}")
        print("\n  Run these commands first:")
        print("    python scripts/evaluate_5_fruits.py")
        print("    python scripts/predict_image.py data/fruits-360/Test/Banana\\ 1/100_100.jpg")
        return 2

    missing_optional = [k for k in results if k not in required_keys and not results[k]]
    if missing_optional:
        print("\n  Optional assets missing (not blocking compilation):")
        for k in missing_optional:
            print(f"    {k}")
        print("  Run demo to generate them:")
        print("    python scripts/run_demo.py data/fruits-262/banana --max 5")

    print("\n  All required assets present. LaTeX report ready to compile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
