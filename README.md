## Datasets

The project expects two datasets placed manually:

```
data/
  fruits-360/
    Training/
    Test/
    test-multiple_fruits/
  fruits-262/
```

### Fruits-360

- GitHub: https://github.com/Horea94/Fruit-Images-Dataset
- Kaggle: https://www.kaggle.com/moltean/fruits
- Download and extract so that `data/fruits-360/Training/` exists with one subfolder per class.

### Fruits-262

- Kaggle: https://www.kaggle.com/datasets/aelchimminut/fruits262
- Used only for visual proof-of-concept — not for training or parameter tuning.

> If you don't have Kaggle credentials, download the zip files from the dataset pages on Kaggle and extract them manually.

---

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Quick Start

### 1. List available fruit classes

```bash
python scripts/list_fruits.py
# Filter by prefix
python scripts/list_fruits.py --filter ban
```

### 2. Train on 5 fruits

Edit `config.yaml` to set the `selected_fruits` list, then:

```bash
python scripts/train_5_fruits.py
```

This saves `outputs/models/prototypes_5_fruits.pkl`.

### 3. Evaluate on Fruits-360 Test

```bash
python scripts/evaluate_5_fruits.py
```

Outputs:

- `outputs/reports/metrics_5_fruits.json`
- `outputs/reports/evaluation_5_fruits.csv`
- `outputs/reports/confusion_matrix_5_fruits.png`
- `outputs/examples/correct/` and `outputs/examples/wrong/`

### 4. Predict a single image

```bash
python scripts/predict_image.py path/to/image.jpg
```

## Configuration

All main parameters live in `config.yaml`:

| Parameter              | Default    | Description                           |
| ---------------------- | ---------- | ------------------------------------- |
| `selected_fruits`      | 5 fruits   | Which classes to train/evaluate       |
| `pyramid_levels`       | 4          | Number of Laplacian Pyramid levels    |
| `block_sizes`          | [8,8,4,4]  | Block size per pyramid level          |
| `image_size`           | 100        | Resize input to N×N before processing |
| `background_threshold` | 230        | Gray ≥ this → background pixel        |
| `distance_metric`      | normalized | `euclidean` or `normalized`           |

---

## Algorithm

### 1. Grayscale conversion

`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` — library call, allowed by assignment.

### 2. Background removal

Simple threshold: pixels with gray value ≥ `background_threshold` are background. Works well on the white-background images in Fruits-360.

### 3. Laplacian Pyramid (own code)

We build a Gaussian pyramid by iteratively blurring and downsampling. The Laplacian pyramid is computed as the difference between consecutive Gaussian levels:

```
L[k] = G[k] − upsample(G[k+1])
L[last] = G[last]   (residual)
```

Each level captures a band of spatial frequencies:

- Level 0: fine details (edges, surface texture)
- Higher levels: coarser structure

### 4. Block-based feature extraction (own code)

For each pyramid level, the image is divided into non-overlapping blocks. Per block we compute:

- Mean absolute response
- Standard deviation
- Variance
- Min-max range
- Energy (mean squared value)
- Mean
- Kurtosis (4th standardised moment)

Features from all levels are concatenated into a single descriptor vector per image.

### 5. Prototype creation (training)

For each class, we average feature vectors over all training images → class prototype (mean + std for normalization).

### 6. Classification

At prediction time, each block's feature vector is compared to all class prototypes using normalized Euclidean distance. The nearest prototype wins. Background blocks (low foreground fraction) are excluded.

For image-level classification, majority vote over non-background blocks is used.

### 7. Segmentation output

Block labels are upsampled to pixel resolution. A color overlay is generated for visualization — colors are only used in the display, not in the classification pipeline.

---

## Limitations

- **No color information.** Fruits that look very different in color but similar in grayscale texture (e.g., a red apple vs a green apple) will likely be confused.
- **Block-based segmentation is not pixel-accurate.** Block boundaries are visible in the output mask.
- **Fruits-262 is much harder.** Real-world images have cluttered backgrounds, varying scale, lighting changes, and partial occlusion — performance will be lower than on Fruits-360.
- **Fixed prototype.** The method uses a single mean prototype per class and does not model intra-class variation beyond the std normalization.

---

## Scientific Honesty

This project does not aim for state-of-the-art segmentation. The goal is to understand and document the advantages and disadvantages of a specific classical processing chain (grayscale + Laplacian Pyramid + block statistics + nearest-prototype). We report results honestly and discuss where and why the method fails.

---

## Project Structure

```
fruit_laplacian_segmentation/
├── config.yaml
├── requirements.txt
├── src/
│   ├── config.py       — load configuration
│   ├── data_utils.py   — dataset loading, background removal
│   ├── pyramid.py      — Laplacian Pyramid (OWN IMPLEMENTATION)*3
│   ├── features.py     — block-based feature extraction (OWN IMPLEMENTATION)*5
│   ├── train.py        — prototype training*1
│   ├── predict.py      — block-level classification
│   ├── evaluate.py     — evaluation pipeline*2
│   ├── metrics.py      — accuracy, precision, recall, F1, confusion matrix
│   └── visualize.py    — segmentation overlays, plots
├── scripts/
│   ├── list_fruits.py
│   ├── train_5_fruits.py
│   ├── evaluate_5_fruits.py
│   ├── predict_image.py
│   └── run_demo.py
├── docs/
│   ├── algorithm_notes.md
│   └── experiment_log_template.md
├── docs/
│   ├── technical_explanation.md   — full technical reference for understanding/defense
│   ├── project_report_for_teacher.md  — formal academic report (Markdown)
│   ├── project_report_for_teacher.tex — LaTeX source for PDF submission
│   ├── references.bib             — BibTeX references
│   └── figures/                   — report figures (generated by prepare_report_assets.py)
└── outputs/            — generated results (not tracked in git)
```

---

## Generating the Teacher Report

After training and evaluation, run the following to prepare all report assets and compile the PDF:

```bash
# 1. Train (if not already done)
python scripts/train_5_fruits.py

# 2. Evaluate and generate figures
python scripts/evaluate_5_fruits.py



# 3. Compile the LaTeX report to PDF
cd docs
pdflatex project_report_for_teacher.tex
bibtex project_report_for_teacher
pdflatex project_report_for_teacher.tex
pdflatex project_report_for_teacher.tex
```

The compiled PDF will be at `docs/project_report_for_teacher.pdf`.

---

## References

1. Burt, P.J. and Adelson, E.H. (1983). _The Laplacian pyramid as a compact image code._ IEEE Transactions on Communications, 31(4), 532–540.

2. Mureşan, H. and Oltean, M. (2018). _Fruit recognition from images using deep learning._ Acta Universitatis Sapientiae, Informatica, 10(1), 26–42. (Fruits-360 dataset paper)

3. Fruits-262 dataset: https://www.kaggle.com/datasets/aelchimminut/fruits262

4. Sokolova, M. and Lapalme, G. (2009). _A systematic analysis of performance measures for classification tasks._ Information Processing & Management, 45(4), 427–437.
