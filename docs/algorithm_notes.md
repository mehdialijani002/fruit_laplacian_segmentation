# Algorithm Notes — Fruit Segmentation via Laplacian Pyramid Features

## Group 4 constraints
- Input: RGB image
- Conversion to grayscale: allowed via library (OpenCV `cvtColor`)
- Laplacian Pyramid: must be implemented as own code
- Feature extraction: block-based, grayscale only
- No color features (no RGB/HSV/YCrCb channels)
- No deep learning

## Pipeline overview

```
RGB image
    │
    ▼
cv2.cvtColor (BGR → gray)      [library allowed]
    │
    ▼
Background removal              [own code: gray ≥ threshold → background]
    │
    ▼
Laplacian Pyramid (L levels)    [OWN IMPLEMENTATION]
    ├── Level 0: finest detail (full resolution)
    ├── Level 1: medium detail (½ resolution)
    ├── ...
    └── Level L-1: residual (coarsest)
    │
    ▼
Block-based feature extraction  [OWN IMPLEMENTATION]
    Per level, per block:
    - mean absolute response
    - standard deviation
    - variance
    - min-max range
    - energy (mean squared)
    - mean
    - second moment
    │
    ▼
Per-class feature aggregation   [OWN IMPLEMENTATION]
    prototype_mean = mean over all training blocks
    prototype_std  = std over all training blocks
    │
    ▼
Nearest-prototype classification [OWN IMPLEMENTATION]
    distance = normalized Euclidean (feature - proto_mean) / proto_std
    │
    ▼
Block-level label map → upsample → pixel label map
    │
    ▼
Color overlay for visualization [colors used only here, not in features]
```

## Why Laplacian Pyramid?
The LP decomposes an image into frequency bands. Fruits differ in texture across scales:
- Fine details (edges, bumps, seeds) → captured in fine LP levels
- Coarse structure → captured in residual level
Block-based statistics summarize the local texture in each band.

## Limitations
- No color: fruits with similar textures but different colors (e.g., red vs green apple) may be confused.
- Block-based segmentation is not pixel-accurate.
- Fruits-262 images have real-world backgrounds, scale variation, and partial occlusion, making them much harder.
- Performance may vary significantly across fruit classes.
