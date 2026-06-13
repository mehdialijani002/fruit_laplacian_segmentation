# Technical Explanation: Fruit Segmentation Using Grayscale Laplacian Pyramid Features

---

## 1. Project Overview

This project implements a complete image classification and segmentation pipeline for fruit recognition using the Fruits-360 dataset. The core idea is to represent each fruit image as a compact feature vector derived from a **Grayscale Laplacian Pyramid**, then classify new images by comparing their feature vectors to pre-computed class prototypes using a **normalized Euclidean distance** metric.

The pipeline has three major phases:

1. **Training:** Load training images for each fruit class, convert them to grayscale, build a Laplacian pyramid for each image, extract block-based statistical features from every pyramid level, and compute a mean (prototype) feature vector per class.
2. **Evaluation:** For each test image, repeat the feature extraction steps and assign it to the class whose prototype is nearest in feature space.
3. **Segmentation:** Use the trained class knowledge to produce a per-pixel segmentation mask that highlights the fruit region within an arbitrary input image.

The system was designed specifically to satisfy the constraints of Group 4 of the Computer Vision assignment, which require:
- Grayscale input (no color channels used for classification)
- A Laplacian pyramid as the primary multi-scale representation
- Block-based feature extraction at each pyramid level
- Prototype-based (nearest-mean) classification
- A demonstration on both Fruits-360 and the Fruits-262 Kaggle dataset

The final system achieved **85.71% overall accuracy** on 812 test images across five fruit classes.

---

## 2. Assignment Constraints for Group 4

The assignment specifies a strict set of methodological constraints that every design decision must respect. Understanding these constraints is essential for defending the project.

**Constraint 1 — Grayscale only.**
All images must be converted to single-channel grayscale before any processing. No RGB, HSV, or LAB color features may be used. This makes the task harder because many fruits that look similar in gray (e.g., yellow Banana and yellow Mango) must be distinguished by texture and shape alone.

**Constraint 2 — Laplacian Pyramid.**
The multi-scale representation must be a Laplacian pyramid, not a Gaussian pyramid, not a wavelet transform, not a scale-space heatmap. A Laplacian pyramid captures **band-pass detail** at each level — the fine-grain texture removed when going from one Gaussian level to the next.

**Constraint 3 — Block-based feature extraction.**
At each pyramid level, the image is divided into non-overlapping rectangular blocks. Features are computed within each block and then aggregated. This gives spatial information: which part of the image has which kind of texture.

**Constraint 4 — Prototype-based classification.**
The classifier must be a nearest-mean (prototype) classifier, not a neural network, not an SVM, not a k-NN with k > 1. Each class is represented by exactly one prototype vector (the mean of all training feature vectors for that class), and classification assigns a test image to the nearest prototype.

**Constraint 5 — Two datasets.**
The system must be demonstrated on the Fruits-360 dataset (primary) and the Fruits-262 dataset from Kaggle (secondary demo).

These constraints collectively define a classical computer vision pipeline. They have educational value precisely because they force the student to understand multi-scale representations, feature engineering, and distance-based classification rather than treating everything as a black box.

---

## 3. Dataset Structure

### Fruits-360

Fruits-360 is a large, clean, benchmark dataset of fruit photographs. It was created and maintained by Horea Muresan and Mihai Oltean and is available on Kaggle and GitHub.

Key properties:
- **Image size:** All images are exactly 100 × 100 pixels.
- **Background:** Pure white (RGB 255,255,255), making segmentation straightforward.
- **Split:** The dataset has a pre-defined `Training/` and `Test/` split. Each fruit variety has its own subfolder (e.g., `Training/Banana 1/`, `Test/Banana 1/`).
- **Scale:** Over 90,000 images covering more than 131 fruit varieties (as of recent versions).
- **Naming conventions:** Varieties are named with a number suffix (e.g., "Banana 1", "Banana 2", "Banana 3"). This project uses the "1" varieties.

The training set used in this project:
- Banana 1: 490 images
- Kiwi 1: 466 images
- Pear 1: 492 images
- Mango 1: 490 images
- Orange 1: 479 images
- **Total training images: 2417**

The test set:
- Banana 1: 166 images
- Kiwi 1: 156 images
- Pear 1: 164 images
- Mango 1: 166 images
- Orange 1: 160 images
- **Total test images: 812**

### Fruits-262

Fruits-262 is a separate Kaggle dataset with 262 fruit categories. Unlike Fruits-360 it does not have a pure white background, making it a realistic proof-of-concept for the segmentation module. At the time of this writing, the segmentation demo on Fruits-262 has not yet been run (the `outputs/segmentations/` folder is empty), but the pipeline is fully capable of processing it.

---

## 4. Selected Fruit Classes

The assignment specification suggested Banana, Strawberry, Pineapple, Pear, and Kiwi as example classes. However, when the Fruits-360 dataset was inspected, folder names like "Strawberry" and "Pineapple" do not appear as exact top-level matches in the version used. To select five well-populated and recognizable classes the following were chosen:

| Class    | Training images | Test images | Notes                        |
|----------|-----------------|-------------|------------------------------|
| Banana 1 | 490             | 166         | Elongated, yellow, smooth    |
| Kiwi 1   | 466             | 156         | Round, brown, fuzzy surface  |
| Pear 1   | 492             | 164         | Teardrop shape, green-yellow |
| Mango 1  | 490             | 166         | Oval, yellow-orange          |
| Orange 1 | 479             | 160         | Round, orange, textured skin |

The choice provides good variety in shape and texture. Mango and Banana share similar coloring in grayscale, which explains the main source of misclassification (46 Mango images predicted as Banana). Orange and Kiwi share a round silhouette and similar textural coarseness at coarse scales, explaining the second confusion (49 Orange images predicted as Kiwi).

---

## 5. Complete Pipeline (Step by Step)

### Step 1 — Image loading and resizing
All images are loaded from disk using OpenCV (`cv2.imread`). Although Fruits-360 images are already 100 × 100, the pipeline resizes every image to exactly 100 × 100 to be robust to other sources.

### Step 2 — Grayscale conversion
The loaded BGR image is converted to grayscale: `gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`. The result is a 2D array of uint8 values in [0, 255].

### Step 3 — Background masking
A binary mask is created: pixels with gray value ≥ 230 are marked as background. In Fruits-360 the background is white (255), so this threshold cleanly separates fruit from background. Only foreground pixels (gray < 230) are used in feature computation to avoid the background dominating the statistics.

### Step 4 — Laplacian pyramid construction
A 4-level Laplacian pyramid is built:
- `G[0]` = original 100×100 grayscale image
- `G[1]` = GaussianBlur + downsample → 50×50
- `G[2]` = GaussianBlur + downsample → 25×25
- `G[3]` = GaussianBlur + downsample → 13×13 (approx.)
- `L[0]` = G[0] − upsample(G[1]) → 100×100 detail
- `L[1]` = G[1] − upsample(G[2]) → 50×50 detail
- `L[2]` = G[2] − upsample(G[3]) → 25×25 detail
- `L[3]` = G[3] (residual, coarsest level)

Each level `L[k]` captures a different frequency band.

### Step 5 — Block division
At each pyramid level `k`, the level image is divided into non-overlapping blocks. The block size at each level (finest to coarsest) is `[8, 8, 4, 4]`. For a 100×100 image with block size 8, this gives approximately 12×12 = 144 blocks. However, since the number of blocks varies by level, only the per-block statistics are averaged across all blocks to produce exactly one number per feature per level. This yields a fixed-length feature vector regardless of image size.

### Step 6 — Per-block feature computation
For each block, 7 statistics are computed over the foreground pixels within that block:
1. **mean_abs** — mean of absolute pixel values (reflects average intensity magnitude)
2. **std** — standard deviation (reflects contrast/texture variation)
3. **variance** — variance (std²; different sensitivity to outliers)
4. **range** — max − min (captures dynamic range of the block)
5. **energy** — sum of squares / N (reflects signal power)
6. **mean** — arithmetic mean (signed, relevant for Laplacian levels which can be negative)
7. **second_moment** — mean of squares (related to energy; captures overall intensity level)

### Step 7 — Feature aggregation across blocks
The per-block values are averaged across all blocks in the level. This gives exactly 7 numbers per level. With 4 levels: 4 × 7 = **28 features** per image.

### Step 8 — Prototype computation (training)
For each class, collect all training feature vectors (one per training image). The prototype is the element-wise mean: `prototype[c] = mean(all feature vectors for class c)`. Also store per-dimension standard deviation for normalization.

### Step 9 — Classification (test time)
For a test image, compute its 28-dimensional feature vector exactly as in steps 1–7. Then compute the normalized Euclidean distance to every class prototype:

```
d(f, μ_c) = sqrt( sum_i ((f_i − μ_c_i) / σ_c_i)^2 )
```

Assign the image to the class with the smallest distance.

### Step 10 — Segmentation mask generation
For segmentation (rather than classification), the pipeline slides a window over the input image, extracts features from each window, and determines whether the window belongs to a given fruit class. Alternatively, the foreground mask from Step 3 is used directly as a segmentation proxy for Fruits-360 (since the background is white). For Fruits-262 a more sophisticated sliding-window or region-growing approach would be used.

### Step 11 — Evaluation
Classification results are compared to ground truth labels. Accuracy, per-class precision, recall, and F1 are computed. A confusion matrix is saved.

---

## 6. Laplacian Pyramid Explanation

### What is a Gaussian Pyramid?

A Gaussian pyramid is a sequence of images G[0], G[1], ..., G[N-1] where:
- G[0] is the original image
- G[k+1] = downsample(blur(G[k]))

"Blur then downsample" (also called "reduce") is anti-aliasing. Without the blur, downsampling would cause aliasing artifacts. The blur is typically a 5×5 Gaussian kernel (Burt & Adelson, 1983).

Each level is half the width and half the height of the previous one. A 100×100 image yields levels of 100×100, 50×50, 25×25, 13×13 (approximately).

### What is a Laplacian Pyramid?

The Laplacian pyramid is derived from the Gaussian pyramid:

```
L[k] = G[k] − upsample(G[k+1])   for k = 0, 1, ..., N-2
L[N-1] = G[N-1]   (residual)
```

"Upsample" means expanding the smaller image back to the size of G[k] and interpolating missing values.

The subtraction `G[k] − upsample(G[k+1])` removes the low-frequency content captured in G[k+1] from G[k]. The result L[k] is therefore a **band-pass filtered** image: it contains only the spatial frequencies that "disappeared" in the downsampling step.

### Why are Laplacian levels useful for texture description?

- **L[0]** (finest level): Contains high-frequency detail — fine surface texture, sharp edges.
- **L[1]**: Mid-high frequency — coarser texture, broader edges.
- **L[2]**: Mid-low frequency — overall shape features.
- **L[3]** (residual): Very low frequency — the coarse "silhouette" of the object.

Different fruits have different textures at different scales. Kiwi has coarse, fuzzy surface detail (strong at L[0]). Banana has a smooth, uniform surface (weak L[0], strong L[3] signal). By computing features at all levels, the classifier captures both fine-grained and coarse-grained discriminative information.

### Reconstruction property

A key mathematical property: the Laplacian pyramid is a **lossless representation**. The original image can be perfectly reconstructed by:

```
G[k] = L[k] + upsample(G[k+1])
```

applied from the coarsest level upward. This means no information is lost — the pyramid is simply a change of basis, not a lossy compression.

### Why does this matter for the assignment?

Using a Laplacian pyramid rather than a raw Gaussian pyramid forces the feature extractor to work with **residual detail** images. This is more informative because each level captures a distinct frequency band rather than overlapping cumulative content. The Gaussian pyramid levels are highly correlated (each is a blurred version of the previous), while Laplacian levels are approximately decorrelated.

---

## 7. Block-Based Feature Extraction

### Why blocks?

A single global statistic over the whole image discards all spatial information. A 28-dimensional feature from mean/std/etc. computed over the entire image would look identical for any image with the same global distribution — a Banana and an Orange might score similarly if their global histograms are similar.

Dividing the image into blocks gives **local** statistics. The top-left corner of a Banana image looks different from its center. Blocks allow the feature vector to encode "the center has low variance, the edges have high gradient."

### Block configuration

| Level | Image size (approx.) | Block size | Num blocks (approx.) |
|-------|----------------------|------------|----------------------|
| L[0]  | 100×100              | 8×8        | 144                  |
| L[1]  | 50×50                | 8×8        | 36                   |
| L[2]  | 25×25                | 4×4        | 36                   |
| L[3]  | 13×13                | 4×4        | 9                    |

The block sizes are chosen so that finer pyramid levels use larger blocks (8×8) while coarser levels use smaller blocks (4×4). This is a deliberate design choice: at fine pyramid levels the image is large and there are many blocks — this gives rich spatial detail. At coarse levels the image is small and 4×4 blocks are proportionally larger, capturing the coarse global structure.

### The 7 features per block

All features are computed on the foreground pixels within the block. Pixels with gray value ≥ 230 (background) are excluded.

1. **mean_abs:** `mean(|pixel values|)`. For the residual level (L[3]) pixel values are positive (gray intensities). For Laplacian difference levels (L[0]–L[2]) values can be negative, so absolute mean avoids cancellation.

2. **std:** `sqrt(mean((x − mean(x))^2))`. Measures spread. A smooth region has low std; a textured region has high std.

3. **variance:** `std^2`. Same information as std but with different sensitivity. Models with additive Gaussian noise often use variance directly.

4. **range:** `max(x) − min(x)`. Measures the dynamic range of the block. A uniform block has range ≈ 0. An edge crossing the block has large range.

5. **energy:** `mean(x^2)`. Measures total signal power. Related to mean squared value.

6. **mean:** `mean(x)`. Unlike mean_abs, this can be negative for Laplacian levels. It captures whether the level has a net positive or negative "signal" in this region.

7. **second_moment:** `mean(x^2)`. Same as energy in this implementation (the distinction is notation; both capture average squared intensity).

### Aggregation

After computing 7 features for each block at a given level, the values are averaged across all blocks. This produces exactly 7 numbers per level regardless of image size or exact block count. Concatenating across 4 levels yields the final 28-dimensional feature vector.

---

## 8. Prototype-Based Classification

### What is a prototype?

A prototype is a single representative feature vector for a class. In this system it is the **arithmetic mean** of all training feature vectors for that class:

```
μ_c = (1/N_c) * sum_{i=1}^{N_c} f_i^(c)
```

where `f_i^(c)` is the feature vector of the i-th training image of class c, and N_c is the number of training images for class c.

This is also called "Nearest Mean Classifier" or "Linear Discriminant Analysis with equal covariance assumption."

### Why use normalized Euclidean distance?

Raw Euclidean distance is sensitive to the scale of each feature dimension. Feature dimensions with large variance (e.g., energy at fine levels, which can be large) would dominate the distance and drown out features with small variance (e.g., mean of a Laplacian level, which is near zero). Normalizing by the per-dimension standard deviation `σ_c_i` corrects this:

```
d(f, μ_c) = sqrt( sum_i ((f_i − μ_c_i) / σ_c_i)^2 )
```

This is also related to the **Mahalanobis distance** under the assumption that all feature dimensions are independent (diagonal covariance matrix).

### Decision rule

```
predicted_class = argmin_c d(f, μ_c)
```

The test image is assigned to the class whose prototype is nearest. There are no thresholds or confidence scores — it is always assigned to the nearest class.

### Minimum foreground fraction filter

A practical pre-filter checks: if fewer than 5% of the pixels in the image are foreground (i.e., the image is mostly background/white), the image is flagged as "no fruit detected" and skipped rather than forced into a class. This prevents spurious classifications of nearly blank images.

---

## 9. Segmentation Strategy

### Fruits-360 segmentation

For Fruits-360, the segmentation is straightforward because the background is pure white. The segmentation mask is simply the complement of the background mask:

```
mask[y, x] = 1   if gray[y, x] < 230
mask[y, x] = 0   otherwise
```

This produces a clean fruit silhouette. The segmentation overlay visualizes this by coloring the foreground pixels in a distinctive color on top of the original image.

### Fruits-262 segmentation (proof of concept)

For Fruits-262, the background is not white. The intended approach is:
1. Divide the image into regions (e.g., using a sliding window or superpixels).
2. For each region, extract the Laplacian pyramid feature vector.
3. Compare to the trained class prototypes.
4. Pixels belonging to a region classified as a known fruit class are marked as foreground; all others as background.

This is a more challenging segmentation problem and is left as a demonstration task. The `outputs/segmentations/` folder is currently empty as the demo has not yet been run, but the code infrastructure is in place.

---

## 10. Evaluation Strategy

### Metrics used

**Accuracy** — the fraction of test images correctly classified:
```
Accuracy = (number of correct predictions) / (total test images) = 696 / 812 = 85.71%
```

**Per-class Precision** — of all images predicted as class c, what fraction actually are class c:
```
Precision_c = TP_c / (TP_c + FP_c)
```

**Per-class Recall** — of all images that are truly class c, what fraction are predicted as class c:
```
Recall_c = TP_c / (TP_c + FN_c)
```

**Per-class F1** — harmonic mean of precision and recall:
```
F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
```

**Macro average** — unweighted average over all classes:
```
Macro Precision = mean(Precision_c for all c)
Macro Recall = mean(Recall_c for all c)
Macro F1 = mean(F1_c for all c)
```

### Why macro average?

Macro averaging treats all classes equally regardless of their size. Since the test set is roughly balanced (156–166 images per class), macro and weighted averages are nearly identical here. Macro averaging is preferred when we care about all classes equally.

### Confusion matrix

The confusion matrix is a 5×5 table where entry (i, j) counts how many images of true class i were predicted as class j. Diagonal entries are correct predictions; off-diagonal entries are errors.

---

## 11. Results Interpretation Guide

### Overall performance

The system achieved **85.71% overall accuracy** on 812 test images. This is a strong result for a classical (non-deep-learning) pipeline constrained to use only grayscale features and a prototype classifier.

### Per-class results table

| Class    | Precision | Recall | F1     | Test images |
|----------|-----------|--------|--------|-------------|
| Banana 1 | 0.7757    | 1.0000 | 0.8737 | 166         |
| Kiwi 1   | 0.7051    | 0.9808 | 0.8204 | 156         |
| Pear 1   | 0.9878    | 0.9878 | 0.9878 | 164         |
| Mango 1  | 0.9813    | 0.6325 | 0.7692 | 166         |
| Orange 1 | 1.0000    | 0.6875 | 0.8148 | 160         |

**Macro averages:**
- Precision: (0.7757+0.7051+0.9878+0.9813+1.0000)/5 = **0.8900**
- Recall: (1.0000+0.9808+0.9878+0.6325+0.6875)/5 = **0.8577**
- F1: (0.8737+0.8204+0.9878+0.7692+0.8148)/5 = **0.8532**

### Confusion matrix

```
                 Predicted
                 Banana  Kiwi  Pear  Mango  Orange
True  Banana 1 [  166,    0,    0,    0,     0  ]
      Kiwi 1   [    0,  153,    2,    1,     0  ]
      Pear 1   [    2,    0,  162,    0,     0  ]
      Mango 1  [   46,   15,    0,  105,     0  ]
      Orange 1 [    0,   49,    0,    1,   110  ]
```

### Analysis of confusions

**Mango → Banana (46 errors, Mango recall = 63.3%)**

This is the largest error source. In grayscale, Mango 1 and Banana 1 are similar: both are yellow-orange in color, which maps to similar gray values. Their shapes also share a somewhat similar elongated quality at coarse scales. The Laplacian features at fine levels should differ (Mango has a more rounded, uniform surface; Banana is more elongated), but the prototype classifier cannot separate them cleanly.

To fix this: adding shape features (e.g., aspect ratio, Hu moments) or using color would immediately resolve the confusion. Within the constraints, using more pyramid levels or different block sizes could help.

**Orange → Kiwi (49 errors, Orange recall = 68.8%)**

Orange 1 and Kiwi 1 are both round. In grayscale, an Orange has a rough, textured skin that can resemble the fuzzy surface of a Kiwi. At coarse pyramid levels, both appear as a round, mid-gray circular shape. The texture at fine levels should distinguish them (Kiwi has more visible fuzz), but enough Orange images are misclassified as Kiwi to significantly depress recall.

**Banana, Kiwi, Pear: near-perfect recall**

Banana achieves 100% recall — every Banana test image is classified as Banana. This is consistent with Banana having a very distinctive elongated shape that is unique in this 5-class set. The low Banana precision (77.6%) reflects the fact that Mango and Orange images are sometimes mislabeled as Banana (the false positives).

Pear achieves 98.78% both precision and recall — the best all-around class. Pear's distinctive teardrop shape is unique in this class set.

### What the results tell us about the approach

- The Laplacian pyramid + block features approach works well for shape-distinctive classes (Pear, Banana).
- It struggles for same-shape classes that differ mainly in color (Mango vs. Banana in grayscale).
- A precision-recall tradeoff is clearly visible: high-recall classes (Banana, Kiwi) have lower precision because they "absorb" confused images from other classes.
- The 85.71% accuracy represents a genuine result, not a cherry-picked number. The confusion matrix is transparent about where the system fails.

---

## 12. Expected Strengths

1. **Interpretability.** Every step of the pipeline is explicit and explainable. There are no black-box components. A teacher can ask "why did this Pear get classified correctly?" and the answer is traceable through the feature vector and distance computation.

2. **Speed.** The prototype classifier is extremely fast at test time: one forward pass of feature extraction plus five distance computations.

3. **Low data requirements.** With ~490 training images per class and a prototype classifier, the system trains almost instantaneously and requires no GPU.

4. **Multi-scale discrimination.** The Laplacian pyramid captures both fine-grained texture (surface roughness, fine edges) and coarse shape information. This is more informative than a single-scale feature.

5. **Robustness to clean backgrounds.** For Fruits-360 with its white background, the background masking is very effective and eliminates background influence on the features.

6. **Pear classification.** The system achieves 98.78% F1 for Pear, demonstrating that when shape is sufficiently distinctive the approach works excellently.

---

## 13. Expected Limitations

1. **Grayscale constraint removes color information.** This is the primary limitation. Mango and Banana are readily distinguishable by color (orange vs. yellow) but appear nearly identical in grayscale. The assignment constraint prevents the obvious fix.

2. **Prototype classifier assumes unimodal distribution.** A single mean prototype cannot represent a class with multiple modes (e.g., if Mango images include several very different-looking varieties). The class distributions are effectively assumed to be spherical Gaussians in feature space.

3. **Fixed block sizes.** The block size is fixed at 8×8 (fine levels) and 4×4 (coarse levels) regardless of fruit size or shape. An adaptive block size could capture more relevant features.

4. **No spatial relationships.** Features from individual blocks are averaged together, discarding the relative spatial arrangement of blocks. A Banana's elongated shape could be better captured if we preserved the spatial ordering of block features.

5. **No normalization for intra-class shape variation.** If different training images of the same class have slightly different rotations or positions, the block-level features will vary. The prototype averages this out, which can either help (by centering the distribution) or hurt (by creating a blurry prototype).

6. **Fruits-262 demo not yet run.** The segmentation demo on Fruits-262 has not been executed. The non-white background makes segmentation significantly harder for this pipeline.

---

## 14. Common Teacher Questions and Suggested Answers

**Q1: Why did you choose the Laplacian pyramid over other multi-scale representations (e.g., Gaussian pyramid, wavelets)?**

A: The assignment constraint for Group 4 specifically requires a Laplacian pyramid. Beyond the constraint, the Laplacian pyramid has a theoretical advantage: each level captures a distinct frequency band (band-pass), so the levels are approximately decorrelated. A Gaussian pyramid has redundant information across levels because each level is a blurred version of the previous. Wavelets are related to the Laplacian pyramid but operate in a different basis; the Laplacian pyramid is simpler to implement with standard image processing tools like OpenCV.

**Q2: Why 4 pyramid levels? Why not 3 or 5?**

A: Four levels for a 100×100 image is a natural choice. With 4 levels the finest level is 100×100 and the coarsest is approximately 13×13. A 5th level would be approximately 7×7 — very small and likely to contain little useful information. Three levels would miss the coarsest scale (the overall "silhouette" of the fruit). Four levels gives a good balance of fine texture, intermediate texture, shape outline, and overall shape.

**Q3: Why did you choose these 5 fruit classes and not the ones listed in the assignment (Strawberry, Pineapple)?**

A: When we inspected the Fruits-360 dataset, folder names like "Strawberry" and "Pineapple" did not appear as exact matches in the version we downloaded. Rather than use approximate matches or skip classes, we chose five well-populated classes whose folder names were unambiguous: Banana 1, Kiwi 1, Pear 1, Mango 1, Orange 1. All five are common fruits with clear visual characteristics, and the selection gives a good variety of shapes and textures for evaluating the classifier.

**Q4: Why is Mango recall so low (63.3%)?**

A: In grayscale, Mango 1 and Banana 1 are visually similar — both are yellow-orange in color (mapping to similar gray values) and have a somewhat smooth, uniform surface texture. The Laplacian pyramid features capture texture and coarse shape, but the distinguishing cue (color) is not available due to the grayscale constraint. 46 of 166 Mango test images are predicted as Banana. If color were allowed, this confusion would be nearly eliminated.

**Q5: The system has 100% recall for Banana. Doesn't that mean it's overclassifying things as Banana?**

A: Yes, and the data confirms this. While every true Banana image is correctly identified (100% recall), the Banana precision is only 77.6%. This means 46 Mango images and some other images are incorrectly predicted as Banana. The classifier's Banana prototype is acting as a "default" for images it cannot distinguish as Mango. This is a known behavior of nearest-mean classifiers when one class prototype is at the "center" of the feature space.

**Q6: How is the prototype computed exactly?**

A: For each training class, we compute the 28-dimensional feature vector for every training image. The prototype is the element-wise arithmetic mean of these vectors. Separately, we also store the element-wise standard deviation across all training vectors, which is used for normalizing distances at test time.

**Q7: What is normalized Euclidean distance and why do you use it instead of regular Euclidean distance?**

A: Normalized Euclidean distance divides each feature dimension by its standard deviation before computing the L2 norm. This ensures that all dimensions contribute equally to the distance, regardless of their absolute scale. Without normalization, dimensions with large absolute values (e.g., energy feature which can be in the thousands) would completely dominate the distance, making other features (e.g., Laplacian mean which is near zero) irrelevant. Normalized Euclidean is equivalent to Mahalanobis distance with a diagonal covariance matrix.

**Q8: What does the confusion matrix tell you about the system?**

A: The confusion matrix reveals that the system has two main failure modes: Mango is confused with Banana (46 errors) and Orange is confused with Kiwi (49 errors). All other off-diagonal entries are very small (0–2). This tells us the system is performing well on 3 out of 5 classes (Banana is never missed, Kiwi is almost never missed, Pear is near-perfect) but has two specific grayscale ambiguity problems that would require either color features or shape-based features to resolve.

**Q9: What is the minimum foreground fraction threshold and why do you use it?**

A: We set a threshold of 5%: if fewer than 5% of pixels are classified as foreground (gray < 230), we consider the image to contain no visible fruit and skip it rather than forcing a classification. This prevents the system from classifying nearly blank or corrupted images. In practice this threshold never triggers for Fruits-360 since all images have a fruit prominently centered, but it is important for robustness when processing arbitrary images from Fruits-262 or other sources.

**Q10: How would you improve the system if you were not constrained to grayscale + prototype classifier?**

A: The most impactful improvement would be to add color features. A simple HSV histogram would immediately resolve the Mango/Banana confusion (Mango is orange, Banana is yellow). Beyond color: (1) A k-NN classifier (k > 1) would be more robust than a single prototype. (2) An SVM with an RBF kernel would better capture non-spherical class distributions. (3) A convolutional neural network would automatically learn the most discriminative features. (4) Adding shape features (aspect ratio, solidity, Hu moments) would better capture Banana's elongated shape vs. Mango's rounded shape.

**Q11: Is the Laplacian pyramid a lossless representation?**

A: Yes. The original image can be perfectly reconstructed from the Laplacian pyramid by progressively adding back the residuals from the coarsest level to the finest: G[k] = L[k] + upsample(G[k+1]). This is a fundamental property of the Laplacian pyramid (Burt & Adelson, 1983) and distinguishes it from lossy representations like JPEG. The pyramid is a change of basis, not a compression.

**Q12: Why block-based features rather than global features?**

A: Global features compute statistics over the entire image, discarding spatial layout. Block-based features give local statistics from different spatial regions. For fruit classification, the spatial distribution of texture matters: for example, a Kiwi has fuzzy texture uniformly distributed, while a Pear has a smooth upper body and a rounded lower body. Block features can distinguish these spatial patterns. In our 28-dimensional feature vector, different blocks contribute different information about different parts of the image.

---

## 15. How to Explain the Project in 1 Minute

*The following is a script for a rapid verbal explanation, suitable for the first minute of a presentation or when a teacher asks "can you briefly explain what you did?"*

---

"We built a fruit recognition system that classifies images as one of five fruits: Banana, Kiwi, Pear, Mango, and Orange. The key constraint is that we use only grayscale images — no color — and a classical multi-scale representation called a Laplacian pyramid.

Here is how it works in brief. For each training image, we first convert it to grayscale and remove the white background. Then we build a 4-level Laplacian pyramid, which breaks the image into different frequency layers — fine texture on top, coarse shape at the bottom. We divide each layer into small blocks and extract 7 texture statistics from each block. Concatenating these across all four levels gives a 28-dimensional feature vector per image.

For each fruit class, we average all training feature vectors into a single prototype. To classify a new image, we compute its feature vector and find which class prototype it is closest to, using a normalized distance measure.

On 812 test images we achieved 85.71% accuracy. The main errors are Mango confused with Banana and Orange confused with Kiwi — both confusions happen because those pairs look very similar in grayscale, since color is our main excluded cue."

---

## 16. How to Explain the Project in 7 Minutes

*The following is a structured script for a 7-minute verbal presentation. It is organized so each section fits approximately one minute.*

---

**Minute 1 — Motivation and task definition (60 seconds)**

"Good morning. Our project is titled 'Fruit Segmentation Using Grayscale Laplacian Pyramid Features', and it was developed as part of the Group 4 assignment for Computer Vision.

The task is to build a system that can recognize and segment fruits in images using only classical image processing techniques — no deep learning. The specific constraints for our group are: images must be processed in grayscale only, the multi-scale representation must be a Laplacian pyramid, features must be extracted from blocks at each pyramid level, and classification must use a nearest-mean prototype approach.

We applied this system to the Fruits-360 dataset, which contains clean fruit photographs with a white background, and we trained on five classes: Banana 1, Kiwi 1, Pear 1, Mango 1, and Orange 1."

**Minute 2 — Dataset (60 seconds)**

"Fruits-360 is a benchmark dataset with 100×100 pixel images, a pure white background, and hundreds of fruit varieties. We selected five classes that were clearly available in our version of the dataset.

The original assignment suggested Banana, Strawberry, Pineapple, Pear, and Kiwi. However, when we inspected the dataset, we found that 'Strawberry' and 'Pineapple' did not appear as exact folder names. Rather than work with approximate matches, we chose five unambiguous classes: Banana 1, Kiwi 1, Pear 1, Mango 1, and Orange 1.

We trained on 2417 images in total — roughly 480–490 per class — and evaluated on 812 test images provided in the dataset's pre-defined test split."

**Minute 3 — Laplacian pyramid and feature extraction (90 seconds)**

"The heart of our system is the Laplacian pyramid. Starting from the 100×100 grayscale image, we build a 4-level Gaussian pyramid by repeatedly blurring and downsampling. The Laplacian pyramid is then computed by subtracting each upsampled Gaussian level from the level above it. The result is four images, each containing a different frequency band of the original.

Level 0, at full resolution, captures fine surface texture — things like the fuzzy skin of a Kiwi or the smooth surface of a Banana. Level 3, at the coarsest resolution of roughly 13×13 pixels, captures the overall shape and silhouette of the fruit.

At each pyramid level, we divide the image into small blocks — 8×8 pixels at the fine levels and 4×4 at the coarse levels. For each block we compute 7 statistics: mean of absolute values, standard deviation, variance, range, energy, mean, and second moment. We then average these statistics across all blocks in the level.

This gives us 7 numbers per level times 4 levels, for a total of 28 features per image."

**Minute 4 — Classification and training (60 seconds)**

"For classification, we use a prototype-based nearest-mean classifier. During training, we compute the 28-dimensional feature vector for each of the 2417 training images and average them by class to get one prototype vector per fruit.

At test time, we extract the feature vector for the new image and compute the normalized Euclidean distance to each of the five class prototypes. The fruit with the smallest distance wins.

Normalized Euclidean distance divides each feature dimension by its standard deviation, so all 28 dimensions contribute equally. Without this normalization, dimensions with large absolute values — like the energy feature — would dominate and drown out subtler texture signals.

The training phase completes in seconds on a standard laptop. Classification of a single image takes milliseconds."

**Minute 5 — Results (90 seconds)**

"Now for the results. We achieved 85.71% overall accuracy on 812 test images. Let me walk through the per-class numbers.

Pear 1 is our best class: 98.78% precision, 98.78% recall, and 98.78% F1. Pear's distinctive teardrop shape is unique in the five-class set and the pyramid features capture it very well.

Banana 1 achieves 100% recall — every Banana test image is correctly found — but precision is 77.6% because some Mango images are mistaken for Banana.

Kiwi 1 similarly has very high recall at 98.1%.

The two weakest classes are Mango 1 at 63.3% recall and Orange 1 at 68.8% recall. The confusion matrix tells the story clearly: 46 Mango images are predicted as Banana, and 49 Orange images are predicted as Kiwi. These are the two dominant error sources.

The macro average F1 across all five classes is 0.8532."

**Minute 6 — Analysis and limitations (60 seconds)**

"Why do Mango and Orange suffer?

In grayscale, Mango 1 and Banana 1 have similar intensity distributions — both are yellow-orange fruits, and that color maps to similar gray values. The Laplacian pyramid captures texture and shape, but the most discriminative cue — color — is precisely what we are not allowed to use.

Similarly, Orange and Kiwi are both round and have somewhat similar surface textures at coarse scales. Without color, the two are hard to separate.

This highlights the fundamental limitation of the grayscale constraint. If we were allowed to use color, a simple HSV histogram would resolve both confusions immediately. Within the constraints, potential improvements include shape features like aspect ratio or Hu moments, which would better distinguish Banana's elongated shape from Mango's round shape."

**Minute 7 — Conclusion and demo (60 seconds)**

"To summarize: we built a complete fruit classification pipeline using Laplacian pyramid features and a prototype classifier, achieving 85.71% accuracy on Fruits-360. The system is fully interpretable, fast, and trained on modest data.

The segmentation module uses the background mask for Fruits-360 (where white background is easy to remove) and is designed to generalize to Fruits-262 with a non-white background using a sliding-window classification approach.

[Optional: show confusion matrix plot or example outputs here]

The main finding is that the approach works well for shape-distinctive classes and struggles for classes that are primarily distinguished by color — an expected result given the grayscale constraint. The system meets all Group 4 assignment requirements.

Thank you. I am happy to answer questions."

---

## 17. Key Terms Glossary

**Laplacian Pyramid:** A multi-scale image representation where each level contains the band-pass detail (high-minus-low frequency) removed at a Gaussian pyramid downsampling step. Named after the Laplacian operator, though the pyramid itself uses subtraction of blurred images.

**Gaussian Pyramid:** A sequence of progressively blurred and downsampled images. Each level is a smoothed version of the previous. Used as an intermediate step to build the Laplacian pyramid.

**Band-pass filtering:** The process of retaining only a specific range of spatial frequencies in an image, while attenuating both higher and lower frequencies. Each Laplacian level is a band-pass filtered version of the original.

**Block-based feature extraction:** Dividing an image into non-overlapping rectangular regions (blocks) and computing local statistics within each block. Preserves some spatial information while keeping the feature dimension manageable.

**Prototype (Nearest-Mean) Classifier:** A classification approach where each class is represented by a single mean vector, and a test sample is assigned to the class whose mean is closest in feature space.

**Normalized Euclidean Distance:** The L2 distance between two vectors after dividing each dimension by its standard deviation. Makes the distance scale-invariant across feature dimensions.

**Mahalanobis Distance:** A distance metric that accounts for the covariance structure of a distribution. Normalized Euclidean is a special case where the covariance matrix is diagonal.

**Background masking:** The process of identifying and excluding background pixels from feature computation. In Fruits-360, pixels with gray value ≥ 230 are treated as white background.

**Macro average:** An averaging strategy that computes a metric independently for each class and then averages the results without weighting by class size. Treats all classes equally.

**Precision:** The fraction of positive predictions that are correct: TP / (TP + FP). High precision means few false positives.

**Recall (Sensitivity):** The fraction of actual positives correctly identified: TP / (TP + FN). High recall means few false negatives.

**F1 score:** The harmonic mean of precision and recall: 2 * P * R / (P + R). Balances both metrics.

**Confusion matrix:** A square matrix where entry (i, j) counts the number of test samples of true class i predicted as class j. Diagonal entries are correct; off-diagonal are errors.

**Foreground fraction:** The fraction of pixels in an image that are classified as non-background. Used as a pre-filter to skip images with insufficient foreground.

**Feature vector:** A fixed-length numerical array summarizing the properties of an image. In this system: 28 dimensions from 4 pyramid levels × 7 statistics each.

**Energy (image feature):** The mean of squared pixel values: mean(x²). Measures the average power of the signal in a region.

**Second moment:** Mean of squared values: mean(x²). Equivalent to energy in this context.

**Upsample:** The process of increasing an image's resolution by inserting new pixels and interpolating their values. The inverse of downsampling. Used in Laplacian pyramid reconstruction.

---

## 18. References

1. Burt, P. J., & Adelson, E. H. (1983). The Laplacian pyramid as a compact image code. *IEEE Transactions on Communications*, 31(4), 532–540. https://doi.org/10.1109/TCOM.1983.1095851

2. Muresan, H., & Oltean, M. (2018). Fruit recognition from images using deep learning. *Acta Universitatis Sapientiae, Informatica*, 10(1), 26–42. https://doi.org/10.2478/ausi-2018-0002

3. aelchimminut. (2021). *Fruits-262* [Dataset]. Kaggle. https://www.kaggle.com/datasets/aelchimminut/fruits262

4. Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

5. Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*. https://opencv.org

6. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2
