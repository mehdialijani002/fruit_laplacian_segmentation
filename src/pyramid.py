import cv2
import numpy as np


def gaussian_downsample(image: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
 
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    h, w = blurred.shape[:2]
    downsampled = cv2.resize(blurred, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_LINEAR)
    return downsampled


def gaussian_upsample(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
  
    h, w = target_shape
    upsampled = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return upsampled


def build_gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
   
    pyramid = [image.astype(np.float32)]
    for _ in range(levels - 1):
        pyramid.append(gaussian_downsample(pyramid[-1]))
    return pyramid


def build_laplacian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
  
    gauss = build_gaussian_pyramid(image, levels)
    laplacian: list[np.ndarray] = []
    for k in range(levels - 1):
        upsampled = gaussian_upsample(gauss[k + 1], target_shape=gauss[k].shape[:2])
        lap = gauss[k].astype(np.float32) - upsampled.astype(np.float32)
        laplacian.append(lap)
    laplacian.append(gauss[-1].astype(np.float32))  # residual
    return laplacian
