"""
blob_finder.py
Finds the center(s) of color blobs in an image.

Dependencies:
    pip install Pillow scipy numpy
"""

from PIL import Image
import numpy as np
from scipy import ndimage
from typing import Sequence


def find_blob_centers(
    image_path: str,
    colors: Sequence[tuple[int, int, int]],
    tolerance: int = 20,
    min_blob_size: int = 50,
) -> dict[tuple[int, int, int], list[tuple[float, float]]]:
    """
    Find the centers of color blobs in an image.

    Args:
        image_path:    Path to the image file.
        colors:        List of RGB tuples to search for, e.g. [(255, 0, 0), (0, 255, 0)].
        tolerance:     Max Euclidean distance in RGB space for a pixel to match a color.
        min_blob_size: Minimum number of pixels for a region to count as a blob.

    Returns:
        A dict mapping each input color to a list of (row, col) centroid coordinates.
        If no blobs are found for a color, its list is empty.

    Example:
        >>> centers = find_blob_centers(
        ...     "photo.png",
        ...     colors=[(255, 0, 0), (0, 0, 255)],
        ...     tolerance=30,
        ...     min_blob_size=100,
        ... )
        >>> for color, centroids in centers.items():
        ...     print(f"Color {color}: {centroids}")
    """
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img, dtype=np.int32)  # shape: (H, W, 3)

    results: dict[tuple[int, int, int], list[tuple[float, float]]] = {}

    for color in colors:
        target = np.array(color, dtype=np.int32)

        # Euclidean distance from each pixel to the target color
        diff = pixels - target                          # (H, W, 3)
        dist = np.sqrt((diff ** 2).sum(axis=2))        # (H, W)
        mask = dist <= tolerance                        # boolean mask

        # Label connected regions (8-connectivity)
        struct = ndimage.generate_binary_structure(2, 2)
        labeled, num_features = ndimage.label(mask, structure=struct)

        centroids: list[tuple[float, float]] = []
        for label_id in range(1, num_features + 1):
            blob = labeled == label_id
            size = blob.sum()
            if size < min_blob_size:
                continue
            cy, cx = ndimage.center_of_mass(blob)
            centroids.append((round(cy, 2), round(cx, 2)))

        results[tuple(color)] = centroids  # type: ignore[index]

    return results


# ---------------------------------------------------------------------------
# Quick demo — run as a script to test on any image
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python blob_finder.py <image_path> [r,g,b ...]")
        print("Example: python blob_finder.py photo.png 255,0,0 0,255,0")
        sys.exit(1)

    path = sys.argv[1]
    raw_colors = sys.argv[2:] or ["255,0,0"]  # default: red
    color_list = [tuple(int(c) for c in s.split(",")) for s in raw_colors]

    centers = find_blob_centers(path, color_list)  # type: ignore[arg-type]

    for color, blobs in centers.items():
        if blobs:
            print(f"Color {color}: {len(blobs)} blob(s)")
            for i, (row, col) in enumerate(blobs, 1):
                print(f"  Blob {i}: center = row {row}, col {col}")
        else:
            print(f"Color {color}: no blobs found")
