import argparse
import os

import cv2


def sample_pixels(gray, margin, map_scale):
    height, width = gray.shape
    samples = []

    for y in range(0, height, margin):
        for x in range(0, width, margin):
            terrain_height = (float(gray[y, x]) / 255.0) * map_scale
            samples.append((x / margin, terrain_height, y / margin))

    return samples


def build_triangles(samples, width, height, margin):
    """
    samples: list of (x, y, r) in row-major order
    width, height: original image dimensions
    margin: sampling step

    returns: list of triangles (each triangle is a tuple of 3 indices)
    """

    cols = (width + margin - 1) // margin
    rows = (height + margin - 1) // margin

    def idx(x_idx, y_idx):
        return y_idx * cols + x_idx

    triangles = []

    for y in range(rows - 1):
        for x in range(cols - 1):
            i0 = idx(x, y)
            i1 = idx(x + 1, y)
            i2 = idx(x, y + 1)
            i3 = idx(x + 1, y + 1)

            triangles.append((i0, i1, i2))
            triangles.append((i1, i3, i2))

    return triangles


def main(image_path, margin, map_scale, output_path, blur_sigma=0.0):
    margin = int(margin)
    map_scale = float(map_scale)
    blur_sigma = float(blur_sigma)

    if margin <= 0:
        raise ValueError("margin must be greater than 0")

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Failed to load height map image from '{image_path}'")

    if blur_sigma > 0:
        gray = cv2.GaussianBlur(
            gray, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    height, width = gray.shape
    samples = sample_pixels(gray, margin, map_scale)
    tris = build_triangles(samples, width, height, margin)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        for x, y, z in samples:
            f.write(f"{x},{y},{z}\n")

        for tri in tris:
            f.write(f"v{tri[0]},v{tri[1]},v{tri[2]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a .tri terrain mesh from a grayscale height map.")
    parser.add_argument("image_path")
    parser.add_argument("output_path")
    parser.add_argument("--margin", type=int, required=True)
    parser.add_argument("--map-scale", type=float, required=True)
    parser.add_argument("--blur-sigma", type=float, default=0.0)
    args = parser.parse_args()

    main(
        image_path=args.image_path,
        margin=args.margin,
        map_scale=args.map_scale,
        output_path=args.output_path,
        blur_sigma=args.blur_sigma,
    )
